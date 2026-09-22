"""
LLM 访问层：可切换 provider + 磁盘缓存。

设计动机(面试讲解点)
--------------------
把 LLM 调用封在一层后面, 换来四件事:

1. **演示确定性**。所有调用按 hash(model+prompt+params) 落盘。
   面试现场用 GRAPHRAG_LLM=replay, 零网络、零调用、每次结果完全一致 ——
   演示要的是可复现, 不是开盲盒。
2. **评测可复现**。缓存命中即确定性, 同一份 prompt 的评测分数不会随机漂移,
   于是"这一版 prompt 涨了 3 分"才是真的涨了, 而不是采样噪声。
3. **成本可控**。消融实验会把同一批 prompt 反复跑十几轮, 没有缓存就是十几倍开销。
4. **provider 可换**。Claude / 本地 Ollama / 纯回放, 改一个环境变量。

环境变量
--------
    GRAPHRAG_LLM     claude | deepseek | ollama | replay
                     (默认 claude; 缺对应 key 时自动降级 replay)
    GRAPHRAG_CACHE   缓存目录, 默认 <repo>/data/llm_cache
    ANTHROPIC_API_KEY
    DEEPSEEK_API_KEY
    DEEPSEEK_MODEL   默认 deepseek-chat
    OLLAMA_HOST      默认 http://127.0.0.1:11434
    OLLAMA_MODEL     默认 qwen2.5:7b-instruct

能力差异(直接影响架构, 不是口味问题)
------------------------------------
                    Claude          DeepSeek / Ollama
  严格 schema 约束   原生支持        只有 JSON mode -> 需自己做校验+重试
  字符级 citations   原生支持        无 -> 需自己做"标记+回溯对齐"
  成本              基准            约 1/20

  因此 generate/ 下同时实现两套引用机制: Claude 走原生 citations,
  其余 provider 走"标记+span 对齐"。两者在同一套评测上对比,
  差多少分是实测出来的 —— 这本身就是一条评测结论。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compat import project_root, write_text

# 默认模型。抽取类任务量大且不难 -> Haiku; 生成类要质量 -> Opus。
MODEL_EXTRACT = os.environ.get("GRAPHRAG_MODEL_EXTRACT", "claude-haiku-4-5")
MODEL_ANSWER = os.environ.get("GRAPHRAG_MODEL_ANSWER", "claude-opus-5")

# 近似计价 (USD / 1M token), 只用于把成本打印出来, 不作账务用途
# 近似值, 仅用于把成本量级打印出来; 以各家官网为准, 不作账务用途。
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # DeepSeek 便宜约一到两个数量级, 且价格调整频繁 —— 这里只作量级参考
    "deepseek-chat": (0.3, 1.2),
    "deepseek-reasoner": (0.6, 2.4),
}

CAPABILITIES: dict[str, dict[str, bool]] = {
    # provider -> 能力位。上层据此选择实现路径, 而不是靠 if provider == "claude"
    "claude":   {"strict_schema": True,  "native_citations": True},
    "deepseek": {"strict_schema": False, "native_citations": False},
    "ollama":   {"strict_schema": False, "native_citations": False},
    "replay":   {"strict_schema": True,  "native_citations": True},
}


@dataclass
class LLMResult:
    text: str
    parsed: Any = None                 # 结构化输出时的 JSON 对象
    citations: list[dict] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_read_tokens: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False
    provider: str = ""

    @property
    def cost_usd(self) -> float:
        if self.from_cache:
            return 0.0
        pin, pout = PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1_000_000


class CacheMissError(RuntimeError):
    """replay 模式下缓存未命中 —— 说明演示前的预跑没覆盖到这个调用。"""


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------

class DiskCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or os.environ.get("GRAPHRAG_CACHE")
                         or project_root() / "data" / "llm_cache")
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        # 两级目录, 避免单目录下上万个文件
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResult | None:
        p = self.path(key)
        if not p.exists():
            self.misses += 1
            return None
        self.hits += 1
        raw = json.loads(p.read_text(encoding="utf-8"))
        return LLMResult(**{**raw, "from_cache": True})

    def put(self, key: str, res: LLMResult) -> None:
        d = {k: v for k, v in res.__dict__.items() if k != "from_cache"}
        write_text(self.path(key), json.dumps(d, ensure_ascii=False, indent=1))

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

class BaseProvider:
    name = "base"

    def resolve_model(self, requested: str) -> str:
        """把上层请求的模型名映射成本 provider 实际会用的模型。

        上层按"角色"请求模型(抽取用便宜的、生成用强的), 用的是 Claude 的模型名。
        DeepSeek/Ollama 会忽略它、用自己的模型 —— 如果不在这里归一,
        缓存键里记的是 claude-haiku、实际跑的是 deepseek-chat, 计价也会按错的表算。
        这类不一致不会报错, 只会让成本统计和缓存命中率悄悄失真。
        """
        return requested

    def complete(self, *, system: str, user: str, model: str,
                 max_tokens: int, schema: dict | None,
                 documents: list[dict] | None) -> LLMResult:
        raise NotImplementedError


class ReplayProvider(BaseProvider):
    """只读缓存。面试现场用这个: 零网络、零花费、结果完全确定。"""
    name = "replay"

    def complete(self, **kw) -> LLMResult:
        raise CacheMissError(
            "replay 模式下缓存未命中。演示前请先用 GRAPHRAG_LLM=claude 预跑一遍，"
            "把结果写进 data/llm_cache/。")


class ClaudeProvider(BaseProvider):
    name = "claude"

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as e:                                # pragma: no cover
            raise RuntimeError("需要 anthropic SDK: pip install anthropic") from e
        self._client = anthropic.Anthropic()

    def complete(self, *, system: str, user: str, model: str,
                 max_tokens: int, schema: dict | None,
                 documents: list[dict] | None) -> LLMResult:
        content: list[dict] = []

        # 把检索到的 chunk 作为 document block 传入并开启 citations,
        # 回答会自带 cited_text 与字符级位置 —— 溯源不必自己实现。
        for d in documents or []:
            content.append({
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain",
                           "data": d["text"]},
                "title": d.get("title") or d.get("id", ""),
                "context": d.get("context", ""),
                "citations": {"enabled": True},
            })
        content.append({"type": "text", "text": user})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
        }
        if schema is not None:
            # 结构化输出与 citations 互斥(同时传会 400), 由调用方保证只用其一
            kwargs["output_config"] = {"format": {"type": "json_schema",
                                                  "schema": schema}}

        t0 = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000

        if resp.stop_reason == "refusal":
            raise RuntimeError(f"模型拒绝生成: {getattr(resp, 'stop_details', None)}")

        texts: list[str] = []
        cites: list[dict] = []
        for block in resp.content:
            if block.type != "text":
                continue
            texts.append(block.text)
            for c in (getattr(block, "citations", None) or []):
                cites.append({
                    "cited_text": getattr(c, "cited_text", ""),
                    "document_index": getattr(c, "document_index", None),
                    "document_title": getattr(c, "document_title", None),
                    "start": getattr(c, "start_char_index", None),
                    "end": getattr(c, "end_char_index", None),
                })
        text = "".join(texts)

        parsed = None
        if schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

        u = resp.usage
        return LLMResult(
            text=text, parsed=parsed, citations=cites, model=model,
            input_tokens=getattr(u, "input_tokens", 0),
            output_tokens=getattr(u, "output_tokens", 0),
            cached_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            latency_ms=latency, provider=self.name)


class OllamaProvider(BaseProvider):
    """本地模型, 完全离线免费。质量弱于 Claude, 但抽取类任务够用。

    不引入额外依赖 —— 直接打 Ollama 的 HTTP 接口(标准库 urllib)。
    """
    name = "ollama"

    def __init__(self) -> None:
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    def resolve_model(self, requested: str) -> str:
        return self.model

    def complete(self, *, system: str, user: str, model: str,
                 max_tokens: int, schema: dict | None,
                 documents: list[dict] | None) -> LLMResult:
        import urllib.request

        # Ollama 没有 document block, 把资料直接拼进 user 消息
        if documents:
            refs = "\n\n".join(
                f"[{i + 1}]（{d.get('title') or d.get('id', '')}）\n{d['text']}"
                for i, d in enumerate(documents))
            user = f"参考资料：\n{refs}\n\n{user}"

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        if schema is not None:
            body["format"] = schema          # Ollama 的结构化输出

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))
        latency = (time.perf_counter() - t0) * 1000

        text = data.get("message", {}).get("content", "")
        parsed = None
        if schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        return LLMResult(
            text=text, parsed=parsed, model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency, provider=self.name)


class DeepSeekProvider(BaseProvider):
    """DeepSeek API。OpenAI 兼容接口。

    为什么用 urllib 而不是 openai SDK:
      请求体就是一个扁平 JSON, 非流式, 无需重试策略以外的东西。
      为省 30 行代码引入一个 SDK, 会让"零依赖兜底"这条路径失效 ——
      本项目的约束是"搬到别人的 Windows 上要能跑", 依赖越少越好。
      若将来需要流式/工具调用, 再换 openai SDK 不迟。

    两个能力缺口(已在 CAPABILITIES 里声明, 由上层处理):
      · 无严格 schema 约束 -> 只能开 JSON mode, 由调用方做 schema 校验与重试
      · 无原生 citations   -> 由 generate/ 的"标记+span 对齐"兜底
    """
    name = "deepseek"
    BASE = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self) -> None:
        self.key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        if not self.key:
            raise RuntimeError("需要 DEEPSEEK_API_KEY")

    def resolve_model(self, requested: str) -> str:
        return self.model

    def complete(self, *, system: str, user: str, model: str,
                 max_tokens: int, schema: dict | None,
                 documents: list[dict] | None) -> LLMResult:
        import urllib.error
        import urllib.request

        if documents:
            refs = "\n\n".join(
                f"[{i + 1}]（{d.get('title') or d.get('id', '')}）\n{d['text']}"
                for i, d in enumerate(documents))
            user = f"参考资料：\n{refs}\n\n{user}"

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,        # 评测要可复现, 一律贪心解码
            "stream": False,
        }
        if schema is not None:
            # 只有 JSON mode, 不保证符合 schema。schema 本身塞进 prompt,
            # 结构校验与重试由调用方(ingest/extract.py)负责。
            body["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            self.BASE,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})

        t0 = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(3):                    # 429/5xx 退避重试
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"DeepSeek HTTP {e.code}: "
                                   f"{e.read().decode('utf-8', 'replace')[:200]}") from e
        else:                                       # pragma: no cover
            raise RuntimeError(f"DeepSeek 重试耗尽: {last_err}")
        latency = (time.perf_counter() - t0) * 1000

        text = data["choices"][0]["message"]["content"] or ""
        parsed = None
        if schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None       # 交由上层做修复重试 —— 这是无严格 schema 的代价

        u = data.get("usage", {})
        return LLMResult(
            text=text, parsed=parsed, model=self.model,
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            cached_read_tokens=u.get("prompt_cache_hit_tokens", 0) or 0,
            latency_ms=latency, provider=self.name)


# --------------------------------------------------------------------------
# 门面
# --------------------------------------------------------------------------

class LLM:
    """统一入口。缓存在 provider 之前 —— 命中就不产生任何调用与费用。"""

    def __init__(self, provider: str | None = None, cache: DiskCache | None = None):
        self.cache = cache or DiskCache()
        name = (provider or os.environ.get("GRAPHRAG_LLM") or "claude").lower()
        # 没有对应 key 时不报错, 降级为只读回放 —— 别人拿到仓库也能跑通演示
        need_key = {"claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                    "deepseek": ("DEEPSEEK_API_KEY",)}
        if name in need_key and not any(os.environ.get(k) for k in need_key[name]):
            name = "replay"
        self.provider_name = name
        self._provider: BaseProvider | None = None
        self.calls = 0
        self.total_cost = 0.0

    @property
    def provider(self) -> BaseProvider:
        if self._provider is None:
            self._provider = {
                "claude": ClaudeProvider,
                "deepseek": DeepSeekProvider,
                "ollama": OllamaProvider,
                "replay": ReplayProvider,
            }[self.provider_name]()
        return self._provider

    def complete(self, *, system: str, user: str,
                 model: str = MODEL_ANSWER, max_tokens: int = 4096,
                 schema: dict | None = None,
                 documents: list[dict] | None = None,
                 cache_tag: str = "") -> LLMResult:
        # 先问 provider "你实际会用哪个模型", 再拿它建缓存键 ——
        # 否则用 DeepSeek 跑时缓存键里记的却是 claude-haiku, 换 provider 会误命中。
        effective = model
        if self.provider_name != "replay":
            try:
                effective = self.provider.resolve_model(model)
            except Exception:
                pass
        key = DiskCache.key({
            "provider": self.provider_name if self.provider_name != "replay" else "claude",
            "model": effective, "system": system, "user": user,
            "schema": schema, "documents": documents, "tag": cache_tag,
        })
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        res = self.provider.complete(system=system, user=user, model=model,
                                     max_tokens=max_tokens, schema=schema,
                                     documents=documents)
        self.cache.put(key, res)
        self.calls += 1
        self.total_cost += res.cost_usd
        return res

    def supports(self, capability: str) -> bool:
        """上层据此选择实现路径, 而不是到处写 if provider == 'claude'。"""
        return CAPABILITIES.get(self.provider_name, {}).get(capability, False)

    def report(self) -> str:
        c = self.cache.stats()
        return (f"provider={self.provider_name}  实际调用={self.calls}  "
                f"缓存命中={c['hits']}/{c['hits'] + c['misses']} "
                f"({c['hit_rate']:.0%})  累计花费≈${self.total_cost:.3f}")
