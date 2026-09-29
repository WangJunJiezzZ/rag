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
    GRAPHRAG_LLM     claude | deepseek | ollama   (真实 provider)
    GRAPHRAG_OFFLINE 1 = 只读缓存, 绝不发起网络调用  <- 面试演示用这个

    GRAPHRAG_LLM=replay 是上面两者的简写, 等价于
    "保持 provider 配置不变, 但强制离线"。

为什么 replay 不是一个 provider
-------------------------------
首版把 replay 做成第四个 provider, 结果它的能力位与真实 provider 不同
(replay 声称支持原生 citations, DeepSeek 不支持), 于是上层构造的请求体不一样,
**缓存键对不上, 用 DeepSeek 跑出来的缓存在 replay 下永远命中不了**。

这个 bug 一直没暴露, 因为从没端到端验证过"用 A 录、用 replay 放"这条路径。

正确的建模: **provider 是"谁来答", offline 是"能不能联网"**, 两者正交。
离线模式下 provider 配置完全不变 —— 能力位、模型名、请求体构造全部一致,
缓存键因此天然匹配。
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

# replay 回放时, 各 provider 实际使用的模型名(缓存键的一部分)
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "deepseek": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    "ollama": os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
}

CAPABILITIES: dict[str, dict[str, bool]] = {
    # provider -> 能力位。上层据此选择实现路径, 而不是靠 if provider == "claude"
    "claude":   {"strict_schema": True,  "native_citations": True},
    "deepseek": {"strict_schema": False, "native_citations": False},
    "ollama":   {"strict_schema": False, "native_citations": False},
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
    alias_hit: bool = False      # 经稳定键兜底命中(检索上下文可能与当前略有出入)
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
        self.alias_hits = 0        # 经稳定键兜底命中的次数

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        # 两级目录, 避免单目录下上万个文件
        return self.root / key[:2] / f"{key}.json"

    def alias_path(self, stable: str) -> Path:
        return self.root / "_alias" / stable[:2] / f"{stable}.txt"

    def put_alias(self, stable: str, primary: str) -> None:
        """语义稳定键 -> 精确键 的指针。

        为什么需要: 精确键 = hash(完整 prompt), 而 prompt 里含检索到的文档。
        检索用到 ONNX 浮点运算, **同一份代码在 macOS 与 Linux 上 top-k 排序
        会有细微差别**(实测: 本地 39/39 命中, 同版本容器只有 29/39)。
        于是"在这台机器录、到那台机器放"就会大面积失效 ——
        这不只影响 Docker 部署, 换一台 Windows 电脑同样会中招。

        稳定键只取**语义身份**(prompt 版本 + 问题 + 图谱来源 + 模型),
        不含检索结果, 因此跨平台一致。
        """
        write_text(self.alias_path(stable), primary)

    def get_by_alias(self, stable: str) -> "LLMResult | None":
        p = self.alias_path(stable)
        if not p.exists():
            return None
        primary = p.read_text(encoding="utf-8").strip()
        return self.get(primary) if primary else None

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
                "alias_hits": self.alias_hits,
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

    NEED_KEY = {"claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                "deepseek": ("DEEPSEEK_API_KEY",)}

    def __init__(self, provider: str | None = None, cache: DiskCache | None = None,
                 offline: bool | None = None):
        self.cache = cache or DiskCache()
        name = (provider or os.environ.get("GRAPHRAG_LLM") or "claude").lower()

        self.offline = (offline if offline is not None
                        else os.environ.get("GRAPHRAG_OFFLINE", "") in ("1", "true", "yes"))

        if name == "replay":
            # 兼容写法: 保持 provider 配置不变, 只是强制离线。
            # 选一个"有凭据"的 provider 作为影子, 这样能力位和模型名
            # 与当初录制时完全一致, 缓存键才对得上。
            self.offline = True
            name = next((p for p in ("deepseek", "claude")
                         if any(os.environ.get(k) for k in self.NEED_KEY[p])),
                        "deepseek")

        # 没凭据且未显式离线 -> 自动转离线, 而不是报错。
        # 别人拿到仓库、一个 key 都没有, 也能靠缓存跑通全部演示。
        if name in self.NEED_KEY and not any(os.environ.get(k)
                                             for k in self.NEED_KEY[name]):
            self.offline = True
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
            }[self.provider_name]()
        return self._provider

    def _model_for_key(self, model: str) -> str:
        """缓存键用的模型名。离线时不实例化 provider(它可能需要 key),
        直接查默认模型表 —— 与在线时 resolve_model() 的结果一致。"""
        if self.offline:
            return PROVIDER_DEFAULT_MODEL.get(self.provider_name, model)
        try:
            return self.provider.resolve_model(model)
        except Exception:
            return model

    def complete(self, *, system: str, user: str,
                 model: str = MODEL_ANSWER, max_tokens: int = 4096,
                 schema: dict | None = None,
                 documents: list[dict] | None = None,
                 cache_tag: str = "",
                 stable_key: str | None = None) -> LLMResult:
        """stable_key: 与检索结果无关的语义身份, 用于跨平台回放兜底。

        在线时**只认精确键** —— prompt 改一个字就该重新调用, 否则评测会读到旧结果。
        离线时精确键未命中才退到稳定键, 并在返回值上标记 `alias_hit`,
        让上层知道这条结果对应的检索上下文可能与当前略有出入。
        """
        effective = self._model_for_key(model)
        key = DiskCache.key({
            "provider": self.provider_name, "model": effective,
            "system": system, "user": user, "schema": schema,
            "documents": documents, "tag": cache_tag,
        })
        hit = self.cache.get(key)
        if hit is not None:
            # 命中精确键时也补写别名 —— 幂等且零成本。
            # 否则别名只会在"未命中→调用"的路径上生成, 而一台已经录满缓存的
            # 机器永远走不到那条路径, 索引就建不起来。
            if stable_key:
                self.cache.put_alias(DiskCache.key({"stable": stable_key}), key)
            return hit
        if self.offline and stable_key:
            # 跨平台兜底: 精确键对不上, 但语义身份一致
            alias = self.cache.get_by_alias(DiskCache.key({"stable": stable_key}))
            if alias is not None:
                self.cache.alias_hits += 1
                alias.alias_hit = True
                return alias
        if self.offline:
            raise CacheMissError(
                f"离线模式(provider={self.provider_name})下缓存未命中。\n"
                f"演示前请先用真实 provider 预跑一遍, 例如:\n"
                f"  GRAPHRAG_LLM={self.provider_name} "
                f"python run.py e2e -- --ab --split train")

        res = self.provider.complete(system=system, user=user, model=model,
                                     max_tokens=max_tokens, schema=schema,
                                     documents=documents)
        self.cache.put(key, res)
        if stable_key:
            self.cache.put_alias(DiskCache.key({"stable": stable_key}), key)
        self.calls += 1
        self.total_cost += res.cost_usd
        return res

    def supports(self, capability: str) -> bool:
        """上层据此选择实现路径, 而不是到处写 if provider == 'claude'。

        **离线时也返回真实 provider 的能力位** —— 否则上层构造的请求体
        与录制时不同, 缓存键对不上。这正是首版 replay 失效的原因。
        """
        return CAPABILITIES.get(self.provider_name, {}).get(capability, False)

    def report(self) -> str:
        c = self.cache.stats()
        mode = "离线" if self.offline else "在线"
        alias = f"  其中稳定键兜底={c['alias_hits']}" if c.get("alias_hits") else ""
        return (f"provider={self.provider_name}({mode})  实际调用={self.calls}  "
                f"缓存命中={c['hits']}/{c['hits'] + c['misses']} "
                f"({c['hit_rate']:.0%}){alias}  累计花费≈${self.total_cost:.3f}")
