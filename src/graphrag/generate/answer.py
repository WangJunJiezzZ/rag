"""
答案生成 —— RAG 的最后一环, 也是幻觉发生的地方。

三件事在这里完成
----------------
  1. 把检索结果组装成 context (含编号, 供引用)
  2. 调 LLM 生成 (prompt 版本可切换 -> 这就是 prompt A/B 的落点)
  3. 抽取引用并**回验**: 声称的引用编号是否存在、引用的原文是否真在该 chunk 里

第 3 步是关键: 模型标了 [2] 不代表 [2] 真支撑这句话。不回验, "带引用"
就只是装饰。回验结果作为独立指标(citation_valid_rate)进评测表。

两套引用机制
------------
  Claude       原生 citations: 把 chunk 作为 document block 传入, 回答自带
               cited_text 与字符级位置。精确, 无需自己实现。
  其余 provider 标记式: prompt 要求输出 [n], 这里解析 [n] 并回查该编号
               对应的 chunk 是否真含相关内容。

两者在同一评测集上对比, 差多少分是实测出来的 —— 这本身是一条结论。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm import LLM, MODEL_ANSWER
from ..prompts import registry

CITE_RE = re.compile(r"\[(\d{1,2})\]")

REFUSAL_MARKERS = ("资料中未提及", "未提及", "无法确定", "没有提及", "未找到",
                   "资料中没有", "无法回答", "缺少", "未披露", "不足以")


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    text: str
    valid: bool = True
    cited_text: str = ""        # Claude 原生 citations 才有
    start: int | None = None
    end: int | None = None


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    context_chunks: list[str] = field(default_factory=list)
    route: str = ""
    graph_paths: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    from_cache: bool = False
    prompt_id: str = ""
    n_invalid_citations: int = 0
    trace: dict = field(default_factory=dict)   # 检索链路痕迹, 供前端展示

    @property
    def citation_valid_rate(self) -> float:
        if not self.citations:
            return 0.0
        return sum(1 for c in self.citations if c.valid) / len(self.citations)


def build_context(chunks: list[dict], max_chars: int = 9000) -> tuple[str, list[dict]]:
    """把 chunk 组装成带编号的 context。

    编号从 1 开始且与传入顺序一致 —— 引用回验依赖这个约定。
    超出预算就截断, 并把被截掉的数量返回给上层记录(而不是静默丢弃)。
    """
    parts: list[str] = []
    used: list[dict] = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        body = c["raw_text"]
        head = f"[{i}]（来源：{c['doc_title']}）"
        block = f"{head}\n{body}"
        if total + len(block) > max_chars and used:
            break
        parts.append(block)
        used.append(c)
        total += len(block)
    return "\n\n".join(parts), used


def _looks_refused(text: str) -> bool:
    head = text[:160]
    return any(m in head for m in REFUSAL_MARKERS)


class AnswerGenerator:
    def __init__(self, llm: LLM, prompt_id: str = "answer/v3_guarded",
                 model: str = MODEL_ANSWER, use_native_citations: bool = True,
                 max_context_chars: int = 9000):
        self.llm = llm
        self.prompt_id = prompt_id
        self.model = model
        self.max_context_chars = max_context_chars
        # 只有支持原生 citations 的 provider 才走 document block 路径
        self.native = use_native_citations and llm.supports("native_citations")

    def answer(self, question: str, chunks: list[dict],
               route: str = "", graph_paths: list[str] | None = None) -> Answer:
        p = registry.get(self.prompt_id)
        context, used = build_context(chunks, self.max_context_chars)

        # 图检索走通时, 把关系路径也作为一段"资料"给模型 ——
        # 否则模型只能看到散落的文档, 需要自己重新推出那条链, 容易断。
        if graph_paths:
            path_block = "关系路径（由知识图谱推导，每一步均有上列文档支撑）：\n" + \
                         "\n".join(f"  {x}" for x in graph_paths[:6])
            context = f"{context}\n\n{path_block}"

        docs = None
        if self.native:
            docs = [{"text": c["raw_text"], "title": c["doc_title"],
                     "id": c["doc_id"]} for c in used]

        system, user = p.render(context=context, question=question)
        res = self.llm.complete(system=system, user=user, model=self.model,
                                max_tokens=2048, documents=docs,
                                cache_tag=self.prompt_id)

        ans = Answer(text=res.text, refused=_looks_refused(res.text),
                     context_chunks=[c["id"] for c in used], route=route,
                     graph_paths=list(graph_paths or []),
                     latency_ms=res.latency_ms, cost_usd=res.cost_usd,
                     from_cache=res.from_cache, prompt_id=self.prompt_id)
        ans.citations = self._collect_citations(res, used)
        ans.n_invalid_citations = sum(1 for c in ans.citations if not c.valid)
        return ans

    # ------------------------------------------------------------------
    def _collect_citations(self, res, used: list[dict]) -> list[Citation]:
        out: list[Citation] = []

        # 路径一: Claude 原生 citations —— 带字符级位置, 直接可信
        for c in res.citations:
            di = c.get("document_index")
            if di is None or not (0 <= di < len(used)):
                continue
            ck = used[di]
            out.append(Citation(index=di + 1, chunk_id=ck["id"], doc_id=ck["doc_id"],
                                text=ck["raw_text"], cited_text=c.get("cited_text", ""),
                                start=c.get("start"), end=c.get("end"), valid=True))
        if out:
            return out

        # 路径二: 标记式 [n] —— 必须回验编号是否存在
        for m in CITE_RE.finditer(res.text):
            idx = int(m.group(1))
            if 1 <= idx <= len(used):
                ck = used[idx - 1]
                out.append(Citation(index=idx, chunk_id=ck["id"],
                                    doc_id=ck["doc_id"], text=ck["raw_text"],
                                    valid=True))
            else:
                # 引用了不存在的编号 —— 这是可被自动检出的幻觉
                out.append(Citation(index=idx, chunk_id="", doc_id="",
                                    text="", valid=False))
        # 去重保序
        seen: set[tuple] = set()
        uniq: list[Citation] = []
        for c in out:
            k = (c.index, c.chunk_id)
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        return uniq


class ExtractiveGenerator:
    """零 LLM 的抽取式基线: 直接返回排名第一的证据原文。

    存在的意义有二:
      1. 无 API key 时端到端流水线仍可跑通
      2. 作为生成环节的下界 —— 回答"LLM 到底在生成环节贡献了多少分"。
         很多 RAG 项目从不设这个对照, 于是分不清是检索好还是模型好。
    """
    prompt_id = "extractive/none"

    def answer(self, question: str, chunks: list[dict], route: str = "",
               graph_paths: list[str] | None = None) -> Answer:
        if not chunks:
            return Answer(text="资料中未提及。", refused=True, route=route)
        top = chunks[0]
        body = top["raw_text"].strip().replace("\n", " ")[:300]
        return Answer(text=f"{body} [1]", route=route,
                      context_chunks=[c["id"] for c in chunks[:5]],
                      graph_paths=list(graph_paths or []),
                      citations=[Citation(1, top["id"], top["doc_id"],
                                          top["raw_text"])],
                      prompt_id=self.prompt_id)
