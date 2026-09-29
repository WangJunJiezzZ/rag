"""
RAG 服务装配 —— 把各组件接成一条可运行的链路。

评测脚本和 HTTP API 共用这一个类, 保证"评测跑的"和"演示跑的"是同一套代码。
两边各写一份装配逻辑, 是评测与线上行为不一致的最常见来源。
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .compat import project_root, read_jsonl
from .embedding import get_embedder
from .generate.answer import Answer, AnswerGenerator, ExtractiveGenerator
from .ingest.chunking import chunk_documents
from .llm import LLM
from .retrieval.bm25 import BM25
from .retrieval.dense import DenseRetriever
from .retrieval.fusion import RRFFusion
from .retrieval.graph_retriever import EntityLinker, GraphRetriever
from .retrieval.router import QueryRouter, RoutedRetriever
from .store.graph import KnowledgeGraph


class _BM25Retriever:
    name = "BM25"

    def __init__(self, chunks):
        self.bm25 = BM25().index([(c.id, c.text) for c in chunks])

    def retrieve(self, query: str, top_k: int):
        return self.bm25.search(query, top_k)


@dataclass
class ServiceConfig:
    chunk_strategy: str = "contextual"
    graph_path: str = "data/index/extracted_graph.json"   # 默认用**抽取**的图
    prompt_id: str = "answer/v3_guarded"
    use_dense: bool = True
    use_graph: bool = True
    use_router: bool = True
    top_k: int = 8
    generator: str = "llm"            # llm | extractive


class RAGService:
    def __init__(self, cfg: ServiceConfig | None = None, llm: LLM | None = None,
                 quiet: bool = False):
        self.cfg = cfg or ServiceConfig()
        root = project_root()
        self.llm = llm or LLM()

        docs = list(read_jsonl(root / "data/synthetic/documents.jsonl"))
        self.docs = {d["id"]: d for d in docs}
        chunks = chunk_documents(docs, self.cfg.chunk_strategy)
        self.chunks = {c.id: c for c in chunks}
        self.doc_to_chunks: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            self.doc_to_chunks[c.doc_id].append(c.id)

        bm25 = _BM25Retriever(chunks)
        lexical = bm25
        self.dense_enabled = False
        if self.cfg.use_dense:
            emb = get_embedder()
            if emb is not None:
                dense = DenseRetriever(emb, name="Dense")
                dense.index([(c.id, c.text) for c in chunks], verbose=not quiet)
                lexical = RRFFusion([bm25, dense], name="BM25+Dense")
                self.dense_enabled = True

        self.graph: KnowledgeGraph | None = None
        self.retriever = lexical
        if self.cfg.use_graph:
            gp = root / self.cfg.graph_path
            if gp.exists():
                self.graph = KnowledgeGraph.load(gp)
                gr = GraphRetriever(self.graph, self.doc_to_chunks,
                                    name="Graph")
                self.graph_retriever = gr
                self.retriever = (
                    RoutedRetriever(lexical, gr,
                                    QueryRouter(EntityLinker(self.graph)),
                                    name="Routed")
                    if self.cfg.use_router else RRFFusion([lexical, gr], name="RRF"))
            elif not quiet:
                print(f"[warn] 找不到图谱 {gp}, 仅使用词面/语义检索")

        self.generator = (ExtractiveGenerator() if self.cfg.generator == "extractive"
                          else AnswerGenerator(self.llm, prompt_id=self.cfg.prompt_id))

    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.cfg.top_k
        hits = self.retriever.retrieve(question, k)
        out: list[dict] = []
        for cid, score in hits:
            c = self.chunks.get(cid)
            if c is None:
                continue
            out.append({"id": c.id, "doc_id": c.doc_id, "doc_title": c.doc_title,
                        "section": c.section, "raw_text": c.raw_text,
                        "offset": c.offset, "score": float(score)})
        return out

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        chunks = self.retrieve(question, top_k)

        route = getattr(getattr(self.retriever, "last_route", None), "primary", "")
        reason = getattr(getattr(self.retriever, "last_route", None), "reason", "")
        paths: list[str] = []
        graphs: list[dict] = []
        linked: list[str] = []
        intent = ""
        gr = getattr(self, "graph_retriever", None)
        if gr is not None and getattr(gr, "last_trace", None):
            tr = gr.last_trace
            paths = [p for p in tr.paths if p][:6]
            graphs = tr.graphs[:3]
            linked = [n for _, n in tr.linked]
            intent = tr.intent

        # 图谱来源必须进身份: 同一问题在抽取图和标准图下检索到的证据不同,
        # 答案也不同, 不能互相顶替。
        ans = self.generator.answer(question, chunks, route=route or "lexical",
                                    graph_paths=paths,
                                    cache_identity=self.cfg.graph_path)
        ans.route = f"{route or 'lexical'}（{reason}）" if reason else ans.route
        ans.latency_ms = (time.perf_counter() - t0) * 1000
        ans.trace = {"route": route or "lexical", "reason": reason,
                     "intent": intent, "linked_entities": linked,
                     "graphs": graphs, "retriever": getattr(self.retriever, "name", ""),
                     "graph_source": self.cfg.graph_path}
        return ans
