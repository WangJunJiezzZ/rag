"""
HTTP API —— 演示界面的后端。

和评测脚本共用 RAGService, 所以"演示看到的"和"评测跑的"是同一套代码。
两边各写一份装配逻辑, 是线上行为与评测结果不一致的最常见来源。
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .compat import project_root, read_jsonl
from .llm import LLM
from .service import RAGService, ServiceConfig

ROOT = project_root()
WEB = ROOT / "web"

app = FastAPI(title="graphrag-lab", docs_url="/api/docs")

# 两套服务实例: 抽取图(真实水平) 与 oracle 图(上界)。
# 前端可切换 —— 让"抽取环节损失了多少"这件事在演示中肉眼可见。
_services: dict[str, RAGService] = {}
_docs: dict[str, dict] = {}


def get_service(graph: str = "extracted", prompt: str = "answer/v3_guarded") -> RAGService:
    key = f"{graph}|{prompt}"
    if key not in _services:
        path = ("data/synthetic/graph.json" if graph == "oracle"
                else "data/index/extracted_graph.json")
        _services[key] = RAGService(
            ServiceConfig(graph_path=path, prompt_id=prompt), quiet=True)
    return _services[key]


@app.on_event("startup")
def _warm() -> None:
    global _docs
    _docs = {d["id"]: d for d in read_jsonl(ROOT / "data/synthetic/documents.jsonl")}
    get_service()          # 预热: 建索引 + 向量化, 避免首次提问时干等


class AskRequest(BaseModel):
    question: str
    graph: str = "extracted"        # extracted | oracle
    prompt: str = "answer/v3_guarded"
    top_k: int = 8


@app.get("/api/health")
def health() -> dict:
    llm = LLM()
    svc = get_service()
    return {
        "ok": True,
        "llm_provider": llm.provider_name,
        "native_citations": llm.supports("native_citations"),
        "chunks": len(svc.chunks),
        "docs": len(svc.docs),
        "graph": svc.graph.stats() if svc.graph else None,
        "dense": svc.dense_enabled,
        "retriever": getattr(svc.retriever, "name", ""),
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    """演示用的三个问题 —— 刻意覆盖能力边界的三个档位。"""
    items = list(read_jsonl(ROOT / "data/eval/eval_set.jsonl"))
    picks = []
    for t, label in [("fact_direct", "① 单跳事实 · 谁都能答对"),
                     ("semantic_only", "② 纯语义 · 词面检索失效，靠向量"),
                     ("hop4_risk", "③ 四跳关系穿透 · 纯检索必然失败，必须走图"),
                     ("negative", "④ 幻觉陷阱 · 语料里根本没有，应当拒答")]:
        for it in items:
            if it["type"] == t:
                picks.append({"label": label, "question": it["question"],
                              "type": t, "gold": it["gold_answer"][:80],
                              "n_evidence": len(it["gold_docs"])})
                break
    return picks


@app.post("/api/ask")
def ask(req: AskRequest) -> JSONResponse:
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    svc = get_service(req.graph, req.prompt)
    t0 = time.perf_counter()
    chunks = svc.retrieve(req.question, req.top_k)
    try:
        ans = svc.ask(req.question, req.top_k)
        text, refused = ans.text, ans.refused
        cits = [asdict(c) for c in ans.citations]
        trace, cost = ans.trace, ans.cost_usd
        err = None
    except Exception as e:                       # LLM 不可用时仍展示检索结果
        text, refused, cits, cost = "", False, [], 0.0
        trace = {"route": "", "reason": "", "graphs": [], "linked_entities": []}
        err = f"{type(e).__name__}: {e}"
        r = getattr(svc.retriever, "last_route", None)
        if r:
            trace["route"], trace["reason"] = r.primary, r.reason
        gr = getattr(svc, "graph_retriever", None)
        if gr and gr.last_trace:
            trace["graphs"] = gr.last_trace.graphs[:3]
            trace["linked_entities"] = [n for _, n in gr.last_trace.linked]
            trace["intent"] = gr.last_trace.intent

    return JSONResponse({
        "question": req.question,
        "answer": text,
        "refused": refused,
        "citations": cits,
        "chunks": chunks,
        "trace": trace,
        "cost_usd": cost,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "error": err,
    })


@app.get("/api/doc/{doc_id}")
def get_doc(doc_id: str) -> dict:
    d = _docs.get(doc_id)
    if not d:
        raise HTTPException(404, f"找不到文档 {doc_id}")
    return d


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
