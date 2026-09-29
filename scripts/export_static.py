"""
导出纯静态演示所需的数据。

为什么能做成静态
----------------
这套系统里只有**生成**环节必须调 LLM, 而生成结果本来就全部进了缓存。
检索侧的三件事 —— BM25、实体链接、图遍历 —— 都是确定性算法,
数据量也小(两百来个 chunk、几十个实体、不到两百条边), 完全可以搬到浏览器里跑。

于是静态版的能力边界是:
  · 预录的那批问题 -> 完整体验(答案 + 引用 + 关系图), 查表即得
  · 访客自己输入的问题 -> BM25 + 图检索实时在浏览器里跑, 但没有生成的答案

dense 向量检索没有搬过去: 要在浏览器里跑 ONNX 得额外加载 10MB WASM,
对一个演示页面不划算。静态版的检索因此略弱于完整版, 页面上会注明。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import read_jsonl, setup_console, write_text   # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]

GRAPHS = {"extracted": "data/index/extracted_graph.json",
          "oracle": "data/synthetic/graph.json"}

# HuggingFace Static Space 靠 README.md 顶部这段 YAML 识别配置。
# 2026 年起 Docker / Gradio Space 需付费, Static 仍免费 —— 所以走静态这条路。
HF_README = """---
title: GraphRAG Lab
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: 可评测、可消融的 GraphRAG（喜羊羊 × 铁甲小宝 角色关系）
---

# GraphRAG Lab

一个**可评测、可消融**的 GraphRAG 检索增强系统。题材是《喜羊羊与灰太狼》和《铁甲小宝》的角色关系——
"小灰灰的爷爷是谁""小让的爷爷封印过哪个机器人""小灰灰和小香香是什么关系"。

> 事实取自维基百科等公开资料，译名采用中国大陆播出版；文档由程序按模板渲染，措辞不是原文引用。
> 角色及形象权利归原著作权方所有，本页面仅作检索技术演示。

## 这个 Demo 要证明什么

> 纯向量 RAG 在什么情况下会失败？加上什么能把分数拉回来？每一步各值多少分？

检索 recall@10：**BM25 80.9% → +向量 87.0% → +图检索与路由 94.4%**

两跳关系题：图检索取 5 块就召回全部证据，BM25 放大到 20 块也只有 82%。

## 关于这个静态版

**检索完全在你的浏览器里实时运行** —— BM25、实体链接、图遍历都是确定性算法，
数据量也小（187 个 chunk / 49 个实体 / 174 条边），用 JS 重写后毫秒级跑完。
与 Python 版逐题校验过（`python run.py verify-static`）：BM25 分数四位小数一致，
路由、图查询模板与发出的 chunk 序列在两份图谱上均 109/109 一致。

- 点**示例按钮** → 完整体验：生成的答案 + 引用溯源 + 关系图
- 自己输入**任意问题** → 检索与关系图照常工作，但没有生成的答案文字（那需要调大模型）
- 未包含向量检索（需额外 10MB ONNX WASM），纯语义类问题会弱于完整版

**最值得试的**：第 ③ 个示例"小灰灰和小香香是什么关系"。关系图会画出

```
小灰灰 ←母亲— 红太狼 ←表妹— 香太狼 —女儿→ 小香香
```

三条关系分别写在两份档案里，**没有任何一份文档同时提到小灰灰和小香香**。
再试试只用台湾译名提问："蝎子蓝蓝的原型是什么？"——实体链接会把它认成蝎子莱莱。
"""


# Agent 回放: 同一道题在两份图谱上各跑一遍的真实录像。
# 只能收录**已在线跑过**的问题 —— 导出时离线重放, 缓存必须逐轮命中。
AGENT_DEMOS = [{
    "qid": "q-0084",
    "question": "小灰灰和小香香是什么关系？",
    "runs": {
        "oracle": {"label": "标准图谱", "verdict": "correct",
                   "note": "答对，引用了标准答案要求的全部 2 份证据。第 2 轮的 explain_relation 已经给出完整路径，"
                           "模型又用两次 read_document 回到原文逐跳核实：更稳妥，但多花了一轮模型调用，"
                           "成本约为另一次的 1.6 倍。"},
        "extracted": {"label": "自动抽取的图谱", "verdict": "correct",
                      "note": "同样答对，路径与标准图一致：v4 抽取 prompt 在这两份档案上没有漏边，"
                              "消歧也没有把「小灰灰」「小香香」这类前两字不同的名字误并。"
                              "模型拿到路径后没有再读原文就直接作答 —— 少一轮调用，"
                              "但结论的可靠性完全押在抽取质量上。"},
    },
}]


def export_agent_traces(out: Path) -> int:
    """离线重放 MCP Agent 的运行, 导出逐步轨迹。

    走进程内 MCP 连接(Client(MCPServer)) —— 与 stdio 子进程发出的工具定义、
    返回的工具结果完全一致, 于是每一轮的请求都能命中当初在线运行时写下的缓存。
    """
    try:
        import anyio
        from mcp import Client
    except ImportError:
        print("[skip] 未安装 mcp, 跳过 Agent 回放 (pip install 'mcp>=2.2')")
        return 0
    from graphrag.llm import LLM, PRICING, CacheMissError
    from graphrag.mcp_app.client import MCPChat
    from graphrag.mcp_app.server import build_server
    from graphrag.mcp_app.tools import GraphRAGTools
    from graphrag.service import ServiceConfig

    gold = {i["id"]: i for i in read_jsonl(ROOT / "data/eval/eval_set.jsonl")}
    pin, pout = PRICING["deepseek-chat"]

    async def run(graph_rel: str, question: str):
        server = build_server(GraphRAGTools(ServiceConfig(graph_path=graph_rel)))
        async with Client(server) as c:
            chat = MCPChat(c, LLM(provider="deepseek", offline=True), verbose=False)
            await chat.setup()
            return await chat.ask(question)

    demos = []
    for spec in AGENT_DEMOS:
        g = gold[spec["qid"]]
        runs = {}
        for gname, meta in spec["runs"].items():
            try:
                turn = anyio.run(run, GRAPHS[gname], spec["question"])
            except CacheMissError:
                print(f"[warn] Agent 回放缓存未命中: {gname} | {spec['question']}\n"
                      f"       先在线跑一遍: python run.py mcp-chat -- "
                      f"--graph {GRAPHS[gname]} -q \"{spec['question']}\"")
                continue
            runs[gname] = {
                **meta,
                "answer": turn.answer,
                "stopped": turn.stopped,
                "llm_calls": turn.llm_calls,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "cost_usd": round((turn.input_tokens * pin
                                   + turn.output_tokens * pout) / 1e6, 4),
                "steps": [{"round": s.round, "tool": s.tool, "args": s.arguments,
                           "result": s.result, "error": s.is_error}
                          for s in turn.steps],
            }
        if runs:
            demos.append({"qid": spec["qid"], "question": spec["question"],
                          "eval_question": g["question"], "type": g["type"],
                          "gold_answer": g["gold_answer"], "gold_docs": g["gold_docs"],
                          "runs": runs})
    write_text(out / "agent_traces.json",
               json.dumps({"model": "deepseek-chat", "demos": demos},
                          ensure_ascii=False, separators=(",", ":")))
    return sum(len(d["runs"]) for d in demos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    help="train / dev / test / all; 只导出缓存里已有答案的题")
    ap.add_argument("--out", default="build/static/data")
    args = ap.parse_args()

    import os
    os.environ["GRAPHRAG_OFFLINE"] = "1"
    from graphrag.ingest.chunking import chunk_documents
    from graphrag.llm import CacheMissError
    from graphrag.service import RAGService, ServiceConfig
    from graphrag.store.graph import KnowledgeGraph

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    docs = list(read_jsonl(ROOT / "data/synthetic/documents.jsonl"))
    chunks = chunk_documents(docs, "contextual")

    # ---- chunk: 检索用 text, 展示用 raw_text ----
    write_text(out / "chunks.json", json.dumps([{
        "i": i, "id": c.id, "d": c.doc_id, "t": c.doc_title,
        "x": c.text, "r": c.raw_text,
    } for i, c in enumerate(chunks)], ensure_ascii=False, separators=(",", ":")))

    # ---- 图谱 ----
    for name, rel in GRAPHS.items():
        kg = KnowledgeGraph.load(ROOT / rel)
        write_text(out / f"graph_{name}.json", json.dumps({
            "entities": [{"id": e.id, "n": e.name, "t": e.type,
                          "a": e.aliases}
                         for e in kg.entities.values()],
            "edges": [{"s": e.src, "r": e.rel, "o": e.dst,
                       "d": e.all_docs()} for e in kg.edges],
        }, ensure_ascii=False, separators=(",", ":")))

    # ---- 预录问答 ----
    items = [i for i in read_jsonl(ROOT / "data/eval/eval_set.jsonl")
             if args.split == "all" or i["split"] == args.split]
    answers: dict[str, dict] = {}
    miss = 0
    for gname, rel in GRAPHS.items():
        svc = RAGService(ServiceConfig(graph_path=rel), quiet=True)
        for it in items:
            try:
                a = svc.ask(it["question"])
            except CacheMissError:
                miss += 1
                continue
            answers[f"{gname}|{it['question']}"] = {
                "text": a.text,
                "refused": a.refused,
                "cites": [{"i": c.index, "c": c.chunk_id, "d": c.doc_id}
                          for c in a.citations],
                "chunks": [c["id"] for c in svc.retrieve(it["question"], 8)],
                "route": a.route,
                "graphs": a.trace.get("graphs", [])[:3],
                "type": it["type"],
            }
    write_text(out / "answers.json",
               json.dumps(answers, ensure_ascii=False, separators=(",", ":")))

    # ---- 示例题 ----
    spec = [("fact_direct", "① 单跳事实 · 谁都能答对"),
            ("semantic_only", "② 纯语义 · 词面检索失效，靠向量"),
            ("relation_path", "③ 关系路径 · 两端有名字、中间全靠图"),
            ("negative", "④ 幻觉陷阱 · 语料里根本没有，应当拒答")]
    examples = []
    for t, label in spec:
        it = next((i for i in items if i["type"] == t
                   and f"extracted|{i['question']}" in answers), None)
        if it:
            examples.append({"label": label, "q": it["question"], "type": t,
                             "n": len(it["gold_docs"])})
    write_text(out / "examples.json",
               json.dumps(examples, ensure_ascii=False))

    # ---- 可回答问题清单 ----
    # 静态版只有这些题有生成的答案。把清单显式交给前端, 让访客
    # **先看到能问什么**, 而不是随便试几个都没答案后就走了。
    TYPE_LABEL = {
        "fact_direct": "单跳事实", "fact_paraphrase": "同义改写",
        "semantic_only": "纯语义", "fact_disambig": "易混实体",
        "hop2": "两跳关系", "hop3": "三跳关系",
        "relation_path": "关系路径", "aggregation": "列举比较",
        "negative": "幻觉陷阱",
    }
    catalog = []
    for it in items:
        if f"extracted|{it['question']}" not in answers:
            continue
        catalog.append({"q": it["question"],
                        "t": it["type"],
                        "label": TYPE_LABEL.get(it["type"], it["type"]),
                        "hops": it["hops"]})
    order = list(TYPE_LABEL)
    catalog.sort(key=lambda x: (order.index(x["t"]) if x["t"] in order else 99,
                                x["q"]))
    write_text(out / "catalog.json",
               json.dumps(catalog, ensure_ascii=False, separators=(",", ":")))

    # ---- 元信息 ----
    write_text(out / "meta.json", json.dumps({
        "chunks": len(chunks), "docs": len(docs),
        "answers": len(answers), "split": args.split,
        "graphs": {k: KnowledgeGraph.load(ROOT / v).stats()
                   for k, v in GRAPHS.items()},
    }, ensure_ascii=False, indent=1))

    # ---- Agent 回放 ----
    n_traces = export_agent_traces(out)

    # ---- 页面模板 + HF Static Space 配置 ----
    site = out.parent
    (site / "js").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "deploy/static/index.html", site / "index.html")
    shutil.copy2(ROOT / "deploy/static/agent.html", site / "agent.html")
    shutil.copy2(ROOT / "deploy/static/engine.js", site / "js/engine.js")
    write_text(site / "README.md", HF_README)
    # GitHub Pages 默认用 Jekyll 处理站点, 会忽略下划线开头的目录、
    # 也会拖慢部署。放一个空的 .nojekyll 关掉它。对其他托管无害。
    (site / ".nojekyll").write_text("", encoding="utf-8")

    total = sum(p.stat().st_size for p in out.glob("*.json"))
    print(f"[ok] 静态数据已导出 -> {out}")
    for p in sorted(out.glob("*.json"), key=lambda x: -x.stat().st_size):
        print(f"     {p.name:<22} {p.stat().st_size/1024:>7.0f} KB")
    print(f"     {'合计':<22} {total/1024:>7.0f} KB")
    print(f"     预录问答 {len(answers)} 条 ({len(items)} 题 × {len(GRAPHS)} 种图谱)"
          + (f", {miss} 条缓存未命中" if miss else ""))
    print(f"     Agent 回放 {n_traces} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
