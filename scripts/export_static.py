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
    # 同一题、同一份抽取图谱: 固定流水线 vs Agent。
    # runs 的键可任取; mode 缺省 "agent", graph 缺省取键名(extracted / oracle)。
    "qid": "q-0082",
    "question": "卡布达搭档的爷爷封印过哪个机器人？",
    "runs": {
        "pipeline": {"mode": "pipeline", "graph": "extracted", "label": "固定流水线",
                     "verdict": "wrong",
                     "note": "拒答。图谱其实找到了完整的 3 跳关系链，这条路径也交给了模型；"
                             "但抽取图谱里多了一条错误的边「卡布达 -搭档-> 卡布达巨人」，"
                             "它的来源 doc-0041 占掉了一个检索名额（固定 8 条），"
                             "写着「封印」的原文 doc-0042 被挤了出去。"
                             "流水线用的是防幻觉 prompt，要求每个结论都有原文支撑：有路径、没原文，于是保守地拒答。"
                             "同一流程在标准图谱上能召回 doc-0042 并答对。"},
        "agent": {"mode": "agent", "graph": "extracted", "label": "Agent（MCP 工具）",
                  "verdict": "correct",
                  "note": "答对，3 份标准证据全部引用。同一份有错误边的图谱，"
                          "Agent 查到「小让的爷爷是高圆寺寅彦」之后，自己决定再查一步「谁被他封印过」。"
                          "它不受固定检索名额的限制，按中间结果选下一步，于是绕开了那条错误支路。"},
    },
}, {
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


# 评测里两边结论不同 / 都答错的题, 自动收录进回放页。
# 下面几道写了失败分析(见 docs/07-Agent.md 第二节), 其余只显示判分结果。
EVAL_NOTES = {
    "q-0052": {"agent": "答错，原因是检索时预设了答案。Agent 的检索词是「管理整片草原 村长」——"
                        "它自己加了「村长」，结果被引向慢羊羊（羊村村长），"
                        "写着「包包大人……青青草原的管理者」的 doc-0009 根本没有出现。"
                        "（用原问题直接检索，doc-0009 排第一。）"},
    "q-0058": {"agent": "没答出来，但其实只差一步。前 2 轮检索了 4 次（含「爱喝酒」「喜欢喝酒」），"
                        "每个检索词都带了「机器人」这个泛词，结果被「阵营一览」等提到大量机器人的文档占满；"
                        "第 3～6 轮挨个查机器人的邻居；第 7 轮只搜「贪杯好饮」，第 3 条就是蝎子莱莱（「该角色爱喝酒」）；"
                        "第 8 轮去读原文——这是最后一轮，读到的内容还没交回给模型就达到了步数上限，已拿到的证据被整个丢掉。"
                        "两个问题：检索词带泛词；步数用完时没有强制模型根据已有信息作答。"},
    "q-0088": {"agent": "答错。答案是「藏之助的搭档金龟次郎与卡布达同属正义阵营」，要比较两个角色的属性；"
                        "工具只能沿关系走，阵营这类字面量节点按设计不能作为中转，Agent 于是认为两人没有关系。"
                        "改进方向：补一个「按属性列出实体」的工具。",
               "pipeline": "答错。抽取图多记的一份来源文档占掉一个检索名额，把「阵营一览」挤出了前 8 条。"},
    "q-0069": {"agent": "答对。没有任何文档直接写「251」：Agent 读了三份档案，"
                        "按「黑太狼第 249 代 → 灰太狼第 250 代 → 小灰灰是灰太狼的儿子」推出第 251 代。",
               "pipeline": "拒答。只拿到了「灰太狼是第 250 代」，推不出小灰灰的代数。"},
    "q-0063": {"agent": "答对。查到妈妈是红太狼之后，又去查了红太狼的武器。",
               "pipeline": "拒答。只取回了人物关系，没有取回红太狼的武器。"},
}


def load_eval_results() -> dict | None:
    """汇总 reports/phase4_agent_*.json(三个划分) —— 页面上的 109 题评测结果。"""
    files = [ROOT / f"reports/phase4_agent_{s}.json" for s in ("train", "dev", "test")]
    if not all(f.exists() for f in files):
        return None
    P, A = [], []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        split = d["split"]
        P += [{**r, "split": split} for r in d["pipeline"]["details"]]
        A += [{**r, "split": split} for r in d["agent"]["details"]]

    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else None

    def summ(R):
        ans = [r for r in R if r["type"] != "negative"]
        neg = [r for r in R if r["type"] == "negative"]
        return {"n": len(R), "correct": sum(r["correct"] for r in R),
                "acc": mean([r["correct"] for r in R]),
                "false_refusal": mean([r["false_refusal"] for r in ans]),
                "trap": mean([r["refusal_correct"] for r in neg]),
                "tools": mean([r["tool_calls"] for r in R]),
                "llm_calls": mean([r["llm_calls"] for r in R]),
                "tokens": mean([r["input_tokens"] + r["output_tokens"] for r in R]),
                "cost_usd": sum(r["cost_usd"] for r in R)}

    by_type: dict[str, dict] = {}
    for r in P:
        by_type.setdefault(r["type"], {"n": 0, "pipeline": 0, "agent": 0})
        by_type[r["type"]]["n"] += 1
        by_type[r["type"]]["pipeline"] += r["correct"]
    for r in A:
        by_type[r["type"]]["agent"] += r["correct"]
    pm = {r["id"]: r for r in P}
    diff = [{"id": r["id"], "type": r["type"], "split": r["split"],
             "pipeline": pm[r["id"]]["correct"], "agent": r["correct"],
             "tools": r["tool_calls"], "tokens": r["input_tokens"] + r["output_tokens"],
             "stopped": r.get("stopped", "")}
            for r in A if r["correct"] != pm[r["id"]]["correct"]
            or (r["correct"] == 0 and pm[r["id"]]["correct"] == 0)]
    return {"pipeline": summ(P), "agent": summ(A), "by_type": by_type, "diff": diff,
            "details": {"pipeline": pm, "agent": {r["id"]: r for r in A}}}


TRACE_MISSES: list[str] = []      # 严格模式据此判定构建是否完整


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
            # 必须在 Client 的任务组里面捕获: 异常一旦穿出任务组, 会被包成 ExceptionGroup,
            # 外面的 except CacheMissError 接不住, 整个构建直接崩(linux 容器里实测)
            try:
                return await chat.ask(question)
            except CacheMissError:
                return None

    def run_pipeline(graph_rel: str, question: str) -> dict:
        """固定流水线(规则路由 -> 一次性检索 -> 一次生成), 与评测跑的是同一个 RAGService。"""
        from graphrag.service import RAGService
        svc = RAGService(ServiceConfig(graph_path=graph_rel), quiet=True)
        calls: list = []
        inner = svc.llm.complete
        svc.llm.complete = lambda **kw: calls.append(inner(**kw)) or calls[-1]
        ans = svc.ask(question)
        hits = svc.retrieve(question)
        tr = svc.graph_retriever.last_trace
        tin = sum(r.input_tokens for r in calls)
        tout = sum(r.output_tokens for r in calls)
        return {"answer": ans.text, "refused": ans.refused, "stopped": "final",
                # 流水线的答案用 [1][4] 编号引用, 页面无法从正文里认出 doc id, 单独给出
                "cited_docs": list(dict.fromkeys(c.doc_id for c in ans.citations)),
                "llm_calls": len(calls), "input_tokens": tin, "output_tokens": tout,
                "cost_usd": round((tin * pin + tout * pout) / 1e6, 4), "steps": [],
                "pipeline": {"route": ans.route, "intent": tr.intent if tr else "",
                             "top_k": svc.cfg.top_k,
                             "retrieved": list(dict.fromkeys(h["doc_id"] for h in hits)),
                             # 实际交给模型的路径(未去重, 与生成时一致)
                             "paths": ans.graph_paths}}

    # ---- 精选案例 + 评测中两边结论不同 / 都答错的题 ----
    ev = load_eval_results()
    specs = [{**sp, "group": "curated"} for sp in AGENT_DEMOS]
    if ev:
        curated = {sp["qid"] for sp in AGENT_DEMOS}
        for d in ev["diff"]:
            if d["id"] in curated:
                continue
            notes = EVAL_NOTES.get(d["id"], {})
            verdict = lambda ok: "correct" if ok else "wrong"          # noqa: E731
            pr, ar = ev["details"]["pipeline"][d["id"]], ev["details"]["agent"][d["id"]]
            specs.append({"qid": d["id"], "question": gold[d["id"]]["question"], "group": "eval",
                          "runs": {
                "pipeline": {"mode": "pipeline", "graph": "extracted", "label": "固定流水线",
                             "verdict": verdict(d["pipeline"]), "note": notes.get("pipeline", ""),
                             "judge_reason": pr.get("judge_reason", "")},
                "agent": {"mode": "agent", "graph": "extracted", "label": "Agent（MCP 工具）",
                          "verdict": verdict(d["agent"]), "note": notes.get("agent", ""),
                          "judge_reason": ar.get("judge_reason", "")}}})

    demos = []
    for spec in specs:
        g = gold[spec["qid"]]
        runs = {}
        for key, meta in spec["runs"].items():
            mode, gname = meta.get("mode", "agent"), meta.get("graph", key)
            try:
                if mode == "pipeline":
                    runs[key] = {**meta, "mode": mode, "graph": gname,
                                 **run_pipeline(GRAPHS[gname], spec["question"])}
                    continue
                turn = anyio.run(run, GRAPHS[gname], spec["question"])
                if turn is None:
                    raise CacheMissError(spec["question"])
            except CacheMissError:
                hint = ("python run.py e2e" if mode == "pipeline" else
                        f"python run.py mcp-chat -- --graph {GRAPHS[gname]} -q \"{spec['question']}\"")
                TRACE_MISSES.append(f"{spec['qid']}:{key}")
                print(f"[warn] Agent 回放缓存未命中: {key}({mode}, {gname}) | {spec['question']}\n"
                      f"       先在线跑一遍: {hint}")
                continue
            runs[key] = {
                **meta, "mode": mode, "graph": gname,
                "alias_rounds": getattr(turn, "alias_rounds", 0),
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
            demos.append({"qid": spec["qid"], "question": spec["question"], "group": spec["group"],
                          "split": g["split"],
                          "eval_question": g["question"], "type": g["type"],
                          "gold_answer": g["gold_answer"], "gold_docs": g["gold_docs"],
                          "runs": runs})
    write_text(out / "agent_traces.json",
               json.dumps({"model": "deepseek-chat", "demos": demos,
                           "eval": {k: v for k, v in ev.items() if k != "details"} if ev else None},
                          ensure_ascii=False, separators=(",", ":")))
    return sum(len(d["runs"]) for d in demos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    help="train / dev / test / all; 只导出缓存里已有答案的题")
    ap.add_argument("--out", default="build/static/data")
    ap.add_argument("--strict", action="store_true",
                    help="任何缓存未命中都以失败退出 —— 自动发布用, 宁可不发也不发残缺的站")
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
    print(f"     Agent 回放 {n_traces} 条"
          + (f", {len(TRACE_MISSES)} 条缓存未命中" if TRACE_MISSES else ""))
    if args.strict and (miss or TRACE_MISSES):
        print(f"[FAIL] 严格模式: 预录问答未命中 {miss} 条, Agent 回放未命中 {len(TRACE_MISSES)} 条"
              f"{' ' + ', '.join(TRACE_MISSES[:10]) if TRACE_MISSES else ''}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
