"""
导出纯静态演示所需的数据。

为什么能做成静态
----------------
这套系统里只有**生成**环节必须调 LLM, 而生成结果本来就全部进了缓存。
检索侧的三件事 —— BM25、实体链接、图遍历 —— 都是确定性算法,
数据量也小(1092 个 chunk、227 实体、477 条边), 完全可以搬到浏览器里跑。

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
short_description: 可评测、可消融的 GraphRAG 检索增强系统（合成数据演示）
---

# GraphRAG Lab

一个**可评测、可消融**的 GraphRAG 检索增强系统。

> ⚠️ 全部语料为程序合成。机构名、基金名、人名、管辖区名及其风险等级均属虚构，
> 不指涉任何真实企业、个人或司法管辖区，不构成任何投资、法律或合规建议。

## 这个 Demo 要证明什么

> 纯向量 RAG 在什么情况下会失败？加上什么能把分数拉回来？每一步各值多少分？

检索 recall@10：**BM25 65.2% → +向量 70.5% → +图检索与路由 86.3%**

## 关于这个静态版

**检索完全在你的浏览器里实时运行** —— BM25、实体链接、图遍历都是确定性算法，
数据量也小（1092 个 chunk / 227 实体 / 477 条边），用 JS 重写后毫秒级跑完。
与 Python 版逐题校验过：BM25 分数四位小数一致，路由与图模板 78/78 一致。

- 点**示例按钮** → 完整体验：生成的答案 + 引用溯源 + 关系图
- 自己输入**任意问题** → 检索与关系图照常工作，但没有生成的答案文字（那需要调大模型）
- 未包含向量检索（需额外 10MB ONNX WASM），纯语义类问题会弱于完整版

**最值得试的**：问完第 ③ 个示例后，把「图谱来源」从「抽取图」切到「标准图」：

```
抽取图：  基金 → 新港城                                （2 节点，链断了）
标准图：  基金 → 管理人 → 董事 → 关联公司 → 高风险辖区    （5 节点，完整）
```

这个差距就是 **LLM 抽取环节的损失**：检索层面差 4.8 分，端到端答对率差 **24.6 分**。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
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
                          "a": e.aliases,
                          "risk": e.props.get("risk")}
                         for e in kg.entities.values()],
            "edges": [{"s": e.src, "r": e.rel, "o": e.dst,
                       "d": e.all_docs()} for e in kg.edges],
        }, ensure_ascii=False, separators=(",", ":")))

    # ---- 预录问答 ----
    items = [i for i in read_jsonl(ROOT / "data/eval/eval_set.jsonl")
             if i["split"] == args.split]
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
            ("hop4_risk", "③ 四跳关系穿透 · 纯检索必然失败，必须走图"),
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

    # ---- 元信息 ----
    write_text(out / "meta.json", json.dumps({
        "chunks": len(chunks), "docs": len(docs),
        "answers": len(answers), "split": args.split,
        "graphs": {k: KnowledgeGraph.load(ROOT / v).stats()
                   for k, v in GRAPHS.items()},
    }, ensure_ascii=False, indent=1))

    # ---- 页面模板 + HF Static Space 配置 ----
    site = out.parent
    (site / "js").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "deploy/static/index.html", site / "index.html")
    shutil.copy2(ROOT / "deploy/static/engine.js", site / "js/engine.js")
    write_text(site / "README.md", HF_README)

    total = sum(p.stat().st_size for p in out.glob("*.json"))
    print(f"[ok] 静态数据已导出 -> {out}")
    for p in sorted(out.glob("*.json"), key=lambda x: -x.stat().st_size):
        print(f"     {p.name:<22} {p.stat().st_size/1024:>7.0f} KB")
    print(f"     {'合计':<22} {total/1024:>7.0f} KB")
    print(f"     预录问答 {len(answers)} 条 ({len(items)} 题 × {len(GRAPHS)} 种图谱)"
          + (f", {miss} 条缓存未命中" if miss else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
