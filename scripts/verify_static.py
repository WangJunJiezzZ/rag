"""
静态站一致性自检 —— Python 检索器 vs 浏览器里的 engine.js, 逐题比对。

为什么需要这个脚本
------------------
静态站把 BM25、实体链接、图检索用 JS 重写了一遍。两份实现只要有一处不一致,
网页上看到的检索结果就和评测报告里的数字对不上 —— 而且没有任何报错。
这里对评测集的每一道题, 在两边各跑一次, 比对:

  · BM25 top-10 的 chunk id 与分数(4 位小数)
  · 路由结果(graph / lexical)与图查询模板(relation_path / relation_chain / neighborhood)
  · 图检索发出的 chunk 序列(两份图谱各一次)

需要本机装有 Node.js(>=18)。没有 node 时跳过, 不算失败。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import read_jsonl, setup_console               # noqa: E402
from graphrag.ingest.chunking import chunk_documents                # noqa: E402
from graphrag.retrieval.bm25 import BM25                            # noqa: E402
from graphrag.retrieval.graph_retriever import EntityLinker, GraphRetriever  # noqa: E402
from graphrag.retrieval.router import QueryRouter                   # noqa: E402
from graphrag.store.graph import KnowledgeGraph                     # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]
GRAPHS = {"extracted": "data/index/extracted_graph.json",
          "oracle": "data/synthetic/graph.json"}

NODE_CHECK = r"""
import { BM25, Graph, graphSearch } from '%s';
import fs from 'fs';
const D = JSON.parse(fs.readFileSync(process.argv[2]));
const D2C = new Map(), CH = new Map();
for (const c of D.chunks){ CH.set(c.id, c); if (!D2C.has(c.d)) D2C.set(c.d, []); D2C.get(c.d).push(c.id); }
const out = {};
const b = new BM25().index(D.chunks.map(c => ({ id: c.id, text: c.x })));
out.bm25 = D.q.map(q => b.search(q, 10).map(([id, s]) => [id, Math.round(s * 1e4) / 1e4]));
out.graph = {};
for (const [name, raw] of Object.entries(D.graphs)) {
  const g = new Graph(raw);
  out.graph[name] = D.q.map(q => {
    const r = graphSearch(g, q, D2C, 8, CH);
    return { intent: r.intent, route: r.route.primary, hits: r.hits.map(h => h[0]) };
  });
}
process.stdout.write(JSON.stringify(out));
"""


def main() -> int:
    node = shutil.which("node")
    if node is None:
        print("[skip] 未找到 node, 跳过静态站一致性自检")
        return 0

    docs = list(read_jsonl(ROOT / "data/synthetic/documents.jsonl"))
    chunks = chunk_documents(docs, "contextual")
    d2c: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        d2c[c.doc_id].append(c.id)
    ctext = {c.id: c.raw_text for c in chunks}
    questions = [i["question"] for i in read_jsonl(ROOT / "data/eval/eval_set.jsonl")]

    bm = BM25().index([(c.id, c.text) for c in chunks])
    py_bm25 = [[[cid, round(s, 4)] for cid, s in bm.search(q, 10)] for q in questions]

    payload = {"chunks": [{"id": c.id, "d": c.doc_id, "x": c.text, "r": c.raw_text}
                          for c in chunks],
               "q": questions, "graphs": {}}
    py_graph: dict[str, list[dict]] = {}
    for name, rel in GRAPHS.items():
        kg = KnowledgeGraph.load(ROOT / rel)
        payload["graphs"][name] = {
            "entities": [{"id": e.id, "n": e.name, "t": e.type, "a": e.aliases}
                         for e in kg.entities.values()],
            "edges": [{"s": e.src, "r": e.rel, "o": e.dst, "d": e.all_docs()}
                      for e in kg.edges]}
        gr = GraphRetriever(kg, d2c, chunk_text=ctext)
        router = QueryRouter(EntityLinker(kg))
        rows = []
        for q in questions:
            hits = gr.retrieve(q, 8)
            rows.append({"intent": gr.last_trace.intent if hits or gr.last_trace.linked else "",
                         "route": router.route(q).primary,
                         "hits": [cid for cid, _ in hits]})
        py_graph[name] = rows

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data.json"
        data.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        script = Path(tmp) / "check.mjs"
        engine = (ROOT / "deploy/static/engine.js").resolve().as_uri()
        script.write_text(NODE_CHECK % engine, encoding="utf-8")
        res = subprocess.run([node, str(script), str(data)], capture_output=True,
                             text=True, encoding="utf-8")
    if res.returncode != 0:
        print(res.stderr)
        return 1
    js = json.loads(res.stdout)

    fails = 0
    ok = sum(1 for a, b in zip(py_bm25, js["bm25"]) if a == b)
    fails += len(questions) - ok
    print(f"BM25 top-10(id + 4 位小数分数) ........ {ok}/{len(questions)}")
    for name in GRAPHS:
        ok = 0
        for q, a, b in zip(questions, py_graph[name], js["graph"][name]):
            if a["route"] == b["route"] and a["hits"] == b["hits"] and \
               (not a["hits"] or a["intent"] == b["intent"]):
                ok += 1
            elif fails < 5:
                print(f"   [diff] {name} | {q} | py={a['intent']}/{a['route']} "
                      f"js={b['intent']}/{b['route']}")
        fails += len(questions) - ok
        print(f"路由 + 图模板 + chunk 序列 ({name:<9}) ... {ok}/{len(questions)}")
    print("[PASS] 静态站与 Python 版逐题一致" if fails == 0
          else f"[FAIL] {fails} 处不一致")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
