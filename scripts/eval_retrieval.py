"""
Phase 1 - 检索评测

两组实验:
  A. 切块策略对比    fixed / section / contextual  x  BM25
  B. 检索路径消融    BM25 / Graph(oracle) / BM25+Graph(RRF)

B 组里的图检索使用 ground-truth 图, 所以得到的是**上界**:
假如抽取环节完美, 图检索最多能拿多少分。Phase 2 换成 LLM 抽取的图
再跑一次, 差值就是抽取环节的损失。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, read_jsonl, write_text     # noqa: E402
from graphrag.ingest.chunking import chunk_documents                  # noqa: E402
from graphrag.retrieval.bm25 import BM25                              # noqa: E402
from graphrag.retrieval.graph_retriever import GraphRetriever         # noqa: E402
from graphrag.retrieval.fusion import RRFFusion                       # noqa: E402
from graphrag.retrieval.router import QueryRouter, RoutedRetriever    # noqa: E402
from graphrag.retrieval.graph_retriever import EntityLinker           # noqa: E402
from graphrag.retrieval.dense import DenseRetriever                   # noqa: E402
from graphrag.embedding import get_embedder                           # noqa: E402
from graphrag.store.graph import KnowledgeGraph                       # noqa: E402
from graphrag.eval.retrieval import (RetrievalEvaluator,              # noqa: E402
                                     format_report, compare_table)

setup_console()
ROOT = Path(__file__).resolve().parents[1]


class BM25Retriever:
    def __init__(self, chunks, name: str):
        self.name = name
        self.bm25 = BM25().index([(c.id, c.text) for c in chunks])

    def retrieve(self, query: str, top_k: int):
        return self.bm25.search(query, top_k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="train/dev/test; 缺省全部")
    ap.add_argument("--strategy", default="contextual", help="B 组使用的切块策略")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--with-extracted", action="store_true",
                    help="(已默认开启, 保留以兼容旧命令)")
    ap.add_argument("--no-extracted", action="store_true",
                    help="不评测抽取图/无消歧图; 缺省时只要文件存在就一起评 —— 上界 vs 实测")
    args = ap.parse_args()

    docs = list(read_jsonl(ROOT / "data/synthetic/documents.jsonl"))
    items = list(read_jsonl(ROOT / "data/eval/eval_set.jsonl"))
    kg = KnowledgeGraph.load(ROOT / "data/synthetic/graph.json")

    print("=" * 76)
    print("A 组  切块策略对比 (检索器固定为 BM25)")
    print("=" * 76)
    group_a = []
    for strategy in ["fixed", "section", "contextual"]:
        chunks = chunk_documents(docs, strategy)
        c2d = {c.id: c.doc_id for c in chunks}
        ev = RetrievalEvaluator(c2d, items, split=args.split)
        rep = ev.evaluate(BM25Retriever(chunks, f"BM25/{strategy}"))
        group_a.append(rep)
        print(f"\n{format_report(rep)}\n  chunk 数 {len(chunks)}")
    print("\n" + compare_table(group_a, k=10))
    print("\n" + compare_table(group_a, k=1).replace("recall@1 对比", "recall@1 对比"))

    print("\n" + "=" * 76)
    print(f"B 组  检索路径消融 (切块策略固定为 {args.strategy})")
    print("=" * 76)
    chunks = chunk_documents(docs, args.strategy)
    c2d = {c.id: c.doc_id for c in chunks}
    d2c: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        d2c[c.doc_id].append(c.id)

    bm25 = BM25Retriever(chunks, "BM25")
    ctext = {c.id: c.raw_text for c in chunks}
    graph = GraphRetriever(kg, d2c, name="Graph(oracle)", chunk_text=ctext)
    graph_nohub = GraphRetriever(kg, d2c, name="Graph(无枢纽阻断)", hub_degree=None,
                                 chunk_text=ctext)
    hybrid = RRFFusion([bm25, graph], name="RRF(朴素)")

    retrievers = [bm25]
    embedder = get_embedder()
    dense = None
    if embedder is not None:
        dense = DenseRetriever(embedder, name="Dense")
        dense.index([(c.id, c.text) for c in chunks], verbose=True)
        retrievers.append(dense)
        lexical = RRFFusion([bm25, dense], name="BM25+Dense")
        retrievers.append(lexical)
    else:
        lexical = bm25

    retrievers += [graph, graph_nohub, hybrid]
    if dense is not None:
        # 与 Routed 同样三路都有, 只差"融合 vs 路由" —— 否则比较的其实是有没有向量
        retrievers.append(RRFFusion([bm25, dense, graph], name="RRF(三路)"))

    if not args.no_extracted:
        for path, label in [("data/index/extracted_graph.json", "Graph(抽取)"),
                            ("data/index/graph_noresolve.json", "Graph(无消歧)")]:
            fp = ROOT / path
            if not fp.exists():
                print(f"[warn] 缺少 {path}, 跳过 —— 先跑 python run.py build-graph")
                continue
            kg2 = KnowledgeGraph.load(fp)
            g2 = GraphRetriever(kg2, d2c, name=label, chunk_text=ctext)
            retrievers.append(g2)
            retrievers.append(RoutedRetriever(
                lexical, g2, QueryRouter(EntityLinker(kg2)),
                name=f"Routed({label[6:-1]})"))
    retrievers.append(RoutedRetriever(lexical, graph, QueryRouter(EntityLinker(kg)),
                                      name="Routed(完整)"))

    ev = RetrievalEvaluator(c2d, items, split=args.split)
    group_b = []
    for r in retrievers:
        t0 = time.perf_counter()
        rep = ev.evaluate(r, keep_detail=args.detail)
        ms = (time.perf_counter() - t0) * 1000 / max(1, len(ev.items))
        group_b.append(rep)
        print(f"\n{format_report(rep)}\n  单次检索 {ms:.1f}ms")

    print("\n" + compare_table(group_b, k=10))
    print("\n" + compare_table(group_b, k=5))

    out = ROOT / "reports" / "phase1_retrieval.json"
    write_text(out, json.dumps({
        "split": args.split or "all", "strategy": args.strategy,
        "group_a_chunking": [_ser(r) for r in group_a],
        "group_b_ablation": [_ser(r) for r in group_b],
    }, ensure_ascii=False, indent=2))
    print(f"\n-> {out}")
    return 0


def _ser(r) -> dict:
    return {"retriever": r.retriever,
            "overall": {"n": r.overall.n, "recall": r.overall.recall,
                        "hit": r.overall.hit, "mrr": r.overall.mrr},
            "by_type": {t: {"n": s.n, "recall": s.recall, "hit": s.hit,
                            "mrr": s.mrr} for t, s in r.by_type.items()},
            "detail": r.detail}


if __name__ == "__main__":
    raise SystemExit(main())
