"""
Phase 2 - 从文档抽取知识图谱, 并评测抽取质量。

    python run.py build-graph                    # 规则抽取(无需 API key) -> graph_rule.json
    python run.py build-graph -- --extractor llm --prompt extract_triples/v4_structural
                                                 # LLM 抽取 -> extracted_graph.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, read_jsonl, write_text       # noqa: E402
from graphrag.embedding import get_embedder                            # noqa: E402
from graphrag.eval.extraction import (evaluate_extraction,             # noqa: E402
                                      format_extraction)
from graphrag.ingest.extract import LLMExtractor, RuleExtractor        # noqa: E402
from graphrag.ingest.pipeline import build_graph                       # noqa: E402
from graphrag.ingest.resolve import EntityResolver                     # noqa: E402
from graphrag.llm import LLM                                           # noqa: E402
from graphrag.store.graph import KnowledgeGraph                        # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="rule", choices=["rule", "llm"])
    ap.add_argument("--prompt", default="extract_triples/v3_fewshot")
    ap.add_argument("--no-resolve", action="store_true", help="关闭实体消歧(消融)")
    ap.add_argument("--no-evidence-check", action="store_true",
                    help="关闭 evidence 回查(消融)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 份文档")
    ap.add_argument("--workers", type=int, default=8,
                    help="抽取并发度 (LLM 模式生效)")
    ap.add_argument("--out", default=None,
                    help="缺省: LLM 抽取写 extracted_graph.json(演示与评测用的那张图); "
                         "规则抽取写 graph_rule.json —— 不能让 `run.py all` 里的规则基线"
                         "覆盖掉 LLM 抽取图, 否则离线演示的缓存全部对不上")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("data/index/extracted_graph.json" if args.extractor == "llm"
                    else "data/index/graph_rule.json")

    docs = list(read_jsonl(ROOT / "data/synthetic/documents.jsonl"))
    if args.limit:
        docs = docs[:args.limit]
    gold = KnowledgeGraph.load(ROOT / "data/synthetic/graph.json")

    llm = LLM()
    if args.extractor == "llm":
        extractor = LLMExtractor(llm, prompt_id=args.prompt)
        print(f"抽取器: LLM  prompt={args.prompt}  provider={llm.provider_name}")
    else:
        extractor = RuleExtractor()
        print("抽取器: 规则（正则）—— 仅在模板化语料上有效，分数不具外推性")

    resolver = None
    if not args.no_resolve:
        resolver = EntityResolver(embedder=get_embedder(),
                                  llm=llm if llm.provider_name != "replay" else None)

    print(f"文档 {len(docs)} 份\n")
    res = build_graph(docs, extractor, resolver,
                      check_evidence=not args.no_evidence_check,
                      workers=args.workers if args.extractor == "llm" else 1)

    print("\n[抽取]")
    print("  " + res.extract_stats.summary().replace("\n", "\n  "))
    print("\n[消歧]")
    print("  " + res.resolve_stats.summary().replace("\n", "\n  "))
    for ex in res.resolve_stats.examples[:8]:
        print(f"    {ex}")

    print(f"\n[图谱] {res.graph.stats()}")
    print(f"[耗时] {res.seconds:.1f}s\n")

    rep = evaluate_extraction(res.graph, gold)
    print(format_extraction(rep, extractor.name))

    out = ROOT / args.out
    res.graph.save(out)
    write_text(ROOT / "reports" / f"phase2_extraction_{extractor.name.replace(':', '_')}.json",
               json.dumps({
                   "extractor": extractor.name,
                   "prompt": args.prompt if args.extractor == "llm" else None,
                   "resolve": not args.no_resolve,
                   "evidence_check": not args.no_evidence_check,
                   "extract_stats": res.extract_stats.__dict__,
                   "resolve_stats": res.resolve_stats.__dict__,
                   "precision": rep.precision, "recall": rep.recall, "f1": rep.f1,
                   "by_relation": rep.by_relation,
                   "unmapped_entities": rep.unmapped_entities,
               }, ensure_ascii=False, indent=2, default=str))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
