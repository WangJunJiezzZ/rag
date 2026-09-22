"""
Phase 3 - 端到端评测 + prompt A/B

    python run.py e2e                              # 默认 prompt, 抽取图
    python run.py e2e -- --ab                      # 三版 answer prompt 横向对比
    python run.py e2e -- --split train             # 只在 train 上调 prompt
    python run.py e2e -- --generator extractive    # 零 LLM 基线

**调 prompt 只许用 train, 最终分数报 test。** 脚本会在跨划分使用时给出警告。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, read_jsonl, write_text    # noqa: E402
from graphrag.eval.endtoend import (aggregate, compare_prompts,      # noqa: E402
                                    format_e2e, score_answer)
from graphrag.llm import LLM, CacheMissError                         # noqa: E402
from graphrag.service import RAGService, ServiceConfig               # noqa: E402
from graphrag.store.graph import KnowledgeGraph                      # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]


def alias_map() -> dict[str, list[str]]:
    """答案判分要别名感知 —— 模型回答"澜图资管"不该被判成没提到
    "澜图资产管理有限公司"。"""
    kg = KnowledgeGraph.load(ROOT / "data/synthetic/graph.json")
    return {e.name: [e.name, *e.aliases] for e in kg.entities.values()}


def run_one(cfg: ServiceConfig, items: list[dict], amap: dict,
            label: str, quiet: bool = False):
    llm = LLM()
    svc = RAGService(cfg, llm=llm, quiet=quiet)
    rows, missed = [], 0
    for i, it in enumerate(items, 1):
        if not quiet and (i % 10 == 0 or i == len(items)):
            print(f"\r  {label}: {i}/{len(items)}", end="", flush=True)
        try:
            ans = svc.ask(it["question"])
        except CacheMissError:
            missed += 1
            continue
        rows.append(score_answer(it, ans, amap))
    if not quiet:
        print()
    if missed:
        print(f"  [warn] {missed}/{len(items)} 题因 replay 缓存未命中被跳过。"
              f"演示前请先用真实 provider 预跑一遍。")
    return aggregate(rows, label, cfg.prompt_id, llm.report())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="train / dev / test")
    ap.add_argument("--ab", action="store_true", help="三版 answer prompt 对比")
    ap.add_argument("--prompt", default="answer/v3_guarded")
    ap.add_argument("--generator", default="llm", choices=["llm", "extractive"])
    ap.add_argument("--graph", default="data/index/extracted_graph.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = list(read_jsonl(ROOT / "data/eval/eval_set.jsonl"))
    if args.split:
        items = [i for i in items if i["split"] == args.split]
    if args.limit:
        items = items[:args.limit]
    amap = alias_map()

    if args.split == "test":
        print("[提示] 正在 test 上评测。test 只应在最终定稿时跑一次，"
              "反复在 test 上迭代等于把它变成 train。\n")
    elif not args.split:
        print("[提示] 未指定 --split，正在全量评测。调 prompt 请用 --split train。\n")

    # 没有可用 provider 且缓存为空时自动降级为抽取式基线 ——
    # 否则首次运行会看到一屏 cache miss, 分不清是环境问题还是代码问题。
    generator = args.generator
    if generator == "llm":
        probe = LLM()
        if probe.provider_name == "replay":
            cache_root = probe.cache.root
            has_cache = cache_root.exists() and any(cache_root.rglob("*.json"))
            if not has_cache:
                print("[降级] 未检测到可用的 LLM provider，且回放缓存为空。\n"
                      "       本次改用 extractive 抽取式基线（零 LLM）跑通端到端链路。\n"
                      "       要评测真实生成质量，请设置 ANTHROPIC_API_KEY 或 "
                      "DEEPSEEK_API_KEY 后重跑：\n"
                      "         python run.py e2e -- --ab --split train\n")
                generator = "extractive"
                args.ab = False

    prompts = (["answer/v1_naive", "answer/v2_cited", "answer/v3_guarded"]
               if args.ab else [args.prompt])
    reps = []
    for pid in prompts:
        cfg = ServiceConfig(prompt_id=pid, graph_path=args.graph,
                            generator=generator)
        label = ("extractive(基线)" if generator == "extractive"
                 else pid.split("/")[-1])
        rep = run_one(cfg, items, amap, label, quiet=len(prompts) > 1)
        reps.append(rep)
        print(f"\n{format_e2e(rep)}\n")

    if len(reps) > 1:
        print("=" * 76)
        print(compare_prompts(reps))
        print("=" * 76)

    out = ROOT / "reports" / f"phase3_e2e{'_ab' if args.ab else ''}.json"
    write_text(out, json.dumps([{
        "prompt": r.prompt_id, "name": r.name,
        "overall": r.overall.__dict__,
        "by_type": {t: s.__dict__ for t, s in r.by_type.items()},
        "details": r.details,
    } for r in reps], ensure_ascii=False, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
