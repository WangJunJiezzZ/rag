#!/usr/bin/env python3
"""
Phase 4 - Agent 评测: 固定流水线 vs Agent(通过 MCP 自己编排工具)

    python run.py eval-agent -- --split dev                     只读缓存(默认, 不花钱)
    python run.py eval-agent -- --split dev --online            允许调用 DeepSeek
    python run.py eval-agent -- --split dev --online --limit 20 先跑一小批
    python run.py eval-agent -- --split test --graph data/synthetic/graph.json --online

**默认绝不联网。** 必须显式加 --online 才会调用 API; 联网前先打印预估花费,
累计估算花费超过 --max-usd 后不再发起新题。跑过的题进缓存, 重跑免费。

调 Agent 的 prompt / 工具描述只许看 train、dev; 最终数字报 test。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import read_jsonl, setup_console, write_text   # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]

# 按 mcp-chat 的实测量级估算(每题约 4 轮模型调用、1.4 万 token), 只用于联网前提示
EST_USD_PER_Q = 0.005

TYPE_ORDER = ["fact_direct", "fact_paraphrase", "semantic_only", "fact_disambig",
              "hop2", "hop3", "relation_path", "aggregation", "negative"]
TYPE_LABEL = {"fact_direct": "单跳事实", "fact_paraphrase": "同义改写",
              "semantic_only": "纯语义", "fact_disambig": "易混实体",
              "hop2": "两跳关系", "hop3": "三跳关系", "relation_path": "关系路径",
              "aggregation": "列举比较", "negative": "幻觉陷阱"}


def alias_map() -> dict[str, list[str]]:
    from graphrag.store.graph import KnowledgeGraph
    kg = KnowledgeGraph.load(ROOT / "data/synthetic/graph.json")
    return {e.name: [e.name, *e.aliases] for e in kg.entities.values()}


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "negative"]
    neg = [r for r in rows if r["type"] == "negative"]
    return {
        "n": len(rows),
        "correct": _mean([r["correct"] for r in rows]),
        "answerable_correct": _mean([r["correct"] for r in ans]),
        "false_refusal": _mean([r["false_refusal"] for r in ans]),
        "trap_refusal": _mean([r["refusal_correct"] for r in neg]),
        "citation_grounded": _mean([r["citation_grounded"] for r in rows]),
        "tool_calls": _mean([r["tool_calls"] for r in rows]),
        "llm_calls": _mean([r["llm_calls"] for r in rows]),
        "tokens": _mean([r["input_tokens"] + r["output_tokens"] for r in rows]),
        "cost_usd": sum(r["cost_usd"] for r in rows),
        "max_steps": sum(1 for r in rows if r.get("stopped") == "max_steps"),
    }


def pct(x):
    return "  —  " if x != x else f"{x:5.0%}"


def report(pipe: list[dict], agent: list[dict]) -> str:
    # 只比两边都跑出结果的题, 否则缓存缺口会伪装成分数差
    common = {r["id"] for r in pipe} & {r["id"] for r in agent}
    pipe = [r for r in pipe if r["id"] in common]
    agent = [r for r in agent if r["id"] in common]
    P, A = summarize(pipe), summarize(agent)
    out = [f"共同题数 {len(common)}", "",
           f"{'':18s}{'固定流水线':>10s}{'Agent':>10s}",
           f"{'答对率(全部)':16s}{pct(P['correct']):>12s}{pct(A['correct']):>10s}",
           f"{'  可答题答对率':15s}{pct(P['answerable_correct']):>12s}{pct(A['answerable_correct']):>10s}",
           f"{'  误拒答率':16s}{pct(P['false_refusal']):>12s}{pct(A['false_refusal']):>10s}",
           f"{'  陷阱题正确拒答':13s}{pct(P['trap_refusal']):>12s}{pct(A['trap_refusal']):>10s}",
           f"{'引用落在标准证据':13s}{pct(P['citation_grounded']):>12s}{pct(A['citation_grounded']):>10s}",
           f"{'每题工具调用':15s}{P['tool_calls']:>12.1f}{A['tool_calls']:>10.1f}",
           f"{'每题模型调用':15s}{P['llm_calls']:>12.1f}{A['llm_calls']:>10.1f}",
           f"{'每题 token':16s}{P['tokens']:>12,.0f}{A['tokens']:>10,.0f}",
           f"{'估算总花费':15s}{'$%.4f' % P['cost_usd']:>12s}{'$%.4f' % A['cost_usd']:>10s}",
           f"{'达到步数上限':15s}{'—':>12s}{A['max_steps']:>10d}",
           "", "按题型(答对率):",
           f"  {'题型':10s}{'n':>4s}{'流水线':>9s}{'Agent':>8s}{'差值':>8s}{'Agent 工具/题':>12s}"]
    bp, ba = defaultdict(list), defaultdict(list)
    for r in pipe:
        bp[r["type"]].append(r)
    for r in agent:
        ba[r["type"]].append(r)
    for t in TYPE_ORDER:
        if t not in bp:
            continue
        p, a = _mean([r["correct"] for r in bp[t]]), _mean([r["correct"] for r in ba[t]])
        out.append(f"  {TYPE_LABEL.get(t, t):10s}{len(bp[t]):>4d}{pct(p):>9s}{pct(a):>8s}"
                   f"{a - p:>+8.0%}{_mean([r['tool_calls'] for r in ba[t]]):>12.1f}")
    # 逐题分歧: 这是失败分析的入口
    pm = {r["id"]: r for r in pipe}
    flips = [(r["id"], r["type"], pm[r["id"]]["correct"], r["correct"]) for r in agent
             if r["correct"] != pm[r["id"]]["correct"]]
    out += ["", f"两边结论不同的题 {len(flips)} 道:"]
    for qid, t, p, a in flips:
        out.append(f"  {qid} {TYPE_LABEL.get(t, t):8s} 流水线{'✓' if p else '✗'} → Agent{'✓' if a else '✗'}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="固定流水线 vs Agent")
    ap.add_argument("--split", help="train / dev / test; 不填需加 --all")
    ap.add_argument("--all", action="store_true", help="全量 109 题")
    ap.add_argument("--graph", default="data/index/extracted_graph.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--online", action="store_true", help="允许调用 DeepSeek(会花钱)")
    ap.add_argument("--max-usd", type=float, default=0.5,
                    help="实际 API 花费上限(缓存命中不计)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true",
                    help="关系类题也只用关键词判分(默认用 LLM 判官, 见 eval/judge.py)")
    args = ap.parse_args()
    if not args.split and not args.all:
        ap.error("请指定 --split train/dev/test, 或 --all")

    import anyio
    from graphrag.eval.agent import run_agent, run_pipeline
    from graphrag.eval.endtoend import score_answer
    from graphrag.llm import LLM

    items = list(read_jsonl(ROOT / "data/eval/eval_set.jsonl"))
    if args.split:
        items = [i for i in items if i["split"] == args.split]
    if args.limit:
        items = items[:args.limit]
    amap = alias_map()
    score = lambda it, ans: score_answer(it, ans, amap)       # noqa: E731

    if not args.online:
        os.environ["GRAPHRAG_OFFLINE"] = "1"     # 固定流水线那一侧也不许联网
    llm = LLM(provider="deepseek", offline=not args.online)
    if args.online and llm.offline:
        print("[FAIL] 指定了 --online, 但 LLM 仍处于离线模式: "
              "检查 .env 里的 DEEPSEEK_API_KEY 与 GRAPHRAG_OFFLINE")
        return 1

    gname = "标准图谱" if "synthetic" in args.graph else "抽取图谱"
    print(f"题目 {len(items)} 道 ({args.split or 'all'}) · {gname} · "
          f"{'在线' if args.online else '只读缓存'}")
    if args.online:
        print(f"预估: 未命中缓存的题每道约 ${EST_USD_PER_Q}, 最多约 "
              f"${EST_USD_PER_Q * len(items):.2f}; 实际花费超过 ${args.max_usd} 后停止开新题")

    print("\n[1/2] 固定流水线 (从端到端评测缓存重放) ...")
    pipe = run_pipeline(items, args.graph, score)
    if pipe["missed"]:
        print(f"  [warn] {pipe['missed']} 题缓存未命中被跳过; 先跑 python run.py e2e")

    print("[2/2] Agent ...")

    def progress(done, total, spent):
        print(f"\r  {done}/{total}  实际花费 ${spent:.4f}", end="", flush=True)

    res = anyio.run(lambda: run_agent(items, args.graph, llm, score,
                                      concurrency=args.concurrency,
                                      max_steps=args.max_steps,
                                      budget_usd=args.max_usd if args.online else None,
                                      progress=progress))
    print()
    if res["missed"]:
        print(f"  [info] {res['missed']} 题没有缓存" +
              ("" if args.online else "; 加 --online 才会实际调用"))
    if res["skipped_budget"]:
        print(f"  [warn] {res['skipped_budget']} 题因达到花费上限 ${args.max_usd} 未发起")
    print(f"  {llm.report()}")

    if not res["rows"]:
        print("\n没有可比较的结果。")
        return 0
    # ---- 关系类题改由 LLM 判官判"最终结论"; 两边用同一个判官 ----
    judged = 0
    if not args.no_judge:
        from graphrag.eval.judge import Judge
        from graphrag.llm import CacheMissError
        judge = Judge(llm)
        byid = {it["id"]: it for it in items}
        for rows in (pipe["rows"], res["rows"]):
            for r in rows:
                try:
                    judge.rescore(r, byid[r["id"]], r["answer"])
                except CacheMissError:
                    r["judge_ok"] = False        # 离线且判官没缓存: 保留关键词结果
                judged += 1 if r.get("judge_ok") else 0
        flips = [(r["id"], sysname, r["kw_correct"], r["correct"])
                 for sysname, rows in (("流水线", pipe["rows"]), ("Agent", res["rows"]))
                 for r in rows if r.get("judge_ok") and r["kw_correct"] != r["correct"]]
        print(f"\nLLM 判官复判 {judged} 条关系类回答; 与关键词结论不同 {len(flips)} 条:")
        for qid, who, kw, jv in flips:
            print(f"  {qid} {who:<4} 关键词{'✓' if kw else '✗'} → 判官{'✓' if jv else '✗'}")

    print("\n" + "=" * 64 + "\n" + report(pipe["rows"], res["rows"]) + "\n" + "=" * 64)
    tools = Counter(t for r in res["rows"] for t in r["tools"])
    print("Agent 工具使用分布: " + ", ".join(f"{k} {v}" for k, v in tools.most_common()))

    suffix = ("_oracle" if "synthetic" in args.graph else "") + f"_{args.split or 'all'}"
    out = ROOT / "reports" / f"phase4_agent{suffix}.json"
    write_text(out, json.dumps({
        "graph": args.graph, "split": args.split or "all", "model": "deepseek-chat",
        "scoring": "keyword" if args.no_judge else "keyword + judge/v1_rubric(关系类题)",
        "pipeline": {"summary": summarize(pipe["rows"]), "details": pipe["rows"]},
        "agent": {"summary": summarize(res["rows"]), "details": res["rows"],
                  "missed": res["missed"], "skipped_budget": res["skipped_budget"]},
    }, ensure_ascii=False, indent=1))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
