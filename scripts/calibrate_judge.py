#!/usr/bin/env python3
"""
校准 LLM 判官: 在标注集上比对判官与标注的一致率。

    python run.py calibrate-judge                只读缓存(默认)
    python run.py calibrate-judge -- --online    允许调用 DeepSeek(约 41 条, < $0.02)

标注集 data/eval/judge_labels.jsonl 每行绑定一条**具体的回答全文**, 标注的是
"这条回答的最终结论对不对"。labeler / human_reviewed 字段如实记录是谁标的、
有没有人工复核过 —— AI 给出的标注不能当人工标注报告。

判官只有在一致率达标后才用于正式评测; 不一致的每一条都要打印出来逐条看。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import read_jsonl, setup_console, write_text   # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/eval/judge_labels.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="允许调用 DeepSeek")
    ap.add_argument("--prompt", default="judge/v2_rubric")
    ap.add_argument("--min-agree", type=float, default=0.95)
    args = ap.parse_args()
    if not args.online:
        os.environ["GRAPHRAG_OFFLINE"] = "1"

    from graphrag.eval.judge import Judge, load_labels
    from graphrag.llm import LLM, CacheMissError

    items = {i["id"]: i for i in read_jsonl(ROOT / "data/eval/eval_set.jsonl")}
    labels = load_labels(LABELS)
    llm = LLM(provider="deepseek", offline=not args.online)
    if args.online and llm.offline:
        print("[FAIL] --online 但 LLM 仍离线: 检查 DEEPSEEK_API_KEY / GRAPHRAG_OFFLINE")
        return 1
    judge = Judge(llm, args.prompt)

    rows, missed, broken = [], 0, 0
    for lab in labels:
        try:
            v = judge.judge(items[lab["id"]], lab["answer"])
        except CacheMissError:
            missed += 1
            continue
        if not v["ok"]:
            broken += 1
        rows.append({**{k: lab[k] for k in ("id", "system", "label", "kw_correct", "note")},
                     "set": lab.get("set", "calibration"),
                     "judge": v["correct"], "reason": v["reason"], "ok": v["ok"]})

    if missed:
        print(f"[info] {missed} 条没有缓存" + ("" if args.online else "; 加 --online 才会调用"))
    if not rows:
        return 0
    ok = [r for r in rows if r["ok"]]
    agree = sum(1 for r in ok if r["judge"] == r["label"]) / len(ok)
    kw_agree = sum(1 for r in rows if r["kw_correct"] == r["label"]) / len(rows)
    tp = sum(1 for r in ok if r["label"] == 1 and r["judge"] == 1)
    tn = sum(1 for r in ok if r["label"] == 0 and r["judge"] == 0)
    fp = sum(1 for r in ok if r["label"] == 0 and r["judge"] == 1)   # 判官放水
    fn = sum(1 for r in ok if r["label"] == 1 and r["judge"] == 0)   # 判官误伤

    humans = sum(1 for lab in labels if lab.get("human_reviewed"))
    print(f"判官 {args.prompt}  标注 {len(labels)} 条(人工复核 {humans} 条)  "
          f"判官输出无法解析 {broken} 条")
    print(f"  判官 vs 标注 一致率 {agree:.1%}   (关键词判分 vs 标注: {kw_agree:.1%})")
    # 分集合报: 写规则时见过的(calibration) 与 写完规则后才出现的(holdout) 必须分开看,
    # 否则在"考过的题"上拿满分会掩盖泛化问题
    by_set = {}
    for name in sorted({r["set"] for r in ok}):
        sub = [r for r in ok if r["set"] == name]
        by_set[name] = sum(1 for r in sub if r["judge"] == r["label"]) / len(sub)
        print(f"    {name:<12} {sum(1 for r in sub if r['judge'] == r['label'])}/{len(sub)} = {by_set[name]:.1%}")
    print(f"  混淆: 标注对&判对 {tp}  标注错&判错 {tn}  放水(标注错判对) {fp}  误伤(标注对判错) {fn}")
    dis = [r for r in ok if r["judge"] != r["label"]]
    print(f"\n不一致 {len(dis)} 条:")
    for r in dis:
        print(f"  {r['id']} {r['system']:<8} 标注={'对' if r['label'] else '错'} "
              f"判官={'对' if r['judge'] else '错'}\n     判官理由: {r['reason']}\n     标注备注: {r['note']}")
    print(f"\n{llm.report()}")

    tag = args.prompt.split("/")[-1]
    write_text(ROOT / "reports" / f"judge_calibration_{tag}.json", json.dumps({
        "prompt": args.prompt, "n": len(ok), "agreement": agree, "kw_agreement": kw_agree,
        "agreement_by_set": by_set,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "human_reviewed": humans, "rows": rows}, ensure_ascii=False, indent=1))
    passed = agree >= args.min_agree
    print(("[PASS]" if passed else "[FAIL]") + f" 一致率 {agree:.1%} "
          + ("≥" if passed else "<") + f" 门槛 {args.min_agree:.0%}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
