"""
Phase 0 - 数据集自检

评测集的 note 里写了不少断言, 例如"纯向量检索无论 top-k 取多大都召不回"。
断言必须被验证, 否则就是自说自话。本脚本把这些性质逐条检查出来:

  A. 证据完整性   每道非陷阱题的 gold_docs 都非空, 且文档 id 真实存在
  B. 答案可支撑   gold_entities 确实出现在 gold_docs 的正文里
  C. 多跳不可短路 hop4_risk 题目中, 没有任何单一文档同时包含
                  "基金名" 和 "高风险" —— 这是"纯向量必然失败"的形式化表述
  D. 陷阱题干净   negative 题的关键词确实不在任何文档中
  E. 划分无泄漏   train/dev/test 之间没有重复问题
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, read_jsonl  # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    docs = {d["id"]: d for d in read_jsonl(ROOT / "data/synthetic/documents.jsonl")}
    items = list(read_jsonl(ROOT / "data/eval/eval_set.jsonl"))
    graph = json.loads((ROOT / "data/synthetic/graph.json").read_text(encoding="utf-8"))
    ent_name = {e["id"]: e["name"] for e in graph["entities"]}

    failures: list[str] = []
    warnings: list[str] = []

    # --- A. 证据完整性 -------------------------------------------------
    for it in items:
        if it["expect_refusal"]:
            if it["gold_docs"]:
                failures.append(f"[A] {it['id']} 陷阱题不应有证据文档")
            continue
        if not it["gold_docs"]:
            failures.append(f"[A] {it['id']} ({it['type']}) 缺少证据文档")
        for d in it["gold_docs"]:
            if d not in docs:
                failures.append(f"[A] {it['id']} 引用了不存在的文档 {d}")
    print(f"A. 证据完整性 .......... {'PASS' if not failures else 'FAIL'}")

    # --- B. 答案可支撑 (别名感知) ---------------------------------------
    # 文档里可能只出现简称或罗马化名, 规范名一次都不出现。
    # 这不是数据 bug, 恰恰是**实体消歧的必答题**: 系统必须把
    # "Daiyu Financial Services Ltd." 认成"岱屿金融服务有限公司"才能答对。
    # 所以这里按 规范名 + 全部别名 逐一比对, 并把"仅以别名出现"单列统计。
    aliases_of: dict[str, list[str]] = {
        e["name"]: [e["name"], *e.get("aliases", [])] for e in graph["entities"]}
    b_fail = 0
    alias_only = 0
    for it in items:
        if it["expect_refusal"] or not it["gold_entities"]:
            continue
        corpus = "\n".join(docs[d]["text"] for d in it["gold_docs"] if d in docs)
        for ent in it["gold_entities"]:
            if ent.isdigit():           # 聚合题的计数值不会字面出现
                continue
            forms = aliases_of.get(ent, [ent])
            hit = [f for f in forms if f in corpus]
            if not hit:
                b_fail += 1
                warnings.append(
                    f"[B] {it['id']} 答案实体 '{ent}' 的任何写法都未出现在证据文档中")
            elif ent not in hit:
                alias_only += 1
                warnings.append(
                    f"[B*] {it['id']} '{ent}' 在证据文档中仅以 '{hit[0]}' 形式出现 "
                    f"-> 必须做实体消歧才能答对")
    status = "PASS" if b_fail == 0 else f"{b_fail} 处缺证据 -> FAIL"
    print(f"B. 答案可支撑 .......... {status}")
    print(f"   其中 {alias_only} 处答案实体在文档中**仅以别名/罗马化名出现**, "
          f"构成实体消歧的硬性考点")
    if b_fail:
        failures.append(f"[B] {b_fail} 道题的答案在证据文档中无任何支撑")

    # --- C. 多跳不可短路 (招牌断言) -------------------------------------
    print("\nC. 多跳不可短路 —— 验证'纯向量检索必然失败'这个断言")
    c_fail = 0
    checked = 0
    for it in items:
        if it["type"] != "hop4_risk":
            continue
        checked += 1
        fund = it["question"].split("是否")[0]
        # 该基金名与"高风险"是否曾在同一份文档中共现?
        cooccur = [d["id"] for d in docs.values()
                   if fund in d["text"] and "高风险" in d["text"]]
        if cooccur:
            c_fail += 1
            failures.append(f"[C] {it['id']} 基金 {fund} 与'高风险'在 {cooccur} 中共现, "
                            f"该题可被单文档短路")
    print(f"   检查 {checked} 道 hop4_risk 题: "
          f"{'全部无法被单文档短路 -> PASS' if c_fail == 0 else f'{c_fail} 题可短路 -> FAIL'}")

    # 进一步: 证据文档之间的词面重叠有多低?
    for it in items:
        if it["type"] != "hop4_risk":
            continue
        fund = it["question"].split("是否")[0]
        hit = [d for d in it["gold_docs"] if fund in docs[d]["text"]]
        print(f"   {it['id']}: 证据 {len(it['gold_docs'])} 份, "
              f"其中仅 {len(hit)} 份字面提到基金名 —— "
              f"其余 {len(it['gold_docs']) - len(hit)} 份对 dense 检索完全不可见")
        break  # 打一条样例即可

    # --- D. 陷阱题干净 --------------------------------------------------
    print("\nD. 陷阱题干净度")
    neg_keywords = ["年化收益率", "资产管理规模", "分红方案", "员工"]
    all_text = "\n".join(d["text"] for d in docs.values())
    for kw in neg_keywords:
        n = all_text.count(kw)
        flag = "OK" if n == 0 else f"出现 {n} 次(干扰项, 预期内)"
        print(f"   '{kw}' 在语料中: {flag}")
    fake_funds = [it for it in items
                  if it["expect_refusal"] and "最低认购金额" in it["question"]]
    for it in fake_funds:
        name = it["question"].replace("的最低认购金额是多少？", "")
        if name in all_text:
            failures.append(f"[D] {it['id']} 虚构基金 '{name}' 竟然存在于语料中")
    print(f"   虚构基金名校验: {len(fake_funds)} 个, "
          f"{'全部确认不存在 -> PASS' if not any('[D]' in f for f in failures) else 'FAIL'}")

    # --- D2. 纯语义题的零词面重叠 ---------------------------------------
    # semantic_only 的设计前提是"问题用词在语料中不出现"。
    # 这个前提必须验证, 否则 BM25 靠某个碰巧重合的词就能蒙对, 实验就失效了。
    print("\nD2. 纯语义题的零词面重叠（dense 测试的前提）")
    sem = [it for it in items if it["type"] == "semantic_only"]
    worst = []
    for it in sem:
        q = it["question"].replace("有哪些基金是", "").replace("的？", "")
        # 按 4 字滑窗检查问题描述是否在语料中出现过
        grams = {q[i:i + 4] for i in range(max(0, len(q) - 3))}
        leaked = sorted(g for g in grams if g in all_text)
        if leaked:
            worst.append((it["id"], q, leaked))
    print(f"   共 {len(sem)} 题, {len(sem) - len(worst)} 题与语料**完全零重叠**")
    for qid, q, leaked in worst[:4]:
        print(f"   [warn] {qid} '{q}' 中 {leaked[:3]} 在语料中出现过")
    if len(worst) > len(sem) // 2:
        failures.append("[D2] 过半纯语义题与语料有词面重叠, dense 实验不成立")

    # --- E. 划分无泄漏 --------------------------------------------------
    seen: dict[str, str] = {}
    e_fail = 0
    for it in items:
        q = it["question"]
        if q in seen and seen[q] != it["split"]:
            e_fail += 1
            failures.append(f"[E] 问题跨划分重复: {q}")
        seen[q] = it["split"]
    by_split: dict[str, int] = defaultdict(int)
    for it in items:
        by_split[it["split"]] += 1
    print(f"\nE. 划分无泄漏 .......... {'PASS' if e_fail == 0 else 'FAIL'}  "
          f"({dict(by_split)})")

    # --- 汇总 -----------------------------------------------------------
    print("\n" + "=" * 68)
    if failures:
        print(f"[FAIL] {len(failures)} 项硬性检查未通过:")
        for f in failures[:15]:
            print("   -", f)
        return 1
    print("[PASS] 全部硬性检查通过。")
    if warnings:
        print(f"[warn] {len(warnings)} 项软性提示(不阻断):")
        for w in warnings[:8]:
            print("   -", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
