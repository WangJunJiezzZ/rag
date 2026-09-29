"""
Phase 0 - 数据集自检

评测集的 note 里写了不少断言, 例如"没有任何一份文档同时提到小灰灰和小香香"。
断言必须被验证, 否则就是自说自话。本脚本把这些性质逐条检查出来:

  A. 证据完整性   每道非陷阱题的 gold_docs 都非空, 且文档 id 真实存在
  B. 答案可支撑   gold_entities 确实出现在 gold_docs 的正文里(推理型答案除外)
  C. 多跳不可短路 relation_path 题目中, 没有任何单一文档同时包含两端的角色
                  —— 这是"纯向量检索必然失败"的形式化表述
  D. 陷阱题干净   negative 题的关键词在语料中的出现情况(同类信息是干扰项)
  D2. 纯语义零重叠 semantic_only 题与语料没有 4 字以上的词面重叠
  E. 划分无泄漏   train/dev/test 之间没有重复问题
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, read_jsonl  # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]


def _squash(s: str) -> str:
    return re.sub(r"[\s　]+", "", s)


def main() -> int:
    docs = {d["id"]: d for d in read_jsonl(ROOT / "data/synthetic/documents.jsonl")}
    items = list(read_jsonl(ROOT / "data/eval/eval_set.jsonl"))
    graph = json.loads((ROOT / "data/synthetic/graph.json").read_text(encoding="utf-8"))

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
    # 文档里可能只出现别名, 规范名一次都不出现(如剧情文档只写"高圆寺博士")。
    # 这不是数据 bug, 恰恰是**实体消歧的必答题**。
    # 所以这里按 规范名 + 全部别名 逐一比对, 并把"仅以别名出现"单列统计。
    aliases_of: dict[str, list[str]] = {
        e["name"]: [e["name"], *e.get("aliases", [])] for e in graph["entities"]}
    b_fail = 0
    alias_only = 0
    for it in items:
        if it["expect_refusal"] or not it["gold_entities"] or it.get("derived"):
            continue
        corpus = _squash("\n".join(docs[d]["text"] for d in it["gold_docs"] if d in docs))
        for ent in it["gold_entities"]:
            if ent.isdigit():           # 计数值不会字面出现
                continue
            forms = aliases_of.get(ent, [ent])
            hit = [f for f in forms if _squash(f) in corpus]
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
    planted = graph.get("planted", {}).get("alias_only", [])
    print(f"   语料中植入 {len(planted)} 处'证据只写别名': "
          + "、".join(f"{p['alias']}→{p['canonical']}" for p in planted))
    if b_fail:
        failures.append(f"[B] {b_fail} 道题的答案在证据文档中无任何支撑")

    # --- C. 多跳不可短路 (招牌断言) -------------------------------------
    print("\nC. 多跳不可短路 —— 验证'纯向量检索必然失败'这个断言")

    def mentions(name: str, text: str) -> bool:
        return any(f in text for f in aliases_of.get(name, [name]))

    c_fail = 0
    shortcut_hops: dict[str, list[str]] = defaultdict(list)
    for it in items:
        if it["type"] not in ("relation_path", "hop2", "hop3"):
            continue
        # 两端: 路径起点实体 与 标准答案实体
        path = it["gold_path"]
        if not path:
            continue
        names_in_q = [n for n in aliases_of if n in it["question"]]
        names_in_q = [n for n in names_in_q
                      if not any(n != m and n in m for m in names_in_q)]   # 最长匹配
        if it["type"] == "relation_path":
            ends = names_in_q[:2]
        else:
            ends = names_in_q[:1] + [e for e in it["gold_entities"] if e in aliases_of][:1]
        if len(ends) < 2:
            continue
        both = [d["id"] for d in docs.values()
                if mentions(ends[0], d["text"]) and mentions(ends[1], d["text"])]
        if both:
            if it["type"] == "relation_path":
                c_fail += 1
                failures.append(f"[C] {it['id']} {ends[0]} 与 {ends[1]} 在 {both} 中共现, "
                                f"该题可被单文档短路")
            else:
                shortcut_hops[it["type"]].append(it["id"])
    n_rel = sum(1 for it in items if it["type"] == "relation_path")
    print(f"   检查 {n_rel} 道 relation_path 题: "
          f"{'两端角色从不共现于同一文档 -> PASS' if c_fail == 0 else f'{c_fail} 题可短路 -> FAIL'}")
    for t, ids in shortcut_hops.items():
        print(f"   [info] {t} 中有 {len(ids)} 题的两端在某份文档里共现"
              f"(可被单文档部分短路, 不阻断): {', '.join(ids[:6])}")

    # --- D. 陷阱题干净 --------------------------------------------------
    print("\nD. 陷阱题干扰项")
    all_text = "\n".join(d["text"] for d in docs.values())
    for it in items:
        if not it["expect_refusal"]:
            continue
        for kw in it.get("trap_keywords", []):
            n = all_text.count(kw)
            flag = "未出现" if n == 0 else f"出现 {n} 次(同类信息, 是干扰项)"
            print(f"   {it['id']} '{kw}': {flag}")

    # --- D2. 纯语义题的零词面重叠 ---------------------------------------
    # semantic_only 的设计前提是"问题用词在语料中不出现"。
    # 这个前提必须验证, 否则 BM25 靠某个碰巧重合的词就能蒙对, 实验就失效了。
    print("\nD2. 纯语义题的零词面重叠（dense 测试的前提）")
    sem = [it for it in items if it["type"] == "semantic_only"]
    worst = []
    for it in sem:
        q = re.sub(r"^(哪只狼|哪只羊|哪个机器人|哪位角色|谁)", "", it["question"]).rstrip("？")
        grams = {q[i:i + 4] for i in range(max(0, len(q) - 3))}
        leaked = sorted(g for g in grams if g in all_text)
        if leaked:
            worst.append((it["id"], q, leaked))
    print(f"   共 {len(sem)} 题, {len(sem) - len(worst)} 题与语料**完全零重叠**(4 字滑窗)")
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
