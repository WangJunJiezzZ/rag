"""
抽取质量评测：把抽出来的图和 ground-truth 图逐条比对。

比对的难点在实体对齐
--------------------
抽取出的名称是"藏之助", 标准答案里是"吉祥寺藏之助"。
直接比字符串会把正确的抽取判成错误。所以先用 ground-truth 的
名称+别名索引把抽取实体映射回标准实体 id, 再比三元组。

**注意这会让分数偏乐观**, 而且已经被实测证实:

    早期基金语料上: 规则抽取的 F1 = 99.8%, 但用同一张抽取图跑图检索,
    recall@10 比 oracle 掉了 7.6 点。

原因就在这里 —— F1 的计算过程用标准答案的别名索引把"藏之助"对齐回了
"吉祥寺藏之助", 于是消歧失败被**掩盖**了; 而图检索没有这个外挂,
两个未合并的节点之间的路径是断的。

所以本报告额外输出 `entity_dedup_ratio`: 抽取实体数 / 标准实体数。
它 > 1 就说明存在未合并的别名, 数值直接反映消歧的欠账。
**只看 F1 会对系统能力产生系统性高估。**
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..store.graph import KnowledgeGraph, normalize_name


# 对称关系: 「A 的丈夫是 B」与「B 的妻子是 A」是同一条事实, 比对时不区分方向
SYMMETRIC = {"SPOUSE_OF", "FRIEND_OF"}


def _key(s: str, r: str, d: str) -> tuple[str, str, str]:
    if r in SYMMETRIC and d < s:
        s, d = d, s
    return (s, r, d)


def _lit(x: str) -> str:
    """字面量比对前的归一: 去空白与标点。
    "我一定会回来的！" 与 "我一定会回来的"、"3 分钟" 与 "3分钟" 是同一个值。"""
    return "lit:" + normalize_name(x[4:])


@dataclass
class ExtractionReport:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_extracted: int = 0
    n_gold: int = 0
    n_correct: int = 0
    unmapped_entities: int = 0
    mapped_entities: int = 0
    entity_dedup_ratio: float = 1.0      # 抽取实体数 / 标准实体数; >1 = 有未合并别名
    collapsed_gold: int = 0              # 多少个标准实体被抽成了 >1 个节点
    by_relation: dict = field(default_factory=dict)
    missed_examples: list[str] = field(default_factory=list)
    spurious_examples: list[str] = field(default_factory=list)


def _gold_index(gold: KnowledgeGraph) -> dict[str, str]:
    idx: dict[str, str] = {}
    for e in gold.entities.values():
        for form in [e.name, *e.aliases]:
            idx.setdefault(normalize_name(form), e.id)
            idx.setdefault(form.strip().lower(), e.id)
    return idx


def evaluate_extraction(extracted: KnowledgeGraph,
                        gold: KnowledgeGraph) -> ExtractionReport:
    idx = _gold_index(gold)
    rep = ExtractionReport()

    def to_gold(eid: str) -> str | None:
        if eid.startswith("lit:"):
            return _lit(eid)
        e = extracted.entities.get(eid)
        if e is None:
            return None
        for form in [e.name, *e.aliases]:
            hit = idx.get(normalize_name(form)) or idx.get(form.strip().lower())
            if hit:
                return hit
        return None

    mapping: dict[str, str | None] = {eid: to_gold(eid) for eid in extracted.entities}
    rep.mapped_entities = sum(1 for v in mapping.values() if v)
    rep.unmapped_entities = sum(1 for v in mapping.values() if not v)

    # 消歧欠账: 有多少标准实体被拆成了多个抽取节点
    inv: dict[str, int] = defaultdict(int)
    for gid in mapping.values():
        if gid:
            inv[gid] += 1
    rep.collapsed_gold = sum(1 for v in inv.values() if v > 1)
    rep.entity_dedup_ratio = (len(extracted.entities) / len(gold.entities)
                              if gold.entities else 1.0)

    def norm_lit(x: str) -> str:
        return _lit(x) if x.startswith("lit:") else x

    gold_set: set[tuple] = {_key(e.src, e.rel, norm_lit(e.dst)) for e in gold.edges}
    ext_set: set[tuple] = set()
    for e in extracted.edges:
        s = mapping.get(e.src)
        d = to_gold(e.dst) if e.dst.startswith("lit:") else mapping.get(e.dst)
        if s and d:
            ext_set.add(_key(s, e.rel, d))
        else:
            rep.n_extracted += 0     # 无法对齐的三元组不计入分子分母, 单独报告

    correct = ext_set & gold_set
    rep.n_extracted = len(ext_set)
    rep.n_gold = len(gold_set)
    rep.n_correct = len(correct)
    rep.precision = len(correct) / len(ext_set) if ext_set else 0.0
    rep.recall = len(correct) / len(gold_set) if gold_set else 0.0
    rep.f1 = (2 * rep.precision * rep.recall / (rep.precision + rep.recall)
              if rep.precision + rep.recall else 0.0)

    # 按关系拆分 —— 定位是哪一类关系抽不好
    per: dict[str, dict] = defaultdict(lambda: {"gold": 0, "ext": 0, "ok": 0})
    for s, r, d in gold_set:
        per[r]["gold"] += 1
    for s, r, d in ext_set:
        per[r]["ext"] += 1
    for s, r, d in correct:
        per[r]["ok"] += 1
    for r, v in per.items():
        v["precision"] = v["ok"] / v["ext"] if v["ext"] else 0.0
        v["recall"] = v["ok"] / v["gold"] if v["gold"] else 0.0
    rep.by_relation = dict(per)

    def fmt(t: tuple) -> str:
        return f"{gold.name(t[0])} -[{t[1]}]-> {gold.name(t[2])}"
    rep.missed_examples = [fmt(t) for t in list(gold_set - ext_set)[:8]]
    rep.spurious_examples = [fmt(t) for t in list(ext_set - gold_set)[:8]]
    return rep


def format_extraction(rep: ExtractionReport, name: str) -> str:
    lines = [f"抽取器: {name}",
             "-" * 68,
             f"  三元组  抽出 {rep.n_extracted}  标准 {rep.n_gold}  正确 {rep.n_correct}",
             f"  precision {rep.precision:.1%}   recall {rep.recall:.1%}   "
             f"F1 {rep.f1:.1%}",
             f"  实体     对齐 {rep.mapped_entities}  "
             f"无法对齐 {rep.unmapped_entities}",
             f"  消歧欠账 抽取/标准 实体数比 {rep.entity_dedup_ratio:.2f}  "
             f"({rep.collapsed_gold} 个标准实体被拆成多个节点)",
             f"           ^ 这一项 >1.00 说明有别名未合并。F1 用标准答案做了对齐，"
             f"会掩盖它；图检索不会。",
             "-" * 68,
             f"  {'关系':<16}{'标准':>6}{'抽出':>6}{'正确':>6}{'precision':>11}{'recall':>9}"]
    for r, v in sorted(rep.by_relation.items(), key=lambda x: -x[1]["gold"]):
        lines.append(f"  {r:<16}{v['gold']:>6}{v['ext']:>6}{v['ok']:>6}"
                     f"{v['precision']:>11.1%}{v['recall']:>9.1%}")
    if rep.missed_examples:
        lines.append("\n  漏抽示例:")
        lines += [f"    - {x}" for x in rep.missed_examples[:4]]
    if rep.spurious_examples:
        lines.append("  多抽示例:")
        lines += [f"    - {x}" for x in rep.spurious_examples[:4]]
    return "\n".join(lines)
