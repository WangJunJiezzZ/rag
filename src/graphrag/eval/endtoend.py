"""
端到端评测 —— 检索之后, 生成之前的所有努力最终体现在这里。

四个指标, 缺一不可
------------------
  answer_correct      可答题: 标准答案里的实体是否都出现在回答中(别名感知)
  refusal_correct     陷阱题: 是否正确拒答          <- 幻觉抑制的主指标
  false_refusal       可答题: 拒答了且没答对         <- 拒答护栏的代价
  citation_valid      引用编号是否真实存在
  citation_grounded   引用的文档是否确实在标准证据集里

为什么必须同时看 refusal_correct 和 false_refusal
-------------------------------------------------
只看拒答率, 一个永远回答"资料中未提及"的系统能拿满分。
只看准确率, 一个从不拒答的系统在陷阱题上会全错但总分未必难看。
**两个指标必须一起报, 单看任何一个都能被轻易刷分。**
这是 prompt 调优最容易自欺的地方。
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class E2EScore:
    n: int = 0
    correct: float = 0.0
    entity_recall: float = 0.0
    refusal_correct: float = 0.0
    false_refusal: float = 0.0
    citation_valid: float = 0.0
    citation_grounded: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class E2EReport:
    name: str
    prompt_id: str
    overall: E2EScore
    by_type: dict[str, E2EScore] = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)
    llm_report: str = ""


def _alias_forms(name: str, alias_map: dict[str, list[str]]) -> list[str]:
    return alias_map.get(name, [name])


def _squash(s: str) -> str:
    """判分前去掉空白: 标准答案写"3 分钟", 模型常写"3分钟", 两者是同一个答案。"""
    return "".join((s or "").split()).replace("　", "")


# 引用标记不是答案内容。不去掉的话, 标准答案是数字的题(如"8""10")
# 会被"[8]"或"doc-0018"命中 —— 只要引用了某份文档就算答对。
# 修复时核对过: 固定流水线在两份图谱上 109 题的判分结果均无变化。
_CITE_MARK = re.compile(r"doc-\d{4}|\[\d+\]")


def score_answer(item: dict, ans, alias_map: dict[str, list[str]]) -> dict:
    text = _squash(_CITE_MARK.sub("", ans.text))
    row: dict = {"id": item["id"], "type": item["type"],
                 "refused": bool(ans.refused)}

    if item["expect_refusal"]:
        row["refusal_correct"] = 1.0 if ans.refused else 0.0
        row["correct"] = row["refusal_correct"]
        row["entity_recall"] = float("nan")
        row["false_refusal"] = float("nan")
    else:
        golds = item["gold_entities"]
        hit = 0
        for g in golds:
            if any(f and _squash(f) in text for f in _alias_forms(g, alias_map)):
                hit += 1
        row["entity_recall"] = hit / len(golds) if golds else 1.0
        # 全部关键实体出现才算答对 —— 多跳题答一半不算对
        row["correct"] = 1.0 if golds and hit == len(golds) else 0.0
        # 误拒答 = 拒答了**而且**没答对。答对了但带一句"资料未直接写明"的保留说明,
        # 不算误拒答 —— 否则 v3 在关系路径题上 5/5 答对也会被计成 100% 误拒答。
        row["false_refusal"] = 1.0 if ans.refused and not row["correct"] else 0.0
        row["refusal_correct"] = float("nan")

    cits = ans.citations
    row["citation_valid"] = (sum(1 for c in cits if c.valid) / len(cits)
                             if cits else float("nan"))
    gold_docs = set(item["gold_docs"])
    grounded = [c for c in cits if c.valid and c.doc_id]
    row["citation_grounded"] = (
        sum(1 for c in grounded if c.doc_id in gold_docs) / len(grounded)
        if grounded and gold_docs else float("nan"))
    row["latency_ms"] = ans.latency_ms
    row["cost_usd"] = ans.cost_usd
    return row


def _mean(vals: list[float]) -> float:
    xs = [v for v in vals if v == v]          # 过滤 NaN
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(rows: list[dict], name: str, prompt_id: str,
              llm_report: str = "") -> E2EReport:
    def fold(rs: list[dict]) -> E2EScore:
        return E2EScore(
            n=len(rs),
            correct=_mean([r["correct"] for r in rs]),
            entity_recall=_mean([r["entity_recall"] for r in rs]),
            refusal_correct=_mean([r["refusal_correct"] for r in rs]),
            false_refusal=_mean([r["false_refusal"] for r in rs]),
            citation_valid=_mean([r["citation_valid"] for r in rs]),
            citation_grounded=_mean([r["citation_grounded"] for r in rs]),
            latency_ms=_mean([r["latency_ms"] for r in rs]),
            cost_usd=sum(r["cost_usd"] for r in rs))
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    return E2EReport(name=name, prompt_id=prompt_id, overall=fold(rows),
                     by_type={t: fold(rs) for t, rs in by.items()},
                     details=rows, llm_report=llm_report)


TYPE_ORDER = ["fact_direct", "fact_paraphrase", "semantic_only", "fact_disambig",
              "hop2", "hop3", "relation_path", "aggregation", "negative"]


def format_e2e(rep: E2EReport) -> str:
    head = (f"{'题型':<18}{'题数':>5}{'答对率':>9}{'实体召回':>10}"
            f"{'正确拒答':>10}{'误拒答':>9}{'引用有效':>10}{'引用命中':>10}")
    lines = [f"生成器: {rep.name}   prompt: {rep.prompt_id}",
             "-" * len(head), head, "-" * len(head)]
    order = [t for t in TYPE_ORDER if t in rep.by_type]
    for t in order:
        s = rep.by_type[t]
        ref = f"{s.refusal_correct:>10.1%}" if t == "negative" else f"{'-':>10}"
        fr = f"{'-':>9}" if t == "negative" else f"{s.false_refusal:>9.1%}"
        lines.append(f"{t:<18}{s.n:>5}{s.correct:>9.1%}{s.entity_recall:>10.1%}"
                     f"{ref}{fr}{s.citation_valid:>10.1%}{s.citation_grounded:>10.1%}")
    o = rep.overall
    lines.append("-" * len(head))
    lines.append(f"{'总计':<18}{o.n:>5}{o.correct:>9.1%}{o.entity_recall:>10.1%}"
                 f"{o.refusal_correct:>10.1%}{o.false_refusal:>9.1%}"
                 f"{o.citation_valid:>10.1%}{o.citation_grounded:>10.1%}")
    lines.append(f"  平均延迟 {o.latency_ms:.0f}ms   累计花费 ≈ ${o.cost_usd:.3f}")
    if rep.llm_report:
        lines.append(f"  {rep.llm_report}")
    return "\n".join(lines)


def compare_prompts(reps: list[E2EReport]) -> str:
    """prompt A/B 主表。刻意把"答对率"和"拒答"并排放 ——
    它们是一对此消彼长的指标, 分开看必然得出错误结论。"""
    w = max(max(len(r.prompt_id) for r in reps) + 2, 16)
    rows = [("可答题答对率", lambda r: _answerable(r).correct),
            ("可答题实体召回", lambda r: _answerable(r).entity_recall),
            ("陷阱题正确拒答", lambda r: r.by_type.get("negative", E2EScore()).refusal_correct),
            ("可答题误拒答率", lambda r: _answerable(r).false_refusal),
            ("引用有效率", lambda r: r.overall.citation_valid),
            ("引用命中标准证据", lambda r: r.overall.citation_grounded)]
    head = f"{'指标':<20}" + "".join(f"{r.prompt_id.split('/')[-1]:>{w}}" for r in reps)
    out = ["prompt A/B 对比", "-" * len(head), head, "-" * len(head)]
    for label, fn in rows:
        out.append(f"{label:<20}" + "".join(f"{fn(r):>{w}.1%}" for r in reps))
    out.append("-" * len(head))
    out.append(f"{'累计花费(USD)':<20}"
               + "".join(f"{r.overall.cost_usd:>{w}.3f}" for r in reps))
    return "\n".join(out)


def _answerable(rep: E2EReport) -> E2EScore:
    rows = [r for r in rep.details if r["type"] != "negative"]
    if not rows:
        return E2EScore()
    return E2EScore(n=len(rows),
                    correct=_mean([r["correct"] for r in rows]),
                    entity_recall=_mean([r["entity_recall"] for r in rows]),
                    false_refusal=_mean([r["false_refusal"] for r in rows]))
