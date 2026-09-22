"""
检索评测框架。

为什么先单独评检索, 而不是直接看端到端答案对不对
--------------------------------------------------
端到端准确率是个复合指标, 检索错了和生成错了会混在一起, 无法定位。
检索阶段有一个干净、客观、与 LLM 无关的指标:

    recall@k = (top-k 结果覆盖到的证据文档数) / (该题的证据文档总数)

**检索召不回的证据, 生成阶段无论如何也补不回来。** 所以 recall@k 是上限。
先把上限抬上去, 再去优化生成 —— 顺序反了就是在给天花板刷漆。

三个指标各自回答不同问题
------------------------
  recall@k  证据覆盖了多少     -> 多跳题的核心指标(一题要 5-7 份证据)
  hit@k     至少召回一份       -> 单跳题看这个就够
  MRR       第一个正确的排多前 -> rerank 有没有效果, 看这个最灵敏
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


class Retriever(Protocol):
    """任何检索器只要实现这一个方法就能进评测。"""
    name: str

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score), ...], 按分数降序。"""
        ...


@dataclass
class TypeScore:
    n: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0


@dataclass
class Report:
    retriever: str
    ks: tuple[int, ...]
    overall: TypeScore
    by_type: dict[str, TypeScore]
    n_skipped: int = 0
    detail: list[dict] = field(default_factory=list)


class RetrievalEvaluator:
    def __init__(self, chunk_to_doc: dict[str, str], items: Sequence[dict],
                 split: str | None = None):
        self.chunk_to_doc = chunk_to_doc
        self.items = [i for i in items
                      if (split is None or i["split"] == split)
                      and i["gold_docs"]]          # 陷阱题无证据, 不参与检索评测
        self.n_skipped = len(items) - len(self.items)

    def evaluate(self, retriever: Retriever,
                 ks: tuple[int, ...] = (1, 3, 5, 10, 20),
                 keep_detail: bool = False) -> Report:
        max_k = max(ks)
        agg: dict[str, list[dict]] = {}
        detail: list[dict] = []

        for item in self.items:
            hits = retriever.retrieve(item["question"], max_k)
            # chunk 结果按文档去重, 并保留各文档首次出现的名次(用于 MRR)
            doc_rank: dict[str, int] = {}
            for rank, (cid, _) in enumerate(hits, start=1):
                d = self.chunk_to_doc.get(cid)
                if d and d not in doc_rank:
                    doc_rank[d] = rank

            gold = set(item["gold_docs"])
            row: dict = {"type": item["type"]}
            for k in ks:
                found = {d for d, r in doc_rank.items() if r <= k} & gold
                row[f"recall@{k}"] = len(found) / len(gold)
                row[f"hit@{k}"] = 1.0 if found else 0.0
            first = min((r for d, r in doc_rank.items() if d in gold), default=None)
            row["rr"] = 1.0 / first if first else 0.0

            agg.setdefault(item["type"], []).append(row)
            if keep_detail:
                detail.append({
                    "id": item["id"], "type": item["type"],
                    "question": item["question"],
                    "gold_docs": sorted(gold),
                    "retrieved_docs": [d for d, _ in
                                       sorted(doc_rank.items(), key=lambda x: x[1])][:10],
                    "recall@10": row.get("recall@10", 0.0),
                    "missed": sorted(gold - set(doc_rank)),
                })

        def fold(rows: list[dict]) -> TypeScore:
            n = len(rows)
            if n == 0:
                return TypeScore()
            return TypeScore(
                n=n,
                recall={k: sum(r[f"recall@{k}"] for r in rows) / n for k in ks},
                hit={k: sum(r[f"hit@{k}"] for r in rows) / n for k in ks},
                mrr=sum(r["rr"] for r in rows) / n)

        all_rows = [r for rows in agg.values() for r in rows]
        return Report(retriever=retriever.name, ks=ks,
                      overall=fold(all_rows),
                      by_type={t: fold(rows) for t, rows in agg.items()},
                      n_skipped=self.n_skipped, detail=detail)


# --------------------------------------------------------------------------

TYPE_ORDER = ["fact_direct", "fact_paraphrase", "semantic_only",
              "fact_disambig", "hop2",
              "hop3_parent", "hop4_risk", "shared_director", "aggregation"]


def format_report(rep: Report, ks: tuple[int, ...] = (1, 5, 10, 20)) -> str:
    ks = tuple(k for k in ks if k in rep.ks)
    head = f"{'题型':<18}{'题数':>5}" + "".join(f"{'R@' + str(k):>9}" for k in ks) \
           + f"{'hit@10':>9}{'MRR':>8}"
    lines = [f"检索器: {rep.retriever}", "-" * len(head), head, "-" * len(head)]
    order = [t for t in TYPE_ORDER if t in rep.by_type] + \
            [t for t in rep.by_type if t not in TYPE_ORDER]
    for t in order:
        s = rep.by_type[t]
        lines.append(f"{t:<18}{s.n:>5}"
                     + "".join(f"{s.recall[k]:>9.1%}" for k in ks)
                     + f"{s.hit.get(10, 0):>9.1%}{s.mrr:>8.3f}")
    o = rep.overall
    lines.append("-" * len(head))
    lines.append(f"{'总计':<18}{o.n:>5}"
                 + "".join(f"{o.recall[k]:>9.1%}" for k in ks)
                 + f"{o.hit.get(10, 0):>9.1%}{o.mrr:>8.3f}")
    return "\n".join(lines)


def compare_table(reps: list[Report], k: int = 10) -> str:
    """多个检索器横向对比 —— 消融实验的主表。"""
    types = [t for t in TYPE_ORDER if any(t in r.by_type for r in reps)]
    w = max(len(r.retriever) for r in reps) + 2
    head = f"{'题型':<18}" + "".join(f"{r.retriever:>{w}}" for r in reps)
    lines = [f"recall@{k} 对比", "-" * len(head), head, "-" * len(head)]
    for t in types:
        row = f"{t:<18}"
        for r in reps:
            v = r.by_type[t].recall.get(k, 0.0) if t in r.by_type else 0.0
            row += f"{v:>{w}.1%}"
        lines.append(row)
    lines.append("-" * len(head))
    lines.append(f"{'总计':<18}"
                 + "".join(f"{r.overall.recall.get(k, 0.0):>{w}.1%}" for r in reps))
    return "\n".join(lines)
