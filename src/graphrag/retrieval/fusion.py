"""
多路检索结果融合。

RRF (Reciprocal Rank Fusion)
----------------------------
    score(d) = Σ_r  1 / (k + rank_r(d))

为什么用 RRF 而不是把各路分数加权相加:
  BM25 的分数是无界的(几分到几十分), 向量余弦在 [-1,1], 图检索的
  1/hop 又是另一个尺度。**三者不可比**, 直接加权相加等于让尺度最大的
  那一路说了算, 权重调参也调不出稳定结果。
  RRF 只用**名次**, 天然免疫量纲问题, 且无需训练。

k=60 是原论文的经验值。它的作用是压制头部: k 越大, 第 1 名和第 5 名的
差距越小, 越倾向于"被多路同时召回"的文档。本项目的场景正需要这个 ——
一份被 BM25 和图同时指认的文档, 比任一路的第一名更可信。
"""
from __future__ import annotations

from typing import Sequence


class RRFFusion:
    def __init__(self, retrievers: Sequence, name: str = "RRF",
                 k: int = 60, weights: Sequence[float] | None = None,
                 pool_mult: int = 3):
        self.retrievers = list(retrievers)
        self.name = name
        self.k = k
        self.weights = list(weights) if weights else [1.0] * len(self.retrievers)
        self.pool_mult = pool_mult      # 各路先多取一些再融合
        self.last_sources: dict[str, list[str]] = {}

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        pool = top_k * self.pool_mult
        scores: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        for r, w in zip(self.retrievers, self.weights):
            for rank, (cid, _) in enumerate(r.retrieve(query, pool), start=1):
                scores[cid] = scores.get(cid, 0.0) + w / (self.k + rank)
                sources.setdefault(cid, []).append(getattr(r, "name", "?"))
        self.last_sources = sources
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
