"""
Dense 向量检索。

为什么用暴力内积而不是 FAISS/HNSW
---------------------------------
本项目 chunk 数约 1100, 向量 512 维。暴力算一次是 1100x512 的矩阵乘,
在 CPU 上约 0.3ms —— 比建 HNSW 索引的开销还低。
**在这个规模上引入 ANN 是过度优化**, 而且会引入召回率损失这个新变量,
干扰消融实验的结论。

切换点大约在十万级向量。接口(index/search)与 FAISS 一致, 到时换实现即可。
"""
from __future__ import annotations

from ..embedding import BaseEmbedder


class DenseRetriever:
    def __init__(self, embedder: BaseEmbedder, name: str = "Dense"):
        import numpy as np
        self._np = np
        self.embedder = embedder
        self.name = name
        self.ids: list[str] = []
        self.mat = None

    def index(self, items: list[tuple[str, str]], verbose: bool = False):
        self.ids = [cid for cid, _ in items]
        texts = [t for _, t in items]
        if verbose:
            print(f"  向量化 {len(texts)} 个 chunk ...", end="", flush=True)
        self.mat = self.embedder.encode(texts, is_query=False)
        if verbose:
            print(f" 完成 dim={self.mat.shape[1]}")
        return self

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        np = self._np
        if self.mat is None or not len(self.ids):
            return []
        q = self.embedder.encode([query], is_query=True)[0]
        sims = self.mat @ q                       # 已归一化, 内积即余弦
        k = min(top_k, len(self.ids))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def retrieve(self, query: str, top_k: int):
        return self.search(query, top_k)
