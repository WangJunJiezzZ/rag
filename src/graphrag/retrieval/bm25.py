"""
BM25 稀疏检索 —— 纯标准库实现, 零依赖。

中文分词问题
------------
SQLite FTS5 的默认分词器按非字母数字切分, 对中文等于不分词(整句变一个 token)。
引入 jieba 又多一个依赖, 且对角色名这类未登录词效果不稳("灰二太太狼"不在任何词典里)。

这里用**混合分词**:
  · CJK 连续段 -> 字符 bigram    "卡布达巨人" -> 卡布/布达/达巨/巨人
  · ASCII 段   -> 整词(小写)     "AP717" -> ap / 717(数字另算)
  · 数字       -> 整体保留        "10" "0.75"

bigram 对中文角色名的区分度足够, 且完全确定性、零依赖、无需训练。
ASCII 整词这一条是必须的 —— 语料里的英文编号别名(警用机器人又名 "AP717")
只有按词切才检索得到。

为什么自己写而不用 rank_bm25
----------------------------
本项目的核心论点就在检索环节。BM25 一共 40 行, 包进第三方库后
连"为什么这里用 b=0.75"都答不上来, 面试是减分的。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_CJK = re.compile(r"[一-鿿]+")
_ASCII = re.compile(r"[A-Za-z]+")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """混合分词: CJK 字符 bigram + ASCII 整词 + 数字。"""
    tokens: list[str] = []
    for m in _CJK.finditer(text):
        s = m.group()
        if len(s) == 1:
            tokens.append(s)
        else:
            tokens.extend(s[i:i + 2] for i in range(len(s) - 1))
    tokens.extend(m.group().lower() for m in _ASCII.finditer(text))
    tokens.extend(m.group() for m in _NUM.finditer(text))
    return tokens


class BM25:
    """标准 BM25 (Robertson/Sparck Jones)。

    k1=1.5 控制词频饱和: 一个词出现 10 次不该比出现 2 次重要 5 倍。
    b=0.75 控制长度归一: 完全归一(b=1)会过度惩罚长文档, 不归一(b=0)会让长文档
           因为词多而占便宜。0.75 是通用经验值, 本项目 chunk 长度较齐,
           该参数敏感度低 —— 这一点可以在消融实验里验证。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.avg_len = 0.0
        self.n = 0

    def index(self, items: list[tuple[str, str]]) -> "BM25":
        """items: [(chunk_id, text), ...]"""
        for cid, text in items:
            toks = tokenize(text)
            tf = Counter(toks)
            idx = len(self.ids)
            self.ids.append(cid)
            self.tf.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.df[t] += 1
                self.postings[t].append(idx)
        self.n = len(self.ids)
        self.avg_len = sum(self.doc_len) / self.n if self.n else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        # 加 0.5 平滑, 再取 max(., 1e-6) 防止高频词得到负权重
        return max(math.log(1 + (self.n - df + 0.5) / (df + 0.5)), 1e-6)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in q.items():
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for idx in self.postings.get(term, ()):
                f = self.tf[idx][term]
                denom = f + self.k1 * (1 - self.b
                                       + self.b * self.doc_len[idx] / self.avg_len)
                scores[idx] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.ids[i], s) for i, s in ranked]
