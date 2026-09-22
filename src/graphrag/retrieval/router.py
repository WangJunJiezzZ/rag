"""
查询路由 —— 决定一个问题该走哪条检索路径。

为什么需要路由(有实测证据, 不是想当然)
--------------------------------------
Phase 1 消融实验里, 等权 RRF 融合的表现**低于图检索单独使用**:

    recall@10          BM25    Graph   朴素RRF
    hop4_risk          45.0%   78.6%    48.3%   <- 融合把分数拉垮了
    shared_director    31.2%   93.8%    72.9%
    aggregation        37.5%   92.3%    51.1%
    semantic_only      22.8%    0.0%    22.8%   <- 这里反过来, 图完全无能为力

原因很直接: RRF 只看名次, 不看"这一路在这类问题上到底可不可信"。
BM25 对关系类问题返回的是一堆高排名噪音, 等权融合后把图检索的精确结果挤了出去。

所以正确做法不是"都要, 加起来", 而是**先判断这个问题该问谁**。
这也是真实系统里最常被跳过的一步 —— 大多数 RAG 实现直接 top-k 一把梭。

路由策略: 级联(cascade)而非融合
-------------------------------
    主检索器先出结果, 剩余预算由备选检索器补齐(去重)。
    保序、不稀释, 且主检索器失效时自动退化为备选 —— 没有硬失败路径。

当前用规则实现。Phase 3 会加一版 LLM 分类器, 并在同一评测集上对比:
在这个窄领域里, LLM 到底有没有赢过手写规则。**没验证过就上 LLM 是信仰不是工程。**
"""
from __future__ import annotations

from dataclasses import dataclass

from .graph_retriever import (COUNT_HINTS, RISK_HINTS, SHARED_HINTS,
                              EntityLinker)

# 关系类意图: 这类问题的证据天然分散在多份互不重叠的文档里, 只有图走得到
RELATION_HINTS = ("管理人", "托管", "注册在", "母公司", "隶属", "股东",
                  "董事", "关联", "穿透", "旗下")


@dataclass
class Route:
    primary: str                  # "graph" | "lexical"
    reason: str
    entities: list[str]


class QueryRouter:
    """规则路由。输出是可解释的 —— 前端会把 reason 显示出来。"""

    def __init__(self, linker: EntityLinker):
        self.linker = linker

    def route(self, question: str) -> Route:
        linked = self.linker.link(question)
        names = [n for _, n in linked]

        if not linked:
            # 认不出任何实体 -> 图检索没有起点, 只能走词面/语义
            return Route("lexical", "未链接到任何图实体", names)

        if any(h in question for h in SHARED_HINTS):
            return Route("graph", "共同董事类: 证据跨多家主体, 无词面重叠", names)
        if any(h in question for h in RISK_HINTS):
            return Route("graph", "风险穿透类: 需沿关系链多跳", names)
        if any(h in question for h in COUNT_HINTS) and \
           any(h in question for h in RELATION_HINTS):
            return Route("graph", "聚合类: top-k 范式无法覆盖全部证据", names)
        if any(h in question for h in RELATION_HINTS):
            return Route("graph", "关系类: 答案需跨文档拼接", names)

        return Route("lexical", "单跳属性类: 词面/语义检索已足够", names)


class RoutedRetriever:
    """路由 + 级联。主路先发, 备选补齐剩余预算。"""

    def __init__(self, lexical, graph, router: QueryRouter,
                 name: str = "Routed", backup_ratio: float = 0.5):
        self.lexical = lexical
        self.graph = graph
        self.router = router
        self.name = name
        self.backup_ratio = backup_ratio      # 留给备选的预算比例
        self.last_route: Route | None = None

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        route = self.router.route(query)
        self.last_route = route
        primary = self.graph if route.primary == "graph" else self.lexical
        backup = self.lexical if route.primary == "graph" else self.graph

        # 主路拿满预算; 若主路给不满, 剩下的全归备选 —— 主路失效时自动退化
        head = primary.retrieve(query, top_k)
        reserved = int(top_k * self.backup_ratio)
        head = head[: max(top_k - reserved, len(head) - reserved) or top_k]

        seen = {cid for cid, _ in head}
        out = list(head)
        if len(out) < top_k:
            for cid, _ in backup.retrieve(query, top_k):
                if cid not in seen:
                    seen.add(cid)
                    # 备选的分数压到主路之下, 保证主路结果始终靠前
                    out.append((cid, -1.0 - len(out) * 1e-3))
                    if len(out) >= top_k:
                        break
        return out
