"""
图检索器 —— 本项目的招牌功能。

与关键词/向量检索的根本区别
---------------------------
  检索 = 在一堆 chunk 里**找**最像问题的那几块
  图   = 从问题提到的实体出发, 沿关系**走**到证据所在的文档

"走"能到达"找"永远到不了的地方: hop4_risk 那类题里, 关键证据文档
(董事公告、监管名单)与问题零词面重叠, 无论 top-k 取多大都召不回;
而沿着 基金->管理人<-董事->关联公司->辖区->风险 这条链走四步就到了。

三步流水线
----------
  1. 实体链接  从问题文本里认出图中的实体(含别名/罗马化名)
  2. 受控扩展  从这些实体出发 BFS, 按跳数分层收集经过的边
  3. 证据回溯  每条边都带 source_doc, 于是走过的路径直接就是证据清单

为什么先用 ground-truth 图跑(oracle)
------------------------------------
这样得到的是**上界**: 假如抽取环节完美, 图检索最多能拿多少分。
Phase 2 换成 LLM 抽取出来的图再跑一次, 两者的差值就是抽取环节的损失。
先量上界再量损失, 比一上来就报一个混合了两种误差的数字有用得多。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..store.graph import KnowledgeGraph, normalize_name


@dataclass
class GraphHit:
    doc_id: str
    hop: int                       # 产生这份证据的边距离起点几跳
    path: str = ""                 # 人类可读的路径, 用于展示与溯源
    via: str = ""                  # 关系名
    graph: dict = field(default_factory=dict)   # 结构化路径


@dataclass
class GraphTrace:
    """一次图检索的完整痕迹 —— 前端要用它画关系图。"""
    intent: str = ""                                             # 命中的图查询模板
    linked: list[tuple[str, str]] = field(default_factory=list)   # (实体id, 命中的名称写法)
    hits: list[GraphHit] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    graphs: list[dict] = field(default_factory=list)   # 结构化路径, 供前端画图


class EntityLinker:
    """把问题文本里的实体认出来。

    规则式, 零 LLM: 按名称长度倒序做最长匹配, 避免"星海基金"吃掉
    "星海亚洲机会基金"。别名与罗马化名一并纳入索引 ——
    这是实体消歧成果在检索侧的直接兑现。
    """

    def __init__(self, kg: KnowledgeGraph, min_len: int = 2):
        self.kg = kg
        surface: list[tuple[str, str]] = []
        for e in kg.entities.values():
            for form in [e.name, *e.aliases]:
                if len(form) >= min_len:
                    surface.append((form, e.id))
        # 长的优先, 保证最长匹配
        self.surface = sorted(surface, key=lambda x: -len(x[0]))

    def link(self, text: str) -> list[tuple[str, str]]:
        found: dict[str, str] = {}
        consumed = [False] * len(text)
        for form, eid in self.surface:
            start = text.find(form)
            while start != -1:
                if not any(consumed[start:start + len(form)]):
                    for i in range(start, start + len(form)):
                        consumed[i] = True
                    found.setdefault(eid, form)
                    break
                start = text.find(form, start + 1)
        return list(found.items())


# ==========================================================================
# 查询意图 -> 图查询模板
# ==========================================================================
# 无目标的 BFS 扩散会淹没答案: 从一只基金走四跳能碰到几百份文档,
# 真正在证据路径上的那 5 份挤不进 top-10。
# 所以要**目标导向**: 先判断问题在问什么, 再只沿着能到达目标的路径走,
# 并且只返回路径上的边所对应的文档。
#
# 这里用手写规则实现。Phase 2 会换成 LLM 把自然语言翻译成同样形状的查询,
# 届时正好可以比一比: 在这个窄领域里, LLM 到底有没有赢过手写规则。
#   —— 这个对比本身就是一条评测结论, 很多项目直接上 LLM 而从没验证过。

RISK_HINTS = ("高风险", "风险", "制裁", "可疑", "关联方", "穿透", "关联")
SHARED_HINTS = ("共同董事", "共同", "同时担任", "同时是")
COUNT_HINTS = ("几只", "几家", "多少", "数量", "列出", "哪些")


class GraphRetriever:
    """name 里标注图的来源(oracle / extracted), 一眼能看出这是上界还是实测。"""

    def __init__(self, kg: KnowledgeGraph, doc_to_chunks: dict[str, list[str]],
                 name: str = "Graph(oracle)", max_hops: int = 4,
                 max_docs: int = 40, max_paths: int = 12,
                 hub_degree: int | None = 12):
        self.kg = kg
        self.linker = EntityLinker(kg)
        self.doc_to_chunks = doc_to_chunks
        self.name = name
        self.max_hops = max_hops
        self.max_docs = max_docs
        self.max_paths = max_paths
        self.hub_degree = hub_degree     # None = 允许穿透枢纽(消融用)
        self.last_trace: GraphTrace | None = None

    # ------------------------------------------------------------------
    # 模板一: 风险穿透 —— 从种子出发, 找所有能到达"高风险"的路径
    # ------------------------------------------------------------------
    def _q_risk_path(self, seeds: list[str]) -> list[GraphHit]:
        kg = self.kg

        def is_risky(nid: str) -> bool:
            if nid == "lit:高风险":
                return True
            e = kg.entities.get(nid)
            return bool(e and e.props.get("risk") == "高风险")

        hits: dict[str, GraphHit] = {}
        for seed in seeds:
            paths = kg.paths_to(seed, is_risky, max_hops=self.max_hops,
                                max_paths=self.max_paths,
                                hub_degree=self.hub_degree)
            for path in paths:
                desc = kg.path_str(path)
                gdict = kg.path_to_dict(path)
                # 命中的高风险辖区, 其"高风险"这一事实来自监管名单文档,
                # 要把那条 RISK_LEVEL 边的来源一并纳入证据, 否则答案无法自证。
                tail = path[-1].to
                extra = list(kg.out(tail, "RISK_LEVEL"))
                for step_edge in [st.edge for st in path] + extra:
                    for doc in step_edge.all_docs():
                        if doc and doc not in hits:
                            hits[doc] = GraphHit(doc_id=doc, hop=len(path),
                                                 via=step_edge.rel, path=desc,
                                                 graph=gdict)
        return sorted(hits.values(), key=lambda h: (h.hop, h.doc_id))

    # ------------------------------------------------------------------
    # 模板二: 共同董事 —— 种子公司的董事, 还在哪些别的公司任职
    # ------------------------------------------------------------------
    def _q_shared_director(self, seeds: list[str]) -> list[GraphHit]:
        kg = self.kg
        hits: dict[str, GraphHit] = {}
        for seed in seeds:
            for e1 in kg.inn(seed, "DIRECTOR_OF"):          # 人 -[董事]-> 种子公司
                person = e1.src
                for e2 in kg.out(person, "DIRECTOR_OF"):     # 这个人的其他任职
                    if e2.dst == seed:
                        continue
                    desc = (f"{kg.name(seed)} <-[DIRECTOR_OF]- {kg.name(person)} "
                            f"-[DIRECTOR_OF]-> {kg.name(e2.dst)}")
                    gdict = {
                        "nodes": [{"id": n, "name": kg.name(n),
                                   "type": kg.entities[n].type if n in kg.entities
                                           else "Unknown",
                                   "risk": (kg.entities[n].props.get("risk")
                                            if n in kg.entities else None)}
                                  for n in (seed, person, e2.dst)],
                        "edges": [{"from": person, "to": seed, "rel": "DIRECTOR_OF",
                                   "reversed": False, "docs": e1.all_docs()},
                                  {"from": person, "to": e2.dst, "rel": "DIRECTOR_OF",
                                   "reversed": False, "docs": e2.all_docs()}],
                    }
                    for edge in (e1, e2):
                        for doc in edge.all_docs():
                            if doc and doc not in hits:
                                hits[doc] = GraphHit(doc_id=doc, hop=1,
                                                     via="DIRECTOR_OF", path=desc,
                                                     graph=gdict)
        return list(hits.values())

    # ------------------------------------------------------------------
    # 模板三: 邻域扩散 —— 兜底, 按跳数分层收集
    # ------------------------------------------------------------------
    def _q_neighborhood(self, seeds: list[str]) -> list[GraphHit]:
        kg = self.kg
        seen_nodes = set(seeds)
        hits: dict[str, GraphHit] = {}
        frontier = [(s, []) for s in seeds]
        for hop in range(1, self.max_hops + 1):
            nxt = []
            for node, path in frontier:
                for step in kg.steps_from(node):
                    full = path + [step]
                    for doc in step.edge.all_docs():
                        if doc and doc not in hits:
                            hits[doc] = GraphHit(doc_id=doc, hop=hop,
                                                 via=step.edge.rel,
                                                 path=kg.path_str(full),
                                                 graph=kg.path_to_dict(full))
                    if step.to not in seen_nodes:
                        seen_nodes.add(step.to)
                        nxt.append((step.to, path + [step]))
            frontier = nxt
            if not frontier:
                break
        return sorted(hits.values(), key=lambda h: (h.hop, h.doc_id))

    # ------------------------------------------------------------------
    def _route(self, question: str) -> tuple[str, object]:
        """规则路由: 问题 -> 图查询模板。命中多个时按特异性排序。"""
        if any(h in question for h in SHARED_HINTS):
            return "shared_director", self._q_shared_director
        if any(h in question for h in RISK_HINTS):
            return "risk_path", self._q_risk_path
        return "neighborhood", self._q_neighborhood

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        linked = self.linker.link(query)
        trace = GraphTrace(linked=linked)
        self.last_trace = trace
        if not linked:
            return []                     # 认不出实体 -> 交给别的检索器兜底

        seeds = [eid for eid, _ in linked]
        intent, handler = self._route(query)
        hits = handler(seeds)
        if not hits:
            # 目标导向没走通(例如问了风险但根本没有风险关联) -> 回退邻域扩散
            intent, hits = "neighborhood(fallback)", self._q_neighborhood(seeds)
        hits = hits[: self.max_docs]
        trace.intent = intent
        trace.hits = hits
        trace.paths = [h.path for h in hits[:12] if h.path]
        seen_g: set[str] = set()
        for h in hits:
            if h.graph and h.path not in seen_g:
                seen_g.add(h.path)
                trace.graphs.append(h.graph)
            if len(trace.graphs) >= 4:
                break

        # ------------------------------------------------------------------
        # 按文档轮转发放预算, 而不是把一份文档的 chunk 一次吐完。
        #
        # 这是评测公平性问题, 不是检索质量问题: 每份文档平均 3.3 个 chunk,
        # 若按文档顺序发放, top_k=10 只能覆盖 3 份文档, 而 BM25 的 10 个 chunk
        # 可以散落在 10 份文档里 —— 同样的 k, 图检索的"文档预算"只有 BM25 的三分之一。
        # 轮转发放后, 前 top_k 个结果尽可能覆盖 top_k 份不同文档,
        # 与 BM25 在同一量纲上比较。
        # ------------------------------------------------------------------
        out: list[tuple[str, float]] = []
        rounds = max((len(self.doc_to_chunks.get(h.doc_id, [])) for h in hits),
                     default=0)
        for r in range(rounds):
            for h in hits:
                chunks = self.doc_to_chunks.get(h.doc_id, [])
                if r < len(chunks):
                    # 跳数越近分越高; 同文档内靠后的 chunk 轻微降权
                    out.append((chunks[r], 1.0 / h.hop - r * 1e-4))
                    if len(out) >= top_k:
                        return out
        return out
