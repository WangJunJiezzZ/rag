"""
图检索器 —— 本项目的招牌功能。

与关键词/向量检索的根本区别
---------------------------
  检索 = 在一堆 chunk 里**找**最像问题的那几块
  图   = 从问题提到的实体出发, 沿关系**走**到证据所在的文档

"走"能到达"找"永远到不了的地方: "小灰灰和小香香是什么关系"这类题里,
关键证据(香太狼的档案)与问题零词面重叠, 无论 top-k 取多大都未必召得回;
而沿着 小灰灰 <-母亲- 红太狼 <-表妹- 香太狼 -女儿-> 小香香 走三步就到了。

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
    # 这份文档里**应该出现**的名称(边两端的实体名、别名、字面量值)。
    # 用来在文档内挑 chunk: 发出去的必须是写着这条关系的那一块, 不是文档的第一块。
    focus: set[str] = field(default_factory=set)


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

    规则式, 零 LLM: 按名称长度倒序做最长匹配, 避免"灰太狼"吃掉
    "灰二太太狼"、"喜羊羊"吃掉"喜羊羊与灰太狼"。别名(又名/译名)一并纳入索引 ——
    这是实体消歧成果在检索侧的直接兑现: 问"蝎子蓝蓝"也能认出蝎子莱莱。
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
# 无目标的 BFS 扩散会淹没答案: 从一个角色走四跳能碰到大半个语料库,
# 真正在证据路径上的那两三份挤不进 top-10。
# 所以要**目标导向**: 问题里有两个角色、问的是"什么关系"时,
# 只沿着连接这两个角色的路径走, 并且只返回路径上的边所对应的文档。
#
# 这里用手写规则实现。Phase 3 会换成 LLM 把自然语言翻译成同样形状的查询,
# 届时正好可以比一比: 在这个窄领域里, LLM 到底有没有赢过手写规则。

# 问题里的关系词 -> 图里的关系类型。
# 这是"关系链"模板的全部知识: 认出问题在问哪几种关系, 遍历时只沿这些边走。
# 不关心顺序(「表哥的老婆」与「老婆的表哥」走同一组边), 顺序交给生成环节 ——
# 检索只负责把证据拿全, 不负责推理。
REL_WORDS: dict[str, tuple[str, ...]] = {
    "爸爸": ("PARENT_OF",), "妈妈": ("PARENT_OF",), "父亲": ("PARENT_OF",),
    "母亲": ("PARENT_OF",), "儿子": ("PARENT_OF",), "女儿": ("PARENT_OF",),
    "爷爷": ("PARENT_OF", "GRANDPARENT_OF"), "奶奶": ("PARENT_OF", "GRANDPARENT_OF"),
    "外公": ("PARENT_OF", "GRANDPARENT_OF"), "外婆": ("PARENT_OF", "GRANDPARENT_OF"),
    "代孙": ("DESCENDANT_OF", "PARENT_OF"),
    "老婆": ("SPOUSE_OF",), "妻子": ("SPOUSE_OF",), "老公": ("SPOUSE_OF",),
    "丈夫": ("SPOUSE_OF",), "岳父": ("SPOUSE_OF", "PARENT_OF"),
    "岳母": ("SPOUSE_OF", "PARENT_OF"),
    "表哥": ("COUSIN_OF",), "表妹": ("COUSIN_OF",), "表姐": ("COUSIN_OF",),
    "表弟": ("COUSIN_OF",), "表姨": ("COUSIN_OF", "PARENT_OF"),
    "二叔": ("UNCLE_OF",), "叔叔": ("UNCLE_OF",), "侄子": ("UNCLE_OF",),
    "搭档": ("PARTNER_OF",),
    "大哥": ("LEADER_OF", "MEMBER_OF"), "老大": ("LEADER_OF", "MEMBER_OF"),
    "首领": ("LEADER_OF",), "成员": ("MEMBER_OF",), "组成": ("MEMBER_OF",),
    "喜欢": ("LIKES",), "好朋友": ("FRIEND_OF",),
    "校长": ("HEAD_OF",), "村长": ("HEAD_OF",), "当家": ("HEAD_OF",),
    "上学": ("STUDENT_OF",), "学生": ("STUDENT_OF",),
    "住着": ("LIVES_IN",), "住在": ("LIVES_IN",),
    "坐落": ("LOCATED_IN",), "位于": ("LOCATED_IN",),
    "封印": ("SEALED_BY", "RELEASED_BY"),
    "许愿": ("WISHED_ON",),
    "原型": ("PROTOTYPE",), "武器": ("WEAPON",), "兵器": ("WEAPON",),
    "爱吃": ("FAVORITE_FOOD",), "口头禅": ("CATCHPHRASE",),
    "职务": ("ROLE",), "身份": ("ROLE",), "做什么": ("ROLE",), "干什么": ("ROLE",),
    "必杀技": ("SPECIAL_MOVE",), "编号": ("UNIT_NO",), "号机": ("UNIT_NO",),
    "变换": ("TRANSFORM_TIME",), "变身": ("TRANSFORM_TIME",),
    "巨人": ("GIANT_FORM",), "阵营": ("FACTION",), "生日": ("BIRTHDAY",),
}

PATH_HINTS = ("什么关系", "有关系", "有什么联系", "什么联系", "有联系", "关联",
              "有什么关系", "是什么人", "的什么人")
COUNT_HINTS = ("哪些", "哪几", "几个", "几只", "几台", "多少", "列出", "都有谁", "有谁")


class GraphRetriever:
    """name 里标注图的来源(oracle / extracted), 一眼能看出这是上界还是实测。"""

    def __init__(self, kg: KnowledgeGraph, doc_to_chunks: dict[str, list[str]],
                 name: str = "Graph(oracle)", max_hops: int = 4,
                 max_docs: int = 40, max_paths: int = 12,
                 hub_degree: int | None = 12,
                 chunk_text: dict[str, str] | None = None):
        self.kg = kg
        self.linker = EntityLinker(kg)
        self.doc_to_chunks = doc_to_chunks
        self.chunk_text = chunk_text or {}
        # 每份档案的主人: 它的 APPEARS_IN 边来源于这份文档。
        # 档案主人的名字在自己档案里到处都是, 拿来给块打分没有区分度。
        self.subject_of: dict[str, str] = {}
        for e in kg.edges:
            if e.rel == "APPEARS_IN":
                for d in e.all_docs():
                    self.subject_of.setdefault(d, e.src)
        self._hub_names = sorted((n for eid, ent in kg.entities.items()
                                  if kg.is_hub(eid, hub_degree)
                                  for n in [ent.name, *ent.aliases]),
                                 key=len, reverse=True)
        self.name = name
        self.max_hops = max_hops
        self.max_docs = max_docs
        self.max_paths = max_paths
        self.hub_degree = hub_degree     # None = 允许穿透枢纽(消融用)
        self.last_trace: GraphTrace | None = None

    # ------------------------------------------------------------------

    def _add(self, hits: dict[str, GraphHit], doc: str, edge, **kw) -> None:
        """同一份文档可能被多条边命中: 第一次命中定 hop/path, focus 取所有边的并集。"""
        if not doc:
            return
        if doc not in hits:
            hits[doc] = GraphHit(doc_id=doc, **kw)
        hits[doc].focus |= self._forms(edge.src) | self._forms(edge.dst)

    # ------------------------------------------------------------------
    # 模板一: 关系路径 —— 问题里的两个实体之间, 有哪些不经过枢纽的路径
    # ------------------------------------------------------------------
    def _q_relation_path(self, seeds: list[str]) -> list[GraphHit]:
        kg = self.kg
        hits: dict[str, GraphHit] = {}
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                paths = kg.find_paths(seeds[i], seeds[j], max_hops=self.max_hops,
                                      limit=self.max_paths, hub_degree=self.hub_degree)
                for path in paths:
                    desc = kg.path_str(path)
                    gdict = kg.path_to_dict(path)
                    for st in path:
                        for doc in st.edge.all_docs():
                            self._add(hits, doc, st.edge, hop=len(path),
                                      via=st.edge.rel, path=desc, graph=gdict)
        return sorted(hits.values(), key=lambda h: (h.hop, h.doc_id))

    # ------------------------------------------------------------------
    # 模板二: 关系链 —— 只沿问题里提到的那几种关系走
    # ------------------------------------------------------------------
    def _q_relation_chain(self, seeds: list[str], rels: set[str]) -> list[GraphHit]:
        """为什么需要它: 无差别的邻域扩散按跳数分层, 第一层就可能把预算用光。
        "灰太狼表哥的老婆是谁"从灰太狼出发, 一跳就连着 7 份文档(妻子、儿子、侄子、
        表哥、二叔、居民登记……), 真正要的第二跳——香太狼的档案——排不进 top-8。
        只沿 COUSIN_OF / SPOUSE_OF 走, 第一层只剩夜太狼和红太狼, 第二层就是香太狼。

        走到的终点实体, 再补上它**自己的档案**(APPEARS_IN 边的来源文档):
        "美羊羊喜欢的那只羊有什么爱好" —— 爱好写在喜羊羊的档案里, 不在任何一条边上。
        """
        kg = self.kg
        hits: dict[str, GraphHit] = {}
        seen = set(seeds)
        frontier = [(s, []) for s in seeds]
        reached: list[tuple[str, int, list]] = []
        for hop in range(1, self.max_hops):
            nxt = []
            for node, path in frontier:
                for step in kg.steps_from(node):
                    if step.edge.rel not in rels:
                        continue
                    full = path + [step]
                    for doc in step.edge.all_docs():
                        self._add(hits, doc, step.edge, hop=hop, via=step.edge.rel,
                                  path=kg.path_str(full), graph=kg.path_to_dict(full))
                    if step.to not in seen and not kg.is_hub(step.to, self.hub_degree):
                        seen.add(step.to)
                        nxt.append((step.to, full))
                        reached.append((step.to, hop, full))
            frontier = nxt
            if not frontier:
                break
        for node, hop, full in reached:
            for e in kg.out(node, "APPEARS_IN"):
                for doc in e.all_docs():
                    if doc not in hits:
                        self._add(hits, doc, e, hop=hop + 1, via="APPEARS_IN",
                                  path=kg.path_str(full), graph=kg.path_to_dict(full))
        return sorted(hits.values(), key=lambda h: (h.hop, h.doc_id))

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
                        if doc not in hits:
                            self._add(hits, doc, step.edge, hop=hop, via=step.edge.rel,
                                      path=kg.path_str(full), graph=kg.path_to_dict(full))
                        else:
                            self._add(hits, doc, step.edge)
                    # 枢纽(节目节点)可以作为终点, 不能再往外扩:
                    # 否则第二跳就把同一部剧的全部角色档案都拉进来
                    if step.to not in seen_nodes and not kg.is_hub(step.to, self.hub_degree):
                        seen_nodes.add(step.to)
                        nxt.append((step.to, path + [step]))
            frontier = nxt
            if not frontier:
                break
        return sorted(hits.values(), key=lambda h: (h.hop, h.doc_id))

    # ------------------------------------------------------------------
    def _route(self, question: str, n_seeds: int) -> tuple[str, object]:
        """规则路由: 问题 -> 图查询模板。按特异性排序。"""
        if n_seeds >= 2 and any(h in question for h in PATH_HINTS):
            return "relation_path", self._q_relation_path
        rels = {r for w, rs in REL_WORDS.items() if w in question for r in rs}
        if rels:
            return "relation_chain", lambda seeds: self._q_relation_chain(seeds, rels)
        return "neighborhood", self._q_neighborhood

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        linked = self.linker.link(query)
        trace = GraphTrace(linked=linked)
        self.last_trace = trace
        if not linked:
            return []                     # 认不出实体 -> 交给别的检索器兜底

        seeds = [eid for eid, _ in linked]
        intent, handler = self._route(query, len(seeds))
        hits = handler(seeds)
        if not hits:
            # 目标导向没走通(两个实体之间确实没有不经枢纽的路径) -> 回退邻域扩散
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
        #
        # 文档内的 chunk 顺序也有讲究: 优先发**写着这条关系**的那一块。
        # 首版直接按原文顺序发, 于是每份档案先发出去的是「XX 角色档案 + 资料来源」
        # 这个标题块 —— 文档级 recall@8 是 100%, 可写着"父亲是黑太狼"的那一块
        # 根本没进上下文, 两跳题答对率只有 25%。recall 按文档算, 掩盖了块级缺失。
        # ------------------------------------------------------------------
        out: list[tuple[str, float]] = []
        ordered = {h.doc_id: self._order_chunks(h) for h in hits}
        rounds = max((len(v) for v in ordered.values()), default=0)
        for r in range(rounds):
            for h in hits:
                chunks = ordered[h.doc_id]
                if r < len(chunks):
                    # 跳数越近分越高; 同文档内靠后的 chunk 轻微降权
                    out.append((chunks[r], 1.0 / h.hop - r * 1e-4))
                    if len(out) >= top_k:
                        return out
        return out

    def _order_chunks(self, h: GraphHit) -> list[str]:
        """按"提到了几个 focus 名称"排序; 没有正文可查时保持原文顺序。

        两处去噪, 都是实测撞上的:
          · 先抹掉枢纽名(节目名)再匹配: 「喜羊羊与灰太狼」里就含着「灰太狼」,
            否则每份档案的第一句都会被当成"提到了灰太狼"
          · 不数档案主人自己的名字: 它只出现在第一块, 数进去等于给第一块加分
        """
        chunks = self.doc_to_chunks.get(h.doc_id, [])
        if not self.chunk_text or not h.focus:
            return list(chunks)
        own = self._forms(self.subject_of[h.doc_id]) if h.doc_id in self.subject_of else set()
        focus = [f for f in h.focus if f and f not in own]

        def score(cid: str) -> int:
            text = self.chunk_text.get(cid, "")
            for n in self._hub_names:
                text = text.replace(n, "")
            return sum(1 for f in focus if f in text)
        return sorted(chunks, key=lambda c: -score(c))      # sorted 是稳定排序

    def _forms(self, nid: str) -> set[str]:
        """一条边在正文里可能的写法。枢纽(节目名)不算: 它出现在每份档案的第一句,
        拿它打分只会把每份档案的第一块顶到最前面 —— 正是要修掉的那个问题。"""
        if nid.startswith("lit:"):
            return {nid[4:]}
        if self.kg.is_hub(nid, self.hub_degree):
            return set()
        e = self.kg.entities.get(nid)
        return {e.name, *e.aliases} if e else set()
