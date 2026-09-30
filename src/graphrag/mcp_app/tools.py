"""
工具层 —— 把 RAGService 拆成一组"原子能力", 供 MCP Server 与 Agent 共用。

为什么不直接暴露一个 ask()
--------------------------
ask() 是写死的流水线: 规则路由 -> 检索 -> 生成。调用方(Claude / DeepSeek /
任何 MCP 客户端)只能"问一句、拿一个答案", 无法在中途判断"这一步没找到, 换条路"。
把能力拆成 查实体 / 查邻居 / 找路径 / 全文检索 / 读原文 之后,
决策权交给了调用方的模型 —— 这正是项目 B(Agentic RAG)要评测的东西:
**模型自己编排工具, 到底比手写规则路由好多少、贵多少。**

ask() 仍然保留, 作为"固定流水线"基线, 同一个 Server 里两种用法都能对比。

设计约束
--------
  · 纯 Python, 不依赖 mcp SDK —— 评测时 Agent 可在进程内直接调用, 不走协议开销
  · 返回值全部是 JSON 友好的 dict/list, 且**刻意截断** —— 工具输出会原样进入
    模型上下文, 一次返回 40 份全文等于把 token 预算烧光。截断长度是可调参数
  · 所有"找不到"都返回结构化的提示而不是抛异常 —— 模型能读懂 hint 并换个写法重试,
    读不懂 Python traceback
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any

from ..service import RAGService, ServiceConfig
from ..store.graph import LITERAL_PREFIX, KnowledgeGraph, normalize_name

# 关系的含义与方向。放进 schema resource, 也写进工具描述 ——
# 模型不知道 "PARENT_OF 是 父母->子女" 就会把方向查反, 把爷爷答成孙子。
RELATIONS: dict[str, str] = {
    "APPEARS_IN":     "角色 -> 节目: 出自哪部剧",
    "PARENT_OF":      "角色 -> 角色: 前者是后者的父亲或母亲",
    "GRANDPARENT_OF": "角色 -> 角色: 前者是后者的爷爷/奶奶/外公/外婆",
    "SPOUSE_OF":      "角色 -> 角色: 夫妻(无方向)",
    "COUSIN_OF":      "角色 -> 角色: 前者是后者的表哥/表姐/表弟/表妹",
    "UNCLE_OF":       "角色 -> 角色: 前者是后者的叔叔(后者是侄子)",
    "DESCENDANT_OF":  "角色 -> 角色: 前者是后者的后代",
    "LIKES":          "角色 -> 角色: 前者喜欢后者",
    "FRIEND_OF":      "角色 -> 角色: 好朋友(无方向)",
    "PARTNER_OF":     "角色 -> 角色: 前者的搭档是后者",
    "SEALED_BY":      "角色 -> 角色: 前者被后者封印",
    "RELEASED_BY":    "角色 -> 角色: 前者的封印被后者解开",
    "HEAD_OF":        "角色 -> 地点: 村长/校长",
    "LIVES_IN":       "角色 -> 地点: 住在",
    "STUDENT_OF":     "角色 -> 地点: 在该学校上学",
    "LOCATED_IN":     "地点 -> 地点: 位于",
    "MEMBER_OF":      "角色 -> 团体: 成员",
    "LEADER_OF":      "角色 -> 团体: 首领/老大",
    "WISHED_ON":      "角色 -> 道具: 向某颗和平星许愿",
    "ALIAS":          "角色 -> 文本: 又名/译名",
    "UNIT_NO":        "角色 -> 文本: 编号(如 1 号机)",
    "PROTOTYPE":      "角色 -> 文本: 原型",
    "ROLE":           "角色 -> 文本: 身份/职务",
    "BIRTHDAY":       "角色 -> 文本: 生日",
    "CATCHPHRASE":    "角色 -> 文本: 口头禅",
    "WEAPON":         "角色 -> 文本: 武器",
    "SPECIAL_MOVE":   "角色 -> 文本: 必杀技",
    "FAVORITE_FOOD":  "角色 -> 文本: 最爱吃的食物",
    "CARRIES":        "角色 -> 文本: 随身物品",
    "TRANSFORM_TIME": "角色 -> 文本: 超级变换形态维持时间",
    "GIANT_FORM":     "角色 -> 文本: 巨人形态",
    "FACTION":        "角色 -> 文本: 阵营(正义/反派/中立)",
    "FIRST_AIRED":    "节目 -> 文本: 首播日期",
    "BROADCASTER":    "节目 -> 文本: 首播电视台",
    "EPISODES":       "节目 -> 文本: 集数",
    "THEME_SONG":     "节目 -> 文本: 主题曲",
    "FIRST_MOVIE":    "节目 -> 文本: 首部电影",
    "QUANTITY":       "道具 -> 文本: 数量",
}

SNIPPET_CHARS = 220        # 检索结果每条只给摘要, 要全文用 read_document
DOC_CHARS = 1500           # read_document 默认上限


class GraphRAGTools:
    """懒加载: 第一次调用工具时才建索引。

    MCP 客户端启动 Server 后会立刻发 initialize 握手; 若在 import 时就建向量索引,
    握手要等好几秒, 部分客户端会判定超时。
    """

    def __init__(self, cfg: ServiceConfig | None = None,
                 service: RAGService | None = None):
        self._cfg = cfg or ServiceConfig()
        self._svc = service

    # ------------------------------------------------------------------
    @property
    def svc(self) -> RAGService:
        if self._svc is None:
            # stdio 传输下 stdout 是协议通道, 初始化期间的进度/告警打印必须改走 stderr。
            # mcp>=2 会自行把 fd 1 转到 stderr, 这里再兜一层, 兼容 1.x 与进程内调用。
            with contextlib.redirect_stdout(sys.stderr):
                self._svc = RAGService(self._cfg, quiet=True)
        return self._svc

    @property
    def kg(self) -> KnowledgeGraph:
        if self.svc.graph is None:
            raise RuntimeError(f"图谱未加载: {self._cfg.graph_path}")
        return self.svc.graph

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _entity_brief(self, eid: str) -> dict:
        kg = self.kg
        if eid.startswith(LITERAL_PREFIX):
            return {"id": eid, "name": kg.name(eid), "type": "Literal"}
        e = kg.entities.get(eid)
        if e is None:
            return {"id": eid, "name": eid, "type": "Unknown"}
        out: dict[str, Any] = {"id": e.id, "name": e.name, "type": e.type}
        if e.aliases:
            out["aliases"] = e.aliases
        return out

    def _resolve(self, ref: str) -> str | None:
        """接受实体 id 或名称/别名。模型两种都会传, 都要认。"""
        kg = self.kg
        if ref in kg.entities:
            return ref
        eid = kg.resolve(ref)
        if eid:
            return eid
        linked = self.svc.graph_retriever.linker.link(ref)
        return linked[0][0] if linked else None

    def _not_found(self, ref: str) -> dict:
        return {"error": f"图中找不到实体「{ref}」",
                "hint": "先用 find_entity 做模糊查找拿到实体 id; "
                        "若仍找不到, 该对象可能只在文档里出现, 改用 search_text。"}

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def find_entity(self, name: str, limit: int = 5) -> dict:
        kg = self.kg
        hits: list[str] = []
        exact = self._resolve(name)
        if exact:
            hits.append(exact)
        # 子串模糊匹配兜底: 问"太狼"时把灰太狼、红太狼、蕉太狼等候选都列出来, 由模型挑
        q = name.strip().lower()
        for e in kg.entities.values():
            if len(hits) >= limit:
                break
            if e.id in hits:
                continue
            if any(q and q in f.lower() for f in [e.name, *e.aliases]):
                hits.append(e.id)
        if not hits:
            return self._not_found(name)
        return {"candidates": [{**self._entity_brief(h), "degree": kg.degree(h)}
                               for h in hits]}

    def get_neighbors(self, entity: str, relation: str | None = None,
                      limit: int = 30) -> dict:
        kg = self.kg
        eid = self._resolve(entity)
        if eid is None:
            return self._not_found(entity)
        if relation and relation not in RELATIONS:
            return {"error": f"未知关系 {relation}", "valid_relations": list(RELATIONS)}
        rows: list[dict] = []
        for e in kg.out(eid, relation):
            rows.append({"direction": "out", "relation": e.rel,
                         "other": self._entity_brief(e.dst), "docs": e.all_docs()})
        for e in kg.inn(eid, relation):
            rows.append({"direction": "in", "relation": e.rel,
                         "other": self._entity_brief(e.src), "docs": e.all_docs()})
        return {"entity": self._entity_brief(eid), "total": len(rows),
                "truncated": len(rows) > limit, "edges": rows[:limit]}

    def find_paths(self, source: str, target: str, max_hops: int = 4,
                   limit: int = 5) -> dict:
        kg = self.kg
        src, dst = self._resolve(source), self._resolve(target)
        if src is None:
            return self._not_found(source)
        if dst is None:
            return self._not_found(target)
        max_hops = max(1, min(max_hops, 5))     # 5 跳以上路径数量指数爆炸, 硬上限
        paths = kg.find_paths(src, dst, max_hops=max_hops, limit=limit)
        paths.sort(key=len)
        return {"source": self._entity_brief(src), "target": self._entity_brief(dst),
                "paths": [{"hops": len(p), "path": kg.path_str(p),
                           "docs": kg.path_docs(p)} for p in paths]}

    def explain_relation(self, a: str, b: str, max_hops: int = 4) -> dict:
        """复用图检索器的关系路径逻辑(含枢纽阻断), 保证与评测口径一致。

        与 find_paths 的区别: 这里**不允许经由枢纽中转**。两个角色同属一部剧,
        经由节目节点永远能连上 —— 那不是关系, 是噪音。
        """
        kg = self.kg
        src, dst = self._resolve(a), self._resolve(b)
        if src is None:
            return self._not_found(a)
        if dst is None:
            return self._not_found(b)
        gr = self.svc.graph_retriever
        paths = kg.find_paths(src, dst, max_hops=max(1, min(max_hops, 5)),
                              limit=gr.max_paths, hub_degree=gr.hub_degree)
        return {"a": self._entity_brief(src), "b": self._entity_brief(dst),
                "paths": [{"hops": len(p), "path": kg.path_str(p),
                           "docs": kg.path_docs(p)} for p in paths],
                "note": ("两者之间没有不经枢纽的关系路径(已阻断连接数>"
                         f"{gr.hub_degree} 的节点中转, 如节目节点)") if not paths else ""}

    def list_by_relation(self, relation: str, value: str | None = None,
                         limit: int = 50) -> dict:
        """按关系(和可选的取值)列出所有匹配的边 —— Agent v2 新增。

        v1 的失败分析里有两类问题都源于缺这个工具(见 docs/07-Agent.md 第三节):
          · 列举题只能逐个查邻居: "正义阵营有哪些机器人"调了 13 次工具
          · 共同属性连不起来: 阵营是字面量节点, 按设计不能作为路径中转,
            "藏之助和卡布达同属正义阵营"这类联系, explain_relation 永远找不到
        """
        kg = self.kg
        if relation not in RELATIONS:
            return {"error": f"未知关系 {relation}", "valid_relations": list(RELATIONS)}
        want_id = self._resolve(value) if value else None
        want = normalize_name(value) if value else ""
        rows: list[dict] = []
        for e in kg.edges:
            if e.rel != relation:
                continue
            if value:
                obj = normalize_name(kg.name(e.dst))
                # 实体按 id 匹配(支持别名); 字面量按规范化后的文本匹配, 允许包含关系
                if not (e.dst == want_id or obj == want or (want and want in obj)):
                    continue
            rows.append({"subject": {"id": e.src, "name": kg.name(e.src)},
                         "object": kg.name(e.dst), "docs": e.all_docs()})
        if not rows:
            return {"relation": relation, "value": value, "total": 0, "rows": [],
                    "hint": "没有匹配的边。不确定取值写法时, 先不填 value 看看这个关系有哪些取值。"}
        return {"relation": relation, "value": value, "total": len(rows),
                "truncated": len(rows) > limit, "rows": rows[:limit]}

    def search_text(self, query: str, top_k: int = 5) -> dict:
        top_k = max(1, min(top_k, 10))
        hits = self.svc.lexical.retrieve(query, top_k)
        out = []
        for cid, score in hits:
            c = self.svc.chunks.get(cid)
            if c is None:
                continue
            out.append({"doc_id": c.doc_id, "title": c.doc_title,
                        "section": c.section, "score": round(float(score), 4),
                        "snippet": c.raw_text[:SNIPPET_CHARS]})
        return {"retriever": getattr(self.svc.lexical, "name", ""), "results": out}

    def read_document(self, doc_id: str, max_chars: int = DOC_CHARS) -> dict:
        d = self.svc.docs.get(doc_id)
        if d is None:
            return {"error": f"不存在文档 {doc_id}",
                    "hint": "doc_id 形如 doc-0001, 从其他工具返回的 docs 字段里取。"}
        text = d["text"]
        return {"doc_id": doc_id, "title": d.get("title", ""),
                "doc_type": d.get("doc_type", ""), "date": d.get("date", ""),
                "truncated": len(text) > max_chars, "text": text[:max_chars]}

    def ask(self, question: str) -> dict:
        """固定流水线(规则路由 + 检索 + 生成)。会调用一次 LLM。"""
        with contextlib.redirect_stdout(sys.stderr):
            ans = self.svc.ask(question)
        return {"answer": ans.text, "refused": ans.refused, "route": ans.route,
                "citations": [{"index": c.index, "doc_id": c.doc_id, "valid": c.valid}
                              for c in ans.citations],
                "graph_paths": ans.graph_paths[:6]}

    # ------------------------------------------------------------------
    # 只读资源
    # ------------------------------------------------------------------
    def schema(self) -> dict:
        s = self.kg.stats()
        return {"graph_source": self._cfg.graph_path,
                "entity_types": s["by_type"],
                "relations": {r: {"meaning": m, "count": s["by_rel"].get(r, 0)}
                              for r, m in RELATIONS.items()},
                "documents": len(self.svc.docs),
                "notes": ["角色名采用中国大陆播出版译名; 又名/台湾译名(如 蝎子蓝蓝)也可作为查询输入",
                          "语料根据维基百科等公开资料整理, 覆盖《喜羊羊与灰太狼》《铁甲小宝》两部剧"]}

    def entities(self, type_: str | None = None) -> list[dict]:
        return [{"id": e.id, "name": e.name, "type": e.type}
                for e in self.kg.entities.values()
                if type_ is None or e.type == type_]
