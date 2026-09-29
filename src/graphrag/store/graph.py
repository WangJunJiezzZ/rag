"""
知识图谱存储与遍历。

为什么不用 Neo4j:
  demo 规模(数千节点/万级边)下, 图遍历在内存里几十行就能写完, 且零外部依赖,
  在任何机器上都能一键跑起来。引入图数据库在这个规模上是过度工程。
  本类的接口(neighbors / find_paths / reach)与 Cypher 的表达力对齐,
  换 Neo4j 只需换这一个文件的实现 —— 这是"接口先行"的取舍, 面试可讲。

两处用途共用同一份实现:
  1) 评测时加载 ground-truth 图 (data/synthetic/graph.json)
  2) 运行时加载从文档中抽取出来的图 (data/index/extracted_graph.json)
  对比这两张图, 就得到抽取环节自身的 precision / recall。
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..compat import write_text

LITERAL_PREFIX = "lit:"


@dataclass
class Entity:
    id: str
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    props: dict = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    source_doc: str = ""
    confidence: float = 1.0
    # 同一条事实被多份文档支撑时, 其余来源记在这里。
    # **必须是正式字段**: 首版把它塞在 __dict__ 里, asdict() 序列化时被静默丢弃,
    # 存盘再载入后多来源溯源信息全部消失, 图检索凭空少掉一批证据文档。
    # 这类"内存里对、落盘后错"的 bug 不会报错, 只会让分数莫名其妙地低。
    extra_docs: list[str] = field(default_factory=list)

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.rel, self.dst)

    def all_docs(self) -> list[str]:
        return [d for d in [self.source_doc, *self.extra_docs] if d]


@dataclass
class Step:
    """路径上的一步。direction=+1 表示沿边正向走, -1 表示逆向走。"""
    edge: Edge
    direction: int

    @property
    def frm(self) -> str:
        return self.edge.src if self.direction > 0 else self.edge.dst

    @property
    def to(self) -> str:
        return self.edge.dst if self.direction > 0 else self.edge.src


Path_ = list[Step]


class KnowledgeGraph:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        self._edges: list[Edge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self._name_index: dict[str, str] = {}      # 规范化名称 -> entity id
        self.meta: dict = {}

    # ---------------- 构建 ----------------

    def add_entity(self, e: Entity) -> Entity:
        existing = self.entities.get(e.id)
        if existing:
            for a in e.aliases:
                if a not in existing.aliases:
                    existing.aliases.append(a)
            existing.props.update(e.props)
            self._index_names(existing)
            return existing
        self.entities[e.id] = e
        self._index_names(e)
        return e

    def add_edge(self, edge: Edge, dedupe: bool = True) -> bool:
        """返回 True 表示新增了一条此前不存在的边。"""
        if dedupe and edge.key() in self._edge_keys:
            # 同一事实被多份文档支撑 —— 保留第一条, 但记录额外来源
            for existing in self._out[edge.src]:
                if existing.key() == edge.key():
                    if edge.source_doc and edge.source_doc != existing.source_doc \
                       and edge.source_doc not in existing.extra_docs:
                        existing.extra_docs.append(edge.source_doc)
                    break
            return False
        self._edges.append(edge)
        self._edge_keys.add(edge.key())
        self._out[edge.src].append(edge)
        self._in[edge.dst].append(edge)
        return True

    def _index_names(self, e: Entity) -> None:
        for n in [e.name, *e.aliases]:
            self._name_index.setdefault(normalize_name(n), e.id)

    # ---------------- 查询 ----------------

    @property
    def edges(self) -> list[Edge]:
        return self._edges

    def by_type(self, t: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.type == t]

    def name(self, eid: str) -> str:
        if eid.startswith(LITERAL_PREFIX):
            return eid[len(LITERAL_PREFIX):]
        e = self.entities.get(eid)
        return e.name if e else eid

    def resolve(self, text: str) -> str | None:
        """按规范名或别名查实体 id。"""
        return self._name_index.get(normalize_name(text))

    def out(self, eid: str, rel: str | None = None) -> list[Edge]:
        return [e for e in self._out.get(eid, []) if rel is None or e.rel == rel]

    def inn(self, eid: str, rel: str | None = None) -> list[Edge]:
        return [e for e in self._in.get(eid, []) if rel is None or e.rel == rel]

    def one_out(self, eid: str, rel: str) -> str | None:
        es = self.out(eid, rel)
        return es[0].dst if es else None

    def neighbors(self, eid: str, rel: str | None = None,
                  direction: str = "both") -> list[str]:
        res: list[str] = []
        if direction in ("out", "both"):
            res += [e.dst for e in self.out(eid, rel)]
        if direction in ("in", "both"):
            res += [e.src for e in self.inn(eid, rel)]
        return res

    def steps_from(self, eid: str) -> Iterator[Step]:
        """从某节点可走的所有边(无向)。

        **字面量节点不得作为中转。** 否则会产生这样的伪路径:
            卡布达 -[TRANSFORM_TIME]-> "3 分钟" <-[TRANSFORM_TIME]- 某机器人 -> ...
        两个角色仅因某个属性值相同就被判定为存在关联。
        字面量(原型/编号/阵营)是**属性值**不是实体, 数值相同不构成任何关系。
        它可以是路径的终点(例如"走到了'正义'阵营"), 但永远不能是中间节点。
        """
        if eid.startswith(LITERAL_PREFIX):
            return
        for e in self._out.get(eid, []):
            yield Step(e, +1)
        for e in self._in.get(eid, []):
            yield Step(e, -1)

    def reach(self, start: str, max_hops: int,
              predicate: Callable[[str], bool] | None = None,
              rel_whitelist: Iterable[str] | None = None,
              ) -> list[tuple[str, Path_]]:
        """从 start 出发做 BFS, 返回 (命中的实体id, 最短路径)。

        无向遍历 —— 关系穿透天然是无向的:
        "灰太狼->小灰灰"(父子) 与 "红太狼->小灰灰"(母子) 方向相同, 但从小灰灰出发
        要逆着边才能走到父母 —— 亲属链天然需要双向走。
        """
        allow = set(rel_whitelist) if rel_whitelist else None
        seen = {start}
        queue: deque[tuple[str, Path_]] = deque([(start, [])])
        hits: list[tuple[str, Path_]] = []
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for step in self.steps_from(node):
                if allow and step.edge.rel not in allow:
                    continue
                nxt = step.to
                if nxt in seen:
                    continue
                seen.add(nxt)
                new_path = path + [step]
                if predicate is None or predicate(nxt):
                    hits.append((nxt, new_path))
                queue.append((nxt, new_path))
        return hits

    def degree(self, eid: str) -> int:
        return len(self._out.get(eid, [])) + len(self._in.get(eid, []))

    def entity_degree(self, eid: str) -> int:
        """只数连向**实体**的边。字面量边(口头禅、原型、编号……)不算。

        枢纽判定要用这个而不是 degree(): 卡布达有十几条字面量边,
        但它不是枢纽 —— 字面量本来就不能中转, 数进去只会把普通角色误判成枢纽。
        """
        return (sum(1 for e in self._out.get(eid, []) if not e.dst.startswith(LITERAL_PREFIX))
                + len(self._in.get(eid, [])))

    def is_hub(self, eid: str, hub_degree: int | None) -> bool:
        return hub_degree is not None and self.entity_degree(eid) > hub_degree

    def paths_to(self, start: str, predicate: Callable[[str], bool],
                 max_hops: int = 4, max_paths: int = 12,
                 hub_degree: int | None = None) -> list[Path_]:
        """枚举 start 出发、终点满足 predicate 的**多条**简单路径。

        与 reach() 的关键区别: reach 对每个节点只保留最短路径, 于是
        一条短路径会把经过同一节点的长路径整条吞掉 —— 而长的那条可能才是
        真正要展示的关系。这里不按节点去重。

        hub_degree: 度数超过该阈值的节点**不允许作为中转**(仍可作为终点)。
          本项目的节目节点(《喜羊羊与灰太狼》《铁甲小宝》)各连着十几个角色,
          若允许穿透, 同一部剧里任意两个角色都会被判定为"存在关联"
          —— 这是关系图谱里最常见的假阳性来源。
          阻断枢纽中转是图分析里的标准做法, 代价是可能漏掉真实的弱关联,
          所以它是一个可消融的参数, 而不是写死的规则。
        """
        out: list[Path_] = []
        stack: list[tuple[str, Path_, set[str]]] = [(start, [], {start})]
        while stack and len(out) < max_paths:
            node, path, visited = stack.pop()
            if len(path) >= max_hops:
                continue
            if path and self.is_hub(node, hub_degree):
                continue                       # 枢纽节点: 到此为止, 不再往外扩
            for step in self.steps_from(node):
                nxt = step.to
                if nxt in visited:
                    continue
                new_path = path + [step]
                if predicate(nxt):
                    out.append(new_path)
                    if len(out) >= max_paths:
                        break
                else:
                    stack.append((nxt, new_path, visited | {nxt}))
        out.sort(key=len)
        return out

    def find_paths(self, src: str, dst: str, max_hops: int = 4,
                   limit: int = 8, hub_degree: int | None = None) -> list[Path_]:
        """枚举 src -> dst 的所有不超过 max_hops 跳的简单路径(短的在前)。

        按层 BFS 而不是 DFS: DFS 会先钻进一条深路径把 limit 用完,
        最短的那条反而没被找到。hub_degree 的含义同 paths_to()。
        """
        out: list[Path_] = []
        frontier: list[tuple[str, Path_, set[str]]] = [(src, [], {src})]
        for _ in range(max_hops):
            nxt: list[tuple[str, Path_, set[str]]] = []
            for node, path, visited in frontier:
                if path and self.is_hub(node, hub_degree):
                    continue
                for step in self.steps_from(node):
                    to = step.to
                    if to in visited:
                        continue
                    new_path = path + [step]
                    if to == dst:
                        out.append(new_path)
                        if len(out) >= limit:
                            return out
                    else:
                        nxt.append((to, new_path, visited | {to}))
            frontier = nxt
            if not frontier:
                break
        return out

    def path_docs(self, path: Path_) -> list[str]:
        """一条路径涉及的全部来源文档 —— 图检索结果的溯源依据。"""
        docs: list[str] = []
        for step in path:
            for d in step.edge.all_docs():
                if d not in docs:
                    docs.append(d)
        return docs

    def path_to_dict(self, path: Path_) -> dict:
        """结构化的路径, 供前端画图。字符串版只适合日志。"""
        if not path:
            return {"nodes": [], "edges": []}
        nodes: list[dict] = []
        seen: set[str] = set()

        def push(nid: str) -> None:
            if nid in seen:
                return
            seen.add(nid)
            e = self.entities.get(nid)
            nodes.append({
                "id": nid, "name": self.name(nid),
                "type": "Literal" if nid.startswith(LITERAL_PREFIX)
                        else (e.type if e else "Unknown"),
            })

        push(path[0].frm)
        edges: list[dict] = []
        for step in path:
            push(step.to)
            edges.append({"from": step.frm, "to": step.to,
                          "rel": step.edge.rel,
                          "reversed": step.direction < 0,
                          "docs": step.edge.all_docs()})
        return {"nodes": nodes, "edges": edges}

    def path_str(self, path: Path_) -> str:
        if not path:
            return ""
        parts = [self.name(path[0].frm)]
        for step in path:
            arrow = f"-[{step.edge.rel}]->" if step.direction > 0 else f"<-[{step.edge.rel}]-"
            parts.append(f" {arrow} {self.name(step.to)}")
        return "".join(parts)

    # ---------------- 序列化 ----------------

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        kg.meta = {k: v for k, v in raw.items() if k not in ("entities", "edges")}
        for e in raw.get("entities", []):
            kg.add_entity(Entity(**e))
        for e in raw.get("edges", []):
            kg.add_edge(Edge(**{k: v for k, v in e.items()
                                if k in ("src", "rel", "dst", "source_doc",
                                         "confidence", "extra_docs")}))
        return kg

    def save(self, path: str | Path) -> None:
        write_text(path, json.dumps({
            **self.meta,
            "entities": [asdict(e) for e in self.entities.values()],
            "edges": [asdict(e) for e in self._edges],
        }, ensure_ascii=False, indent=2))

    def stats(self) -> dict:
        by_type: dict[str, int] = defaultdict(int)
        for e in self.entities.values():
            by_type[e.type] += 1
        by_rel: dict[str, int] = defaultdict(int)
        for e in self._edges:
            by_rel[e.rel] += 1
        return {"entities": len(self.entities), "edges": len(self._edges),
                "by_type": dict(by_type), "by_rel": dict(by_rel)}


# --------------------------------------------------------------------------

_PUNCT = "（）()。，,、．. 　\t\r\n·:：“”\"'-—_/\\《》「」！!？?…"


def normalize_name(s: str) -> str:
    """名称规范化 —— 实体消歧的第一道(也是最廉价的一道)工序。

    只做确定性的规则归一(大小写/标点/空白); 语义层面的归一(本名 <-> 又名 <-> 译名)
    由 ingest/resolve.py 用声明别名 + embedding + LLM 处理。
    """
    s = (s or "").strip().lower()
    for ch in _PUNCT:
        s = s.replace(ch, "")
    return s
