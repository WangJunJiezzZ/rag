"""
实体消歧 —— 把同一实体的不同写法归并成一个节点。

为什么这一步是图谱的生死线
--------------------------
抽取出来的名称是散的:
    "高圆寺寅彦" / "高圆寺博士"      "吉祥寺藏之助" / "藏之助"
不归并, 图里就是两个孤立节点, **所有经过它的多跳查询全部失效**。
本项目刻意植入了 3 处"证据文档只写别名"(见 verify_dataset.py 的 B 项)。

四级级联, 成本递增
------------------
    0. 声明别名   文档里明确写着「A又名B」 -> 直接合并。这是最强的证据, 且免费
    1. 规则归一   normalize_name(): 去标点/空白/大小写。O(n), 免费
    2. 向量召回   同类型实体两两算相似度, 只保留候选对。O(n²) 但纯矩阵运算, 毫秒级
    3. LLM 裁决   只处理词面说不清、且向量相似度 >= low 的候选对

为什么不直接全量 LLM 两两比较: n=50 时就是 1200 多次调用, 而其中 99% 的对比是
显然的("卡布达" vs "暖羊羊")。**把 LLM 用在它不可替代的地方, 而不是所有地方。**

两条一票否决 (cannot-link)
--------------------------
**单一信号永远不足以触发合并**, 但单一信号足以**阻止**合并:

  · 词面不相容  前两字不同的两个名称, 向量再像也不合并。
                喜羊羊/美羊羊/懒羊羊、灰太狼/红太狼/蕉太狼在向量空间里挨得极近,
                只看向量会把整个羊村并成一只羊。
  · 有关系边    同一份文档里"A 是 B 的爷爷", A 和 B 就一定不是同一个实体。
                「高圆寺寅彦」与「高圆寺让」前三个字相同、向量高度相似,
                LLM 看名字也可能犹豫 —— 但抽取结果里有一条爷孙关系边, 这就够了。

子串不等于简称
--------------
公司名里"星海资管"⊂"星海资产管理有限公司"几乎总是简称; 角色名里却经常相反:
    「和平星」⊂「射手座和平星」      「卡布达」⊂「卡布达巨人」
前者是总称与个体, 后者是机器人与它的变身形态。照搬"子串即简称"的规则, 实测踩了两次:
  · 第一次把三颗和平星并成了一颗 -> 加上"前两字相同"的限制
  · 第二次 LLM 抽取把「卡布达巨人」抽成了实体, 与「卡布达」前两字相同, 又被并掉
所以子串现在只算**弱证据**: 前两字相同的名称进入灰区, 交给 LLM 裁决;
真正的别名(「慢羊羊村长」「藏之助」)靠文档里声明的「又名」合并, 不靠猜。
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..llm import LLM
from ..prompts import registry
from ..store.graph import normalize_name


@dataclass
class ResolveStats:
    mentions: int = 0
    canonical: int = 0
    merged_by_alias: int = 0      # 文档声明的「又名」
    merged_by_rule: int = 0
    merged_by_vector: int = 0
    merged_by_llm: int = 0
    llm_calls: int = 0
    llm_rejected: int = 0
    blocked_by_lexical: int = 0   # 向量说像但词面一票否决 —— 避免的错合并
    blocked_by_relation: int = 0  # 两者之间有关系边 -> 一票否决
    blocked_transitive: int = 0   # 被 cannot-link 拦下的传递性错合并
    gray_unresolved: int = 0      # 灰区但无 LLM 可用 -> 留作未合并
    cost_usd: float = 0.0
    examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"提及 {self.mentions} 个名称 -> 归并为 {self.canonical} 个实体\n"
                f"  声明别名合并 {self.merged_by_alias} | 规则/词面合并 {self.merged_by_rule} | "
                f"向量合并 {self.merged_by_vector} | LLM 裁决合并 {self.merged_by_llm}\n"
                f"  词面一票否决 {self.blocked_by_lexical} 对 | "
                f"关系边一票否决 {self.blocked_by_relation} 对 | "
                f"cannot-link 拦下传递性错合并 {self.blocked_transitive} 次\n"
                f"  LLM 调用 {self.llm_calls} 次 (驳回 {self.llm_rejected}), "
                f"灰区未解决 {self.gray_unresolved}, 花费 ≈ ${self.cost_usd:.3f}")


class UnionFind:
    """带 cannot-link 约束的并查集。

    为什么需要约束: 合并是**传递**的。a≡b 且 b≡c 就会得到 a≡c,
    哪怕 a 与 c 明确不同。例如
        高圆寺让 ≡ 小让          (对, 文档声明的又名)
        高圆寺让 ≡ 高圆寺博士    (错, 但前三个字相同, 向量也像)
        => 爷爷和孙子被传递性地并成一个节点, 爷孙关系变成了自环。
    修法是约束聚类里的标准做法(cannot-link): 合并前检查两个簇的成员之间
    是否存在已知的"明确不同"关系, 有就拒绝合并。
    单点的判断可能出错, 但簇级的一票否决能把错误控制住。
    """

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.members: dict[str, set[str]] = {}
        self.cannot: set[tuple[str, str]] = set()
        self.blocked_transitive = 0

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        self.members.setdefault(x, {x})
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def forbid(self, a: str, b: str) -> None:
        self.cannot.add((a, b) if a < b else (b, a))

    def _conflicts(self, ra: str, rb: str) -> bool:
        ma, mb = self.members.get(ra, {ra}), self.members.get(rb, {rb})
        if len(ma) > len(mb):
            ma, mb = mb, ma
        for x in ma:
            for y in mb:
                if ((x, y) if x < y else (y, x)) in self.cannot:
                    return True
        return False

    def union(self, a: str, b: str, prefer: str | None = None) -> bool:
        """prefer: 指定用哪个名字做簇代表(声明别名时, 档案主人的名字是规范名)。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._conflicts(ra, rb):
            self.blocked_transitive += 1
            return False
        if prefer is not None:
            keep = self.find(prefer)
            ra, rb = (keep, rb if keep == ra else ra)
        elif len(rb) > len(ra):
            # 名字长的做代表 —— 全称比简称信息量大
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members.setdefault(ra, {ra}).update(self.members.pop(rb, {rb}))
        return True


class EntityResolver:
    def __init__(self, embedder=None, llm: LLM | None = None,
                 high: float = 0.93, low: float = 0.72,
                 prompt_id: str = "resolve_entity/v1_adjudicate"):
        self.embedder = embedder
        self.llm = llm
        self.high = high          # 保留参数以兼容旧调用; 向量不再单独拍板
        self.low = low            # 低于此: 直接判不同, 不送 LLM
        self.prompt_id = prompt_id
        self.stats = ResolveStats()

    # ------------------------------------------------------------------
    def resolve(self, mentions: dict[str, str],
                contexts: dict[str, str] | None = None,
                neighbors: dict[str, set[str]] | None = None,
                aliases: list[tuple[str, str]] | None = None,
                related: set[tuple[str, str]] | None = None) -> dict[str, str]:
        """mentions: {名称 -> 类型}
        aliases:  [(规范名, 别名)] —— 文档声明的「又名」
        related:  {(a, b)} —— 抽取结果里有关系边直接相连的名称对, 必不相同
        """
        st = self.stats
        st.mentions = len(mentions)
        uf = UnionFind()
        contexts = contexts or {}
        related = related or set()
        for a, b in related:
            uf.forbid(a, b)

        # ---- 第 0 级: 文档声明的别名 ----
        for canon, alias in aliases or []:
            if alias in mentions and canon in mentions and uf.union(canon, alias, prefer=canon):
                st.merged_by_alias += 1
                self._note(f"[又名] {alias} -> {canon}")

        # ---- 第 1 级: 规则归一 ----
        by_norm: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name, typ in mentions.items():
            by_norm[(typ, normalize_name(name))].append(name)
        for names in by_norm.values():
            for other in names[1:]:
                if uf.union(names[0], other):
                    st.merged_by_rule += 1

        by_type: dict[str, list[str]] = defaultdict(list)
        for name, typ in mentions.items():
            by_type[typ].append(name)

        for typ, names in by_type.items():
            reps = sorted({uf.find(n) for n in names})
            if len(reps) < 2:
                continue

            # ---- 第 2 级 a: 强词面证据 —— 不依赖向量, 也不花钱 ----
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    a, b = reps[i], reps[j]
                    if uf.find(a) == uf.find(b):
                        continue
                    v0 = lexical_verdict(a, b)
                    if v0 == "incompatible":
                        uf.forbid(a, b)
                    elif v0 == "strong" and uf.union(a, b):
                        st.merged_by_rule += 1
                        self._note(f"[词面-强] {a} == {b}")

            # ---- 第 2 级 b: 词面说不清的, 才动用向量 / LLM ----
            reps = sorted({uf.find(n) for n in names})
            if self.embedder is not None and len(reps) >= 2:
                for a, b, sim in self._candidate_pairs(reps):
                    if uf.find(a) == uf.find(b):
                        continue
                    if _pair(a, b) in related or _pair(uf.find(a), uf.find(b)) in related:
                        st.blocked_by_relation += 1
                        continue
                    v = lexical_verdict(a, b)
                    if v == "incompatible":
                        st.blocked_by_lexical += 1   # 向量说像, 词面一票否决
                        continue
                    # 向量只负责**提名**候选, 不负责拍板: 词面说不清(weak)时,
                    # 相似度再高也要 LLM 裁决。「卡布达」与「卡布达巨人」向量几乎一样,
                    # 但一个是机器人, 一个是它的变身形态。
                    if v == "strong":
                        if uf.union(a, b):
                            st.merged_by_vector += 1
                            self._note(f"[向量 {sim:.2f} + 词面] {a} == {b}")
                    else:
                        self._maybe_llm(uf, a, b, contexts, sim)

        mapping = {n: uf.find(n) for n in mentions}
        st.canonical = len(set(mapping.values()))
        st.blocked_transitive = uf.blocked_transitive
        return mapping

    def _note(self, msg: str) -> None:
        if len(self.stats.examples) < 12:
            self.stats.examples.append(msg)

    def _maybe_llm(self, uf: "UnionFind", a: str, b: str,
                   contexts: dict[str, str], sim: float) -> None:
        verdict = self._adjudicate(a, b, contexts)
        if verdict is None:
            self.stats.gray_unresolved += 1
        elif verdict:
            if uf.union(a, b):
                self.stats.merged_by_llm += 1
                self._note(f"[LLM {sim:.2f}] {a} == {b}")
        else:
            self.stats.llm_rejected += 1

    # ------------------------------------------------------------------
    def _candidate_pairs(self, names: list[str]) -> list[tuple[str, str, float]]:
        vecs = self.embedder.encode(names, is_query=False)
        sims = vecs @ vecs.T
        out: list[tuple[str, str, float]] = []
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sims[i, j])
                if s >= self.low:
                    out.append((names[i], names[j], s))
        out.sort(key=lambda x: -x[2])       # 高相似度优先合并, 减少链式误并
        return out

    def _adjudicate(self, a: str, b: str, contexts: dict[str, str]) -> bool | None:
        """None = 无法判定(没有 LLM 可用)。"""
        if self.llm is None:
            return None
        p = registry.get(self.prompt_id)
        system, user = p.render(name_a=a, context_a=contexts.get(a, "（无上下文）"),
                                name_b=b, context_b=contexts.get(b, "（无上下文）"))
        try:
            res = self.llm.complete(
                system=system, user=user, max_tokens=512,
                schema={"type": "object",
                        "properties": {"same": {"type": "boolean"},
                                       "confidence": {"type": "number"},
                                       "reason": {"type": "string"}},
                        "required": ["same", "confidence"],
                        "additionalProperties": False},
                cache_tag=self.prompt_id)
        except Exception:
            return None
        self.stats.llm_calls += 1
        self.stats.cost_usd += res.cost_usd
        d = res.parsed or {}
        if not isinstance(d, dict) or "same" not in d:
            return None
        return bool(d["same"]) and float(d.get("confidence", 0)) >= 0.7


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


# --------------------------------------------------------------------------
# 词面守卫 —— 向量的安全带
# --------------------------------------------------------------------------

_CJK_RANGE = ("一", "鿿")


def _script(s: str) -> str:
    has_cjk = any(_CJK_RANGE[0] <= c <= _CJK_RANGE[1] for c in s)
    has_ascii = any("a" <= c.lower() <= "z" for c in s)
    if has_cjk and has_ascii:
        return "mixed"
    return "cjk" if has_cjk else "ascii"


_DISCRIMINATOR_RE = re.compile(r"[（(]\s*([0-9０-９]{1,3})\s*[）)]")


def _discriminator(name: str) -> str | None:
    """抽出名称里的**区分标记**, 如「卡布达（2）」中的 2。

    规范化会把括号当标点洗掉, 但区分标记是**区分性信息, 不是噪音**。
    规则: 两个名称的区分标记必须完全一致(同为无, 或同值), 否则不得合并。
    """
    m = _DISCRIMINATOR_RE.search(name)
    if not m:
        return None
    return m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def lexical_verdict(a: str, b: str) -> str:
    """词面证据的三值判定: strong / weak / incompatible。

    **词面既是守卫也是证据。**
      strong        本身就足以合并, 不需要向量或 LLM 背书
                    (规范化后完全相同, 如「蝎子莱莱」与「蝎子 莱莱」)
      incompatible  一票否决, 向量再像也不合并
                    (前两字不同, 或区分标记不一致, 或文种不同)
      weak          说不清 -> 交给 LLM 裁决
                    (前两字相同: 慢羊羊 vs 慢羊羊村长, 卡布达 vs 卡布达巨人,
                     高圆寺让 vs 高圆寺博士 —— 有的是同一实体, 有的不是)
    """
    if _discriminator(a) != _discriminator(b):
        return "incompatible"
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return "incompatible"
    if na == nb:
        return "strong"
    if _script(a) != _script(b):
        # 跨文种(如 AP717 vs 警用机器人)没有任何廉价的词面信号 ——
        # 只认文档声明的「又名」, 不猜。
        return "incompatible"
    # 品牌前缀取前两个字: 只比首字会让「小灰灰」「小香香」「小让」互相进入灰区
    same_head = na[:2] == nb[:2]
    if not same_head:
        return "incompatible"
    # 子串只是弱证据: 「慢羊羊」⊂「慢羊羊村长」是同一只羊,
    # 「卡布达」⊂「卡布达巨人」却是机器人和它的变身形态 —— 词面分不出来, 交给 LLM。
    return "weak"
