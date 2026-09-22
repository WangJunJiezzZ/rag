"""
实体消歧 —— 把同一实体的不同写法归并成一个节点。

为什么这一步是图谱的生死线
--------------------------
抽取出来的名称是散的:
    "星海资产管理有限公司" / "星海资管" / "Starsea Asset Management Pte. Ltd."
不归并, 图里就是三个孤立节点, **所有跨文档的多跳查询全部失效**。
本项目已验证: 有 12 道题的答案实体在证据文档中只以别名出现。

三级级联, 成本递增
------------------
    1. 规则归一   normalize_name(): 去后缀/标点/大小写。O(n), 免费, 覆盖大多数
    2. 向量召回   同类型实体两两算相似度, 只保留候选对。O(n²) 但纯矩阵运算, 毫秒级
    3. LLM 裁决   只处理落在灰区 [low, high] 的候选对

为什么不直接全量 LLM 两两比较: n=200 时是 2 万次调用, 而其中 99% 的对比是
显然的("星海资管" vs "李明远")。**把 LLM 用在它不可替代的地方, 而不是所有地方。**

一次真实的事故: 向量相似度 1.00 却是两家不同公司
--------------------------------------------------
首次跑通时出现:
    [向量 1.00] Daiyu Capital Management Ltd. == Dixing Capital Management Ltd.
    [向量 1.00] Daiyu Capital Management Ltd. == Heming Financial Services Ltd.
原因: bge-small-zh 是**中文**模型, 对纯英文串的表示几乎坍缩到同一点,
余弦相似度全是 1.00。若只看向量, 所有英文公司名会被并成一个节点, 整张图作废。

由此得出本模块的核心原则:
    **单一信号永远不足以触发合并。**
  · 同文种(都中文): 向量 + 词面守卫, 两者一致才合并
  · 都英文:         向量不可信, 改用 token Jaccard
  · 跨文种:         没有任何廉价信号 -> 用"共享邻居"做分块, 再交 LLM 裁决

跨文种: 先用拼音, 不要一上来就找 LLM
-------------------------------------
首版用"共享邻居"分块 + LLM 裁决, 实测结果很难看:
    LLM 调用 228 次 -> 只合并 1 对; 24 个实体仍被拆成中文名 + 罗马化名两个节点。
看模型的回复才发现问题不在模型:
    {"same": false, "reason": "中文名「柏观」与罗马化名「Jinyue」不对应"}
    —— 它判断是对的, 是**分块根本没把正确的对配出来**。
共享辖区这种信号太弱, 生成的几乎全是错误候选。

缺的信号其实非常显然: **拼音**。
    岱屿 -> daiyu, 而 "Daiyu Financial Services Ltd." 的首 token 就是 daiyu。
pypinyin 是纯 Python、3.5MB、完全离线的确定性库, 一次转换就能得到强证据。

教训: **有确定性方法可用时, 不要让 LLM 去做它。**
228 次 LLM 调用 + 几乎全错的候选, 被一个离线库替代了 ——
这不是模型不行, 是把模型用在了不该用的地方。

pypinyin 未安装时自动退回"共享邻居 + LLM"路径, 不阻断运行。

自然人的特殊规则
----------------
同名自然人在真实数据里极常见。没有强标识(证件号)时一律不合并 ——
漏合并只是少一条边, **错合并会把两个人的关系网连成一张图, 伪造出不存在的关联路径**。
在反洗钱场景里, 后者是要出事的。这条规则写死, 不交给模型判断。
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
    merged_by_rule: int = 0
    merged_by_vector: int = 0
    merged_by_llm: int = 0
    llm_calls: int = 0
    llm_rejected: int = 0
    blocked_by_lexical: int = 0   # 向量说像但词面一票否决 —— 避免的错合并
    blocked_transitive: int = 0   # 被 cannot-link 拦下的传递性错合并
    gray_unresolved: int = 0      # 灰区但无 LLM 可用 -> 留作未合并
    cost_usd: float = 0.0
    examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"提及 {self.mentions} 个名称 -> 归并为 {self.canonical} 个实体\n"
                f"  规则/词面合并 {self.merged_by_rule} | 向量合并 {self.merged_by_vector} | "
                f"LLM 裁决合并 {self.merged_by_llm}\n"
                f"  词面一票否决 {self.blocked_by_lexical} 对 | "
                f"cannot-link 拦下传递性错合并 {self.blocked_transitive} 次\n"
                f"  LLM 调用 {self.llm_calls} 次 (驳回 {self.llm_rejected}), "
                f"灰区未解决 {self.gray_unresolved}, 花费 ≈ ${self.cost_usd:.3f}")


class UnionFind:
    """带 cannot-link 约束的并查集。

    为什么需要约束: 合并是**传递**的。a≡b 且 b≡c 就会得到 a≡c,
    哪怕 a 与 c 明确不同。实测踩过这个坑 ——
        岱屿资本管理 ≡ Daiyu Capital       (对)
        岱屿金融服务 ≡ Daiyu Capital       (错, 但品牌拼音相同)
        => 两家不同公司被传递性地并成一个节点, 连带 3 个标准实体混作一团。

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
        # 簇小的那边遍历, 控制开销
        if len(ma) > len(mb):
            ma, mb = mb, ma
        for x in ma:
            for y in mb:
                if ((x, y) if x < y else (y, x)) in self.cannot:
                    return True
        return False

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._conflicts(ra, rb):
            self.blocked_transitive += 1
            return False
        # 名字长的做代表 —— 全称比简称信息量大
        if len(rb) > len(ra):
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
        self.high = high          # 高于此: 直接合并, 不问 LLM
        self.low = low            # 低于此: 直接判不同
        self.prompt_id = prompt_id
        self.stats = ResolveStats()

    # ------------------------------------------------------------------
    def resolve(self, mentions: dict[str, str],
                contexts: dict[str, str] | None = None,
                neighbors: dict[str, set[str]] | None = None) -> dict[str, str]:
        """mentions: {名称 -> 类型}。neighbors: {名称 -> 邻居名称集合}, 用于跨文种分块。"""
        st = self.stats
        st.mentions = len(mentions)
        uf = UnionFind()
        contexts = contexts or {}
        neighbors = neighbors or {}

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
            if typ == "Person":
                continue          # 自然人不做模糊合并, 见模块文档
            reps = sorted({uf.find(n) for n in names})
            if len(reps) < 2:
                continue
            cjk = [r for r in reps if _script(r) == "cjk"]
            ascii_ = [r for r in reps if _script(r) == "ascii"]

            # ---- 第 2 级 a0: 强词面证据 —— 不依赖向量, 也不花钱 ----
            # 必须先于向量路径执行: 强证据对不该排队等 LLM。
            for i in range(len(cjk)):
                for j in range(i + 1, len(cjk)):
                    a, b = cjk[i], cjk[j]
                    if uf.find(a) == uf.find(b):
                        continue
                    v0 = lexical_verdict(a, b)
                    if v0 == "incompatible":
                        uf.forbid(a, b)
                    elif v0 == "strong" and uf.union(a, b):
                        st.merged_by_rule += 1
                        self._note(f"[词面-强] {a} == {b}")

            # ---- 第 2 级 b: 词面说不清的, 才动用向量 / LLM ----
            if self.embedder is not None and len(cjk) >= 2:
                for a, b, sim in self._candidate_pairs(cjk):
                    if uf.find(a) == uf.find(b):
                        continue
                    v = lexical_verdict(a, b)
                    if v == "incompatible":
                        st.blocked_by_lexical += 1   # 向量说像, 词面一票否决
                        uf.forbid(a, b)
                        continue
                    if v == "strong" or sim >= self.high:
                        if uf.union(a, b):
                            st.merged_by_vector += 1
                            self._note(f"[向量 {sim:.2f}] {a} == {b}")
                    elif sim >= self.low:
                        self._maybe_llm(uf, a, b, contexts, sim)

            # ---- 第 2 级 b: 同为英文 —— 向量不可信, 用 token Jaccard ----
            for i in range(len(ascii_)):
                for j in range(i + 1, len(ascii_)):
                    a, b = ascii_[i], ascii_[j]
                    if uf.find(a) == uf.find(b):
                        continue
                    if _token_jaccard(a, b) >= 0.85 and \
                       lexical_verdict(a, b) != "incompatible":
                        if uf.union(a, b):
                            st.merged_by_rule += 1
                            self._note(f"[词面] {a} == {b}")

            # ---- 第 3 级: 跨文种 ----
            # 先用拼音拿强证据(免费、离线、确定性), 拿不到再退回共享邻居 + LLM。
            for a in cjk:
                for b in ascii_:
                    if uf.find(a) == uf.find(b):
                        continue
                    v = lexical_verdict(a, b)
                    if v == "incompatible":
                        uf.forbid(a, b)
                    elif v == "strong":
                        if uf.union(a, b):
                            st.merged_by_rule += 1
                            self._note(f"[拼音] {a} == {b}")
                    elif v == "weak":
                        na = neighbors.get(a, set())
                        if na and (na & neighbors.get(b, set())):
                            self._maybe_llm(uf, a, b, contexts, sim=0.0, cross=True)

        mapping = {n: uf.find(n) for n in mentions}
        st.canonical = len(set(mapping.values()))
        st.blocked_transitive = uf.blocked_transitive
        return mapping

    def _note(self, msg: str) -> None:
        if len(self.stats.examples) < 12:
            self.stats.examples.append(msg)

    def _maybe_llm(self, uf: "UnionFind", a: str, b: str,
                   contexts: dict[str, str], sim: float,
                   cross: bool = False) -> None:
        verdict = self._adjudicate(a, b, contexts)
        if verdict is None:
            self.stats.gray_unresolved += 1
        elif verdict:
            if uf.union(a, b):
                self.stats.merged_by_llm += 1
                tag = "LLM跨文种" if cross else f"LLM {sim:.2f}"
                self._note(f"[{tag}] {a} == {b}")
        else:
            self.stats.llm_rejected += 1

    # ------------------------------------------------------------------
    def _candidate_pairs(self, names: list[str]) -> list[tuple[str, str, float]]:
        import numpy as np
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


# --------------------------------------------------------------------------
# 词面守卫 —— 向量的安全带
# --------------------------------------------------------------------------

_CJK_RANGE = ("\u4e00", "\u9fff")

try:                                    # 可选依赖, 缺了就降级
    from pypinyin import Style, lazy_pinyin

    def _romanize(text: str, n_chars: int = 3) -> str:
        """取中文名的品牌部分(前 n 个汉字)转拼音, 用于跨文种比对。

        只转品牌: "岱屿金融服务有限公司" 整体转出来是
        daiyujinrongfuwuyouxiangongsi, 而英文名是 "Daiyu Financial Services Ltd."
        —— 只有品牌部分是音译的, 后缀是意译的, 整体比对必然对不上。
        """
        cjk = "".join(c for c in text if _CJK_RANGE[0] <= c <= _CJK_RANGE[1])
        if not cjk:
            return ""
        return "".join(lazy_pinyin(cjk[:n_chars], style=Style.NORMAL)).lower()

    HAS_PINYIN = True

# 中英文公司类型后缀对照。
# 品牌部分是**音译**(岱屿->Daiyu), 后缀部分是**意译**(资本管理->Capital Management),
# 所以只比拼音会把"岱屿资本管理"和"岱屿金融服务"判成同一家 —— 实测正是这样炸的。
# 这张表是跨语言实体匹配的标准配置(双语术语表), 不是 ground truth,
# 生产系统里由业务方维护, 或改用多语言 embedding 模型。
_SUFFIX_MAP: list[tuple[str, tuple[str, ...]]] = [
    ("资产管理", ("asset", "assets")),
    ("资本管理", ("capital",)),
    ("投资控股", ("investment", "investments")),
    ("国际控股", ("international",)),
    ("环球投资", ("global",)),
    ("金融服务", ("financial", "finance")),
    ("信托",     ("trust",)),
    ("银行",     ("bank",)),
]


def _suffix_tokens(cjk_name: str) -> tuple[str, ...] | None:
    """中文名 -> 期望出现在英文名里的关键词。认不出返回 None(不作判据)。"""
    for zh, en in _SUFFIX_MAP:
        if zh in cjk_name:
            return en
    return None
except ImportError:                     # pragma: no cover
    def _romanize(text: str, n_chars: int = 3) -> str:
        return ""

    HAS_PINYIN = False


def _script(s: str) -> str:
    has_cjk = any(_CJK_RANGE[0] <= c <= _CJK_RANGE[1] for c in s)
    has_ascii = any("a" <= c.lower() <= "z" for c in s)
    if has_cjk and has_ascii:
        return "mixed"
    return "cjk" if has_cjk else "ascii"


_DISCRIMINATOR_RE = re.compile(r"[（(]\s*([0-9０-９]{1,3})\s*[）)]")


def _discriminator(name: str) -> str | None:
    """抽出名称里的**区分标记**, 如「澜图资产管理有限公司（2）」中的 2。

    为什么单独处理: 规范化会把括号当标点洗掉, 于是
    「澜图资管」和「澜图资产管理有限公司（2）」看起来品牌前缀一致, 被判为可合并 ——
    但它们是两家不同主体。区分标记是**区分性信息, 不是噪音**,
    规范化不能把它一起洗掉。

    现实中的同类标记: 「XX有限公司（北京）」「XX（上海）有限公司」「XX银行深圳分行」。
    规则: 两个名称的区分标记必须完全一致(同为无, 或同值), 否则不得合并。
    一方有、一方无 -> 归属不明, 同样不合并。
    """
    m = _DISCRIMINATOR_RE.search(name)
    if not m:
        return None
    return m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def lexical_verdict(a: str, b: str) -> str:
    """词面证据的三值判定: strong / weak / incompatible。

    首版把词面只当作"守卫"(二值), 结果是: 简称明明匹配上了, 却因为向量相似度
    落在灰区而去排队等 LLM 裁决 —— 没有 API key 时整个消歧环节合并数为 0,
    消歧欠账 1.17。

    正确的理解是: **词面既是守卫也是证据。**
      strong        本身就足以合并, 不需要向量或 LLM 背书
                    (连续子串 / 中文缩写子序列, 且品牌前缀与区分标记一致)
      incompatible  一票否决, 向量再像也不合并
                    (区分标记不一致, 或品牌前缀不同)
      weak          说不清 -> 才需要向量高分或 LLM 裁决
    """
    # 区分标记不一致 -> 一票否决, 优先于其他所有判据
    da, db = _discriminator(a), _discriminator(b)
    if da != db:
        return "incompatible"

    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return "incompatible"
    if na == nb or na in nb or nb in na:
        return "strong"                   # 简称 / 全称(连续子串)
    # 中文简称通常是"品牌 + 各语义段首字": 资产管理->资管, 投资控股->投资。
    # 连续子串匹配不了这种缩写, 但**子序列**可以。
    # 加上"品牌前缀一致"的约束, 避免把任意两个短名判成缩写关系。
    same_brand = na[:2] == nb[:2]
    if _script(a) == "cjk" == _script(b) and same_brand:
        short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(short) >= 3 and _is_subsequence(short, long_):
            return "strong"               # 中文缩写: 资产管理 -> 资管
    # ---- 跨文种: 拼音比对 ----
    sa, sb = _script(a), _script(b)
    if {sa, sb} == {"cjk", "ascii"}:
        cjk, ascii_ = (a, b) if sa == "cjk" else (b, a)
        roman = _romanize(cjk)
        first = ascii_.lower().replace(".", " ").split()
        if roman and first:
            head = first[0]
            brand_ok = roman.startswith(head) or head.startswith(_romanize(cjk, 2))
            if brand_ok:
                # 品牌对上还不够: 同品牌下有"资本管理/金融服务/环球投资"等多家主体,
                # 只比品牌会把它们全并成一坨(Union-Find 还会传递性扩散)。
                want = _suffix_tokens(cjk)
                if want is None:
                    return "weak"          # 认不出后缀 -> 不下强判断
                rest = set(first[1:])
                return "strong" if rest & set(want) else "incompatible"
        return "incompatible" if HAS_PINYIN else "weak"

    # 同文种: 品牌前缀不同 -> 否决; 相同 -> 交给向量/LLM
    if sa == "cjk" == sb:
        return "weak" if same_brand else "incompatible"
    ta, tb = a.lower().split(), b.lower().split()
    if ta and tb and ta[0] == tb[0]:
        return "weak"
    return "incompatible"


def _token_jaccard(a: str, b: str) -> float:
    ta = {t for t in a.lower().replace(".", " ").split() if t}
    tb = {t for t in b.lower().replace(".", " ").split() if t}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_subsequence(short: str, long_: str) -> bool:
    it = iter(long_)
    return all(c in it for c in short)
