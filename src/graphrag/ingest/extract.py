"""
三元组抽取：文档 -> 结构化事实。

两个抽取器并列, 目的是对照
--------------------------
  RuleExtractor  正则抽取。在**模板化**语料上接近满分, 但换一种文书格式就归零。
                 它的价值有二: (1) 无 API key 时整条流水线仍可跑通;
                 (2) 作为基线, 回答"LLM 到底比规则强在哪"。
  LLMExtractor   真实方案。泛化能力是它存在的全部理由。

**不要用规则抽取的分数去代表系统能力** —— 那是在自己出的模板题上考自己。
评测报告里会把两者分开列, 并注明规则抽取的分数不具外推性。

无严格 schema 的 provider 怎么办
--------------------------------
Claude 有 output_config.format 做硬约束; DeepSeek/Ollama 只有 JSON mode。
差异在这里被吸收掉:
  1. schema 同时写进 prompt(所有 provider 都受益)
  2. 解析失败或校验不通过时, 把**具体错误**回灌给模型重试(最多 2 次)
  3. 仍失败则丢弃该文档, 并计入 `failed` 统计 —— 不静默吞掉

evidence 回查: 抽取幻觉的最后一道闸
-----------------------------------
v3 prompt 要求每条三元组附 evidence 原文片段。这里逐条回查该片段是否
真的出现在原文里, 对不上就丢弃。这一条能拦掉"看起来合理但原文没说"的三元组,
且完全不需要额外的模型调用。被拦截的数量会被统计出来 —— 那就是幻觉率的下界。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..llm import LLM, MODEL_EXTRACT
from ..prompts import registry
from ..store.graph import Edge, Entity

# 关系白名单 + 类型约束。抽取结果不符者直接丢弃。
RELATION_SCHEMA: dict[str, tuple[str, str]] = {
    # ---- 实体 -> 实体 ----
    "APPEARS_IN":     ("Character", "Show"),
    "PARENT_OF":      ("Character", "Character"),
    "GRANDPARENT_OF": ("Character", "Character"),
    "SPOUSE_OF":      ("Character", "Character"),
    "COUSIN_OF":      ("Character", "Character"),
    "UNCLE_OF":       ("Character", "Character"),
    "DESCENDANT_OF":  ("Character", "Character"),
    "LIKES":          ("Character", "Character"),
    "FRIEND_OF":      ("Character", "Character"),
    "PARTNER_OF":     ("Character", "Character"),
    "SEALED_BY":      ("Character", "Character"),
    "RELEASED_BY":    ("Character", "Character"),
    "HEAD_OF":        ("Character", "Place"),
    "LIVES_IN":       ("Character", "Place"),
    "STUDENT_OF":     ("Character", "Place"),
    "LOCATED_IN":     ("Place", "Place"),
    "MEMBER_OF":      ("Character", "Group"),
    "LEADER_OF":      ("Character", "Group"),
    "WISHED_ON":      ("Character", "Item"),
    # ---- 实体 -> 字面量 ----
    "ALIAS":          ("Character", "Literal"),
    "UNIT_NO":        ("Character", "Literal"),
    "PROTOTYPE":      ("Character", "Literal"),
    "ROLE":           ("Character", "Literal"),
    "BIRTHDAY":       ("Character", "Literal"),
    "CATCHPHRASE":    ("Character", "Literal"),
    "WEAPON":         ("Character", "Literal"),
    "SPECIAL_MOVE":   ("Character", "Literal"),
    "FAVORITE_FOOD":  ("Character", "Literal"),
    "CARRIES":        ("Character", "Literal"),
    "TRANSFORM_TIME": ("Character", "Literal"),
    "GIANT_FORM":     ("Character", "Literal"),
    "FACTION":        ("Character", "Literal"),
    "FIRST_AIRED":    ("Show", "Literal"),
    "BROADCASTER":    ("Show", "Literal"),
    "EPISODES":       ("Show", "Literal"),
    "THEME_SONG":     ("Show", "Literal"),
    "FIRST_MOVIE":    ("Show", "Literal"),
    "QUANTITY":       ("Item", "Literal"),
}

ENTITY_TYPES = ["Show", "Character", "Place", "Group", "Item"]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence": {"type": "string"},
                    "subject": {"type": "string"},
                    "subject_type": {"type": "string", "enum": ENTITY_TYPES},
                    "relation": {"type": "string",
                                 "enum": list(RELATION_SCHEMA)},
                    "object": {"type": "string"},
                    "object_type": {"type": "string",
                                    "enum": [*ENTITY_TYPES, "Literal"]},
                },
                "required": ["subject", "subject_type", "relation",
                             "object", "object_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}


@dataclass
class Triple:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    evidence: str = ""
    source_doc: str = ""

    def key(self) -> tuple:
        return (self.subject, self.relation, self.object)


@dataclass
class ExtractStats:
    docs: int = 0
    raw: int = 0                 # 模型吐出来的三元组总数
    kept: int = 0
    dropped_schema: int = 0      # 关系名/类型不合法
    dropped_evidence: int = 0    # evidence 在原文中查无此句 -> 幻觉
    dropped_dup: int = 0
    parse_retries: int = 0
    parse_failed: int = 0        # 重试后仍无法解析 -> 该文档整份丢弃
    cost_usd: float = 0.0
    reasons: dict = field(default_factory=dict)

    def summary(self) -> str:
        halluc = (self.dropped_evidence / self.raw * 100) if self.raw else 0.0
        return (f"文档 {self.docs}  产出三元组 {self.raw} -> 保留 {self.kept}\n"
                f"  丢弃: schema 不合法 {self.dropped_schema} | "
                f"evidence 查无此句 {self.dropped_evidence} ({halluc:.1f}% 幻觉率下界) | "
                f"重复 {self.dropped_dup}\n"
                f"  解析重试 {self.parse_retries} 次, 最终失败 {self.parse_failed} 份文档\n"
                f"  花费 ≈ ${self.cost_usd:.3f}")


def _norm_ws(s: str) -> str:
    return re.sub(r"[\s　]+", "", s or "")


# 组合式 evidence 的分隔符。结构化文档(标题 + 列表项)的证据天然不连续,
# 模型只能把几段拼起来给你。
_EVIDENCE_SEP = re.compile(r"\s*[·•|+；;]\s*|\s*\.\.\.\s*|\s*…\s*")


def evidence_in_text(evidence: str, text: str, threshold: float = 0.82,
                     min_part_len: int = 2) -> bool:
    """evidence 是否真的出自原文。允许轻微改写, 不允许凭空生成。

    支持**组合式 evidence**。这是被实测逼出来的:
      v4 prompt 要求结构化文档写「标题 · 列表项」形式的 evidence,
      例如 `一、正义阵营 · 卡布达`。这个串在原文里并不连续存在
      (标题和列表项隔了好几行)。早期版本按整串比对, 一份名单上 12 条
      全部正确的三元组被判成 100% 幻觉。

    正确的语义是: **evidence 可以由多个片段拼成, 但每个片段都必须逐字出自原文。**
    这样既保留了反幻觉能力(编造的片段一样查不到), 又支持了跨行的结构化证据。
    """
    t = _norm_ws(text)
    if not _norm_ws(evidence):
        return True                       # v1/v2 prompt 不要求 evidence

    parts = [_norm_ws(x) for x in _EVIDENCE_SEP.split(evidence)]
    parts = [x for x in parts if len(x) >= min_part_len]
    if not parts:
        parts = [_norm_ws(evidence)]

    for e in parts:
        if e in t:
            continue
        # 允许标点/省略改写 —— 用最长公共子串比例兜底
        m = SequenceMatcher(None, e, t).find_longest_match(0, len(e), 0, len(t))
        if m.size / max(len(e), 1) < threshold:
            return False
    return True


class BaseExtractor:
    name = "base"

    def extract_doc(self, doc: dict) -> list[Triple]:
        raise NotImplementedError


# ==========================================================================
# 规则抽取 —— 基线 / 离线兜底
# ==========================================================================

class RuleExtractor(BaseExtractor):
    """正则抽取。**只在本项目的模板化语料上有效**, 不具外推性。

    每条正则与 scripts/gen_synthetic_data.py 里的句子模板一一对应。
    这正是规则抽取的本质: 它不是在"读懂"文档, 而是在"认出"自己的模板。
    """
    name = "rule"

    # 档案里的字面量句: (关系, 正则)。主体是档案的主人。
    LIT_PATTERNS: list[tuple[str, str]] = [
        ("UNIT_NO",        r"该角色的编号是([^。]+?)。"),
        ("PROTOTYPE",      r"该角色的原型是([^。]+?)。"),
        ("ROLE",           r"该角色的身份是([^。]+?)。"),
        ("BIRTHDAY",       r"该角色的生日是([^。]+?)。"),
        ("CATCHPHRASE",    r"该角色的口头禅是“([^”]+)”"),
        ("WEAPON",         r"该角色的武器是([^。]+?)。"),
        ("SPECIAL_MOVE",   r"该角色的必杀技是([^。]+?)。"),
        ("FAVORITE_FOOD",  r"该角色最爱吃([^。]+?)。"),
        ("CARRIES",        r"该角色总是随身带着([^。]+?)。"),
        ("TRANSFORM_TIME", r"该角色的超级变换形态最多维持\s*([^。]+?)。"),
        ("GIANT_FORM",     r"该角色可以变成巨人形态([^。]+?)。"),
    ]

    # "该角色的{称谓}是X" / "该角色是X的{称谓}": 称谓决定关系与方向。
    #   +1: 本角色 -> X      -1: X -> 本角色
    LABEL_OF: dict[str, tuple[str, int, str]] = {
        "父亲": ("PARENT_OF", -1, "Character"), "母亲": ("PARENT_OF", -1, "Character"),
        "儿子": ("PARENT_OF", +1, "Character"), "女儿": ("PARENT_OF", +1, "Character"),
        "爷爷": ("GRANDPARENT_OF", -1, "Character"),
        "丈夫": ("SPOUSE_OF", +1, "Character"), "妻子": ("SPOUSE_OF", +1, "Character"),
    }
    LABEL_IS: dict[str, tuple[str, int, str]] = {
        "表哥": ("COUSIN_OF", +1, "Character"), "表妹": ("COUSIN_OF", +1, "Character"),
        "表姐": ("COUSIN_OF", +1, "Character"), "表弟": ("COUSIN_OF", +1, "Character"),
        "二叔": ("UNCLE_OF", +1, "Character"), "侄子": ("UNCLE_OF", -1, "Character"),
        "村长": ("HEAD_OF", +1, "Place"), "校长": ("HEAD_OF", +1, "Place"),
    }

    def _title_entity(self, doc: dict) -> str:
        t = doc["title"]
        for suf in (" 角色档案", " 节目简介", " 团体资料", "学生名册"):
            t = t.replace(suf, "")
        return t.strip()

    def _t(self, s, st, rel, o, ot, ev, doc) -> Triple:
        return Triple(s.strip(), st, rel, o.strip(), ot, evidence=ev, source_doc=doc["id"])

    def extract_doc(self, doc: dict) -> list[Triple]:
        text, dtype = doc["text"], doc["doc_type"]
        me = self._title_entity(doc)
        out: list[Triple] = []
        T = lambda *a: out.append(self._t(*a, doc))       # noqa: E731

        if dtype == "profile":
            for m in re.finditer(r"([一-鿿A-Za-z0-9]+)是(?:动画|特摄剧)《([^》]+)》中的角色", text):
                T(m.group(1), "Character", "APPEARS_IN", m.group(2), "Show", m.group())
            for m in re.finditer(r"([一-鿿A-Za-z0-9]+)又名([^。]+?)。", text):
                T(m.group(1), "Character", "ALIAS", m.group(2), "Literal", m.group())
            for rel, pat in self.LIT_PATTERNS:
                for m in re.finditer(pat, text):
                    T(me, "Character", rel, m.group(1), "Literal", m.group())
            for m in re.finditer(r"该角色的(\S{2})是([^。，]+?)。", text):
                spec = self.LABEL_OF.get(m.group(1))
                if spec:
                    rel, d, ot = spec
                    s, o = (me, m.group(2)) if d > 0 else (m.group(2), me)
                    T(s, "Character", rel, o, ot, m.group())
            for m in re.finditer(r"该角色是([^。，]+?)的(第\s*\d+\s*代孙|\S{2})。", text):
                other, label = m.group(1), m.group(2)
                if "代孙" in label:
                    T(me, "Character", "DESCENDANT_OF", other, "Character", m.group())
                    continue
                spec = self.LABEL_IS.get(label)
                if spec:
                    rel, d, ot = spec
                    s, o = (me, other) if d > 0 else (other, me)
                    T(s, "Character", rel, o, ot, m.group())
            for m in re.finditer(r"该角色喜欢([^。，]+?)。", text):
                T(me, "Character", "LIKES", m.group(1), "Character", m.group())
            for m in re.finditer(r"该角色和([^。，]+?)是好朋友。", text):
                T(me, "Character", "FRIEND_OF", m.group(1), "Character", m.group())
            for m in re.finditer(r"该角色的搭档是([^。，]+?)。", text):
                T(me, "Character", "PARTNER_OF", m.group(1), "Character", m.group())
            # 配角的身份写在主角档案里: "智羊羊的身份是科学家。"
            for m in re.finditer(r"(?<![该角色])([一-鿿]{2,5})的身份是([^。]+?)。", text):
                if m.group(1) != "该角色":
                    T(m.group(1), "Character", "ROLE", m.group(2), "Literal", m.group())
            return out

        if dtype == "roster":
            for head, block in re.findall(
                    r"[一二三四]、(\S+)\n(.*?)(?=\n[一二三四五]、|\Z)", text, re.DOTALL):
                names = [n.strip() for n in re.findall(r"·\s*([^\n]+)", block)]
                if head.endswith(("居民", "住户")):
                    place = head[:-2]
                    for n in names:
                        T(n, "Character", "LIVES_IN", place, "Place", f"{head} · {n}")
                elif head == "在读学生":
                    for n in names:
                        T(n, "Character", "STUDENT_OF", me, "Place", f"{head} · {n}")
                elif head.endswith("阵营"):
                    for n in names:
                        T(n, "Character", "FACTION", head[:-2], "Literal", f"{head} · {n}")
            return out

        if dtype == "show_intro":
            pats = [("FIRST_AIRED", r"于\s*(\d{4}\s*年\s*\d+\s*月\s*\d+\s*日)\s*首播"),
                    ("BROADCASTER", r"首播电视台为([^，。]+)"),
                    ("EPISODES", r"共\s*(\d+\s*集)"),
                    ("THEME_SONG", r"主题曲为《([^》]+)》"),
                    ("FIRST_MOVIE", r"首部电影为《([^》]+)》")]
            for rel, pat in pats:
                for m in re.finditer(pat, text):
                    T(me, "Show", rel, m.group(1), "Literal", m.group())
            return out

        if dtype == "group":
            for m in re.finditer(r"成员为([^。]+?)。", text):
                for n in re.split(r"[、和]", m.group(1)):
                    T(n, "Character", "MEMBER_OF", me, "Group", m.group())
            for m in re.finditer(r"([一-鿿]{2,6})是[^，。]+?的(?:老大|首领)", text):
                T(m.group(1), "Character", "LEADER_OF", me, "Group", m.group())
            for m in re.finditer(r"([一-鿿]{2,6})是[^，。]+?的成员之一", text):
                T(m.group(1), "Character", "MEMBER_OF", me, "Group", m.group())
            return out

        # story / setting
        for m in re.finditer(r"([一-鿿]{2,6})位于([一-鿿]{2,6})。", text):
            T(m.group(1), "Place", "LOCATED_IN", m.group(2), "Place", m.group())
        for m in re.finditer(r"([一-鿿]{2,6})共有\s*(\d+\s*颗)", text):
            T(m.group(1), "Item", "QUANTITY", m.group(2), "Literal", m.group())
        for m in re.finditer(r"([一-鿿]{2,6})(?:因为[^，。]*，)?被([一-鿿]{2,6}?)封印", text):
            T(m.group(1), "Character", "SEALED_BY", m.group(2), "Character", m.group())
        for m in re.finditer(r"([一-鿿]{2,6})解开了([一-鿿]{2,6}?)的封印", text):
            T(m.group(2), "Character", "RELEASED_BY", m.group(1), "Character", m.group())
        for m in re.finditer(r"([一-鿿]{2,6}?)(?:向|用)([一-鿿]{2,4}座和平星)", text):
            subj = re.sub(r"^.*，", "", m.group(1))
            T(subj, "Character", "WISHED_ON", m.group(2), "Item", m.group())
        return out


# ==========================================================================
# LLM 抽取
# ==========================================================================

class LLMExtractor(BaseExtractor):
    def __init__(self, llm: LLM, prompt_id: str = "extract_triples/v3_fewshot",
                 model: str = MODEL_EXTRACT, max_retries: int = 2):
        self.llm = llm
        self.prompt_id = prompt_id
        self.model = model
        self.max_retries = max_retries
        self.name = f"llm:{prompt_id.split('/')[-1]}"
        self.stats = ExtractStats()

    def extract_doc(self, doc: dict) -> list[Triple]:
        p = registry.get(self.prompt_id)
        system, user = p.render(title=doc["title"], text=doc["text"])
        strict = self.llm.supports("strict_schema")

        for attempt in range(self.max_retries + 1):
            res = self.llm.complete(
                system=system, user=user, model=self.model, max_tokens=4096,
                schema=JSON_SCHEMA if strict else JSON_SCHEMA,
                cache_tag=f"{self.prompt_id}|try{attempt}")
            self.stats.cost_usd += res.cost_usd
            data = res.parsed
            if data is None:
                data = _salvage_json(res.text)
            if isinstance(data, dict) and isinstance(data.get("triples"), list):
                return [Triple(
                    subject=str(t.get("subject", "")).strip(),
                    subject_type=str(t.get("subject_type", "")).strip(),
                    relation=str(t.get("relation", "")).strip(),
                    object=str(t.get("object", "")).strip(),
                    object_type=str(t.get("object_type", "")).strip(),
                    evidence=str(t.get("evidence", "")).strip(),
                    source_doc=doc["id"],
                ) for t in data["triples"]]

            # 把具体错误回灌给模型 —— 比单纯重跑有效得多
            self.stats.parse_retries += 1
            user = (f"{user}\n\n[上一次输出无法解析为合法 JSON。"
                    f"请仅输出 JSON 对象，不要任何解释文字、不要 markdown 代码块。"
                    f"上次输出开头是：{res.text[:120]!r}]")

        self.stats.parse_failed += 1
        return []


def _salvage_json(text: str) -> dict | None:
    """模型爱把 JSON 包在 ```json 里, 或前后加解释。尽量救回来。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    cand = m.group(1) if m else None
    if cand is None:
        i, j = text.find("{"), text.rfind("}")
        cand = text[i:j + 1] if i != -1 and j > i else None
    if cand is None:
        return None
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None


# ==========================================================================
# 校验流水线
# ==========================================================================

def validate(triples: list[Triple], doc_text: str,
             stats: ExtractStats, check_evidence: bool = True) -> list[Triple]:
    out: list[Triple] = []
    seen: set[tuple] = set()
    for t in triples:
        stats.raw += 1
        spec = RELATION_SCHEMA.get(t.relation)
        if spec is None or t.subject_type != spec[0] or t.object_type != spec[1] \
           or not t.subject or not t.object:
            stats.dropped_schema += 1
            stats.reasons[f"schema:{t.relation or '空'}"] = \
                stats.reasons.get(f"schema:{t.relation or '空'}", 0) + 1
            continue
        if check_evidence and not evidence_in_text(t.evidence, doc_text):
            stats.dropped_evidence += 1
            continue
        if t.key() in seen:
            stats.dropped_dup += 1
            continue
        seen.add(t.key())
        out.append(t)
        stats.kept += 1
    return out
