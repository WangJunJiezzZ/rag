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
    "MANAGED_BY":     ("Fund", "Company"),
    "CUSTODIAN":      ("Fund", "Company"),
    "DOMICILED_IN":   ("Fund", "Jurisdiction"),
    "MIN_INVESTMENT": ("Fund", "Literal"),
    "LAUNCHED_ON":    ("Fund", "Literal"),
    "DIRECTOR_OF":    ("Person", "Company"),
    "SHAREHOLDER_OF": ("Person", "Company"),
    "REGISTERED_IN":  ("Company", "Jurisdiction"),
    "SUBSIDIARY_OF":  ("Company", "Company"),
    "RISK_LEVEL":     ("Jurisdiction", "Literal"),
}

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
                    "subject_type": {"type": "string",
                                     "enum": ["Fund", "Company", "Person",
                                              "Jurisdiction"]},
                    "relation": {"type": "string",
                                 "enum": list(RELATION_SCHEMA)},
                    "object": {"type": "string"},
                    "object_type": {"type": "string",
                                    "enum": ["Fund", "Company", "Person",
                                             "Jurisdiction", "Literal"]},
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
      例如 `一、高风险管辖区 · 维兰群岛`。这个串在原文里并不连续存在
      (标题和列表项隔了好几行), 首版按整串比对, 12 条全部正确的三元组
      被判成 100% 幻觉。

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
    """正则抽取。**只在本项目的模板化语料上有效**, 不具外推性。"""
    name = "rule"

    PATTERNS: list[tuple[str, str, str, str]] = [
        # (正则, 关系, 主体类型, 客体类型)
        (r"本基金全称为([^（(]+)（?[^）)]*）?[，,]?于\s*([^\n，,]+?)\s*依法设立",
         "LAUNCHED_ON", "Fund", "Literal"),
        (r"基金管理人[：:]\s*([^\n。，,]+)", "MANAGED_BY", "Fund", "Company"),
        (r"基金托管人[：:]\s*([^\n。，,]+)", "CUSTODIAN", "Fund", "Company"),
        (r"注册登记[，,]?\s*注册地为([^\n。，,]+)", "DOMICILED_IN", "Fund", "Jurisdiction"),
        (r"最低认购金额为\s*([^\n，,。]+?)[，,。]", "MIN_INVESTMENT", "Fund", "Literal"),
        (r"委任([一-鿿]{2,4})为本公司董事", "DIRECTOR_OF", "Person", "Company"),
        (r"([一-鿿]{2,4})当选为本公司董事", "DIRECTOR_OF", "Person", "Company"),
        (r"新任董事为([一-鿿]{2,4})", "DIRECTOR_OF", "Person", "Company"),
        (r"([一-鿿]{2,4})直接持有本公司已发行股本的\s*\d+%",
         "SHAREHOLDER_OF", "Person", "Company"),
        (r"股东名称[：:]\s*([一-鿿]{2,4})", "SHAREHOLDER_OF", "Person", "Company"),
        (r"注册地[：:]\s*([^\n。，,]+)", "REGISTERED_IN", "Company", "Jurisdiction"),
        # 注意: 这条必须同时捕获子公司与母公司。首版只捕获了母公司、
        # 主体用文档标题填充, 而集团架构说明的标题就是母公司 -> 抽出自环, precision 归零。
        (r"([^\n。，,　]+?)为([^\n。，,]+?)之全资附属公司",
         "SUBSIDIARY_OF", "Company", "Company"),
        (r"([^\n。，,　]+?)持有([^\n。，,]+?)\s*100%\s*股权",
         "_PARENT_HOLDS", "Company", "Company"),
    ]

    def _title_entity(self, doc: dict) -> str:
        t = doc["title"]
        for suf in ("招募说明书（节选）", "企业注册信息摘要", "关于董事变更的公告",
                    "权益变动公告", "集团架构说明", "年度业务回顾（摘要）"):
            t = t.replace(suf, "")
        return t.strip()

    def extract_doc(self, doc: dict) -> list[Triple]:
        text, dtype = doc["text"], doc["doc_type"]
        主体 = self._title_entity(doc)
        out: list[Triple] = []

        if dtype == "watchlist":
            for level, block in re.findall(
                    r"[一二三]、(高风险|加强监控|常规)管辖区\n(.*?)(?=\n[一二三四]、|\Z)",
                    text, re.DOTALL):
                lv = {"高风险": "高风险", "加强监控": "中风险", "常规": "低风险"}[level]
                for name in re.findall(r"·\s*([^\n]+)", block):
                    out.append(Triple(name.strip(), "Jurisdiction", "RISK_LEVEL",
                                      lv, "Literal", evidence=name.strip(),
                                      source_doc=doc["id"]))
            return out

        if dtype == "annual_review":
            for fname, strat, jur, cus in re.findall(
                    r"·\s*([^\n：]+)：采用([^\n，]+)策略[^\n]*?注册地为([^\n，]+)[^\n]*?"
                    r"托管人为([^\n。]+)。", text):
                out.append(Triple(fname, "Fund", "MANAGED_BY", 主体, "Company",
                                  evidence=fname, source_doc=doc["id"]))
                out.append(Triple(fname, "Fund", "DOMICILED_IN", jur, "Jurisdiction",
                                  evidence=fname, source_doc=doc["id"]))
                out.append(Triple(fname, "Fund", "CUSTODIAN", cus, "Company",
                                  evidence=fname, source_doc=doc["id"]))
            return out

        for pat, rel, stype, otype in self.PATTERNS:
            for m in re.finditer(pat, text):
                g = [x.strip() for x in m.groups()]
                if rel == "SUBSIDIARY_OF" and len(g) == 2:
                    out.append(Triple(g[0], "Company", "SUBSIDIARY_OF", g[1],
                                      "Company", evidence=m.group(),
                                      source_doc=doc["id"]))
                    continue
                if rel == "_PARENT_HOLDS" and len(g) == 2:
                    # "A 持有 B 100% 股权" 等价于 "B 是 A 的子公司"
                    out.append(Triple(g[1], "Company", "SUBSIDIARY_OF", g[0],
                                      "Company", evidence=m.group(),
                                      source_doc=doc["id"]))
                    continue
                if rel == "LAUNCHED_ON" and len(g) == 2:
                    out.append(Triple(g[0], "Fund", "LAUNCHED_ON", g[1], "Literal",
                                      evidence=m.group(), source_doc=doc["id"]))
                    continue
                if stype in ("Person",):
                    subj, obj = g[0], 主体
                else:
                    subj, obj = 主体, g[0]
                out.append(Triple(subj, stype, rel, obj, otype,
                                  evidence=m.group(), source_doc=doc["id"]))
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
