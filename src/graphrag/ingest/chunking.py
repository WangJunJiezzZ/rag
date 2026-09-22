"""
切块策略。三种实现并列, 目的是做**对照实验**而不是选一个。

为什么切块值得单独做实验
------------------------
本项目的招募说明书里, "本基金全称为XX基金"在第一节,
"最低认购金额为10万新元"在第四节。朴素定长切块会把两者切开,
于是第四节那个 chunk 长这样:

    "合资格投资者参与本基金的最低认购金额为 10 万新元..."

它不含任何基金名 —— 对"星海亚洲机会基金最低认购额是多少"这个提问,
无论 dense 还是 BM25 都无法把它与正确的基金关联起来。
这就是**孤儿 chunk**, 是真实 RAG 系统里最常见也最隐蔽的失分点。

三种策略
--------
  fixed       定长 + overlap。最朴素, 作为基线。
  section     按章节标题切。尊重文档结构, 但孤儿问题依然存在。
  contextual  section + 给每个 chunk 补上"文档标题 / 章节标题"前缀。
              廉价、确定性、零 LLM 调用, 直接消灭孤儿 chunk。

第四种(LLM 生成上下文摘要)留到 Phase 3 —— 先证明廉价方案能拿多少分,
再决定值不值得为剩下的分数付 LLM 的钱。这是成本意识, 面试要讲。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    id: str                 # doc-0001#c02
    doc_id: str
    doc_title: str
    section: str            # 所属章节标题, 无则为空
    text: str               # **送去检索的文本**(contextual 策略下含补充前缀)
    raw_text: str           # 原文片段, 用于展示与引用定位
    offset: int             # raw_text 在原文中的起始字符下标
    strategy: str

    def to_dict(self) -> dict:
        return asdict(self)


# 章节标题: "第一节　基金概况" / "一、变动概述" / "三、备案说明"
_SECTION_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百]+[节章条]|[一二三四五六七八九十]+、)[^\n]*$",
    re.MULTILINE)


def _split_long(text: str, offset: int, max_chars: int,
                overlap: int) -> list[tuple[str, int]]:
    """按句号等自然边界切长段, 切不动才硬切。返回 (片段, 全局偏移)。"""
    if len(text) <= max_chars:
        return [(text, offset)]
    out: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # 在窗口后 40% 内找最靠后的句子边界, 避免把一句话劈两半
            window = text[start + int(max_chars * 0.6):end]
            m = max((window.rfind(p) for p in "。！？\n"), default=-1)
            if m != -1:
                end = start + int(max_chars * 0.6) + m + 1
        piece = text[start:end].strip()
        if piece:
            out.append((piece, offset + start))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_fixed(doc: dict, max_chars: int = 350, overlap: int = 60) -> list[Chunk]:
    """策略一: 定长 + overlap。完全忽略文档结构。"""
    text = doc["text"]
    return [
        Chunk(id=f"{doc['id']}#c{i:02d}", doc_id=doc["id"], doc_title=doc["title"],
              section="", text=piece, raw_text=piece, offset=off, strategy="fixed")
        for i, (piece, off) in enumerate(_split_long(text, 0, max_chars, overlap))
    ]


def _sections(text: str) -> list[tuple[str, str, int]]:
    """切出 (章节标题, 正文, 全局偏移)。无标题的文档回退为单段。"""
    marks = list(_SECTION_RE.finditer(text))
    if not marks:
        return [("", text, 0)]
    out: list[tuple[str, str, int]] = []
    head = text[: marks[0].start()].strip()
    if head:
        out.append(("", head, 0))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group().strip(), text[m.start():end].strip(), m.start()))
    return out


def chunk_section(doc: dict, max_chars: int = 400, overlap: int = 60) -> list[Chunk]:
    """策略二: 按章节切, 过长的章节再按句子边界细切。"""
    chunks: list[Chunk] = []
    n = 0
    for title, body, off in _sections(doc["text"]):
        for piece, poff in _split_long(body, off, max_chars, overlap):
            chunks.append(Chunk(
                id=f"{doc['id']}#c{n:02d}", doc_id=doc["id"], doc_title=doc["title"],
                section=title, text=piece, raw_text=piece, offset=poff,
                strategy="section"))
            n += 1
    return chunks


def chunk_contextual(doc: dict, max_chars: int = 400, overlap: int = 60) -> list[Chunk]:
    """策略三: 在 section 基础上, 给每个 chunk 补上文档标题与章节标题前缀。

    关键点: 前缀只进 `text`(检索用), 不进 `raw_text`(展示与引用定位用)。
    否则引用会指到一段原文里并不存在的文字上 —— 溯源就失真了。
    """
    chunks: list[Chunk] = []
    for c in chunk_section(doc, max_chars, overlap):
        prefix = doc["title"]
        if c.section and c.section not in c.raw_text[:len(c.section) + 2]:
            prefix = f"{prefix}｜{c.section}"
        c.text = f"【{prefix}】\n{c.raw_text}"
        c.strategy = "contextual"
        chunks.append(c)
    return chunks


STRATEGIES = {
    "fixed": chunk_fixed,
    "section": chunk_section,
    "contextual": chunk_contextual,
}


def chunk_documents(docs: list[dict], strategy: str = "contextual",
                    **kw) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知切块策略 {strategy}; 可选 {list(STRATEGIES)}")
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for d in docs:
        out.extend(fn(d, **kw))
    return out
