"""
Ingest 流水线：文档 -> 三元组 -> 消歧 -> 知识图谱。

    documents.jsonl
         │
         ├─ extract   每份文档独立抽取(可并行/可批量/结果进缓存)
         │
         ├─ validate  schema 校验 + evidence 回查 -> 幻觉三元组在此被拦掉
         │
         ├─ resolve   实体消歧, 把散落的名称归并成规范实体
         │
         └─ build     写成 KnowledgeGraph, 每条边带 source_doc -> 可溯源
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..store.graph import Edge, Entity, KnowledgeGraph
from .extract import BaseExtractor, ExtractStats, Triple, validate
from .resolve import EntityResolver, ResolveStats


@dataclass
class IngestResult:
    graph: KnowledgeGraph
    extract_stats: ExtractStats
    resolve_stats: ResolveStats
    seconds: float


def build_graph(docs: list[dict], extractor: BaseExtractor,
                resolver: EntityResolver | None = None,
                check_evidence: bool = True,
                progress: bool = True,
                workers: int = 8) -> IngestResult:
    """workers: 抽取阶段的并发度。

    每份文档的抽取彼此独立, 是天然可并行的。8 并发时耗时约为串行的 1/7。并发只放在**网络等待**这一段:
    校验、消歧、建图仍是单线程的确定性流程 —— 否则结果会随线程调度而变,
    评测就不可复现了。
    """
    t0 = time.perf_counter()
    stats = getattr(extractor, "stats", None) or ExtractStats()
    all_triples: list[Triple] = []
    lock = threading.Lock()
    done = [0]

    def work(doc: dict) -> tuple[dict, list[Triple]]:
        raw = extractor.extract_doc(doc)
        with lock:
            done[0] += 1
            if progress and (done[0] % 10 == 0 or done[0] == len(docs)):
                el = time.perf_counter() - t0
                rate = done[0] / el if el else 0
                eta = (len(docs) - done[0]) / rate if rate else 0
                print(f"\r  抽取 {done[0]}/{len(docs)}  "
                      f"{rate:.1f} 篇/秒  剩余约 {eta:.0f}s   ",
                      end="", flush=True)
        return doc, raw

    if workers <= 1:
        results = [work(d) for d in docs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, d) for d in docs]
            results = [f.result() for f in futures]   # 保序: 按提交顺序取结果
    if progress:
        print()

    # 校验与后续流程单线程执行, 保证结果与并发度无关 —— 评测必须可复现
    for doc, raw in results:
        stats.docs += 1
        all_triples.extend(validate(raw, doc["text"], stats,
                                    check_evidence=check_evidence))

    # ---- 收集实体提及 ----
    mentions: dict[str, str] = {}
    contexts: dict[str, str] = {}
    for t in all_triples:
        mentions[t.subject] = t.subject_type
        if t.object_type != "Literal":
            mentions[t.object] = t.object_type
        contexts.setdefault(t.subject, t.evidence[:120])
        contexts.setdefault(t.object, t.evidence[:120])

    # 文档声明的「又名」: 字面量客体其实是主体的另一个名字
    declared = [(t.subject, t.object) for t in all_triples if t.relation == "ALIAS"]
    # 有关系边直接相连的两个名称, 必不是同一实体 —— 消歧的 cannot-link
    related = {tuple(sorted((t.subject, t.object))) for t in all_triples
               if t.object_type != "Literal"}

    # 共享邻居签名
    neighbors: dict[str, set[str]] = {}
    for tr in all_triples:
        if tr.object_type != "Literal":
            neighbors.setdefault(tr.subject, set()).add(tr.object)
            neighbors.setdefault(tr.object, set()).add(tr.subject)

    if resolver is not None:
        mapping = resolver.resolve(mentions, contexts, neighbors,
                                   aliases=declared, related=related)
        rstats = resolver.stats
    else:
        mapping = {m: m for m in mentions}
        rstats = ResolveStats(mentions=len(mentions), canonical=len(mentions))

    # ---- 建图 ----
    kg = KnowledgeGraph()
    kg.meta = {"source": "extracted", "extractor": extractor.name}
    alias_of: dict[str, set[str]] = {}
    for m, canon in mapping.items():
        alias_of.setdefault(canon, set()).add(m)
    # 声明过的别名即使没在别处出现, 也要挂到实体上 —— 问题里可能只用别名提问
    if resolver is not None:
        for subj, alias in declared:
            if subj in mapping:
                alias_of[mapping[subj]].add(alias)

    id_of: dict[str, str] = {}
    for canon, aliases in alias_of.items():
        eid = f"E{len(id_of):04d}"
        id_of[canon] = eid
        kg.add_entity(Entity(id=eid, type=mentions.get(canon, "Character"),
                             name=canon,
                             aliases=sorted(a for a in aliases if a != canon)))

    for t in all_triples:
        src = id_of[mapping[t.subject]]
        dst = (f"lit:{t.object}" if t.object_type == "Literal"
               else id_of[mapping[t.object]])
        kg.add_edge(Edge(src=src, rel=t.relation, dst=dst,
                         source_doc=t.source_doc))

    return IngestResult(kg, stats, rstats, time.perf_counter() - t0)
