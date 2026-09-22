"""
Phase 0 - 评测集生成器  (跨平台)

核心思路: 语料是我们从图渲染出来的, 所以每个问题的
  · 标准答案 (gold_answer / gold_entities)
  · 证据文档 (gold_docs)  <- 这是关键, 有了它才能算 recall@k
都能直接从图里算出来, 不需要一条条人工标注。

评测集按"能力"分类, 每一类对应一个具体的检索技术:

  fact_direct      1跳  用文档原词提问          -> 基线, 谁都该答对
  fact_paraphrase  1跳  同义改写但保留实体名    -> 弱语义测试(BM25 靠实体名即可过)
  semantic_only    1跳  **完全不含实体名**的描述性提问 -> 真正的 dense 语义测试
  fact_disambig    1跳  同品牌多只基金易混淆    -> 考验 BM25 + rerank 的词面精度
  hop2             2跳  基金 -> 管理人 -> 辖区  -> 纯向量开始失效
  hop3_parent      3跳  基金 -> 管理人 -> 母公司 -> 辖区
  hop4_risk        4跳  招牌案例: 共同董事穿透到高风险辖区 -> 纯向量必然失败
  shared_director  2跳  哪些公司与 X 有共同董事
  aggregation      聚合 计数类, 检索范式根本答不了
  negative         陷阱 语料中不存在, 正确行为是拒答 -> 考验幻觉抑制

train / dev / test 三分:
  prompt 只允许在 train 上调, dev 用于早停, test 只在最后跑一次。
  否则 prompt 会过拟合评测集 —— 这是 prompt 工程里最常见的自欺。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, write_jsonl, write_text  # noqa: E402
from graphrag.store.graph import KnowledgeGraph, Step               # noqa: E402

setup_console()

ROOT = Path(__file__).resolve().parents[1]

SPLIT_WEIGHTS = [("train", 0.40), ("dev", 0.20), ("test", 0.40)]


def assign_split(question: str) -> str:
    """按**问题文本**的哈希确定性分桶。

    刻意不按自增 id 分桶: 两条不同 id 但文本相同的题会被分到不同划分,
    造成 train/test 泄漏。以文本为键可从根本上杜绝这类泄漏,
    且重新生成评测集时划分保持稳定(便于跨版本对比分数)。
    """
    h = int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, wgt in SPLIT_WEIGHTS:
        acc += wgt
        if h < acc:
            return name
    return SPLIT_WEIGHTS[-1][0]


class EvalBuilder:
    def __init__(self, kg: KnowledgeGraph, seed: int = 7):
        self.kg = kg
        self.rng = random.Random(seed)
        self.items: list[dict] = []
        self._seen: set[str] = set()
        self._seq = 0
        self.skipped_dupes = 0

    def add(self, *, qtype: str, question: str, gold_answer: str,
            gold_entities: list[str], gold_docs: list[str], hops: int,
            requires: str, expect_refusal: bool = False,
            gold_path: str = "", note: str = "") -> None:
        if question in self._seen:
            self.skipped_dupes += 1      # 重复题不带来新信息, 且会污染划分
            return
        self._seen.add(question)
        self._seq += 1
        qid = f"q-{self._seq:04d}"
        self.items.append({
            "id": qid,
            "type": qtype,
            "question": question,
            "gold_answer": gold_answer,
            "gold_entities": gold_entities,
            "gold_docs": sorted(set(gold_docs)),
            "hops": hops,
            "requires": requires,          # vector | hybrid | graph | refusal
            "expect_refusal": expect_refusal,
            "gold_path": gold_path,
            "note": note,
            "split": assign_split(question),
        })

    # ------------------------------------------------------------------
    # 辅助: 取某条边的来源文档
    # ------------------------------------------------------------------
    def docs_of(self, src: str, rel: str, dst: str | None = None) -> list[str]:
        out: list[str] = []
        for e in self.kg.out(src, rel):
            if dst is None or e.dst == dst:
                out.append(e.source_doc)
                out.extend(e.extra_docs)
        return [d for d in out if d]

    # ==================================================================
    # 1. 单跳事实 —— 三种难度
    # ==================================================================
    def build_facts(self, n_each: int = 12) -> None:
        kg, rng = self.kg, self.rng
        funds = kg.by_type("Fund")

        # (a) fact_direct: 用文档原词
        for f in rng.sample(funds, min(n_each, len(funds))):
            minv = kg.one_out(f.id, "MIN_INVESTMENT")
            self.add(
                qtype="fact_direct",
                question=f"{f.name}的最低认购金额是多少？",
                gold_answer=kg.name(minv or ""),
                gold_entities=[kg.name(minv or "")],
                gold_docs=self.docs_of(f.id, "MIN_INVESTMENT"),
                hops=1, requires="vector",
                note="提问用词与文档一致, 是最容易的一类")

        # (b) fact_paraphrase: 刻意避开文档原词, 考验语义检索
        paraphrases = [
            ("{f}最少要投多少钱才能参与？", "MIN_INVESTMENT",
             "避开'认购''金额'等文档原词"),
            ("谁在替{f}保管资产？", "CUSTODIAN", "避开'托管'一词"),
            ("{f}归属哪个司法管辖区？", "DOMICILED_IN", "避开'注册地'一词"),
            ("{f}背后是哪家机构在操盘？", "MANAGED_BY", "避开'管理人'一词"),
        ]
        for tmpl, rel, note in paraphrases:
            for f in rng.sample(funds, min(max(3, n_each // 3), len(funds))):
                tgt = kg.one_out(f.id, rel)
                if not tgt:
                    continue
                self.add(
                    qtype="fact_paraphrase",
                    question=tmpl.format(f=f.name),
                    gold_answer=kg.name(tgt),
                    gold_entities=[kg.name(tgt)],
                    gold_docs=self.docs_of(f.id, rel),
                    hops=1, requires="vector", note=note)

        # (c) fact_disambig: 同品牌多只基金, 招募说明书措辞几乎一致
        by_brand: dict[str, list] = {}
        for f in funds:
            by_brand.setdefault(f.name[:2], []).append(f)
        confusable = [fs for fs in by_brand.values() if len(fs) >= 2]
        rng.shuffle(confusable)
        for group in confusable[:n_each]:
            f = rng.choice(group)
            rel = rng.choice(["CUSTODIAN", "MIN_INVESTMENT", "DOMICILED_IN"])
            tgt = kg.one_out(f.id, rel)
            if not tgt:
                continue
            label = {"CUSTODIAN": "托管银行是哪一家",
                     "MIN_INVESTMENT": "最低认购金额是多少",
                     "DOMICILED_IN": "注册在哪个管辖区"}[rel]
            siblings = "、".join(x.name for x in group if x.id != f.id)
            self.add(
                qtype="fact_disambig",
                question=f"{f.name}的{label}？",
                gold_answer=kg.name(tgt),
                gold_entities=[kg.name(tgt)],
                gold_docs=self.docs_of(f.id, rel),
                hops=1, requires="hybrid",
                note=f"同品牌易混基金: {siblings}; 纯 dense 容易召回错基金的说明书")

    # ==================================================================
    # 1.5 纯语义题 —— 问题中不含任何实体名, 且与文档零词面重叠
    # ==================================================================
    # 为什么需要这一类:
    #   跑完第一版评测才发现, fact_paraphrase 虽然换了说法, 但仍带着基金全名。
    #   BM25 靠"星海/海亚/亚洲"这些 bigram 就能定位文档, 根本没考到语义。
    #   真正的 dense 测试必须满足两条: 问题里没有实体名, 且用词与文档不重叠。
    #
    #   例: 文档写"本基金采用量化对冲策略"
    #       提问"哪些基金用程序化模型捕捉市场中性收益？"
    #       —— "程序化""市场中性"在全部语料里一次都没出现过, BM25 必然 0 分。
    STRATEGY_PARAPHRASE = {
        "亚洲机会":     "聚焦亚太地区增长机遇",
        "全球宏观":     "依据全球经济周期调整头寸",
        "私募信贷":     "向未上市企业直接放贷",
        "新兴市场债券": "买入发展中国家的固定收益工具",
        "科技成长":     "押注创新企业的长期增值",
        "基础设施收益": "靠公用事业与交通类资产获取稳定现金流",
        "房地产收益":   "以不动产租金作为主要回报来源",
        "多空策略":     "同时建立看涨与看跌头寸",
        "量化对冲":     "用程序化模型捕捉市场中性收益",
        "可持续发展":   "把环境与社会责任纳入筛选标准",
        "并购套利":     "从企业收购交易的价差中获利",
        "结构化信贷":   "投资资产证券化产品",
    }

    def build_semantic_only(self) -> None:
        kg = self.kg
        by_strategy: dict[str, list] = {}
        for f in kg.by_type("Fund"):
            by_strategy.setdefault(f.props.get("strategy", ""), []).append(f)

        for strategy, desc in self.STRATEGY_PARAPHRASE.items():
            funds = by_strategy.get(strategy, [])
            if not funds:
                continue
            docs: list[str] = []
            for f in funds:
                docs += self.docs_of(f.id, "MIN_INVESTMENT")   # 招募说明书
            self.add(
                qtype="semantic_only",
                question=f"有哪些基金是{desc}的？",
                gold_answer="、".join(f.name for f in funds),
                gold_entities=[f.name for f in funds],
                gold_docs=docs, hops=1, requires="dense",
                note=f"问题中不含任何实体名, 且'{desc}'在全部语料中零出现; "
                     f"对应文档里写的是'{strategy}'。BM25 在此必然失分, "
                     f"能不能答对完全取决于 dense 向量")

    # ==================================================================
    # 2. 两跳: 基金 -> 管理人 -> 注册辖区
    # ==================================================================
    def build_hop2(self, n: int = 14) -> None:
        kg, rng = self.kg, self.rng
        funds = kg.by_type("Fund")
        for f in rng.sample(funds, min(n, len(funds))):
            mgr = kg.one_out(f.id, "MANAGED_BY")
            if not mgr:
                continue
            jur = kg.one_out(mgr, "REGISTERED_IN")
            if not jur:
                continue
            docs = self.docs_of(f.id, "MANAGED_BY") + self.docs_of(mgr, "REGISTERED_IN")
            self.add(
                qtype="hop2",
                question=f"{f.name}的基金管理人注册在哪个管辖区？",
                gold_answer=kg.name(jur),
                gold_entities=[kg.name(mgr), kg.name(jur)],
                gold_docs=docs, hops=2, requires="graph",
                gold_path=f"{f.name} -[MANAGED_BY]-> {kg.name(mgr)} "
                          f"-[REGISTERED_IN]-> {kg.name(jur)}",
                note="答案需要拼接两份文档: 招募说明书 + 管理人注册摘要")

    # ==================================================================
    # 3. 三跳: 基金 -> 管理人 -> 母公司 -> 辖区
    # ==================================================================
    def build_hop3_parent(self) -> None:
        kg = self.kg
        for p in kg.meta.get("planted", {}).get("subsidiary_chain", []):
            f, mgr, parent, jur = p["fund"], p["manager"], p["parent"], p["parent_jurisdiction"]
            docs = (self.docs_of(f, "MANAGED_BY")
                    + self.docs_of(mgr, "SUBSIDIARY_OF", parent)
                    + self.docs_of(parent, "REGISTERED_IN"))
            self.add(
                qtype="hop3_parent",
                question=f"{kg.name(f)}的管理人隶属于哪家母公司？该母公司注册在哪里？",
                gold_answer=f"母公司为{kg.name(parent)}，注册于{kg.name(jur)}",
                gold_entities=[kg.name(parent), kg.name(jur)],
                gold_docs=docs, hops=3, requires="graph",
                gold_path=f"{kg.name(f)} -[MANAGED_BY]-> {kg.name(mgr)} "
                          f"-[SUBSIDIARY_OF]-> {kg.name(parent)} "
                          f"-[REGISTERED_IN]-> {kg.name(jur)}",
                note="三份文档缺一不可")

    # ==================================================================
    # 4. 四跳招牌案例: 共同董事穿透到高风险辖区
    # ==================================================================
    def build_hop4_risk(self) -> None:
        kg = self.kg
        for p in kg.meta.get("planted", {}).get("high_risk_chain", []):
            f, mgr = p["fund"], p["manager"]
            bridge, risky, jur = p["bridge_person"], p["risky_company"], p["jurisdiction"]
            docs = (self.docs_of(f, "MANAGED_BY")
                    + self.docs_of(bridge, "DIRECTOR_OF", mgr)
                    + self.docs_of(bridge, "DIRECTOR_OF", risky)
                    + self.docs_of(risky, "REGISTERED_IN")
                    + self.docs_of(jur, "RISK_LEVEL"))
            self.add(
                qtype="hop4_risk",
                question=f"{kg.name(f)}是否与高风险管辖区的实体存在关联？如果有，请说明关联路径。",
                gold_answer=(f"存在。{kg.name(f)}的管理人{kg.name(mgr)}与"
                             f"{kg.name(risky)}存在共同董事{kg.name(bridge)}，"
                             f"而{kg.name(risky)}注册于高风险管辖区{kg.name(jur)}。"),
                gold_entities=[kg.name(mgr), kg.name(bridge), kg.name(risky), kg.name(jur)],
                gold_docs=docs, hops=4, requires="graph",
                gold_path=(f"{kg.name(f)} -[MANAGED_BY]-> {kg.name(mgr)} "
                           f"<-[DIRECTOR_OF]- {kg.name(bridge)} "
                           f"-[DIRECTOR_OF]-> {kg.name(risky)} "
                           f"-[REGISTERED_IN]-> {kg.name(jur)} -[RISK_LEVEL]-> 高风险"),
                note="招牌案例: 证据横跨5份文档, 且'高风险'三字与基金名从未共现, "
                     "纯向量检索无论 top-k 取多大都召不回关键文档")

    # ==================================================================
    # 5. 共同董事
    # ==================================================================
    def build_shared_director(self) -> None:
        kg = self.kg
        for p in kg.meta.get("planted", {}).get("shared_director", []):
            person, (c1, c2) = p["person"], p["companies"]
            docs = self.docs_of(person, "DIRECTOR_OF", c1) + \
                   self.docs_of(person, "DIRECTOR_OF", c2)
            self.add(
                qtype="shared_director",
                question=f"{kg.name(c1)}与哪些公司存在共同董事？请列出董事姓名。",
                gold_answer=f"与{kg.name(c2)}存在共同董事{kg.name(person)}。",
                gold_entities=[kg.name(c2), kg.name(person)],
                gold_docs=docs, hops=2, requires="graph",
                gold_path=f"{kg.name(c1)} <-[DIRECTOR_OF]- {kg.name(person)} "
                          f"-[DIRECTOR_OF]-> {kg.name(c2)}",
                note="答案分散在两份不同公司的董事公告里, 两份文档彼此无任何词面重叠")

    # ==================================================================
    # 6. 聚合计数
    # ==================================================================
    def build_aggregation(self, n: int = 10) -> None:
        kg, rng = self.kg, self.rng
        banks = [c for c in kg.by_type("Company") if c.props.get("is_bank")]
        for b in banks:
            funds = [e.src for e in kg.inn(b.id, "CUSTODIAN")]
            if not funds:
                continue
            self.add(
                qtype="aggregation",
                question=f"由{b.name}担任托管人的基金一共有几只？请列出名称。",
                gold_answer=f"共 {len(funds)} 只：" + "、".join(kg.name(x) for x in funds),
                gold_entities=[str(len(funds))] + [kg.name(x) for x in funds],
                gold_docs=[d for x in funds for d in self.docs_of(x, "CUSTODIAN")],
                hops=1, requires="graph",
                note="聚合类问题: top-k 检索范式天然答不全, 必须走图/结构化查询")

        jurs = [j for j in kg.by_type("Jurisdiction")
                if j.props.get("risk") == "高风险"]
        for j in jurs:
            comps = [e.src for e in kg.inn(j.id, "REGISTERED_IN")]
            if not comps:
                continue
            self.add(
                qtype="aggregation",
                question=f"有多少家公司注册在{j.name}？",
                gold_answer=f"共 {len(comps)} 家",
                gold_entities=[str(len(comps))],
                gold_docs=[d for c in comps for d in self.docs_of(c, "REGISTERED_IN")],
                hops=1, requires="graph",
                note="证据分散在几十份注册摘要里, 检索 top-k 不可能覆盖全")

    # ==================================================================
    # 7. 陷阱题: 语料中不存在, 正确行为是拒答
    # ==================================================================
    def build_negative(self, n: int = 12) -> None:
        kg, rng = self.kg, self.rng
        funds = kg.by_type("Fund")
        companies = [c for c in kg.by_type("Company") if not c.props.get("is_bank")]

        templates = [
            ("{f}过去三年的年化收益率是多少？", "语料中从未披露任何业绩数据"),
            ("{f}当前的资产管理规模（AUM）是多少？",
             "语料中只有董事简历提到过'管理规模', 是强干扰项, 但与基金AUM无关"),
            ("{f}在 2026 年的分红方案是什么？", "语料中不含分红信息"),
        ]
        for tmpl, note in templates:
            for f in rng.sample(funds, min(max(2, n // 6), len(funds))):
                self.add(qtype="negative", question=tmpl.format(f=f.name),
                         gold_answer="资料中未提及", gold_entities=[], gold_docs=[],
                         hops=0, requires="refusal", expect_refusal=True, note=note)

        for c in rng.sample(companies, min(3, len(companies))):
            self.add(qtype="negative", question=f"{c.name}目前有多少名员工？",
                     gold_answer="资料中未提及", gold_entities=[], gold_docs=[],
                     hops=0, requires="refusal", expect_refusal=True,
                     note="语料中不含雇员信息")

        # 问一只根本不存在的基金 —— 最强的幻觉陷阱
        real = {f.name for f in funds}
        for _ in range(3):
            for _ in range(50):
                fake = (rng.choice(["星海", "澜图", "青柏", "麓岩", "霁川"])
                        + rng.choice(["黄金机遇", "北极星精选", "长青平衡", "远洋稳健"])
                        + "基金")
                if fake not in real:
                    break
            self.add(qtype="negative", question=f"{fake}的最低认购金额是多少？",
                     gold_answer="资料中未提及该基金", gold_entities=[], gold_docs=[],
                     hops=0, requires="refusal", expect_refusal=True,
                     note="该基金完全不存在; 但品牌前缀真实存在, 向量检索一定会召回"
                          "同品牌的真实基金说明书 -> 极易诱发幻觉")


def main() -> None:
    ap = argparse.ArgumentParser(description="从 ground-truth 图谱生成评测集")
    ap.add_argument("--graph", type=Path, default=ROOT / "data/synthetic/graph.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval/eval_set.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    kg = KnowledgeGraph.load(args.graph)
    b = EvalBuilder(kg, args.seed)
    b.build_facts()
    b.build_semantic_only()
    b.build_hop2()
    b.build_hop3_parent()
    b.build_hop4_risk()
    b.build_shared_director()
    b.build_aggregation()
    b.build_negative()

    write_jsonl(args.out, b.items)

    # 汇总
    by_type: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_req: dict[str, int] = {}
    for it in b.items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
        by_split[it["split"]] = by_split.get(it["split"], 0) + 1
        by_req[it["requires"]] = by_req.get(it["requires"], 0) + 1

    print(f"[ok] 评测集 {len(b.items)} 题 -> {args.out}"
          + (f"  (去重丢弃 {b.skipped_dupes} 题)" if b.skipped_dupes else ""))
    print("\n  按能力分类:")
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
        avg_docs = sum(len(i["gold_docs"]) for i in b.items if i["type"] == k) / v
        print(f"    {k:18s} {v:4d} 题   平均证据文档 {avg_docs:.1f} 份")
    print("\n  按所需检索能力:", "  ".join(f"{k}={v}" for k, v in sorted(by_req.items())))
    print("  按数据集划分:  ", "  ".join(f"{k}={by_split.get(k, 0)}"
                                        for k, _ in SPLIT_WEIGHTS))

    write_text(args.out.parent / "summary.json", json.dumps(
        {"total": len(b.items), "by_type": by_type,
         "by_split": by_split, "by_requires": by_req},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
