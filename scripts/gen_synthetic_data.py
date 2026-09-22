"""
Phase 0 - 合成语料生成器  (跨平台: macOS / Linux / Windows)

设计要点(面试讲解点):

1. **先建图, 再渲染文档**。真实世界是"文档 -> 抽取 -> 图", 这里反着来。
   于是图就是 ground truth, 评测集可以自动生成且 100% 准确, 无需人工标注。

2. **显式植入多跳模式**。随机生成不保证出现"共同董事穿透到高风险辖区"这类
   结构, 所以显式 plant, 保证评测集对每种检索能力都有覆盖。

3. **刻意制造实体歧义**。同一家公司在不同文档里用全称/简称/罗马化名称。
   不做实体消歧, 图就会断成孤立节点 -> 所有多跳查询失败。

4. **刻意制造语义干扰段 (distractor)**。每份招募说明书都含有措辞高度雷同的
   "风险因素""费用结构""赎回安排"章节。问"XX基金最低认购额", 向量检索会召回
   几十份*别的*基金的认购章节 —— 这正是纯向量 RAG 的典型失败模式,
   也是 BM25 混合 + rerank 能把分数拉起来的原因。这是 demo 的核心对照实验。

5. **刻意制造指代与跨句依赖**。"本基金""本公司"大量出现, 逼迫切块策略
   必须补上下文, 否则 chunk 变成孤儿。

所有机构名、人名、辖区名均为虚构; 辖区风险等级为虚构设定,
不指涉任何真实司法管辖区、企业或个人。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, write_text, write_jsonl  # noqa: E402

setup_console()

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic"

# ==========================================================================
# 名称池 (全部虚构)
# ==========================================================================

BRANDS: list[tuple[str, str]] = [
    ("星海", "Starsea"), ("澜图", "Lantu"), ("砺石", "Lishi"), ("青柏", "Qingbai"),
    ("屿川", "Yuchuan"), ("鹤鸣", "Heming"), ("沅曦", "Yuanxi"), ("松岚", "Songlan"),
    ("燧明", "Suiming"), ("洄川", "Huichuan"), ("栖梧", "Qiwu"), ("砚池", "Yanchi"),
    ("麓岩", "Luyan"), ("汐禾", "Xihe"), ("昀舟", "Yunzhou"), ("岱屿", "Daiyu"),
    ("漱玉", "Shuyu"), ("樾川", "Yuechuan"), ("柏观", "Baiguan"), ("徽岚", "Huilan"),
    ("凛川", "Linchuan"), ("瑾樾", "Jinyue"), ("翊澜", "Yilan"), ("岫云", "Xiuyun"),
    ("珩屿", "Hengyu"), ("霁川", "Jichuan"), ("茗谷", "Minggu"), ("砥行", "Dixing"),
]

CORP_SUFFIX = [
    ("资产管理有限公司", "Asset Management Pte. Ltd.", "资管"),
    ("投资控股有限公司", "Investment Holdings Ltd.", "投资"),
    ("国际控股有限公司", "International Holdings Ltd.", "国际"),
    ("资本管理有限公司", "Capital Management Ltd.", "资本"),
    ("环球投资有限公司", "Global Investments Ltd.", "环球"),
    ("金融服务有限公司", "Financial Services Ltd.", "金服"),
]

BANK_BRANDS = [("环宇", "Huanyu"), ("北岸", "Beian"), ("汇川", "Huichuan"),
               ("昭明", "Zhaoming"), ("瑞辰", "Ruichen")]

STRATEGIES = ["亚洲机会", "全球宏观", "私募信贷", "新兴市场债券", "科技成长",
              "基础设施收益", "房地产收益", "多空策略", "量化对冲", "可持续发展",
              "并购套利", "结构化信贷"]

SURNAMES = list("李王张陈林黄吴刘郑谢许蔡洪曾邱周徐孙马朱")
GIVEN = ["明远", "志诚", "嘉铭", "之涵", "舒兰", "沐阳", "允之", "思齐", "景行",
         "若谷", "承业", "牧之", "秉文", "清和", "书尧", "南星", "砚书", "行知",
         "怀瑾", "慕白", "知微", "听澜", "见山", "仲康", "敬亭", "叔夜", "衡之"]

JURISDICTIONS = [
    ("新港城", "低风险", False), ("白鹭湾", "低风险", False),
    ("临海特区", "低风险", False), ("北屿自治区", "低风险", False),
    ("澄江特别行政区", "低风险", False),
    ("卡兰群岛", "中风险", True), ("圣维尔", "中风险", True),
    ("图兰海峡区", "中风险", True), ("米德兰", "中风险", False),
    ("维兰群岛", "高风险", True), ("卡瑟尼亚", "高风险", True),
    ("圣德罗自由区", "高风险", True),
]

# ==========================================================================
# 干扰段落池 (distractor)
# 关键: 这些段落在几十份文档间措辞几乎相同, 向量空间里彼此距离极近。
# 它们不含任何可抽取的事实, 却会大量挤占 top-k 名额。
# ==========================================================================

RISK_FACTORS = [
    ("市场风险", "本基金投资的金融工具价格可能因宏观经济环境、利率水平、"
                 "汇率变动、市场情绪及突发事件而出现大幅波动。在极端市场条件下，"
                 "本基金资产净值可能在短期内显著下跌，投资者可能损失全部本金。"),
    ("流动性风险", "本基金部分投资标的可能缺乏活跃的二级市场。在需要变现时，"
                   "本基金可能无法以合理价格及时处置相关资产，"
                   "从而影响赎回安排的正常执行。"),
    ("信用风险", "本基金持有的债务工具发行人可能出现付息或偿本违约。"
                 "发行人信用评级下调亦可能导致相关资产估值下降。"),
    ("汇率风险", "本基金部分资产以非基准货币计价。汇率的不利变动"
                 "可能侵蚀以基准货币计量的投资回报。"),
    ("集中度风险", "本基金可能在特定行业、地区或单一发行人上形成较高的持仓集中度，"
                   "相关标的的不利变动将对本基金产生放大的影响。"),
    ("关联交易风险", "本基金管理人及其关联方可能同时管理其他投资组合，"
                     "从而产生潜在利益冲突。管理人已建立相应的隔离与披露机制。"),
    ("法律与监管风险", "本基金所涉司法管辖区的法律、税务及监管要求可能发生变化，"
                       "该等变化可能对本基金的运作、成本及回报产生不利影响。"),
    ("估值风险", "对于缺乏公开报价的资产，本基金依赖估值模型及第三方估值服务。"
                 "估值结果可能与实际变现价格存在差异。"),
]

REDEMPTION_CLAUSES = [
    "本基金设有 {lock} 个月锁定期。锁定期届满后，投资者可于每个季度末提交赎回申请，"
    "申请须提前 {notice} 个工作日送达管理人。",
    "投资者可于每月最后一个工作日申请赎回，赎回款项一般于确认后 {settle} 个工作日内支付。"
    "本基金保留在市场异常情况下暂停赎回的权利。",
    "赎回安排采用季度开放机制。单一开放日的赎回总额超过基金资产净值 {gate}% 时，"
    "管理人有权按比例延后处理超出部分。",
]

FEE_CLAUSES = [
    "管理人按基金资产净值的 {mgmt}% 年费率收取管理费，按月计提、按季支付。"
    "托管人按 {cust}% 年费率收取托管费。",
    "本基金的业绩报酬按超过 {hurdle}% 年化门槛收益率部分的 {carry}% 计提，"
    "采用高水位线法，每年结算一次。",
    "除管理费与托管费外，本基金还将承担审计费、法律顾问费、行政管理费及"
    "其他与基金运作直接相关的合理费用。前述费用合计一般不超过基金资产净值的 {other}%。",
]

SCOPE_CLAUSES = [
    "本基金的投资范围包括但不限于：于受认可交易所上市的股票及存托凭证、"
    "政府及公司债务工具、货币市场工具、以及为对冲目的持有的衍生金融工具。",
    "本基金可将不超过 {cap}% 的资产净值投资于非上市股权或流动性受限的资产。"
    "该等投资须经投资委员会事前审批。",
    "本基金一般不进行实物商品交易，亦不以投机为目的持有杠杆敞口。"
    "为管理组合久期与汇率敞口，本基金可使用利率及外汇衍生品。",
]

GOVERNANCE_CLAUSES = [
    "本基金设投资委员会，由不少于三名成员组成，负责审议重大投资决策"
    "及关联交易事项。委员会每月至少召开一次会议。",
    "管理人已建立独立的合规与风险管理职能，直接向董事会报告，"
    "并定期对投资限制的遵守情况进行监控。",
    "本基金的账目由独立审计机构按年度审计，审计报告将于财政年度结束后"
    "四个月内向投资者提供。",
]

BIO_CLAUSES = [
    "在加入本公司前，{name}曾于多家区域性金融机构担任投资及风险管理职务，"
    "在跨境资产配置方面积累了超过 {yrs} 年的从业经验。",
    "{name}持有金融相关专业的硕士学位，并具备所在司法管辖区认可的"
    "从业资格，过往主要覆盖机构客户业务。",
    "{name}此前长期从事结构化产品与另类投资的设计与分销工作，"
    "累计管理规模逾 {aum} 亿单位货币。",
]

DISCLAIMER = [
    "本文件所载信息仅供参考，不构成任何投资建议、要约或要约邀请。",
    "投资涉及风险，过往表现并不代表未来业绩，投资者可能损失全部本金。",
    "本公告的中英文版本如有歧义，以中文版本为准。",
    "如对本文件内容有任何疑问，请咨询您的独立专业顾问。",
]

SYNTHETIC_NOTE = "【本文件为系统演示用合成数据，所述机构、人员及管辖区均属虚构。】"


# ==========================================================================
# 数据模型
# ==========================================================================

@dataclass
class Entity:
    id: str
    type: str                      # Fund | Company | Person | Jurisdiction
    name: str                      # 规范名 canonical
    aliases: list[str] = field(default_factory=list)
    props: dict = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    rel: str
    dst: str                       # entity id 或字面量 "lit:..."
    source_doc: str                # 支撑这条边的文档 -> 溯源的根基
    confidence: float = 1.0


@dataclass
class Document:
    id: str
    doc_type: str
    title: str
    date: str
    text: str


class World:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.entities: dict[str, Entity] = {}
        self.edges: list[Edge] = []
        self.docs: list[Document] = []
        self.planted: dict[str, list] = {}
        self._doc_seq = 0

    def add(self, e: Entity) -> Entity:
        self.entities[e.id] = e
        return e

    def link(self, src: str, rel: str, dst: str, doc: str) -> None:
        self.edges.append(Edge(src=src, rel=rel, dst=dst, source_doc=doc))

    def next_doc_id(self) -> str:
        self._doc_seq += 1
        return f"doc-{self._doc_seq:04d}"

    def emit(self, doc_type: str, title: str, date: str, text: str) -> str:
        did = self.next_doc_id()
        self.docs.append(Document(id=did, doc_type=doc_type, title=title,
                                  date=date, text=text))
        return did


# ==========================================================================
# 渲染工具
# ==========================================================================

def pick_alias(w: World, e: Entity) -> str:
    """随机返回全称/简称/罗马化名 —— 实体消歧的燃料。"""
    r = w.rng.random()
    if r < 0.58 or not e.aliases:
        return e.name
    if r < 0.85:
        return e.aliases[0]
    return e.aliases[-1]


def fill(w: World, tmpl: str) -> str:
    rng = w.rng
    return tmpl.format(
        lock=rng.choice([6, 12, 18, 24]), notice=rng.choice([15, 30, 45]),
        settle=rng.choice([5, 7, 10]), gate=rng.choice([10, 15, 20]),
        mgmt=rng.choice(["1.00", "1.25", "1.50", "1.75", "2.00"]),
        cust=rng.choice(["0.05", "0.08", "0.10", "0.12"]),
        hurdle=rng.choice([4, 5, 6, 8]), carry=rng.choice([10, 15, 20]),
        other=rng.choice(["0.30", "0.45", "0.60"]), cap=rng.choice([10, 15, 20, 30]),
        yrs=rng.randint(8, 25), aum=rng.randint(10, 300), name="",
    )


def risk_section(w: World, n: int = 4) -> str:
    picked = w.rng.sample(RISK_FACTORS, n)
    body = "".join(f"（{i+1}）{title}\n　　{text}\n" for i, (title, text) in enumerate(picked))
    return "风险因素\n" + body


# ==========================================================================
# 文档渲染
# ==========================================================================

def render_prospectus(w: World, fund: Entity) -> str:
    """招募说明书: 长文档, 事实分散在不同章节, 大量'本基金'指代, 含干扰章节。"""
    rng = w.rng
    mgr = w.entities[fund.props["manager"]]
    cus = w.entities[fund.props["custodian"]]
    dom = w.entities[fund.props["domicile"]]
    mgr_name = pick_alias(w, mgr)

    parts = [
        f"{fund.name}\n招募说明书（节选）\n",
        SYNTHETIC_NOTE + "\n",
        "第一节　基金概况\n"
        f"　　本基金全称为{fund.name}（以下简称“本基金”），"
        f"于 {fund.props['launch_date']} 依法设立并完成注册登记，注册地为{dom.name}。\n"
        f"　　本基金为面向合资格投资者发售的开放式集合投资计划，基准货币为"
        f"{fund.props['base_ccy']}，存续期限为无固定期限。\n",
        "第二节　投资目标与策略\n"
        f"　　本基金采用{fund.props['strategy']}策略，"
        f"力求在严格控制下行风险的前提下实现资本的中长期稳健增值。\n"
        f"　　{fill(w, rng.choice(SCOPE_CLAUSES))}\n"
        f"　　{fill(w, rng.choice(SCOPE_CLAUSES))}\n",
        "第三节　参与机构\n"
        f"　　基金管理人：{mgr_name}。管理人负责本基金的投资决策、"
        f"日常运作及信息披露。\n"
        f"　　基金托管人：{cus.name}。托管人负责本基金财产的保管、"
        f"清算交收及对管理人投资运作的监督。\n"
        f"　　{fill(w, rng.choice(GOVERNANCE_CLAUSES))}\n",
        "第四节　认购与赎回安排\n"
        f"　　合资格投资者参与本基金的最低认购金额为 {fund.props['min_investment']}，"
        f"低于该金额的申购指令将不予受理。后续追加认购的最低金额为"
        f"{fund.props['min_topup']}。\n"
        f"　　{fill(w, rng.choice(REDEMPTION_CLAUSES))}\n",
        "第五节　费用与税项\n"
        f"　　{fill(w, rng.choice(FEE_CLAUSES))}\n"
        f"　　{fill(w, rng.choice(FEE_CLAUSES))}\n",
        "第六节　" + risk_section(w, rng.randint(3, 5)),
        "第七节　重要提示\n　　" + "\n　　".join(rng.sample(DISCLAIMER, 3)),
    ]
    return "\n".join(parts)


def render_annual_review(w: World, mgr: Entity, funds: list[Entity]) -> str:
    """管理人年度回顾: 一份文档里提到多只基金 —— 制造跨实体的 chunk 归属难题。"""
    rng = w.rng
    mgr_name = pick_alias(w, mgr)
    lines = [
        f"{mgr_name}\n{rng.randint(2023, 2025)} 年度业务回顾（摘要）\n",
        SYNTHETIC_NOTE + "\n",
        "一、业务概览\n"
        f"　　本公司于报告期内持续深化多策略资产管理业务。"
        f"截至报告期末，本公司管理的集合投资计划共 {len(funds)} 只。\n",
        "二、旗下产品\n",
    ]
    for f in funds:
        cus = w.entities[f.props["custodian"]]
        lines.append(
            f"　　· {f.name}：采用{f.props['strategy']}策略，"
            f"注册地为{w.entities[f.props['domicile']].name}，"
            f"托管人为{cus.name}。\n"
        )
    lines.append(
        "三、风险管理\n"
        f"　　{fill(w, rng.choice(GOVERNANCE_CLAUSES))}\n"
        f"　　{fill(w, rng.choice(GOVERNANCE_CLAUSES))}\n"
    )
    lines.append("四、免责声明\n　　" + "\n　　".join(rng.sample(DISCLAIMER, 2)))
    return "\n".join(lines)


def render_director_notice(w: World, person: Entity, company: Entity, date: str) -> str:
    rng = w.rng
    cname = pick_alias(w, company)
    head = rng.choice([
        f"{cname}\n关于董事变更的公告\n",
        f"{cname}\n董事会人事公告\n",
        f"{cname}\n董事会组成变动通知\n",
    ])
    body = rng.choice([
        f"　　本公司董事会谨此宣布，自 {date} 起，委任{person.name}为本公司董事，"
        f"任期三年，自生效之日起算。",
        f"　　经 {date} 召开的股东大会审议通过，{person.name}当选为本公司董事，"
        f"即日生效。相关变更已按规定向注册机关备案。",
        f"　　本公司于 {date} 完成董事会成员增补，新任董事为{person.name}。"
        f"董事会现由五名成员组成。",
    ])
    bio = fill(w, rng.choice(BIO_CLAUSES)).replace("{name}", person.name)
    bio = rng.choice(BIO_CLAUSES).format(name=person.name,
                                         yrs=rng.randint(8, 25), aum=rng.randint(10, 300))
    return "\n".join([
        head, SYNTHETIC_NOTE + "\n", body + "\n",
        "董事简历\n　　" + bio + "\n",
        "其他事项\n"
        f"　　{fill(w, rng.choice(GOVERNANCE_CLAUSES))}\n",
        "　　" + rng.choice(DISCLAIMER),
    ])


def render_shareholding_notice(w: World, person: Entity, company: Entity,
                               pct: int, date: str) -> str:
    rng = w.rng
    cname = pick_alias(w, company)
    return "\n".join([
        f"{cname}\n权益变动公告\n", SYNTHETIC_NOTE + "\n",
        "一、变动概述\n"
        f"　　截至 {date}，{person.name}直接持有本公司已发行股本的 {pct}%，"
        f"为本公司主要股东之一。本次权益变动不会导致本公司控制权发生变化。\n",
        "二、股东信息\n"
        f"　　股东名称：{person.name}\n"
        f"　　持股比例：{pct}%\n"
        f"　　登记日期：{date}\n",
        "三、其他说明\n"
        f"　　{fill(w, rng.choice(GOVERNANCE_CLAUSES))}\n",
        "　　" + rng.choice(DISCLAIMER),
    ])


def render_registration(w: World, company: Entity) -> str:
    rng = w.rng
    dom = w.entities[company.props["jurisdiction"]]
    cname = pick_alias(w, company)
    return "\n".join([
        f"{cname}\n企业注册信息摘要\n", SYNTHETIC_NOTE + "\n",
        "一、主体信息\n"
        f"　　实体名称：{cname}\n"
        f"　　注册编号：{company.props['reg_no']}\n"
        f"　　注册地：{dom.name}\n"
        f"　　注册日期：{company.props['reg_date']}\n"
        f"　　主体状态：存续\n",
        "二、经营范围\n"
        f"　　本公司系一家根据{dom.name}法律注册成立的有限责任公司，"
        f"业务范围涵盖投资管理、投资顾问及相关配套服务。\n",
        "三、备案说明\n"
        f"　　本摘要依据登记机关公开信息整理，"
        f"如与登记机关记载不一致，以登记机关记载为准。\n",
        "　　" + rng.choice(DISCLAIMER),
    ])


def render_group_structure(w: World, parent: Entity, child: Entity) -> str:
    rng = w.rng
    pn, cn = pick_alias(w, parent), pick_alias(w, child)
    return "\n".join([
        f"{pn}\n集团架构说明\n", SYNTHETIC_NOTE + "\n",
        "一、持股关系\n"
        f"　　{cn}为{pn}之全资附属公司，{pn}持有其 100% 股权。\n"
        f"　　本集团的相关投资业务主要通过上述附属公司开展。\n",
        "二、合并范围\n"
        f"　　该附属公司已纳入本集团合并财务报表范围。\n",
        "三、其他说明\n"
        f"　　{fill(w, rng.choice(GOVERNANCE_CLAUSES))}\n",
        "　　" + rng.choice(DISCLAIMER),
    ])


def render_watchlist(w: World, jurs: list[Entity]) -> str:
    high = [j.name for j in jurs if j.props["risk"] == "高风险"]
    mid = [j.name for j in jurs if j.props["risk"] == "中风险"]
    low = [j.name for j in jurs if j.props["risk"] == "低风险"]
    return "\n".join([
        "跨境反洗钱监管观察名单（年度更新）\n", SYNTHETIC_NOTE + "\n",
        "一、高风险管辖区\n"
        "　　下列管辖区在反洗钱与反恐融资框架方面存在重大缺陷。"
        "涉及下列管辖区的客户、交易对手或关联方，应适用强化尽职调查（EDD），"
        "并由合规部门二次复核：\n"
        + "".join(f"　　　· {n}\n" for n in high),
        "二、加强监控管辖区\n"
        "　　下列管辖区已就其反洗钱框架作出改进承诺，适用中等风险等级，"
        "需执行标准尽职调查并保留审计轨迹：\n"
        + "".join(f"　　　· {n}\n" for n in mid),
        "三、常规管辖区\n"
        "　　下列管辖区按低风险处理：\n"
        + "".join(f"　　　· {n}\n" for n in low),
        "四、适用说明\n"
        "　　本名单按年度复核更新。名单变更自发布之日起生效，"
        "对存量业务关系应于六个月内完成重新评估。\n",
        "　　本名单为虚构数据，仅用于系统演示，不指涉任何真实司法管辖区。",
    ])


# ==========================================================================
# 世界构建
# ==========================================================================

def build_world(seed: int, n_companies: int, n_funds: int, n_persons: int) -> World:
    w = World(seed)
    rng = w.rng

    def rand_date(y0: int = 2021, y1: int = 2025) -> str:
        return f"{rng.randint(y0, y1)} 年 {rng.randint(1, 12)} 月 {rng.randint(1, 28)} 日"

    # ---- 辖区 + 观察名单 (辖区风险的唯一来源文档) ----
    jur_entities = []
    for i, (name, risk, offshore) in enumerate(JURISDICTIONS):
        jur_entities.append(w.add(Entity(
            id=f"J{i:02d}", type="Jurisdiction", name=name,
            props={"risk": risk, "offshore": offshore})))
    high_jurs = [e for e in jur_entities if e.props["risk"] == "高风险"]
    low_jurs = [e for e in jur_entities if e.props["risk"] == "低风险"]

    wl = w.emit("watchlist", "跨境反洗钱监管观察名单（年度更新）", "2025-01-15",
                render_watchlist(w, jur_entities))
    for e in jur_entities:
        w.link(e.id, "RISK_LEVEL", f"lit:{e.props['risk']}", wl)

    # ---- 公司 ----
    brand_pool = BRANDS.copy()
    rng.shuffle(brand_pool)
    companies: list[Entity] = []
    for i in range(n_companies):
        zh, roman = brand_pool[i % len(brand_pool)]
        suffix_zh, suffix_en, short_suffix = rng.choice(CORP_SUFFIX)
        dup = i // len(brand_pool)
        tag = f"（{dup+1}）" if dup else ""
        e = w.add(Entity(
            id=f"C{i:03d}", type="Company",
            name=f"{zh}{suffix_zh}{tag}",
            aliases=[f"{zh}{short_suffix}{tag}", f"{roman} {suffix_en}"],
            props={"jurisdiction": (rng.choice(low_jurs) if rng.random() < 0.65
                                    else rng.choice(jur_entities)).id,
                   "reg_no": f"{rng.randint(1990, 2024)}{rng.randint(100000, 999999)}",
                   "reg_date": rand_date(1998, 2023)}))
        companies.append(e)
        d = w.emit("registration", f"{e.name} 企业注册信息摘要",
                   e.props["reg_date"], render_registration(w, e))
        w.link(e.id, "REGISTERED_IN", e.props["jurisdiction"], d)

    # ---- 托管银行 ----
    banks: list[Entity] = []
    for i, (zh, roman) in enumerate(BANK_BRANDS):
        jur = rng.choice(low_jurs)
        e = w.add(Entity(id=f"B{i:02d}", type="Company",
                         name=f"{zh}银行{jur.name}分行",
                         aliases=[f"{zh}银行", f"{roman} Bank"],
                         props={"jurisdiction": jur.id, "is_bank": True,
                                "reg_no": f"BK{rng.randint(100000, 999999)}",
                                "reg_date": rand_date(1990, 2010)}))
        banks.append(e)
        d = w.emit("registration", f"{e.name} 企业注册信息摘要",
                   e.props["reg_date"], render_registration(w, e))
        w.link(e.id, "REGISTERED_IN", jur.id, d)

    # ---- 自然人 ----
    persons: list[Entity] = []
    seen: set[str] = set()
    for i in range(n_persons):
        while True:
            nm = rng.choice(SURNAMES) + rng.choice(GIVEN)
            if nm not in seen:
                seen.add(nm)
                break
        persons.append(w.add(Entity(id=f"P{i:03d}", type="Person", name=nm)))

    # ---- 基金 ----
    managers = [c for c in companies if "资产管理" in c.name or "资本管理" in c.name]
    if len(managers) < 8:
        managers = companies[:12]
    funds: list[Entity] = []
    used_names: set[str] = set()
    for i in range(n_funds):
        mgr = rng.choice(managers)
        brand = mgr.name[:2]
        for _ in range(40):
            fname = f"{brand}{rng.choice(STRATEGIES)}基金"
            if fname not in used_names:
                break
        else:
            fname = f"{brand}{rng.choice(STRATEGIES)}基金{i}"
        used_names.add(fname)
        ccy = rng.choice(["新元", "美元"])
        f = w.add(Entity(
            id=f"F{i:03d}", type="Fund", name=fname,
            aliases=[fname.replace("基金", ""), brand + "基金"],
            props={"manager": mgr.id, "custodian": rng.choice(banks).id,
                   "domicile": (rng.choice(low_jurs) if rng.random() < 0.7
                                else rng.choice(jur_entities)).id,
                   "min_investment": rng.choice(
                       [f"10 万{ccy}", f"25 万{ccy}", f"50 万{ccy}", f"100 万{ccy}"]),
                   "min_topup": rng.choice([f"1 万{ccy}", f"5 万{ccy}", f"10 万{ccy}"]),
                   "base_ccy": ccy,
                   "launch_date": rand_date(2019, 2025),
                   "strategy": fname[2:].replace("基金", "")}))
        funds.append(f)
        d = w.emit("prospectus", f"{fname} 招募说明书（节选）",
                   f.props["launch_date"], render_prospectus(w, f))
        w.link(f.id, "MANAGED_BY", mgr.id, d)
        w.link(f.id, "CUSTODIAN", f.props["custodian"], d)
        w.link(f.id, "DOMICILED_IN", f.props["domicile"], d)
        w.link(f.id, "MIN_INVESTMENT", f"lit:{f.props['min_investment']}", d)
        w.link(f.id, "LAUNCHED_ON", f"lit:{f.props['launch_date']}", d)

    # ---- 管理人年度回顾 (一文多基金) ----
    by_mgr: dict[str, list[Entity]] = {}
    for f in funds:
        by_mgr.setdefault(f.props["manager"], []).append(f)
    for mgr_id, fs in by_mgr.items():
        if len(fs) < 2:
            continue
        mgr = w.entities[mgr_id]
        d = w.emit("annual_review", f"{mgr.name} 年度业务回顾（摘要）", "2025-03-31",
                   render_annual_review(w, mgr, fs))
        for f in fs:
            w.link(f.id, "MANAGED_BY", mgr_id, d)   # 同一事实的第二个来源

    # ---- 董事 / 股东 / 母子公司 (底噪) ----
    def emit_director(p: Entity, c: Entity) -> None:
        date = rand_date(2022, 2025)
        d = w.emit("director_notice", f"{c.name} 关于董事变更的公告", date,
                   render_director_notice(w, p, c, date))
        w.link(p.id, "DIRECTOR_OF", c.id, d)

    for p in persons:
        for c in rng.sample(companies, rng.randint(1, 2)):
            emit_director(p, c)

    for c in rng.sample(companies, len(companies) // 2):
        p = rng.choice(persons)
        pct = rng.choice([5, 10, 15, 20, 25, 30, 51, 75])
        date = rand_date(2022, 2025)
        d = w.emit("shareholding", f"{c.name} 权益变动公告", date,
                   render_shareholding_notice(w, p, c, pct, date))
        w.link(p.id, "SHAREHOLDER_OF", c.id, d)

    for _ in range(max(5, n_companies // 8)):
        parent, child = rng.sample(companies, 2)
        d = w.emit("group_structure", f"{parent.name} 集团架构说明", "2024-06-30",
                   render_group_structure(w, parent, child))
        w.link(child.id, "SUBSIDIARY_OF", parent.id, d)

    # ======================================================================
    # 显式植入多跳模式 —— 保证评测集对每种检索能力都有覆盖
    # ======================================================================
    planted: dict[str, list] = {"high_risk_chain": [], "shared_director": [],
                                "subsidiary_chain": []}

    # 保证有足够多注册在高风险辖区的公司
    def jur_of(c: Entity) -> Entity:
        return w.entities[c.props["jurisdiction"]]

    offshore_pool = [c for c in companies if jur_of(c).props["risk"] == "高风险"]
    need = 8 - len(offshore_pool)
    if need > 0:
        for c in rng.sample([c for c in companies if c not in offshore_pool], need):
            j = rng.choice(high_jurs)
            c.props["jurisdiction"] = j.id
            w.edges = [e for e in w.edges
                       if not (e.src == c.id and e.rel == "REGISTERED_IN")]
            d = w.emit("registration", f"{c.name} 企业注册信息摘要",
                       c.props["reg_date"], render_registration(w, c))
            w.link(c.id, "REGISTERED_IN", j.id, d)
            offshore_pool.append(c)

    # 模式 A (招牌案例): 基金 -> 管理人 -> 共同董事 -> 高风险辖区公司 -> 高风险
    for fund in rng.sample(funds, min(10, len(funds))):
        mgr = w.entities[fund.props["manager"]]
        risky = rng.choice([c for c in offshore_pool if c.id != mgr.id])
        bridge = rng.choice(persons)
        emit_director(bridge, mgr)
        emit_director(bridge, risky)
        planted["high_risk_chain"].append({
            "fund": fund.id, "manager": mgr.id, "bridge_person": bridge.id,
            "risky_company": risky.id, "jurisdiction": risky.props["jurisdiction"],
        })

    # 模式 B: 两家公司共享董事 (纯关系, 不涉风险)
    for _ in range(8):
        c1, c2 = rng.sample(companies, 2)
        p = rng.choice(persons)
        emit_director(p, c1)
        emit_director(p, c2)
        planted["shared_director"].append({"person": p.id, "companies": [c1.id, c2.id]})

    # 模式 C: 基金 -> 管理人 -> 母公司 -> 注册辖区 (三跳, 不经过自然人)
    for fund in rng.sample(funds, min(6, len(funds))):
        mgr = w.entities[fund.props["manager"]]
        parent = rng.choice([c for c in companies if c.id != mgr.id])
        d = w.emit("group_structure", f"{parent.name} 集团架构说明", "2024-06-30",
                   render_group_structure(w, parent, mgr))
        w.link(mgr.id, "SUBSIDIARY_OF", parent.id, d)
        planted["subsidiary_chain"].append({
            "fund": fund.id, "manager": mgr.id, "parent": parent.id,
            "parent_jurisdiction": parent.props["jurisdiction"]})

    w.planted = planted
    return w


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成语料与 ground-truth 图谱")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--companies", type=int, default=80)
    ap.add_argument("--funds", type=int, default=40)
    ap.add_argument("--persons", type=int, default=60)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    w = build_world(args.seed, args.companies, args.funds, args.persons)

    write_text(args.out / "graph.json", json.dumps({
        "seed": args.seed,
        "entities": [asdict(e) for e in w.entities.values()],
        "edges": [asdict(e) for e in w.edges],
        "planted": w.planted,
    }, ensure_ascii=False, indent=2))
    write_jsonl(args.out / "documents.jsonl", (asdict(d) for d in w.docs))

    n_char = sum(len(d.text) for d in w.docs)
    by_type: dict[str, int] = {}
    for d in w.docs:
        by_type[d.doc_type] = by_type.get(d.doc_type, 0) + 1

    print(f"[ok] 实体 {len(w.entities)}  边 {len(w.edges)}  文档 {len(w.docs)}  "
          f"总字数 {n_char:,}  平均每文档 {n_char // max(1, len(w.docs)):,} 字")
    print("     文档类型:", "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"     植入: 高风险关联链 {len(w.planted['high_risk_chain'])}  "
          f"共同董事 {len(w.planted['shared_director'])}  "
          f"母子公司链 {len(w.planted['subsidiary_chain'])}")
    print(f"     -> {args.out / 'graph.json'}")
    print(f"     -> {args.out / 'documents.jsonl'}")


if __name__ == "__main__":
    main()
