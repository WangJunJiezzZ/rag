"""
Phase 0 - 语料生成器  (跨平台: macOS / Linux / Windows)

题材: 《喜羊羊与灰太狼》与《铁甲小宝》两部动画的角色资料。
事实取自维基百科等公开资料(见 docs/README.md 的来源清单), 译名统一用中国大陆播出版。
**只收录至少一个可靠来源明确写出的事实**; 来源互相矛盾的(票房、黑太狼的结局等)一律不收。

设计要点(面试讲解点):

1. **先建图, 再渲染文档**。真实世界是"文档 -> 抽取 -> 图", 这里反着来。
   于是图就是 ground truth, 评测集的证据文档可以直接算出来, 无需人工标注。

2. **每条关系只写在一份文档里**。"小灰灰的父亲是灰太狼"只写在小灰灰的档案里,
   "灰太狼的父亲是黑太狼"只写在灰太狼的档案里。于是"小灰灰的爷爷是谁"
   必须把两份文档拼起来 —— 没有任何单一文档同时提到小灰灰和黑太狼。

3. **刻意制造别名**。高圆寺寅彦在剧情文档里只以「高圆寺博士」出现,
   吉祥寺藏之助在搭档关系里只以「藏之助」出现。不做实体消歧, 图就断开。
   别名只在角色本人的档案里声明一次(「又名……」), 这是消歧唯一可靠的线索。

4. **刻意制造语义干扰段 (distractor)**。每份档案都含措辞几乎相同的
   「登场说明」「资料说明」章节, 不含任何事实, 却会挤占向量检索的 top-k 名额。
   加上角色名本身高度雷同(喜羊羊/美羊羊/懒羊羊, 灰太狼/红太狼/蕉太狼/香太狼),
   向量空间里它们彼此挨得很近 —— 这正是纯向量 RAG 的典型失败模式。

5. **刻意制造指代与跨章节依赖**。档案第二、三节全部用「该角色」指代,
   不出现角色名。切块不补上下文, 这些 chunk 就成了孤儿。

6. **结构型文档**。居民登记、学生名册、阵营一览是「标题 + 列表项」格式,
   「一、正义阵营」下列出的每一项就是一条阵营事实, 但没有一句话完整写出来 ——
   这是考验抽取 prompt 能否读懂排版的专用文档。
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

SOURCE_NOTE = "【本资料根据维基百科等公开资料整理，角色译名采用中国大陆播出版本，角色及形象权利归原著作权方所有。】"
DATE = "2026-09-29"

# ==========================================================================
# 节目
# ==========================================================================

SHOWS = {
    "XYY": {"name": "喜羊羊与灰太狼", "kind": "动画"},
    "TJ":  {"name": "铁甲小宝", "kind": "特摄剧"},
}

# ==========================================================================
# 角色档案
# 每个字段都对应图里的一条边, 来源文档就是这份档案。
#   aliases   又名(ALIAS)          unit    编号(UNIT_NO)
#   proto     原型(PROTOTYPE)      role    身份(ROLE)
#   birthday  生日(BIRTHDAY)       catch   口头禅(CATCHPHRASE)
#   weapon    武器(WEAPON)         move    必杀技(SPECIAL_MOVE)
#   food      最爱吃(FAVORITE_FOOD) carries 随身物品(CARRIES)
#   time      超级变换时限(TRANSFORM_TIME)  giant  巨人形态(GIANT_FORM)
#   traits    性格特点 —— 只进正文, 不进图(自由文本不适合做三元组)
#   rels      人物关系 (关系, 对象, 称谓)。见 REL_SENTENCE
# ==========================================================================

PROFILES: list[dict] = [
    # ---------------- 喜羊羊与灰太狼: 羊 ----------------
    {"name": "喜羊羊", "show": "XYY",
     "carries": "铃铛",
     "traits": ["是羊族里跑得最快的羊，聪明乐观，总能识破灰太狼的诡计",
                "喜欢踢足球和做实验，性子急，常常没听完话就动手，闹出笑话",
                "脖子上的铃铛是父母送的"],
     "rels": [("PARENT_OF<", "智羊羊", "父亲"), ("PARENT_OF<", "丽羊羊", "母亲")]},
    {"name": "美羊羊", "show": "XYY",
     "traits": ["天真善良，爱美也爱哭，精通营养、美容、服装等一切和美有关的事",
                "能用任何东西编织饰物"],
     "rels": [("LIKES", "喜羊羊", "")]},
    {"name": "懒羊羊", "show": "XYY",
     "carries": "枕头",
     "catch": "天下皆醒我独睡，天下皆勤我独懒",
     "traits": ["是草原上最胖、最懒的羊，所以最常被灰太狼抓走",
                "最爱睡觉和吃零食，运气特别好"]},
    {"name": "沸羊羊", "show": "XYY",
     "traits": ["是一只黑脸羊，最健壮也最鲁莽，做事直率，从不考虑后果",
                "最爱健身"],
     "rels": [("LIKES", "美羊羊", "")]},
    {"name": "暖羊羊", "show": "XYY",
     "role": "班长",
     "traits": ["属于盘羊族，是大肥羊学校的转学生",
                "性格温暖，力大无穷",
                "有一张乌鸦嘴，预言坏事很准"]},
    {"name": "慢羊羊", "show": "XYY",
     "aliases": ["慢羊羊村长"],
     "traits": ["是羊族里最年长的羊，行动比蜗牛还慢，拄着拐杖，记性不好",
                "思考时头上会长出一根聪明草",
                "是个乌龙发明家，发明常让自己吃苦头，危急时刻却又能派上用场"],
     "rels": [("HEAD_OF", "羊村", "村长"), ("HEAD_OF", "大肥羊学校", "校长")]},
    {"name": "包包大人", "show": "XYY",
     "role": "青青草原的管理者",
     "traits": ["是一只秃头大象",
                "正义感强，但做事一板一眼，常被灰太狼骗"]},
    # ---------------- 喜羊羊与灰太狼: 狼 ----------------
    {"name": "灰太狼", "show": "XYY",
     "birthday": "羊历 3475 年 9 月 26 日",
     "catch": "我一定会回来的",
     "traits": ["是青青草原上的发明家，曾制造各种道具去抓羊，却从没成功过",
                "每次被羊打败都会喊出自己的口头禅",
                "怕老婆但很爱她，是好丈夫，也是好爸爸",
                "厨艺很好"],
     "rels": [("PARENT_OF<", "黑太狼", "父亲"), ("PARENT_OF<", "银太狼", "母亲"),
              ("DESCENDANT_OF", "武大狼", "第 250 代孙")]},
    {"name": "红太狼", "show": "XYY",
     "weapon": "平底锅", "move": "千手平底锅",
     "catch": "快去抓羊",
     "traits": ["出身富裕世家，爱打扮",
                "总是派丈夫去抓羊，抓不到就动手"],
     "rels": [("SPOUSE_OF", "灰太狼", "丈夫"),
              ("PARENT_OF<", "爹爹狼", "父亲"), ("PARENT_OF<", "娘娘狼", "母亲")]},
    {"name": "小灰灰", "show": "XYY",
     "carries": "奶嘴",
     "catch": "爸爸，你又骗我",
     "traits": ["温驯天真，和小羊们成了朋友",
                "因为爸爸总抓不到羊，只能一直喝奶",
                "最喜欢看爸爸飞上天，哭声很大",
                "首次出现在 2009 年的电影里，是剧场版的新角色"],
     "rels": [("PARENT_OF<", "灰太狼", "父亲"), ("PARENT_OF<", "红太狼", "母亲")]},
    {"name": "蕉太狼", "show": "XYY",
     "food": "香蕉",
     "traits": ["胆小，从小和父母失散，被猴子收养，吃素",
                "常常阻止二叔抓羊"],
     "rels": [("UNCLE_OF<", "灰太狼", "侄子"), ("FRIEND_OF", "暖羊羊", "")]},
    {"name": "香太狼", "show": "XYY",
     "weapon": "熨斗",
     "traits": ["和表姐一样凶悍"],
     "rels": [("COUSIN_OF", "红太狼", "表妹"), ("SPOUSE_OF", "夜太狼", "丈夫"),
              ("PARENT_OF", "小香香", "女儿"), ("LIKES", "蕉太狼", "")]},
    {"name": "夜太狼", "show": "XYY",
     "traits": ["是狼族里有名的恶狼"],
     "rels": [("COUSIN_OF", "灰太狼", "表哥")]},
    {"name": "灰二太太狼", "show": "XYY",
     "move": "沉默羔羊十二式",
     "traits": ["总叫侄子的乳名「尿太狼」"],
     "rels": [("UNCLE_OF", "灰太狼", "二叔")]},
    {"name": "黑太狼", "show": "XYY",
     "role": "狼族大王",
     "traits": ["曾因照顾发烧的儿子放弃抓羊行动，被当作临阵脱逃开除狼籍",
                "后来在雨天为儿子抓鱼时掉进瀑布，从此下落不明"],
     "rels": [("SPOUSE_OF", "银太狼", "妻子"),
              ("DESCENDANT_OF", "武大狼", "第 249 代孙")]},
    # ---------------- 铁甲小宝: 人类 ----------------
    {"name": "小让", "show": "TJ",
     "aliases": ["高圆寺让"],
     "role": "小学生",
     "traits": ["今年 9 岁",
                "能用友情呼唤指令器让卡布达进行超级变换"],
     "rels": [("GRANDPARENT_OF<", "高圆寺寅彦", "爷爷")]},
    {"name": "高圆寺寅彦", "show": "TJ",
     "aliases": ["高圆寺博士"],
     "role": "机器人科学家",
     "traits": ["是机器人技术的天才，剧中所有 B 机器人都是他制造的"]},
    {"name": "吉祥寺藏之助", "show": "TJ",
     "aliases": ["藏之助"],
     "role": "学生会会长",
     "traits": ["今年 11 岁，成绩好，运动也好",
                "父亲是吉祥寺建设公司的副社长"]},
    {"name": "中野美树", "show": "TJ",
     "role": "女警"},
    # ---------------- 铁甲小宝: 机器人 ----------------
    {"name": "卡布达", "show": "TJ",
     "unit": "1 号机", "proto": "独角仙",
     "catch": "卡布", "weapon": "电光棒", "food": "西瓜",
     "time": "3 分钟", "giant": "卡布达巨人",
     "traits": ["全身红色，普通形态说话稚气可爱，超级变换之后会变成成熟稳重的英雄语气",
                "一般要靠搭档的友情呼唤指令器才能超级变换"],
     "rels": [("PARTNER_OF", "小让", "")]},
    {"name": "金龟次郎", "show": "TJ",
     "unit": "2 号机", "proto": "锹形虫",
     "catch": "男人就应该默默的", "weapon": "金龟大剪钳",
     "time": "5 分钟",
     "traits": ["全身绿色，力气很大"],
     "rels": [("PARTNER_OF", "藏之助", "")]},        # 刻意只写别名
    {"name": "飞翔机器人", "show": "TJ",
     "unit": "3 号机", "role": "支援型机器人",
     "time": "7 分钟",
     "traits": ["能和卡布达飞翔连结，让卡布达获得飞行能力",
                "说话狂妄，性格却像个小孩"],
     "rels": [("LIKES", "田德莉娜", "")]},
    {"name": "蝎子莱莱", "show": "TJ",
     "aliases": ["蝎子蓝蓝"],
     "unit": "4 号机", "proto": "鲎",
     "traits": ["虽然名字里有蝎子，全身却是紫色的鲎造型",
                "爱喝酒"]},
    {"name": "蜘蛛侦探", "show": "TJ",
     "unit": "5 号机", "proto": "蜘蛛", "weapon": "巨爪",
     "traits": ["是个守财奴，擅长数钱",
                "说话带关西口音"]},
    {"name": "丸子龙", "show": "TJ",
     "unit": "6 号机", "proto": "鼠妇", "food": "糯米丸子",
     "traits": ["肚子里藏着和平星的碎片",
                "起初保持中立，后来加入了卡布达一方"]},
    {"name": "呱呱蛙", "show": "TJ",
     "aliases": ["聒聒蛙"],
     "unit": "7 号机", "proto": "青蛙", "role": "科学博士",
     "catch": "从结论上来看",
     "traits": ["说话长篇大论，善于分析敌人"]},
    {"name": "蟑螂恶霸", "show": "TJ",
     "unit": "8 号机", "proto": "眼镜蛇", "weapon": "眼镜蛇尾鞭",
     "traits": ["虽然名字里有蟑螂，原型却不是蟑螂",
                "全身蓝黄配色，自称恶之英雄"]},
    {"name": "鲨鱼辣椒", "show": "TJ",
     "unit": "9 号机", "proto": "鲨鱼", "weapon": "鲨鱼神斧",
     "time": "15 分钟", "giant": "超级鲨鱼巨人",
     "traits": ["全身黑银配色，是战斗力最强的 B 机器人",
                "额头上有一道伤疤"]},
    {"name": "田德莉娜", "show": "TJ",
     "unit": "10 号机", "proto": "瓢虫",
     "traits": ["是唯一的女性机器人",
                "能长时间保持超级形态，但几乎没有攻击力"]},
    {"name": "蜻蜓队长", "show": "TJ",
     "proto": "蜻蜓", "role": "裁判机器人",
     "catch": "绝对裁判的公正漂亮",
     "traits": ["争夺和平星时会突然出现，让大家用各种运动或游戏比赛决出胜负",
                "对犯规者会用从天而降的巨大铁拳加以惩罚"]},
    {"name": "警用机器人", "show": "TJ",
     "aliases": ["AP717"],
     "traits": ["全身蓝色，容易上当，出场次数不多"],
     "rels": [("PARTNER_OF", "中野美树", "")]},
]

# 没有独立档案、只在别人档案里出现的角色
MINOR_CHARACTERS = {
    "智羊羊": {"role": "科学家"}, "丽羊羊": {"role": "歌星"},
    "银太狼": {}, "爹爹狼": {}, "娘娘狼": {}, "小香香": {},
    "武大狼": {},
}

PLACES = ["青青草原", "羊村", "狼堡", "大肥羊学校"]
GROUPS = {"七大恶狼": "XYY", "反派三人组": "TJ"}
ITEMS = ["和平星", "射手座和平星", "蛇夫座和平星"]

# 智羊羊、丽羊羊的身份写在喜羊羊的档案里
MINOR_ROLE_DOC = {"智羊羊": "喜羊羊", "丽羊羊": "喜羊羊"}

# ==========================================================================
# 句子模板。RuleExtractor 的正则与这里一一对应 —— 改一边必须改另一边。
# ==========================================================================

LIT_SENTENCE = {
    "unit":     ("UNIT_NO",        "该角色的编号是{x}。"),
    "proto":    ("PROTOTYPE",      "该角色的原型是{x}。"),
    "role":     ("ROLE",           "该角色的身份是{x}。"),
    "birthday": ("BIRTHDAY",       "该角色的生日是{x}。"),
    "catch":    ("CATCHPHRASE",    "该角色的口头禅是“{x}”。"),
    "weapon":   ("WEAPON",         "该角色的武器是{x}。"),
    "move":     ("SPECIAL_MOVE",   "该角色的必杀技是{x}。"),
    "food":     ("FAVORITE_FOOD",  "该角色最爱吃{x}。"),
    "carries":  ("CARRIES",        "该角色总是随身带着{x}。"),
    "time":     ("TRANSFORM_TIME", "该角色的超级变换形态最多维持 {x}。"),
    "giant":    ("GIANT_FORM",     "该角色可以变成巨人形态{x}。"),
}
BASIC_KEYS = ["unit", "proto", "role", "birthday"]
TRAIT_KEYS = ["catch", "weapon", "move", "food", "carries", "time", "giant"]

# (关系, 对象, 称谓) -> 句子。关系名以 "<" 结尾表示边的方向是 对象 -> 本角色。
REL_SENTENCE = {
    "PARENT_OF<":      "该角色的{label}是{x}。",      # x 是本角色的父/母
    "PARENT_OF":       "该角色的{label}是{x}。",      # x 是本角色的子女
    "GRANDPARENT_OF<": "该角色的{label}是{x}。",
    "SPOUSE_OF":       "该角色的{label}是{x}。",
    "COUSIN_OF":       "该角色是{x}的{label}。",
    "UNCLE_OF":        "该角色是{x}的{label}。",
    "UNCLE_OF<":       "该角色是{x}的{label}。",      # 侄子: 边为 x -UNCLE_OF-> 本角色
    "DESCENDANT_OF":   "该角色是{x}的{label}。",
    "HEAD_OF":         "该角色是{x}的{label}。",
    "LIKES":           "该角色喜欢{x}。",
    "FRIEND_OF":       "该角色和{x}是好朋友。",
    "PARTNER_OF":      "该角色的搭档是{x}。",
}

# ==========================================================================
# 干扰段落池 (distractor)
# 在几十份档案间措辞几乎相同, 不含任何可抽取的事实, 却会大量挤占 top-k 名额。
# ==========================================================================

DEBUT_CLAUSES = [
    "该角色在系列的多部作品中均有登场。不同季度、剧场版及续作中的设定可能存在差异，本档案以原版电视剧集为准。",
    "该角色的形象与性格在播出期间大体保持稳定。部分细节在后续作品中有所调整，本档案不收录续作设定。",
    "该角色的相关剧情分散在多集之中。如需了解具体情节，请以正片内容为准。",
    "该角色在不同地区播出时可能使用不同译名。本档案统一采用中国大陆播出版本的名称。",
]

NOTE_CLAUSES = [
    "本档案根据维基百科等公开资料整理，仅供动画爱好者交流参考。",
    "角色形象及相关权利归原著作权方所有，本档案不用于任何商业用途。",
    "本档案内容如与正片不一致，以正片为准。",
    "如发现本档案有遗漏或错误，欢迎对照正片核实后更正。",
]


# ==========================================================================
# 数据模型
# ==========================================================================

@dataclass
class Entity:
    id: str
    type: str                      # Show | Character | Place | Group | Item
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
        self.by_name: dict[str, str] = {}
        self.edges: list[Edge] = []
        self.docs: list[Document] = []
        self.doc_keys: dict[str, str] = {}       # 语义键 -> 文档 id, 供评测集引用
        self.planted: dict[str, list] = {"alias_only": []}
        self._seq: dict[str, int] = {}

    def add(self, typ: str, name: str, prefix: str,
            aliases: list[str] | None = None, **props) -> Entity:
        n = self._seq.get(prefix, 0) + 1
        self._seq[prefix] = n
        e = Entity(id=f"{prefix}{n:02d}", type=typ, name=name,
                   aliases=list(aliases or []), props=props)
        self.entities[e.id] = e
        self.by_name[name] = e.id
        for a in e.aliases:
            self.by_name[a] = e.id
        return e

    def eid(self, name: str) -> str:
        return self.by_name[name]

    def link(self, src: str, rel: str, dst: str, doc: str) -> None:
        """src/dst 传名称; 以 "lit:" 开头的 dst 是字面量。"""
        s = self.eid(src)
        d = dst if dst.startswith("lit:") else self.eid(dst)
        self.edges.append(Edge(src=s, rel=rel, dst=d, source_doc=doc))

    def emit(self, key: str, doc_type: str, title: str, text: str) -> str:
        did = f"doc-{len(self.docs) + 1:04d}"
        self.docs.append(Document(id=did, doc_type=doc_type, title=title,
                                  date=DATE, text=text))
        self.doc_keys[key] = did
        return did


# ==========================================================================
# 文档渲染
# ==========================================================================

def render_profile(w: World, p: dict) -> tuple[str, list[tuple[str, str, str]]]:
    """角色档案。返回 (正文, 待连接的边)。边在拿到 doc id 之后才能落地。"""
    rng = w.rng
    name = p["name"]
    show = SHOWS[p["show"]]
    edges: list[tuple[str, str, str]] = [(name, "APPEARS_IN", show["name"])]

    basic = [f"{name}是{show['kind']}《{show['name']}》中的角色。"]
    for a in p.get("aliases", []):
        basic.append(f"{name}又名{a}。")
        edges.append((name, "ALIAS", f"lit:{a}"))
    for k in BASIC_KEYS:
        if k in p:
            rel, tmpl = LIT_SENTENCE[k]
            basic.append(tmpl.format(x=p[k]))
            edges.append((name, rel, f"lit:{p[k]}"))

    traits = ["该角色" + t + "。" for t in p.get("traits", [])]
    for k in TRAIT_KEYS:
        if k in p:
            rel, tmpl = LIT_SENTENCE[k]
            traits.append(tmpl.format(x=p[k]))
            edges.append((name, rel, f"lit:{p[k]}"))

    rels: list[str] = []
    for rel, other, label in p.get("rels", []):
        rels.append(REL_SENTENCE[rel].format(x=other, label=label))
        if rel.endswith("<"):
            edges.append((other, rel[:-1], name))
        else:
            edges.append((name, rel, other))
    # 配角的身份写在主角档案里(如喜羊羊档案里写父亲是科学家)
    for minor, host in MINOR_ROLE_DOC.items():
        if host == name:
            role = MINOR_CHARACTERS[minor]["role"]
            rels.append(f"{minor}的身份是{role}。")
            edges.append((minor, "ROLE", f"lit:{role}"))
    if not rels:
        rels.append("该角色与其他角色的关系详见相关条目。")

    parts = [
        f"{name}\n角色档案\n",
        SOURCE_NOTE + "\n",
        "一、基本资料\n　　" + "".join(basic) + "\n",
        "二、性格与特长\n　　" + ("\n　　".join(traits) if traits
                             else "该角色的性格特点详见正片。") + "\n",
        "三、人物关系\n　　" + "\n　　".join(rels) + "\n",
        "四、登场说明\n　　" + rng.choice(DEBUT_CLAUSES) + "\n",
        "五、资料说明\n　　" + "\n　　".join(rng.sample(NOTE_CLAUSES, 2)),
    ]
    return "\n".join(parts), edges


def render_list(title: str, sections: list[tuple[str, list[str]]],
                intro: str, outro: str) -> str:
    """结构型文档: 标题 + 列表项。事实由标题决定, 列表项下没有完整句子。"""
    nums = "一二三四五六"
    body = [f"{title}\n", SOURCE_NOTE + "\n", "　　" + intro + "\n"]
    for i, (head, names) in enumerate(sections):
        body.append(f"{nums[i]}、{head}\n" + "".join(f"　　　· {n}\n" for n in names))
    body.append(f"{nums[len(sections)]}、说明\n　　{outro}")
    return "\n".join(body)


# ==========================================================================
# 世界构建
# ==========================================================================

def build_world(seed: int) -> World:
    w = World(seed)

    # ---- 实体 ----
    for key, s in SHOWS.items():
        w.add("Show", s["name"], "S")
    for p in PROFILES:
        w.add("Character", p["name"], "C", aliases=p.get("aliases", []),
              show=SHOWS[p["show"]]["name"])
    for name in MINOR_CHARACTERS:
        w.add("Character", name, "C", show=SHOWS["XYY"]["name"])
    for name in PLACES:
        w.add("Place", name, "P")
    for name, show in GROUPS.items():
        w.add("Group", name, "G", show=SHOWS[show]["name"])
    for name in ITEMS:
        w.add("Item", name, "I")

    # ---- 节目简介 ----
    d = w.emit("show:喜羊羊与灰太狼", "show_intro", "喜羊羊与灰太狼 节目简介", "\n".join([
        "喜羊羊与灰太狼\n节目简介\n", SOURCE_NOTE + "\n",
        "一、基本信息\n"
        "　　《喜羊羊与灰太狼》是一部中国原创动画，于 2005 年 8 月 3 日首播，"
        "首播电视台为杭州电视台少儿频道。\n"
        "　　第 1 季共 530 集，每集 15 分钟，主题曲为《别看我只是一只羊》。\n"
        "　　系列的首部电影为《喜羊羊与灰太狼之牛气冲天》，于 2009 年上映。\n",
        "二、创作花絮\n"
        "　　主角最初打算叫“懒羊羊”，制作团队觉得“喜羊羊”更正面，于是改成了现在的名字。\n"
        "　　羊的名字都是谐音：喜羊羊取自“喜气洋洋”，懒羊羊取自“懒洋洋”，"
        "沸羊羊取自“沸沸扬扬”。\n"
        "　　狼的名字叫“灰太狼”“红太狼”，是为了和传统故事里较为可怕的“大灰狼”区分开来。\n",
        "三、资料说明\n　　" + NOTE_CLAUSES[0],
    ]))
    w.link("喜羊羊与灰太狼", "FIRST_AIRED", "lit:2005 年 8 月 3 日", d)
    w.link("喜羊羊与灰太狼", "BROADCASTER", "lit:杭州电视台少儿频道", d)
    w.link("喜羊羊与灰太狼", "EPISODES", "lit:530 集", d)
    w.link("喜羊羊与灰太狼", "THEME_SONG", "lit:别看我只是一只羊", d)
    w.link("喜羊羊与灰太狼", "FIRST_MOVIE", "lit:喜羊羊与灰太狼之牛气冲天", d)

    d = w.emit("show:铁甲小宝", "show_intro", "铁甲小宝 节目简介", "\n".join([
        "铁甲小宝\n节目简介\n", SOURCE_NOTE + "\n",
        "一、基本信息\n"
        "　　《铁甲小宝》是日本东映出品的特摄电视剧，于 1997 年 2 月 23 日首播，"
        "首播电视台为朝日电视台，共 52 集。\n"
        "　　本剧是金属英雄系列的第 16 部作品。\n",
        "二、故事梗概\n"
        "　　小学生小让发现了爷爷制造的甲虫机器人卡布达。"
        "卡布达和其他 B 机器人为了争夺神秘的和平星，展开了一连串冒险。\n"
        "　　剧名虽然叫《铁甲小宝》，剧中却没有叫“小宝”的角色。\n",
        "三、资料说明\n　　" + NOTE_CLAUSES[0],
    ]))
    w.link("铁甲小宝", "FIRST_AIRED", "lit:1997 年 2 月 23 日", d)
    w.link("铁甲小宝", "BROADCASTER", "lit:朝日电视台", d)
    w.link("铁甲小宝", "EPISODES", "lit:52 集", d)

    # ---- 角色档案 ----
    for p in PROFILES:
        text, edges = render_profile(w, p)
        d = w.emit(f"profile:{p['name']}", "profile", f"{p['name']} 角色档案", text)
        for s, rel, o in edges:
            w.link(s, rel, o, d)

    # ---- 结构型文档: 居民登记 / 学生名册 / 阵营一览 ----
    sheep = ["喜羊羊", "美羊羊", "懒羊羊", "沸羊羊", "暖羊羊"]
    d = w.emit("list:居民", "roster", "青青草原居民登记", render_list(
        "青青草原居民登记（节选）",
        [("羊村居民", sheep + ["慢羊羊村长"]),          # 刻意只写别名
         ("狼堡住户", ["灰太狼", "红太狼", "小灰灰"])],
        "本登记表按居住地列出青青草原上的主要角色。",
        "本登记表仅收录原版电视剧集中的主要角色，不含客串角色。"))
    for n in sheep:
        w.link(n, "LIVES_IN", "羊村", d)
    w.link("慢羊羊", "LIVES_IN", "羊村", d)
    w.planted["alias_only"].append({"doc": d, "alias": "慢羊羊村长", "canonical": "慢羊羊"})
    for n in ["灰太狼", "红太狼", "小灰灰"]:
        w.link(n, "LIVES_IN", "狼堡", d)

    d = w.emit("list:学生", "roster", "大肥羊学校学生名册", render_list(
        "大肥羊学校学生名册",
        [("在读学生", sheep)],
        "大肥羊学校是羊村小羊们上学的地方。",
        "名册按入学先后排列，转学生列在最后。"))
    for n in sheep:
        w.link(n, "STUDENT_OF", "大肥羊学校", d)

    good = ["卡布达", "金龟次郎", "飞翔机器人", "呱呱蛙", "田德莉娜"]
    bad = ["蟑螂恶霸", "蜘蛛侦探", "蝎子莱莱", "鲨鱼辣椒"]
    neutral = ["丸子龙", "蜻蜓队长"]
    d = w.emit("list:阵营", "roster", "B 机器人阵营一览", render_list(
        "B 机器人阵营一览",
        [("正义阵营", good), ("反派阵营", bad), ("中立阵营", neutral)],
        "下表按剧集前期的立场，列出争夺和平星的各个机器人。",
        "部分机器人在剧情后期改变了立场，以正片为准。"))
    for camp, names in (("正义", good), ("反派", bad), ("中立", neutral)):
        for n in names:
            w.link(n, "FACTION", f"lit:{camp}", d)

    # ---- 团体 ----
    d = w.emit("group:反派三人组", "group", "反派三人组 团体资料", "\n".join([
        "反派三人组\n团体资料\n", SOURCE_NOTE + "\n",
        "一、成员\n"
        "　　反派三人组是特摄剧《铁甲小宝》中的反派团体，"
        "成员为蟑螂恶霸、蜘蛛侦探和蝎子莱莱。\n"
        "　　蟑螂恶霸是反派三人组的老大，蜘蛛侦探和蝎子莱莱都称他为“大哥”。\n",
        "二、事迹\n"
        "　　三个机器人一起过着穷困的日子，一心想利用和平星统治人类。"
        "在鲨鱼辣椒出现之前，他们一直是最主要的反派。\n",
        "三、资料说明\n　　" + NOTE_CLAUSES[2],
    ]))
    for n in ["蟑螂恶霸", "蜘蛛侦探", "蝎子莱莱"]:
        w.link(n, "MEMBER_OF", "反派三人组", d)
    w.link("蟑螂恶霸", "LEADER_OF", "反派三人组", d)

    d = w.emit("group:七大恶狼", "group", "七大恶狼 团体资料", "\n".join([
        "七大恶狼\n团体资料\n", SOURCE_NOTE + "\n",
        "一、成员\n"
        "　　七大恶狼是动画《喜羊羊与灰太狼》中的狼族团体。\n"
        "　　灰二太太狼是七大恶狼的首领，夜太狼是七大恶狼的成员之一。\n",
        "二、资料说明\n　　" + NOTE_CLAUSES[2],
    ]))
    w.link("灰二太太狼", "LEADER_OF", "七大恶狼", d)
    w.link("夜太狼", "MEMBER_OF", "七大恶狼", d)

    # ---- 背景故事与剧情 ----
    d = w.emit("story:饿狼传说", "story", "羊村的建立与饿狼传说", "\n".join([
        "羊村的建立与饿狼传说\n背景故事\n", SOURCE_NOTE + "\n",
        "一、羊村的建立\n"
        "　　羊历 3010 年，绵羊族先祖软绵绵为了躲避狼群来到青青草原，"
        "建起了防御坚固的羊村。羊村位于青青草原。\n",
        "二、饿狼传说\n"
        "　　狼群首领武大狼为了钻过羊村的铁门而拼命减肥，钻过去之后却误吞石头而死。"
        "这就是青青草原流传的“饿狼传说”。\n",
        "三、资料说明\n　　" + NOTE_CLAUSES[0],
    ]))
    w.link("羊村", "LOCATED_IN", "青青草原", d)

    d = w.emit("story:和平星", "setting", "和平星 设定资料", "\n".join([
        "和平星\n设定资料\n", SOURCE_NOTE + "\n",
        "一、设定\n"
        "　　和平星是特摄剧《铁甲小宝》中的神秘物体，和平星共有 13 颗，"
        "分别代表包括蛇夫座在内的十三个星座。\n"
        "　　得到一颗和平星，就可以许一个愿望。另有不具许愿效力的假和平星。\n",
        "二、争夺\n"
        "　　B 机器人们争夺和平星时，裁判机器人会突然出现，"
        "让大家用各种运动或游戏比赛决出胜负。\n",
        "三、资料说明\n　　" + NOTE_CLAUSES[0],
    ]))
    w.link("和平星", "QUANTITY", "lit:13 颗", d)

    d = w.emit("story:卡布达巨人", "story", "卡布达巨人的诞生", "\n".join([
        "卡布达巨人的诞生\n剧情梗概\n", SOURCE_NOTE + "\n",
        "一、起因\n"
        "　　鲨鱼辣椒能变成巨人形态，一般的 B 机器人根本无法与之抗衡。\n",
        "二、经过\n"
        "　　为了对抗巨人形态的鲨鱼辣椒，卡布达向射手座和平星许愿，由此诞生了卡布达巨人。"
        "卡布达坐进驾驶舱，操纵卡布达巨人战斗。\n"
        "　　卡布达巨人普通形态高 6.5 米，超级形态高 10.8 米。\n",
        "三、资料说明\n　　" + NOTE_CLAUSES[2],
    ]))
    w.link("卡布达", "WISHED_ON", "射手座和平星", d)

    d = w.emit("story:鲨鱼辣椒", "story", "鲨鱼辣椒的封印与转变", "\n".join([
        "鲨鱼辣椒的封印与转变\n剧情梗概\n", SOURCE_NOTE + "\n",
        "一、封印\n"
        "　　鲨鱼辣椒因为自我意识扭曲，被高圆寺博士封印。"
        "它额头上的伤疤是高圆寺博士不小心弄的，它因此一直心怀怨恨。\n"
        "　　后来，丸子龙解开了鲨鱼辣椒的封印。\n",
        "二、转变\n"
        "　　卡布达用蛇夫座和平星的愿望消去了鲨鱼辣椒额头的伤疤。"
        "鲨鱼辣椒被打动，从此改邪归正。\n",
        "三、资料说明\n　　" + NOTE_CLAUSES[2],
    ]))
    w.link("鲨鱼辣椒", "SEALED_BY", "高圆寺寅彦", d)       # 正文只写「高圆寺博士」
    w.planted["alias_only"].append({"doc": d, "alias": "高圆寺博士", "canonical": "高圆寺寅彦"})
    w.link("鲨鱼辣椒", "RELEASED_BY", "丸子龙", d)
    w.link("卡布达", "WISHED_ON", "蛇夫座和平星", d)

    pk = w.doc_keys["profile:金龟次郎"]
    w.planted["alias_only"].append({"doc": pk, "alias": "藏之助", "canonical": "吉祥寺藏之助"})
    return w


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="生成语料与 ground-truth 图谱")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    w = build_world(args.seed)

    write_text(args.out / "graph.json", json.dumps({
        "seed": args.seed,
        "entities": [asdict(e) for e in w.entities.values()],
        "edges": [asdict(e) for e in w.edges],
        "planted": w.planted,
        "doc_keys": w.doc_keys,
    }, ensure_ascii=False, indent=2))
    write_jsonl(args.out / "documents.jsonl", (asdict(d) for d in w.docs))

    n_char = sum(len(d.text) for d in w.docs)
    by_type: dict[str, int] = {}
    for d in w.docs:
        by_type[d.doc_type] = by_type.get(d.doc_type, 0) + 1
    ent_types: dict[str, int] = {}
    for e in w.entities.values():
        ent_types[e.type] = ent_types.get(e.type, 0) + 1

    print(f"[ok] 实体 {len(w.entities)}  边 {len(w.edges)}  文档 {len(w.docs)}  "
          f"总字数 {n_char:,}  平均每文档 {n_char // max(1, len(w.docs)):,} 字")
    print("     实体类型:", "  ".join(f"{k}={v}" for k, v in sorted(ent_types.items())))
    print("     文档类型:", "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"     植入: 仅以别名出现的证据 {len(w.planted['alias_only'])} 处")
    print(f"     -> {args.out / 'graph.json'}")
    print(f"     -> {args.out / 'documents.jsonl'}")


if __name__ == "__main__":
    main()
