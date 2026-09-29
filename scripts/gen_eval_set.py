"""
Phase 0 - 评测集生成器  (跨平台)

核心思路: 题目是人写的, 但**证据文档是算出来的**。
每道题声明它依赖图里的哪几条边(path), 或哪几份文档(docs);
脚本从 ground-truth 图里查出这些边的来源文档, 作为 gold_docs。
有了 gold_docs 才能算 recall@k —— 这是整个评测的地基。
题目引用的边在图里不存在时直接报错, 保证题库和图谱不会悄悄脱节。

评测集按"能力"分类, 每一类对应一个具体的检索技术:

  fact_direct      1跳  用文档原词提问          -> 基线, 谁都该答对
  fact_paraphrase  1跳  换说法但保留角色名      -> 弱语义测试(BM25 靠角色名即可过)
  semantic_only    1跳  **完全不含角色名**的描述性提问 -> 真正的 dense 语义测试
  fact_disambig    1跳  名字易混(灰太狼/灰二太太狼/蕉太狼), 或只给别名 -> 词面精度
  hop2             2跳  小灰灰 -> 灰太狼 -> 黑太狼 -> 纯向量开始失效
  hop3             3跳  蕉太狼 -> 灰太狼 -> 红太狼 -> 爹爹狼
  relation_path    招牌  "A 和 B 是什么关系" —— 两端都有名字, 中间环节一个都没有
  aggregation      聚合 列举/比较类, 证据分散在多份文档
  negative         陷阱 语料中不存在, 正确行为是拒答 -> 考验幻觉抑制

train / dev / test 三分:
  prompt 只允许在 train 上调, dev 用于早停, test 只在最后跑一次。
  否则 prompt 会过拟合评测集 —— 这是 prompt 工程里最常见的自欺。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, write_jsonl, write_text  # noqa: E402
from graphrag.store.graph import KnowledgeGraph                     # noqa: E402

setup_console()

ROOT = Path(__file__).resolve().parents[1]

SPLIT_WEIGHTS = [("train", 0.40), ("dev", 0.20), ("test", 0.40)]

REQUIRES = {"fact_direct": "vector", "fact_paraphrase": "vector",
            "semantic_only": "dense", "fact_disambig": "hybrid",
            "hop2": "graph", "hop3": "graph", "relation_path": "graph",
            "aggregation": "graph", "negative": "refusal"}
HOPS = {"hop2": 2, "hop3": 3, "relation_path": 3}


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


# ==========================================================================
# 题库
#   path   依赖的边 (主体, 关系, 客体); 客体以 "lit:" 开头表示字面量
#   docs   直接依赖的文档键(性格特点等不进图的事实)
#   ents   判分用的关键实体 —— 回答里必须全部出现才算对
#   derived=True  答案需要推理得出(如"表姨"), 字面上不出现在证据文档里
# ==========================================================================

def Q(type_: str, question: str, answer: str, ents: list[str], *,
      path: list[tuple[str, str, str]] | None = None,
      docs: list[str] | None = None, note: str = "",
      derived: bool = False, trap: list[str] | None = None) -> dict:
    return {"type": type_, "question": question, "answer": answer, "ents": ents,
            "path": path or [], "docs": docs or [], "note": note,
            "derived": derived, "trap": trap or []}


QUESTIONS: list[dict] = [
    # ================= fact_direct: 用文档原词提问 =================
    Q("fact_direct", "灰太狼的表哥是谁？", "夜太狼", ["夜太狼"],
      path=[("夜太狼", "COUSIN_OF", "灰太狼")]),
    Q("fact_direct", "红太狼的武器是什么？", "平底锅", ["平底锅"],
      path=[("红太狼", "WEAPON", "lit:平底锅")]),
    Q("fact_direct", "灰太狼的口头禅是什么？", "我一定会回来的", ["我一定会回来的"],
      path=[("灰太狼", "CATCHPHRASE", "lit:我一定会回来的")]),
    Q("fact_direct", "红太狼的必杀技是什么？", "千手平底锅", ["千手平底锅"],
      path=[("红太狼", "SPECIAL_MOVE", "lit:千手平底锅")]),
    Q("fact_direct", "灰太狼的生日是哪一天？", "羊历 3475 年 9 月 26 日", ["9 月 26 日"],
      path=[("灰太狼", "BIRTHDAY", "lit:羊历 3475 年 9 月 26 日")]),
    Q("fact_direct", "灰二太太狼是灰太狼的什么人？", "二叔", ["二叔"],
      path=[("灰二太太狼", "UNCLE_OF", "灰太狼")]),
    Q("fact_direct", "《喜羊羊与灰太狼》的主题曲叫什么？", "《别看我只是一只羊》",
      ["别看我只是一只羊"],
      path=[("喜羊羊与灰太狼", "THEME_SONG", "lit:别看我只是一只羊")]),
    Q("fact_direct", "喜羊羊与灰太狼系列的首部电影叫什么？", "《喜羊羊与灰太狼之牛气冲天》",
      ["牛气冲天"],
      path=[("喜羊羊与灰太狼", "FIRST_MOVIE", "lit:喜羊羊与灰太狼之牛气冲天")]),
    Q("fact_direct", "卡布达的原型是什么？", "独角仙", ["独角仙"],
      path=[("卡布达", "PROTOTYPE", "lit:独角仙")]),
    Q("fact_direct", "金龟次郎的原型是什么？", "锹形虫", ["锹形虫"],
      path=[("金龟次郎", "PROTOTYPE", "lit:锹形虫")]),
    Q("fact_direct", "田德莉娜的原型是什么？", "瓢虫", ["瓢虫"],
      path=[("田德莉娜", "PROTOTYPE", "lit:瓢虫")]),
    Q("fact_direct", "卡布达的口头禅是什么？", "“卡布”", ["卡布"],
      path=[("卡布达", "CATCHPHRASE", "lit:卡布")]),
    Q("fact_direct", "呱呱蛙的口头禅是什么？", "“从结论上来看”", ["从结论上来看"],
      path=[("呱呱蛙", "CATCHPHRASE", "lit:从结论上来看")]),
    Q("fact_direct", "鲨鱼辣椒的武器是什么？", "鲨鱼神斧", ["鲨鱼神斧"],
      path=[("鲨鱼辣椒", "WEAPON", "lit:鲨鱼神斧")]),
    Q("fact_direct", "卡布达的搭档是谁？", "小让", ["小让"],
      path=[("卡布达", "PARTNER_OF", "小让")]),
    Q("fact_direct", "丸子龙最爱吃什么？", "糯米丸子", ["糯米丸子"],
      path=[("丸子龙", "FAVORITE_FOOD", "lit:糯米丸子")]),
    Q("fact_direct", "卡布达巨人是卡布达向哪颗和平星许愿诞生的？", "射手座和平星",
      ["射手座和平星"],
      path=[("卡布达", "WISHED_ON", "射手座和平星")]),
    Q("fact_direct", "和平星共有多少颗？", "13 颗", ["13"],
      path=[("和平星", "QUANTITY", "lit:13 颗")]),

    # ================= fact_paraphrase: 换说法, 保留角色名 =================
    Q("fact_paraphrase", "灰太狼的老婆是谁？", "红太狼", ["红太狼"],
      path=[("红太狼", "SPOUSE_OF", "灰太狼")],
      note="文档写的是红太狼的'丈夫是灰太狼', 提问换成了'老婆'"),
    Q("fact_paraphrase", "红太狼生气时拿什么招呼老公？", "平底锅", ["平底锅"],
      path=[("红太狼", "WEAPON", "lit:平底锅")], note="避开'武器'一词"),
    Q("fact_paraphrase", "小灰灰嘴里老叼着什么？", "奶嘴", ["奶嘴"],
      path=[("小灰灰", "CARRIES", "lit:奶嘴")], note="避开'随身带着'"),
    Q("fact_paraphrase", "蕉太狼最喜欢吃哪种水果？", "香蕉", ["香蕉"],
      path=[("蕉太狼", "FAVORITE_FOOD", "lit:香蕉")]),
    Q("fact_paraphrase", "羊村由谁当家？", "慢羊羊（村长）", ["慢羊羊"],
      path=[("慢羊羊", "HEAD_OF", "羊村")], note="避开'村长'一词"),
    Q("fact_paraphrase", "小灰灰发现被爸爸骗了会嚷什么？", "“爸爸，你又骗我”", ["你又骗我"],
      path=[("小灰灰", "CATCHPHRASE", "lit:爸爸，你又骗我")]),
    Q("fact_paraphrase", "《喜羊羊与灰太狼》是哪一年开始在电视上播出的？", "2005 年",
      ["2005"],
      path=[("喜羊羊与灰太狼", "FIRST_AIRED", "lit:2005 年 8 月 3 日")],
      note="避开'首播'一词"),
    Q("fact_paraphrase", "喜羊羊他爸是干什么的？", "科学家（智羊羊）", ["智羊羊", "科学家"],
      path=[("智羊羊", "PARENT_OF", "喜羊羊"), ("智羊羊", "ROLE", "lit:科学家")]),
    Q("fact_paraphrase", "卡布达最喜欢啃什么水果？", "西瓜", ["西瓜"],
      path=[("卡布达", "FAVORITE_FOOD", "lit:西瓜")]),
    Q("fact_paraphrase", "卡布达变身之后能撑多久？", "3 分钟", ["3 分钟"],
      path=[("卡布达", "TRANSFORM_TIME", "lit:3 分钟")], note="避开'超级变换形态'"),
    Q("fact_paraphrase", "金龟次郎靠什么家伙打架？", "金龟大剪钳", ["金龟大剪钳"],
      path=[("金龟次郎", "WEAPON", "lit:金龟大剪钳")]),
    Q("fact_paraphrase", "蟑螂恶霸出手时用的是什么兵器？", "眼镜蛇尾鞭", ["眼镜蛇尾鞭"],
      path=[("蟑螂恶霸", "WEAPON", "lit:眼镜蛇尾鞭")]),
    Q("fact_paraphrase", "小让用什么道具让卡布达超级变身？", "友情呼唤指令器",
      ["友情呼唤指令器"], docs=["profile:小让"]),
    Q("fact_paraphrase", "铁甲小宝里的 B 机器人都是谁造出来的？", "高圆寺寅彦（高圆寺博士）",
      ["高圆寺寅彦"], docs=["profile:高圆寺寅彦"]),
    Q("fact_paraphrase", "《铁甲小宝》一共拍了多少集？", "52 集", ["52"],
      path=[("铁甲小宝", "EPISODES", "lit:52 集")]),
    Q("fact_paraphrase", "喜羊羊这个主角一开始本来打算叫什么名字？", "懒羊羊",
      ["懒羊羊"], docs=["show:喜羊羊与灰太狼"]),

    # ================= fact_disambig: 名字易混, 或只给别名 =================
    Q("fact_disambig", "蕉太狼是灰太狼的什么人？", "侄子", ["侄子"],
      path=[("灰太狼", "UNCLE_OF", "蕉太狼")],
      note="易混: 夜太狼是灰太狼的表哥, 灰二太太狼是灰太狼的二叔; 三只狼的档案措辞几乎一样"),
    Q("fact_disambig", "香太狼的丈夫是谁？", "夜太狼", ["夜太狼"],
      path=[("香太狼", "SPOUSE_OF", "夜太狼")],
      note="易混: 红太狼的档案里同样写着'该角色的丈夫是……'"),
    Q("fact_disambig", "美羊羊喜欢谁？", "喜羊羊", ["喜羊羊"],
      path=[("美羊羊", "LIKES", "喜羊羊")], note="易混: 沸羊羊喜欢美羊羊"),
    Q("fact_disambig", "沸羊羊喜欢谁？", "美羊羊", ["美羊羊"],
      path=[("沸羊羊", "LIKES", "美羊羊")], note="易混: 美羊羊喜欢喜羊羊"),
    Q("fact_disambig", "暖羊羊和懒羊羊，谁是班长？", "暖羊羊", ["暖羊羊"],
      path=[("暖羊羊", "ROLE", "lit:班长")]),
    Q("fact_disambig", "灰二太太狼是哪个团体的首领？", "七大恶狼", ["七大恶狼"],
      path=[("灰二太太狼", "LEADER_OF", "七大恶狼")],
      note="灰二太太狼 与 灰太狼 只差两个字, 最长匹配必须先认出长的那个"),
    Q("fact_disambig", "卡布达和金龟次郎，谁的超级变换形态撑得更久？", "金龟次郎（5 分钟）",
      ["金龟次郎"],
      path=[("卡布达", "TRANSFORM_TIME", "lit:3 分钟"),
            ("金龟次郎", "TRANSFORM_TIME", "lit:5 分钟")]),
    Q("fact_disambig", "蝎子莱莱的原型真的是蝎子吗？", "不是，是鲎", ["鲎"],
      path=[("蝎子莱莱", "PROTOTYPE", "lit:鲎")], note="名字与原型对不上, 专钓望文生义"),
    Q("fact_disambig", "蟑螂恶霸的原型真的是蟑螂吗？", "不是，是眼镜蛇", ["眼镜蛇"],
      path=[("蟑螂恶霸", "PROTOTYPE", "lit:眼镜蛇")], note="名字与原型对不上, 专钓望文生义"),
    Q("fact_disambig", "蝎子蓝蓝的原型是什么？", "鲎（蝎子蓝蓝即蝎子莱莱）", ["鲎"],
      path=[("蝎子莱莱", "ALIAS", "lit:蝎子蓝蓝"), ("蝎子莱莱", "PROTOTYPE", "lit:鲎")],
      note="只给别名: 必须把'蝎子蓝蓝'认成'蝎子莱莱'"),
    Q("fact_disambig", "铁甲小宝里真的有叫“小宝”的角色吗？主角叫什么？",
      "没有叫小宝的角色，主角叫小让", ["小让"], docs=["show:铁甲小宝"]),
    Q("fact_disambig", "卡布达和鲨鱼辣椒分别是几号机？", "卡布达 1 号机，鲨鱼辣椒 9 号机",
      ["1 号机", "9 号机"],
      path=[("卡布达", "UNIT_NO", "lit:1 号机"), ("鲨鱼辣椒", "UNIT_NO", "lit:9 号机")]),

    # ================= semantic_only: 不含角色名, 用词与文档不重叠 =================
    Q("semantic_only", "哪只狼是被猴子带大的素食者？", "蕉太狼", ["蕉太狼"],
      docs=["profile:蕉太狼"], note="文档写的是'被猴子收养，吃素'"),
    Q("semantic_only", "哪只羊体型最圆、最不爱干活？", "懒羊羊", ["懒羊羊"],
      docs=["profile:懒羊羊"], note="文档写的是'最胖、最懒'"),
    Q("semantic_only", "哪只羊一开口说坏事就会应验？", "暖羊羊", ["暖羊羊"],
      docs=["profile:暖羊羊"], note="文档写的是'乌鸦嘴，预言坏事很准'"),
    Q("semantic_only", "哪只羊特别热衷锻炼肌肉？", "沸羊羊", ["沸羊羊"],
      docs=["profile:沸羊羊"], note="文档写的是'最爱健身'"),
    Q("semantic_only", "谁动作慢吞吞，走路还得撑着棍子？", "慢羊羊", ["慢羊羊"],
      docs=["profile:慢羊羊"], note="文档写的是'行动比蜗牛还慢，拄着拐杖'"),
    Q("semantic_only", "哪位角色负责管理整片草原？", "包包大人", ["包包大人"],
      docs=["profile:包包大人"], note="文档写的是'青青草原的管理者'"),
    Q("semantic_only", "哪只羊最爱打扮，还会用各种材料编小首饰？", "美羊羊", ["美羊羊"],
      docs=["profile:美羊羊"], note="文档写的是'爱美……编织饰物'"),
    Q("semantic_only", "哪个机器人专门在比赛里当公证人？", "蜻蜓队长", ["蜻蜓队长"],
      docs=["profile:蜻蜓队长"], note="文档写的是'裁判机器人'"),
    Q("semantic_only", "哪个机器人是队伍里仅有的女生？", "田德莉娜", ["田德莉娜"],
      docs=["profile:田德莉娜"], note="文档写的是'唯一的女性机器人'"),
    Q("semantic_only", "哪个机器人讲起话来又臭又长，还很会研究对手？", "呱呱蛙", ["呱呱蛙"],
      docs=["profile:呱呱蛙"], note="文档写的是'说话长篇大论，善于分析敌人'"),
    Q("semantic_only", "哪个机器人是爱钱如命的财迷？", "蜘蛛侦探", ["蜘蛛侦探"],
      docs=["profile:蜘蛛侦探"], note="文档写的是'守财奴，擅长数钱'"),
    Q("semantic_only", "哪个机器人贪杯好饮？", "蝎子莱莱", ["蝎子莱莱"],
      docs=["profile:蝎子莱莱"], note="文档写的是'爱喝酒'"),

    # ================= hop2: 两份文档拼起来才能答 =================
    Q("hop2", "小灰灰的爷爷是谁？", "黑太狼", ["黑太狼"],
      path=[("灰太狼", "PARENT_OF", "小灰灰"), ("黑太狼", "PARENT_OF", "灰太狼")]),
    Q("hop2", "小灰灰的奶奶是谁？", "银太狼", ["银太狼"],
      path=[("灰太狼", "PARENT_OF", "小灰灰"), ("银太狼", "PARENT_OF", "灰太狼")]),
    Q("hop2", "小灰灰的外公是谁？", "爹爹狼", ["爹爹狼"],
      path=[("红太狼", "PARENT_OF", "小灰灰"), ("爹爹狼", "PARENT_OF", "红太狼")]),
    Q("hop2", "小灰灰的外婆是谁？", "娘娘狼", ["娘娘狼"],
      path=[("红太狼", "PARENT_OF", "小灰灰"), ("娘娘狼", "PARENT_OF", "红太狼")]),
    Q("hop2", "小灰灰的妈妈用什么打小灰灰的爸爸？", "平底锅", ["平底锅"],
      path=[("红太狼", "PARENT_OF", "小灰灰"), ("红太狼", "WEAPON", "lit:平底锅")]),
    Q("hop2", "灰太狼表哥的老婆是谁？", "香太狼", ["香太狼"],
      path=[("夜太狼", "COUSIN_OF", "灰太狼"), ("香太狼", "SPOUSE_OF", "夜太狼")]),
    Q("hop2", "香太狼是小灰灰的什么人？", "表姨（香太狼是小灰灰妈妈红太狼的表妹）", ["表姨"],
      path=[("红太狼", "PARENT_OF", "小灰灰"), ("香太狼", "COUSIN_OF", "红太狼")],
      derived=True),
    Q("hop2", "懒羊羊上学的学校，校长是谁？", "慢羊羊", ["慢羊羊"],
      path=[("懒羊羊", "STUDENT_OF", "大肥羊学校"), ("慢羊羊", "HEAD_OF", "大肥羊学校")]),
    Q("hop2", "灰太狼的二叔是哪个团体的首领？", "七大恶狼", ["七大恶狼"],
      path=[("灰二太太狼", "UNCLE_OF", "灰太狼"), ("灰二太太狼", "LEADER_OF", "七大恶狼")]),
    Q("hop2", "美羊羊喜欢的那只羊有什么爱好？", "踢足球和做实验（她喜欢喜羊羊）",
      ["喜羊羊", "足球"],
      path=[("美羊羊", "LIKES", "喜羊羊")], docs=["profile:喜羊羊"]),
    Q("hop2", "小灰灰是武大狼的第几代孙？", "第 251 代（灰太狼是第 250 代孙）", ["251"],
      path=[("灰太狼", "PARENT_OF", "小灰灰"), ("灰太狼", "DESCENDANT_OF", "武大狼")],
      derived=True),
    Q("hop2", "金龟次郎的搭档在学校担任什么职务？", "学生会会长（吉祥寺藏之助）",
      ["学生会会长"],
      path=[("金龟次郎", "PARTNER_OF", "吉祥寺藏之助"),
            ("吉祥寺藏之助", "ROLE", "lit:学生会会长")],
      note="金龟次郎的档案里搭档只写作'藏之助', 不做消歧就接不上"),
    Q("hop2", "卡布达搭档的爷爷叫什么？", "高圆寺寅彦", ["高圆寺寅彦"],
      path=[("卡布达", "PARTNER_OF", "小让"), ("高圆寺寅彦", "GRANDPARENT_OF", "小让")]),
    Q("hop2", "蜘蛛侦探口中的“大哥”，原型是什么？", "眼镜蛇（大哥是蟑螂恶霸）", ["眼镜蛇"],
      path=[("蟑螂恶霸", "LEADER_OF", "反派三人组"), ("蟑螂恶霸", "PROTOTYPE", "lit:眼镜蛇")]),
    Q("hop2", "飞翔机器人喜欢的机器人，原型是什么？", "瓢虫（它喜欢田德莉娜）", ["瓢虫"],
      path=[("飞翔机器人", "LIKES", "田德莉娜"), ("田德莉娜", "PROTOTYPE", "lit:瓢虫")]),
    Q("hop2", "警用机器人的搭档是做什么的？", "女警（中野美树）", ["女警"],
      path=[("警用机器人", "PARTNER_OF", "中野美树"), ("中野美树", "ROLE", "lit:女警")]),
    Q("hop2", "解开鲨鱼辣椒封印的机器人最爱吃什么？", "糯米丸子（丸子龙）", ["糯米丸子"],
      path=[("鲨鱼辣椒", "RELEASED_BY", "丸子龙"), ("丸子龙", "FAVORITE_FOOD", "lit:糯米丸子")]),
    Q("hop2", "小让的爷爷封印过哪个机器人？", "鲨鱼辣椒", ["鲨鱼辣椒"],
      path=[("高圆寺寅彦", "GRANDPARENT_OF", "小让"), ("鲨鱼辣椒", "SEALED_BY", "高圆寺寅彦")],
      note="剧情文档里只写'高圆寺博士', 必须和小让档案里的'高圆寺寅彦'对上"),
    Q("hop2", "被高圆寺博士封印的那个机器人用什么武器？", "鲨鱼神斧", ["鲨鱼神斧"],
      path=[("鲨鱼辣椒", "SEALED_BY", "高圆寺寅彦"), ("鲨鱼辣椒", "WEAPON", "lit:鲨鱼神斧")]),
    Q("hop2", "反派三人组老大的武器是什么？", "眼镜蛇尾鞭", ["眼镜蛇尾鞭"],
      path=[("蟑螂恶霸", "LEADER_OF", "反派三人组"),
            ("蟑螂恶霸", "WEAPON", "lit:眼镜蛇尾鞭")]),

    # ================= hop3: 三份文档 / 三条边 =================
    Q("hop3", "灰太狼表哥家的女儿叫什么？", "小香香", ["小香香"],
      path=[("夜太狼", "COUSIN_OF", "灰太狼"), ("香太狼", "SPOUSE_OF", "夜太狼"),
            ("香太狼", "PARENT_OF", "小香香")]),
    Q("hop3", "蕉太狼二叔的岳父是谁？", "爹爹狼", ["爹爹狼"],
      path=[("灰太狼", "UNCLE_OF", "蕉太狼"), ("红太狼", "SPOUSE_OF", "灰太狼"),
            ("爹爹狼", "PARENT_OF", "红太狼")]),
    Q("hop3", "懒羊羊的校长当家的村子坐落在哪里？", "青青草原", ["青青草原"],
      path=[("懒羊羊", "STUDENT_OF", "大肥羊学校"), ("慢羊羊", "HEAD_OF", "大肥羊学校"),
            ("慢羊羊", "HEAD_OF", "羊村"), ("羊村", "LOCATED_IN", "青青草原")]),
    Q("hop3", "卡布达搭档的爷爷封印过哪个机器人？", "鲨鱼辣椒", ["鲨鱼辣椒"],
      path=[("卡布达", "PARTNER_OF", "小让"), ("高圆寺寅彦", "GRANDPARENT_OF", "小让"),
            ("鲨鱼辣椒", "SEALED_BY", "高圆寺寅彦")]),
    Q("hop3", "小灰灰爷爷的妻子是谁？", "银太狼", ["银太狼"],
      path=[("灰太狼", "PARENT_OF", "小灰灰"), ("黑太狼", "PARENT_OF", "灰太狼"),
            ("黑太狼", "SPOUSE_OF", "银太狼")]),

    # ================= relation_path: 招牌题, 两端有名字, 中间全靠图 =================
    Q("relation_path", "小灰灰和小香香是什么关系？",
      "表亲：小灰灰的妈妈红太狼，是小香香妈妈香太狼的表姐", ["红太狼", "香太狼"],
      path=[("红太狼", "PARENT_OF", "小灰灰"), ("香太狼", "COUSIN_OF", "红太狼"),
            ("香太狼", "PARENT_OF", "小香香")],
      note="招牌案例: 三份证据里, 没有一份同时提到小灰灰和小香香"),
    Q("relation_path", "蕉太狼和爹爹狼有什么关系？",
      "蕉太狼的二叔灰太狼娶了红太狼，爹爹狼是红太狼的父亲", ["灰太狼", "红太狼"],
      path=[("灰太狼", "UNCLE_OF", "蕉太狼"), ("红太狼", "SPOUSE_OF", "灰太狼"),
            ("爹爹狼", "PARENT_OF", "红太狼")]),
    Q("relation_path", "黑太狼和爹爹狼是什么关系？",
      "儿女亲家：黑太狼的儿子灰太狼娶了爹爹狼的女儿红太狼", ["灰太狼", "红太狼"],
      path=[("黑太狼", "PARENT_OF", "灰太狼"), ("红太狼", "SPOUSE_OF", "灰太狼"),
            ("爹爹狼", "PARENT_OF", "红太狼")]),
    Q("relation_path", "小让和鲨鱼辣椒有什么关系？",
      "小让的爷爷高圆寺寅彦（高圆寺博士）封印过鲨鱼辣椒", ["高圆寺"],
      path=[("高圆寺寅彦", "GRANDPARENT_OF", "小让"), ("鲨鱼辣椒", "SEALED_BY", "高圆寺寅彦")],
      note="剧情文档只写'高圆寺博士', 小让档案只写'高圆寺寅彦'"),
    Q("relation_path", "藏之助和卡布达之间有什么联系？",
      "藏之助的搭档金龟次郎与卡布达同属正义阵营", ["金龟次郎"],
      path=[("金龟次郎", "PARTNER_OF", "吉祥寺藏之助"), ("金龟次郎", "FACTION", "lit:正义"),
            ("卡布达", "FACTION", "lit:正义")],
      note="字面量不得作为中转: 图检索只能走到金龟次郎, '同属正义阵营'要靠阵营一览这份文档"),

    # ================= aggregation: 列举 / 比较, 证据分散 =================
    Q("aggregation", "羊村里住着哪些角色？", "喜羊羊、美羊羊、懒羊羊、沸羊羊、暖羊羊、慢羊羊",
      ["喜羊羊", "美羊羊", "懒羊羊", "沸羊羊", "暖羊羊", "慢羊羊"],
      path=[(n, "LIVES_IN", "羊村")
            for n in ["喜羊羊", "美羊羊", "懒羊羊", "沸羊羊", "暖羊羊", "慢羊羊"]]),
    Q("aggregation", "狼堡里住着哪几只狼？", "灰太狼、红太狼、小灰灰",
      ["灰太狼", "红太狼", "小灰灰"],
      path=[(n, "LIVES_IN", "狼堡") for n in ["灰太狼", "红太狼", "小灰灰"]]),
    Q("aggregation", "反派三人组是由哪三个机器人组成的？", "蟑螂恶霸、蜘蛛侦探、蝎子莱莱",
      ["蟑螂恶霸", "蜘蛛侦探", "蝎子莱莱"],
      path=[(n, "MEMBER_OF", "反派三人组") for n in ["蟑螂恶霸", "蜘蛛侦探", "蝎子莱莱"]]),
    Q("aggregation", "正义阵营有哪些机器人？", "卡布达、金龟次郎、飞翔机器人、呱呱蛙、田德莉娜",
      ["卡布达", "金龟次郎", "飞翔机器人", "呱呱蛙", "田德莉娜"],
      path=[(n, "FACTION", "lit:正义")
            for n in ["卡布达", "金龟次郎", "飞翔机器人", "呱呱蛙", "田德莉娜"]]),
    Q("aggregation", "铁甲小宝里带编号的 B 机器人一共有几台？", "10 台（1 号机到 10 号机）",
      ["10"],
      path=[(n, "UNIT_NO", f"lit:{i} 号机") for i, n in enumerate(
          ["卡布达", "金龟次郎", "飞翔机器人", "蝎子莱莱", "蜘蛛侦探", "丸子龙",
           "呱呱蛙", "蟑螂恶霸", "鲨鱼辣椒", "田德莉娜"], 1)],
      note="证据是 10 份不同的档案, top-k 检索很难凑齐"),
    Q("aggregation", "卡布达、金龟次郎、飞翔机器人、鲨鱼辣椒里，谁的超级变换形态维持得最久？",
      "鲨鱼辣椒（15 分钟）", ["鲨鱼辣椒"],
      path=[("卡布达", "TRANSFORM_TIME", "lit:3 分钟"),
            ("金龟次郎", "TRANSFORM_TIME", "lit:5 分钟"),
            ("飞翔机器人", "TRANSFORM_TIME", "lit:7 分钟"),
            ("鲨鱼辣椒", "TRANSFORM_TIME", "lit:15 分钟")]),
    Q("aggregation", "两部剧里，哪两个角色最爱吃水果？", "蕉太狼爱吃香蕉，卡布达爱吃西瓜",
      ["蕉太狼", "卡布达"],
      path=[("蕉太狼", "FAVORITE_FOOD", "lit:香蕉"), ("卡布达", "FAVORITE_FOOD", "lit:西瓜")],
      note="跨节目"),
    Q("aggregation", "《铁甲小宝》比《喜羊羊与灰太狼》早几年首播？", "早 8 年（1997 年与 2005 年）",
      ["8"],
      path=[("铁甲小宝", "FIRST_AIRED", "lit:1997 年 2 月 23 日"),
            ("喜羊羊与灰太狼", "FIRST_AIRED", "lit:2005 年 8 月 3 日")],
      derived=True, note="跨节目, 需要算术"),
    Q("aggregation", "大肥羊学校有哪些学生？", "喜羊羊、美羊羊、懒羊羊、沸羊羊、暖羊羊",
      ["喜羊羊", "美羊羊", "懒羊羊", "沸羊羊", "暖羊羊"],
      path=[(n, "STUDENT_OF", "大肥羊学校")
            for n in ["喜羊羊", "美羊羊", "懒羊羊", "沸羊羊", "暖羊羊"]]),

    # ================= negative: 语料里没有, 正确行为是拒答 =================
    Q("negative", "懒羊羊的生日是哪一天？", "资料中未提及", [],
      trap=["生日"], note="语料里只有灰太狼的生日 —— 同类信息是强干扰项"),
    Q("negative", "美羊羊的口头禅是什么？", "资料中未提及", [],
      trap=["口头禅"], note="好几个角色都有口头禅, 唯独美羊羊没有"),
    Q("negative", "红太狼最喜欢什么颜色？", "资料中未提及", [], trap=["颜色"]),
    Q("negative", "灰太狼一共抓到过多少只羊？", "资料中未提及", [], trap=["抓到过"]),
    Q("negative", "喜羊羊的妹妹叫什么名字？", "资料中未提及", [], trap=["妹妹"],
      note="现实中确有其人, 但不在语料里 —— 考的是'只依据资料'"),
    Q("negative", "卡布达有多重？", "资料中未提及", [], trap=["重量", "公斤"],
      note="语料里只有卡布达巨人的身高, 是强干扰项"),
    Q("negative", "中国大陆版里是谁给卡布达配的音？", "资料中未提及", [], trap=["配音"]),
    Q("negative", "鲨鱼辣椒最爱吃什么？", "资料中未提及", [], trap=[],
      note="爱吃糯米丸子的是丸子龙, 极易张冠李戴"),
    Q("negative", "小让的妈妈叫什么名字？", "资料中未提及", [], trap=["妈妈"]),
    Q("negative", "飞翔机器人的原型是什么？", "资料中未提及", [], trap=[],
      note="其余 B 机器人都写了原型, 唯独它没有 —— 模型很容易照着名字猜'鸟'"),
    Q("negative", "蜻蜓队长是几号机？", "资料中未提及", [], trap=[],
      note="其余机器人大多有编号, 蜻蜓队长没有"),
    Q("negative", "《铁甲小宝》在中国大陆是哪一年开始播出的？", "资料中未提及", [],
      trap=["中国大陆首播", "引进"]),
]


# ==========================================================================

class EvalBuilder:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.doc_keys: dict[str, str] = kg.meta.get("doc_keys", {})
        self.items: list[dict] = []
        self._seen: set[str] = set()

    def _eid(self, name: str) -> str:
        eid = self.kg.resolve(name)
        if eid is None:
            raise SystemExit(f"[错误] 题库引用了图中不存在的实体「{name}」")
        return eid

    def edge_docs(self, s: str, rel: str, o: str) -> list[str]:
        src = self._eid(s)
        dst = o if o.startswith("lit:") else self._eid(o)
        for e in self.kg.out(src, rel):
            if e.dst == dst:
                return e.all_docs()
        raise SystemExit(f"[错误] 题库引用了图中不存在的边: {s} -[{rel}]-> {o}")

    def path_str(self, path: list[tuple[str, str, str]]) -> str:
        return "  ;  ".join(f"{s} -[{r}]-> {o.removeprefix('lit:')}" for s, r, o in path)

    def add(self, spec: dict) -> None:
        q = spec["question"]
        if q in self._seen:
            raise SystemExit(f"[错误] 重复题目: {q}")
        self._seen.add(q)
        docs: list[str] = []
        for s, r, o in spec["path"]:
            docs += self.edge_docs(s, r, o)
        for key in spec["docs"]:
            if key not in self.doc_keys:
                raise SystemExit(f"[错误] 题库引用了不存在的文档键 {key}")
            docs.append(self.doc_keys[key])
        t = spec["type"]
        self.items.append({
            "id": f"q-{len(self.items) + 1:04d}",
            "type": t,
            "question": q,
            "gold_answer": spec["answer"],
            "gold_entities": spec["ents"],
            "gold_docs": sorted(set(docs)),
            "hops": HOPS.get(t, 0 if t == "negative" else 1),
            "requires": REQUIRES[t],
            "expect_refusal": t == "negative",
            "gold_path": self.path_str(spec["path"]) if t in HOPS else "",
            "derived": spec["derived"],
            "trap_keywords": spec["trap"],
            "note": spec["note"],
            "split": assign_split(q),
        })


def main() -> None:
    ap = argparse.ArgumentParser(description="从题库 + ground-truth 图谱生成评测集")
    ap.add_argument("--graph", type=Path, default=ROOT / "data/synthetic/graph.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval/eval_set.jsonl")
    args = ap.parse_args()

    kg = KnowledgeGraph.load(args.graph)
    b = EvalBuilder(kg)
    for spec in QUESTIONS:
        b.add(spec)

    write_jsonl(args.out, b.items)

    by_type: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_req: dict[str, int] = {}
    for it in b.items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
        by_split[it["split"]] = by_split.get(it["split"], 0) + 1
        by_req[it["requires"]] = by_req.get(it["requires"], 0) + 1

    print(f"[ok] 评测集 {len(b.items)} 题 -> {args.out}")
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
