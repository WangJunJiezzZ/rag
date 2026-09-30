"""
GraphRAG MCP Server —— 把知识图谱 + 文档检索暴露成标准 MCP 服务。

MCP 的三种原语, 这里各用一次
----------------------------
  Tools      模型**主动调用**的函数, 有副作用或需要参数   -> 查实体/邻居/路径/检索/读原文/ask
  Resources  客户端**按需读取**的只读上下文, 由 URI 定位  -> 图谱 schema、实体清单
  Prompts    服务端提供的**可复用提示模板**, 由用户选用  -> 角色关系调查

选哪种原语的判断标准是"谁来决定何时使用":
模型决定 -> tool; 应用/客户端决定 -> resource; 用户决定 -> prompt。
schema 放在 resource 而不是 tool, 是因为客户端可以在对话开始前一次性读进来,
不必让模型花一轮工具调用去"问自己有哪些关系"。

传输
----
  python run.py mcp-server                     stdio (默认; 被客户端作为子进程拉起)
  python run.py mcp-server -- --http --port 8765   Streamable HTTP (可被远程客户端连接)

stdio 下 **stdout 是协议通道**: 任何 print 都会混进 JSON-RPC 流, 客户端解析失败。
mcp>=2 会把 fd 1 转到 stderr 兜底; tools.py 在初始化期间也显式重定向。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated

if __package__ in (None, ""):          # 允许 `python src/graphrag/mcp_app/server.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.mcpserver import MCPServer          # noqa: E402
from mcp.types import ToolAnnotations                # noqa: E402
from pydantic import Field                           # noqa: E402

from graphrag.mcp_app.tools import RELATIONS, GraphRAGTools   # noqa: E402
from graphrag.service import ServiceConfig                     # noqa: E402

INSTRUCTIONS = """\
这是《喜羊羊与灰太狼》《铁甲小宝》两部动画的角色关系知识图谱(节目/角色/地点/团体/道具)
及其来源文档的查询服务。语料根据维基百科等公开资料整理, 译名采用中国大陆播出版。
推荐用法:
  1. 先 find_entity 把问题里的名称解析成实体 id(支持又名、台湾译名)
  2. 关系类问题(父母、爷爷、表哥、搭档、A 和 B 是什么关系)用 get_neighbors / explain_relation
  3. 描述性问题(性格、剧情)或图里找不到的对象, 用 search_text
  4. 需要引用原文时用 read_document
每条图谱边都带 docs(来源文档 id), 回答时应引用它们。"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _j(obj) -> str:
    """工具结果以紧凑 JSON 文本返回。

    返回 dict 时 SDK 会按 indent=2 序列化。实测本项目 5 类工具结果, 缩进版
    字符数多 21%~70%(平均 44%; token 增幅小于字符增幅, 连续空格常被合并)。
    工具输出每一轮都会原样进入模型上下文, 是 Agent 成本的大头, 能省则省。
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


PROFILES = ("v1", "v2")


def build_server(tools: GraphRAGTools | None = None, profile: str = "v1") -> MCPServer:
    """profile 决定暴露哪些工具。

    v1  7 个工具, 与第一轮评测录制缓存时完全一致 —— 工具定义是缓存键的一部分,
        改一个字 v1 的全部离线回放都会失效, 所以 v1 冻结不动
    v2  v1 + list_by_relation(按关系 / 属性列举), 注册在最后, 见 docs/07-Agent.md
    """
    if profile not in PROFILES:
        raise ValueError(f"未知 profile {profile}, 可选 {PROFILES}")
    t = tools or GraphRAGTools()
    mcp = MCPServer(name="graphrag-lab", version="0.1.0", instructions=INSTRUCTIONS)

    # ------------------------------------------------------------------ tools
    # 描述文字就是模型选工具的唯一依据 —— 写"什么时候该用我", 而不只是"我做什么"。

    @mcp.tool(annotations=READ_ONLY)
    def find_entity(
        name: Annotated[str, Field(description="实体名称、又名或译名, 如「灰太狼」「蝎子蓝蓝」「高圆寺博士」")],
        limit: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> str:
        """按名称模糊查找图谱实体, 返回候选的 id、类型、别名。
        任何图查询之前先用它把名称解析成 id; 名称有歧义时会返回多个候选。"""
        return _j(t.find_entity(name, limit))

    @mcp.tool(annotations=READ_ONLY)
    def get_neighbors(
        entity: Annotated[str, Field(description="实体 id(推荐)或名称")],
        relation: Annotated[str | None, Field(
            description="只看某一种关系; 不填则返回全部。可选: " + ", ".join(RELATIONS))] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 30,
    ) -> str:
        """列出实体的一跳关系(出边与入边), 每条边带来源文档 id。
        适合: 角色的父母/配偶/搭档/原型/口头禅, 团体有哪些成员, 某地住着谁。
        注意方向: PARENT_OF 的出边指向子女, 入边来自父母。"""
        return _j(t.get_neighbors(entity, relation, limit))

    @mcp.tool(annotations=READ_ONLY)
    def find_paths(
        source: Annotated[str, Field(description="起点实体 id 或名称")],
        target: Annotated[str, Field(description="终点实体 id 或名称")],
        max_hops: Annotated[int, Field(ge=1, le=5)] = 4,
    ) -> str:
        """查找两个实体之间不超过 max_hops 跳的全部路径(包括经由节目等枢纽节点的路径)。
        回答「A 和 B 是什么关系」时优先用 explain_relation。"""
        return _j(t.find_paths(source, target, max_hops))

    @mcp.tool(annotations=READ_ONLY)
    def explain_relation(
        a: Annotated[str, Field(description="第一个角色的 id 或名称")],
        b: Annotated[str, Field(description="第二个角色的 id 或名称")],
        max_hops: Annotated[int, Field(ge=1, le=5)] = 4,
    ) -> str:
        """关系解释: 找出两个角色之间的亲属/搭档/封印等关系路径, 每一跳带来源文档。
        用于「A 和 B 是什么关系」「X 是 Y 的什么人」; 已屏蔽经由节目节点这类枢纽的假关联。"""
        return _j(t.explain_relation(a, b, max_hops))

    @mcp.tool(annotations=READ_ONLY)
    def search_text(
        query: Annotated[str, Field(description="自然语言查询, 用文档里可能出现的措辞效果更好")],
        top_k: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> str:
        """在全部文档中做关键词+语义混合检索, 返回片段摘要。
        适合: 性格特点、剧情经过、节目花絮, 以及图谱里查不到的对象。要全文请再调 read_document。"""
        return _j(t.search_text(query, top_k))

    @mcp.tool(annotations=READ_ONLY)
    def read_document(
        doc_id: Annotated[str, Field(description="文档 id, 形如 doc-0001")],
        max_chars: Annotated[int, Field(ge=200, le=6000)] = 1500,
    ) -> str:
        """读取一份来源文档的原文(可截断), 用于核实证据与引用。"""
        return _j(t.read_document(doc_id, max_chars))

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def ask(question: Annotated[str, Field(description="完整的自然语言问题")]) -> str:
        """一站式问答: 走内置的固定流水线(规则路由 -> 检索 -> LLM 生成), 返回带引用的答案。
        会产生一次 LLM 调用。想自己控制检索步骤时, 用上面的原子工具。"""
        return _j(t.ask(question))

    if profile == "v2":
        @mcp.tool(annotations=READ_ONLY)
        def list_by_relation(
            relation: Annotated[str, Field(description="关系名, 如 FACTION(阵营)、UNIT_NO(编号)、LIVES_IN(住在)。可选: "
                                                        + ", ".join(RELATIONS))],
            value: Annotated[str | None, Field(
                description="只看取值为它的边, 如「正义」「羊村」; 不填则列出该关系的全部边")] = None,
            limit: Annotated[int, Field(ge=1, le=100)] = 50,
        ) -> str:
            """按关系(和取值)一次列出所有匹配的角色。
            适合: 列举题(某阵营有哪些角色、谁住在某地、带编号的有几台), 以及比较两个角色是否有共同属性
            (是否同一阵营 / 同一团体)。不要为了列举而逐个调用 get_neighbors。"""
            return _j(t.list_by_relation(relation, value, limit))

    # -------------------------------------------------------------- resources
    @mcp.resource("graphrag://schema", name="schema", mime_type="application/json",
                  description="图谱的实体类型、关系含义与方向、规模统计")
    def schema() -> str:
        return json.dumps(t.schema(), ensure_ascii=False, indent=2)

    @mcp.resource("graphrag://entities/{type_}", name="entities",
                  mime_type="application/json",
                  description="某类实体的清单; type_ 取 Show / Character / Place / Group / Item / all")
    def entities(type_: str) -> str:
        return json.dumps(t.entities(None if type_ == "all" else type_),
                          ensure_ascii=False)

    # ---------------------------------------------------------------- prompts
    @mcp.prompt(title="角色关系调查")
    def relation_investigation(a: str, b: str) -> str:
        """调查两个角色之间的关系, 输出带证据的结论。"""
        return (f"请调查「{a}」和「{b}」之间是什么关系:\n"
                "1. 用 find_entity 确认两个角色\n"
                "2. 用 explain_relation 找到两者之间的关系路径; 没有则明确说明\n"
                "3. 对每条路径, 用 read_document 核实关键一跳的原文\n"
                "4. 输出: 结论(用一句话说清是什么关系, 如'表姨''儿女亲家') + 每条路径 + 引用的文档 id。"
                "不要使用图谱与文档之外的信息, 包括你自己对这部动画的记忆。")

    return mcp


def main() -> None:
    ap = argparse.ArgumentParser(description="GraphRAG MCP Server")
    ap.add_argument("--http", action="store_true", help="用 Streamable HTTP 代替 stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--profile", default="v1", choices=PROFILES,
                    help="v1 = 7 个工具(冻结); v2 = 增加 list_by_relation")
    ap.add_argument("--graph", default=ServiceConfig.graph_path,
                    help="图谱文件; 换成 data/synthetic/graph.json 即 oracle 图")
    args = ap.parse_args()

    server = build_server(GraphRAGTools(ServiceConfig(graph_path=args.graph)), args.profile)
    if args.http:
        print(f"MCP (Streamable HTTP) -> http://{args.host}:{args.port}/mcp",
              file=sys.stderr)
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
