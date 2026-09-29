"""
DeepSeek 驱动的 MCP Client(Host) —— 看清 MCP 与 function calling 各管哪一层。

    ┌─────────── Host(本文件) ───────────┐         ┌──── MCP Server ────┐
    │  DeepSeek  <── function calling ──>  翻译层  <── MCP(JSON-RPC) ──>  tools/resources │
    └────────────────────────────────────┘         └────────────────────┘

  · function calling 是**模型层**能力: 模型输出"我想调 f(x)", 但它不知道 f 在哪、怎么调
  · MCP 是**工具接入层**协议: 规定工具如何被发现(list_tools)、描述(JSON Schema)、
    调用(call_tool)、返回(content 块)。与模型无关 —— 同一个 Server,
    Claude Desktop、Cursor、这个 DeepSeek 客户端都能直接用
  · Host 负责两边翻译: MCP Tool -> OpenAI tools 格式; tool_calls -> call_tool; 结果 -> role=tool 消息

没有 MCP 时, 每接一个新工具都要在每个应用里各写一遍 schema 和调用代码(M 个应用 × N 个工具);
有了 MCP, 工具方写一次 Server, 应用方写一次 Client, 变成 M + N。

用法
----
  python run.py mcp-chat                          交互式, 自动拉起本地 stdio Server
  python run.py mcp-chat -- -q "小灰灰和小香香是什么关系？"
  python run.py mcp-chat -- --url http://127.0.0.1:8765/mcp   连接已启动的 HTTP Server
  python run.py mcp-chat -- --graph data/synthetic/graph.json -q "..."   换成标准图做对照
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anyio                                            # noqa: E402  (mcp 的依赖)
from mcp import Client, StdioServerParameters          # noqa: E402
from mcp.types import Tool                             # noqa: E402

from graphrag.compat import setup_console              # noqa: E402
from graphrag.llm import LLM, CacheMissError           # noqa: E402

SYSTEM = """你是一名动画角色关系调查助手, 通过工具查询知识图谱与来源文档来回答问题。
规则:
- 只依据工具返回的内容作答; 工具没查到的, 明确说"未找到依据", 不要编造,
  也不要拿你自己对这部动画的记忆来补
- 关系类问题优先用图谱工具; 描述性问题用 search_text
- 答案末尾列出引用的文档 id, 格式: 依据: doc-xxxx, doc-yyyy

图谱 schema:
{schema}"""

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"


def mcp_tool_to_openai(t: Tool) -> dict:
    """MCP Tool -> OpenAI/DeepSeek function 定义。

    两边都用 JSON Schema 描述参数, 所以翻译几乎是改个字段名 ——
    这也是 MCP 能做到模型无关的原因: 它复用了各家 function calling 的公约数。
    """
    return {"type": "function",
            "function": {"name": t.name, "description": t.description or "",
                         "parameters": t.input_schema}}


@dataclass
class Step:
    tool: str
    arguments: dict
    result_chars: int
    is_error: bool
    result: str = ""            # 原样的工具输出 —— 回放页面要逐步展示
    round: int = 0              # 第几轮模型调用发起的; 同一轮可能并行调多个工具


@dataclass
class Turn:
    answer: str
    steps: list[Step] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stopped: str = ""           # "final" | "max_steps"


class MCPChat:
    def __init__(self, client: Client, llm: LLM, max_steps: int = 8,
                 verbose: bool = True):
        self.client = client
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose
        self.tools: list[dict] = []
        self.system = ""

    async def setup(self) -> None:
        listed = await self.client.list_tools()
        self.tools = [mcp_tool_to_openai(t) for t in listed.tools]
        # Resource 由 Host 决定何时读: 这里在对话开始前读一次 schema 塞进 system prompt,
        # 省掉模型"先调个工具问问有哪些关系"的一轮往返。
        schema = await self.client.read_resource("graphrag://schema")
        self.system = SYSTEM.format(schema=schema.contents[0].text)
        if self.verbose:
            print(f"[mcp] 已连接, 发现 {len(self.tools)} 个工具: "
                  + ", ".join(t["function"]["name"] for t in self.tools), file=sys.stderr)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    async def ask(self, question: str, history: list[dict] | None = None) -> Turn:
        messages = [{"role": "system", "content": self.system},
                    *(history or []), {"role": "user", "content": question}]
        turn = Turn(answer="")
        for rnd in range(1, self.max_steps + 1):
            # LLM.chat 是同步 HTTP; 放到线程里跑, 不阻塞 MCP 会话的事件循环
            res = await anyio.to_thread.run_sync(
                lambda: self.llm.chat(messages=messages, tools=self.tools))
            turn.llm_calls += 1
            turn.input_tokens += res.input_tokens
            turn.output_tokens += res.output_tokens

            if not res.tool_calls:
                turn.answer, turn.stopped = res.text, "final"
                break

            # assistant 消息必须原样带上 tool_calls, 后面的 tool 消息靠 id 与之配对
            messages.append({"role": "assistant", "content": res.text or "",
                             "tool_calls": res.tool_calls})
            for tc in res.tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError as e:
                    # 模型偶尔输出坏 JSON; 把错误作为工具结果回给它, 让它自己改, 不要崩
                    text, is_err, args = f"参数不是合法 JSON: {e}", True, {}
                else:
                    r = await self.client.call_tool(name, args)
                    text = "\n".join(getattr(c, "text", "") for c in r.content)
                    is_err = bool(r.is_error)
                turn.steps.append(Step(name, args, len(text), is_err, text, rnd))
                self._log(f"  -> {name}({json.dumps(args, ensure_ascii=False)})"
                          f"  {'[error] ' if is_err else ''}{len(text)} 字符")
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": text})
        else:
            turn.stopped = "max_steps"
            turn.answer = f"(已达到最大步数 {self.max_steps}, 未能给出最终答案)"
        return turn


async def _main(args: argparse.Namespace) -> int:
    llm = LLM(provider="deepseek")
    if llm.offline:
        # 离线有两种来源, 提示必须说清是哪一种 —— 否则 key 明明配好了,
        # 用户会被一句"未设置 key"带去查错方向
        if not os.environ.get("DEEPSEEK_API_KEY"):
            why = "未设置 DEEPSEEK_API_KEY(写进仓库根目录的 .env)"
        else:
            why = (f"GRAPHRAG_OFFLINE={os.environ.get('GRAPHRAG_OFFLINE', '')!r} 强制离线; "
                   "要调用 DeepSeek 请在 .env 里删掉这一行")
        print(f"[warn] 离线模式: {why}。只能回放已缓存的对话, "
              f"新问题会报缓存未命中。", file=sys.stderr)

    server_args = [str(SERVER_SCRIPT)] + (["--graph", args.graph] if args.graph else [])
    target = (args.url if args.url else
              StdioServerParameters(command=sys.executable, args=server_args))
    async with Client(target) as client:
        chat = MCPChat(client, llm, max_steps=args.max_steps)
        await chat.setup()

        async def one(q: str, history: list[dict]) -> None:
            try:
                turn = await chat.ask(q, history)
            except CacheMissError as e:
                print(f"[FAIL] {e}", file=sys.stderr)
                return
            print(f"\n{turn.answer}\n")
            print(f"[{len(turn.steps)} 次工具调用 · {turn.llm_calls} 次模型调用 · "
                  f"{turn.input_tokens}+{turn.output_tokens} tokens · {turn.stopped}]",
                  file=sys.stderr)
            history += [{"role": "user", "content": q},
                        {"role": "assistant", "content": turn.answer}]

        history: list[dict] = []
        if args.question:
            await one(args.question, history)
        else:
            print("输入问题, 空行退出。", file=sys.stderr)
            while True:
                try:
                    q = input("> ").strip()
                except EOFError:
                    break
                if not q:
                    break
                await one(q, history)
    print(llm.report(), file=sys.stderr)
    return 0


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="DeepSeek 驱动的 MCP 客户端")
    ap.add_argument("-q", "--question", help="只问一个问题后退出")
    ap.add_argument("--url", help="连接 Streamable HTTP Server; 不填则拉起本地 stdio Server")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--graph", help="透传给本地 Server 的图谱文件; "
                                    "data/synthetic/graph.json 即标准(oracle)图, 用于区分数据错误与模型错误")
    return anyio.run(_main, ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
