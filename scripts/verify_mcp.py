#!/usr/bin/env python3
"""
MCP 链路自检 —— 不需要任何 API key。

用一个"按剧本出牌"的假 LLM 代替 DeepSeek, 驱动真实的 Client 循环
走过真实的 MCP 协议(stdio 子进程), 验证:
  1. 协议层   工具/资源/prompt 能被发现, schema 翻译成 OpenAI 格式正确
  2. 循环层   tool_calls -> call_tool -> role=tool 消息 -> 最终答案, 消息配对正确
  3. 失败路径 坏 JSON 参数、越界参数、未知实体、超过最大步数 —— 都不崩, 都回给模型
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anyio                                                  # noqa: E402
from mcp import Client, StdioServerParameters                # noqa: E402

from graphrag.compat import setup_console                    # noqa: E402
from graphrag.llm import LLMResult                           # noqa: E402
from graphrag.mcp_app.client import SERVER_SCRIPT, MCPChat   # noqa: E402

setup_console()


def call(name: str, args: dict | str, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name,
                         "arguments": args if isinstance(args, str)
                         else json.dumps(args, ensure_ascii=False)}}


class ScriptedLLM:
    """每次 chat() 按顺序吐出预设回复, 同时记下收到的消息供断言。"""

    def __init__(self, script: list[LLMResult]):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def chat(self, *, messages, tools=None, max_tokens=2048):
        self.seen.append(json.loads(json.dumps(messages)))
        return self.script.pop(0) if self.script else LLMResult(text="(剧本用完)")


FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILS.append(msg)


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])
    async with Client(params) as client:
        print("1. 协议层")
        llm = ScriptedLLM([])
        chat = MCPChat(client, llm, verbose=False)
        await chat.setup()
        names = {t["function"]["name"] for t in chat.tools}
        check(names >= {"find_entity", "get_neighbors", "find_paths", "explain_relation",
                        "search_text", "read_document", "ask"}, f"发现 {len(names)} 个工具")
        fe = next(t for t in chat.tools if t["function"]["name"] == "find_entity")
        check(fe["function"]["parameters"].get("required") == ["name"],
              "MCP input_schema 原样翻译为 OpenAI parameters")
        check("PARENT_OF" in chat.system, "schema resource 已注入 system prompt")
        prompts = await client.list_prompts()
        check(any(p.name == "relation_investigation" for p in prompts.prompts), "prompt 可发现")

        print("2. 正常循环: 查实体 -> 关系解释 -> 作答")
        llm = ScriptedLLM([
            LLMResult(text="", tool_calls=[call("find_entity", {"name": "小灰灰"}, "c1")]),
            LLMResult(text="", tool_calls=[call("explain_relation",
                                                {"a": "小灰灰", "b": "小香香"}, "c2")]),
            LLMResult(text="两者是表亲。依据: doc-0012"),
        ])
        chat.llm = llm
        turn = await chat.ask("小灰灰和小香香是什么关系？")
        check(turn.stopped == "final" and [s.tool for s in turn.steps]
              == ["find_entity", "explain_relation"], "两次工具调用后给出最终答案")
        last = llm.seen[-1]
        check([m["role"] for m in last[-4:]] == ["assistant", "tool", "assistant", "tool"],
              "assistant(tool_calls) 与 tool 消息成对出现")
        check(last[-1]["tool_call_id"] == "c2", "tool 消息通过 tool_call_id 与调用配对")
        rel = json.loads(last[-1]["content"])
        check("paths" in rel and rel["paths"], "工具结果以 JSON 文本回传给模型, 且找到了路径")

        print("3. 失败路径")
        llm = ScriptedLLM([
            LLMResult(text="", tool_calls=[
                call("find_entity", "{not json", "b1"),                 # 坏 JSON
                call("find_entity", {"name": "x", "limit": 999}, "b2"),  # 越界
                call("get_neighbors", {"entity": "查无此人"}, "b3"),      # 未知实体
            ]),
            LLMResult(text="未找到依据。"),
        ])
        chat.llm = llm
        turn = await chat.ask("测试")
        errs = [s.is_error for s in turn.steps]
        check(errs[:2] == [True, True], "坏 JSON / 越界参数 -> is_error, 不崩溃")
        tool_msgs = [m for m in llm.seen[-1] if m["role"] == "tool"]
        check("find_entity" in tool_msgs[2]["content"] and "hint" in tool_msgs[2]["content"],
              "未知实体 -> 返回带 hint 的结构化提示, 引导模型换工具")

        llm = ScriptedLLM([LLMResult(text="", tool_calls=[
            call("find_entity", {"name": "太狼"}, f"l{i}")]) for i in range(10)])
        chat.llm, chat.max_steps = llm, 3
        turn = await chat.ask("死循环测试")
        check(turn.stopped == "max_steps" and len(turn.steps) == 3,
              "模型一直调工具 -> 在 max_steps 处截停")

    print("-" * 60)
    print("[PASS] MCP 链路自检通过" if not FAILS else f"[FAIL] {len(FAILS)} 项未通过")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
