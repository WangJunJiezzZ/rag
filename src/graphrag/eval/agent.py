"""
Agent 评测 —— 让模型自己编排工具, 和写死的固定流水线比, 到底好多少、贵多少。

对比的两方
----------
  pipeline  RAGService.ask: 规则路由 -> 一次性检索 top-k -> 一次生成
  agent     DeepSeek 通过 MCP 调 7 个原子工具, 自己决定查什么、查几步、何时停

公平性约束(不满足任何一条, 对比就没有意义)
------------------------------------------
  · 同一套判分: 直接复用 endtoend.score_answer —— 实体全中才算对, 别名感知,
    拒答用同一组关键词只看开头 160 字
  · 同一个模型: 两边都是 deepseek-chat, temperature=0
  · 同一份图谱: 由 --graph 指定, 两边一起换
  · 成本口径一致: 都按 token × 同一张价目表估算。缓存命中时实际花费为 0,
    但对比的是"这套方法要花多少", 所以一律按 token 折算

Agent 独有的指标
----------------
  tool_calls / llm_calls  每题调了几次工具、几轮模型
  max_steps               达到步数上限仍未作答的题数 —— 死循环/反复验证的信号
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..generate.answer import Citation, _looks_refused
from ..llm import PRICING, CacheMissError

DOC_ID = re.compile(r"doc-\d{4}")


def est_cost(tin: int, tout: int, model: str = "deepseek-chat") -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return (tin * pin + tout * pout) / 1e6


@dataclass
class AgentAnswer:
    """把 Agent 的一轮对话包装成 score_answer 认得的形状(duck typing Answer)。"""
    text: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


def to_answer(turn, known_docs: set[str], latency_ms: float) -> AgentAnswer:
    # Agent 按 system prompt 在末尾写"依据: doc-xxxx"; 引用有效 = 文档真实存在
    ids = list(dict.fromkeys(DOC_ID.findall(turn.answer)))
    cits = [Citation(index=i + 1, chunk_id="", doc_id=d, text="", valid=d in known_docs)
            for i, d in enumerate(ids)]
    # 步数用尽没给出答案, 按拒答计 —— 它确实没回答
    refused = turn.stopped == "max_steps" or _looks_refused(turn.answer)
    return AgentAnswer(text=turn.answer, refused=refused, citations=cits,
                       latency_ms=latency_ms,
                       cost_usd=est_cost(turn.input_tokens, turn.output_tokens))


async def run_agent(items: list[dict], graph_rel: str, llm, score, *,
                    concurrency: int = 4, max_steps: int = 8,
                    budget_usd: float | None = None, progress=None,
                    agent: str = "v1") -> dict:
    """在进程内 MCP 连接上并发跑 Agent。

    与 mcp-chat 走的是同一个 MCPChat + 同一个 Server, 所以请求体逐字节一致,
    之前手动跑过的题直接命中缓存。
    """
    import anyio
    from mcp import Client

    from ..mcp_app.client import AGENT_PROFILES, MCPChat
    from ..mcp_app.server import build_server
    from ..mcp_app.tools import GraphRAGTools
    from ..service import ServiceConfig

    tools = GraphRAGTools(ServiceConfig(graph_path=graph_rel))
    known = set(tools.svc.docs)
    rows: list[dict] = []
    state = {"spent": 0.0, "missed": 0, "skipped_budget": 0, "done": 0}
    limiter = anyio.Semaphore(concurrency)

    async with Client(build_server(tools, AGENT_PROFILES[agent]["server_profile"])) as c:
        chat = MCPChat(c, llm, max_steps=max_steps, verbose=False, profile=agent)
        await chat.setup()

        async def one(it: dict) -> None:
            async with limiter:
                # 预算在**发起前**检查: 已发出的题照常收尾, 不再开新题。
                # 必须按**实际 API 花费**(llm.total_cost, 缓存命中计 0)判断 ——
                # 首版按 token 估算值判断, 全部命中缓存时也会"超支", 把后面的题静默跳过。
                if budget_usd is not None and llm.total_cost >= budget_usd:
                    state["skipped_budget"] += 1
                    return
                t0 = time.perf_counter()
                try:
                    turn = await chat.ask(it["question"])
                except CacheMissError:
                    state["missed"] += 1
                    return
                ans = to_answer(turn, known, (time.perf_counter() - t0) * 1000)
                state["spent"] += ans.cost_usd
                row = score(it, ans)
                row.update({
                    "answer": turn.answer, "stopped": turn.stopped,
                    "tool_calls": len(turn.steps), "llm_calls": turn.llm_calls,
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "tools": [s.tool for s in turn.steps],
                    "tool_errors": sum(1 for s in turn.steps if s.is_error),
                })
                rows.append(row)
                state["done"] += 1
                if progress:
                    progress(state["done"], len(items), llm.total_cost)

        async with anyio.create_task_group() as tg:
            for it in items:
                tg.start_soon(one, it)

    order = {it["id"]: i for i, it in enumerate(items)}
    rows.sort(key=lambda r: order[r["id"]])
    return {"rows": rows, **state}


def run_pipeline(items: list[dict], graph_rel: str, score) -> dict:
    """固定流水线(与 eval_endtoend 同一个 RAGService), 顺带记下每题的 token。"""
    from ..service import RAGService, ServiceConfig

    svc = RAGService(ServiceConfig(graph_path=graph_rel), quiet=True)
    calls: list = []
    inner = svc.llm.complete

    def spy(**kw):
        res = inner(**kw)
        calls.append(res)
        return res

    svc.llm.complete = spy
    rows, missed = [], 0
    for it in items:
        calls.clear()
        try:
            ans = svc.ask(it["question"])
        except CacheMissError:
            missed += 1
            continue
        tin = sum(r.input_tokens for r in calls)
        tout = sum(r.output_tokens for r in calls)
        ans.cost_usd = est_cost(tin, tout)
        row = score(it, ans)
        row.update({"answer": ans.text, "llm_calls": len(calls), "tool_calls": 0,
                    "input_tokens": tin, "output_tokens": tout})
        rows.append(row)
    return {"rows": rows, "missed": missed}
