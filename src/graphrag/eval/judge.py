"""
LLM 判官 —— 关系类题的最终结论判分。

为什么关键词判分对关系类题不够
------------------------------
关键词判分检查"标准答案里的实体是否出现在回答中"。对单跳事实题够用,
对关系类题会系统性放水: 回答把中间人的名字都列了出来, 结论却是"无法确定",
照样判对。Agent 评测时人工核对 41 条多跳回答, 发现 3 例:
  q-0062  写了"按此路径应为娘娘狼", 结论却是"无法回答"
  q-0087  两个关键词都在, 结论是"无法断定二者存在何种关系"
  q-0088  结论是"没有直接关系", 漏了"同属正义阵营"
加关键词治标不治本 —— q-0087 在收紧关键词之后仍然被判对。

作用范围
--------
只接管 JUDGE_TYPES 里的可答题, 其余题型仍用关键词。判官与关键词的结果都保留在
行里(kw_correct / correct), 报告里可以同时看到两种口径。

判官本身也要被评测
------------------
  python run.py calibrate-judge   与标注集(data/eval/judge_labels.jsonl)比对一致率
判官的输出进缓存 —— 同一条回答永远得到同一个判定, 评测可复现。
"""
from __future__ import annotations

import json
import re

from ..llm import LLM
from ..prompts import registry

JUDGE_TYPES = {"hop2", "hop3", "relation_path"}
DEFAULT_PROMPT = "judge/v2_rubric"   # v1 在留出集上误伤 1 例, 见 prompts/judge/v2_rubric.md

_VERDICT = re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"')


class Judge:
    def __init__(self, llm: LLM, prompt_id: str = DEFAULT_PROMPT):
        self.llm = llm
        self.prompt = registry.get(prompt_id)
        self.prompt_id = prompt_id
        self.calls = 0

    def applies(self, item: dict) -> bool:
        return item["type"] in JUDGE_TYPES and not item["expect_refusal"]

    def judge(self, item: dict, answer: str) -> dict:
        system, user = self.prompt.render(question=item["question"],
                                          gold=item["gold_answer"],
                                          answer=answer.strip())
        res = self.llm.complete(system=system, user=user, max_tokens=300,
                                schema={"verdict": "string", "reason": "string"},
                                cache_tag=self.prompt_id)
        self.calls += 1
        data = res.parsed if isinstance(res.parsed, dict) else None
        if data is None:
            # JSON mode 偶尔吐出带前后缀的文本; 抠出 verdict 字段兜底
            m = _VERDICT.search(res.text or "")
            data = {"verdict": m.group(1), "reason": res.text.strip()[:200]} if m else None
        if not data or data.get("verdict") not in ("correct", "incorrect"):
            return {"ok": False, "correct": float("nan"), "reason": (res.text or "")[:200],
                    "input_tokens": res.input_tokens, "output_tokens": res.output_tokens}
        return {"ok": True, "correct": 1.0 if data["verdict"] == "correct" else 0.0,
                "reason": str(data.get("reason", ""))[:300],
                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens}

    def rescore(self, row: dict, item: dict, answer: str) -> dict:
        """用判官结论覆盖关键词结论; 两种口径都留在行里。判官失败时保留关键词结果。"""
        if not self.applies(item):
            return row
        v = self.judge(item, answer)
        row["kw_correct"] = row["correct"]
        row["judge_reason"] = v["reason"]
        row["judge_ok"] = v["ok"]
        if v["ok"]:
            row["correct"] = v["correct"]
            row["false_refusal"] = 1.0 if row["refused"] and not v["correct"] else 0.0
        return row


def load_labels(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
