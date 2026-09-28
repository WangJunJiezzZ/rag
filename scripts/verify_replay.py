"""
离线回放自检 —— 演示前必跑。

为什么需要这个脚本
------------------
"用真实 provider 录制、用离线模式回放"这条路径曾经是坏的, 而且**一直没被发现**:
之前只在缓存为空时测过 replay(看它是否正确报错), 从没测过缓存命中的情况。
**只测失败路径、不测成功路径**, 是很典型的测试盲区。

根因是建模错误: 把 replay 当成第四个 provider, 它的能力位与真实 provider 不同,
导致上层构造的请求体不同, 缓存键对不上。现已重构为
"provider(谁来答) + offline(能不能联网)"两个正交维度。

这个脚本把成功路径固化成检查项: 强制离线, 逐题验证缓存命中。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import read_jsonl, setup_console    # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--all", action="store_true", help="验证全部题目，而非每类一道")
    args = ap.parse_args()

    os.environ["GRAPHRAG_OFFLINE"] = "1"      # 强制离线，绝不发起网络调用
    from graphrag.llm import LLM, CacheMissError
    from graphrag.service import RAGService, ServiceConfig

    llm = LLM()
    print(f"provider = {llm.provider_name}   offline = {llm.offline}")
    print(f"缓存 {sum(1 for _ in llm.cache.root.rglob('*.json'))} 条\n")

    items = [i for i in read_jsonl(ROOT / "data/eval/eval_set.jsonl")
             if i["split"] == args.split]
    if not args.all:
        seen, picked = set(), []
        for it in items:
            if it["type"] not in seen:
                seen.add(it["type"])
                picked.append(it)
        items = picked

    svc = RAGService(ServiceConfig(), llm=llm, quiet=True)
    hit = miss = 0
    for it in items:
        try:
            a = svc.ask(it["question"])
            hit += 1
            if not args.all:
                print(f"  [ok] {it['type']:<16} {len(a.text):>4}字  "
                      f"引用{len(a.citations)}条  "
                      f"{'拒答' if a.refused else '作答'}")
        except CacheMissError:
            miss += 1
            print(f"  [--] {it['type']:<16} {it['question'][:34]}... 未命中")

    total = hit + miss
    print(f"\n{hit}/{total} 命中"
          + ("   [PASS] 离线演示可用" if miss == 0
             else f"\n[FAIL] {miss} 题未命中 —— 演示时这些题答案区会是空的。\n"
                  f"       先跑: GRAPHRAG_LLM={llm.provider_name} "
                  f"python run.py e2e -- --ab --split {args.split}"))
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
