#!/usr/bin/env python3
"""
graphrag-lab 任务入口 (跨平台)

不用 Makefile, 因为 Windows 默认没有 make。所有任务统一走:

    python run.py <task> [args...]

    python run.py phase0       生成语料 + 评测集 + 自检 (一条龙)
    python run.py gen          只生成合成语料
    python run.py eval-set     只生成评测集
    python run.py verify       只跑数据集自检
    python run.py retrieval    检索评测（切块策略对比 + 检索路径消融）
    python run.py build-graph  抽取知识图谱 + 抽取质量评测
    python run.py e2e          端到端评测 + prompt A/B
    python run.py verify-replay 离线回放自检（演示前必跑）
    python run.py serve        启动 Web 演示
    python run.py report       生成 HTML 评测报告
    python run.py all          全流程一条龙
    python run.py doctor       检查运行环境(Python 版本 / 依赖 / 编码)
    python run.py tasks        列出全部任务
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from graphrag.compat import setup_console  # noqa: E402

setup_console()

MIN_PY = (3, 10)

TASKS: dict[str, tuple[str, list[str]]] = {
    "gen":      ("生成合成语料与 ground-truth 图谱", ["scripts/gen_synthetic_data.py"]),
    "eval-set": ("从图谱生成评测集",                 ["scripts/gen_eval_set.py"]),
    "verify":   ("数据集自检",                       ["scripts/verify_dataset.py"]),
    "retrieval":   ("检索评测: 切块策略对比 + 检索路径消融", ["scripts/eval_retrieval.py"]),
    "build-graph": ("从文档抽取知识图谱并评测抽取质量", ["scripts/build_graph.py"]),
    "e2e":         ("端到端评测 + prompt A/B", ["scripts/eval_endtoend.py"]),
    "fetch-model": ("下载 ONNX 向量模型 (约 24MB)", ["scripts/fetch_model.py"]),
    "verify-replay": ("离线回放自检 —— **演示前必跑**", ["scripts/verify_replay.py"]),
    "prepare-hf":   ("生成 HuggingFace Space 部署目录", ["scripts/prepare_hf.py"]),
    "report":      ("汇总全部评测结果, 生成 HTML 报告", ["scripts/make_report.py"]),
    "serve":       ("启动 Web 演示 (默认 http://127.0.0.1:8000)", ["scripts/serve.py"]),
}

CHAINS: dict[str, tuple[str, list[str]]] = {
    "phase0": ("Phase 0 一条龙: 语料 -> 评测集 -> 自检", ["gen", "eval-set", "verify"]),
    "phase1": ("Phase 1: 检索评测", ["retrieval"]),
    "phase2": ("Phase 2: 抽取图谱 + 检索复评", ["build-graph", "retrieval"]),
    "all":    ("全流程: 数据 -> 图谱 -> 检索 -> 端到端 -> 报告",
               ["gen", "eval-set", "verify", "build-graph", "retrieval",
                "e2e", "report"]),
}


def run_script(rel: str, extra: list[str]) -> int:
    """用当前解释器执行脚本 —— 避免 Windows 上 'python' 指向别的版本。"""
    cmd = [sys.executable, str(ROOT / rel), *extra]
    print(f"\n$ {Path(sys.executable).name} {rel} {' '.join(extra)}".rstrip())
    return subprocess.call(cmd, cwd=str(ROOT))


def doctor() -> int:
    import platform
    print("运行环境自检")
    print("-" * 60)
    ok = True
    print(f"  Python      : {platform.python_version()}  ({sys.executable})")
    if sys.version_info < MIN_PY:
        print(f"    [FAIL] 需要 Python >= {MIN_PY[0]}.{MIN_PY[1]}")
        ok = False
    else:
        print("    [ok]")
    print(f"  平台        : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  stdout 编码 : {sys.stdout.encoding}")
    if (sys.stdout.encoding or "").lower().replace("-", "") not in ("utf8", "utf8mb4"):
        print("    [warn] 非 UTF-8; compat.setup_console() 已尝试纠正")
    else:
        print("    [ok] 中文输出安全")
    print("  中文输出测试: 星海亚洲机会基金 · 高风险管辖区 · ✓")

    print("\n  依赖 (Phase 0 全部为 Python 标准库, 无需安装):")
    for mod, need in [("json", True), ("sqlite3", True), ("hashlib", True),
                      ("numpy", False), ("anthropic", False),
                      ("sentence_transformers", False), ("fastapi", False)]:
        try:
            __import__(mod)
            print(f"    [ok]   {mod}")
        except ImportError:
            tag = "FAIL" if need else "未装"
            print(f"    [{tag}] {mod}" + ("" if need else "  (后续阶段才需要)"))
            if need:
                ok = False

    print("\n  凭据 (只显示是否存在，不打印内容):")
    import os
    env_file = ROOT / ".env"
    print(f"    [{'ok' if env_file.exists() else '--'}] .env 文件"
          + ("" if env_file.exists() else "  (可从 .env.example 复制)"))
    any_key = False
    for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        v = os.environ.get(name, "")
        if v:
            any_key = True
            masked = f"{v[:6]}…{v[-4:]}" if len(v) > 12 else "已设置"
            print(f"    [ok] {name} = {masked}  (长度 {len(v)})")
        else:
            print(f"    [--] {name} 未设置")
    provider = os.environ.get("GRAPHRAG_LLM", "claude(默认)")
    print(f"    GRAPHRAG_LLM = {provider}")
    if not any_key:
        print("    -> 无可用凭据，LLM 相关任务将自动降级为 replay / extractive")

    print("\n  数据文件:")
    for rel in ["data/synthetic/documents.jsonl", "data/synthetic/graph.json",
                "data/eval/eval_set.jsonl"]:
        p = ROOT / rel
        print(f"    [{'ok' if p.exists() else '--'}] {rel}"
              + (f"  ({p.stat().st_size:,} bytes)" if p.exists() else "  (尚未生成)"))
    print("-" * 60)
    print("[PASS] 环境可用" if ok else "[FAIL] 环境不满足要求")
    return 0 if ok else 1


def usage() -> None:
    print(__doc__.strip())
    print("\n可用任务:")
    for name, (desc, _) in TASKS.items():
        print(f"  {name:12s} {desc}")
    for name, (desc, steps) in CHAINS.items():
        print(f"  {name:12s} {desc}   [{' -> '.join(steps)}]")
    print(f"  {'doctor':12s} 检查运行环境")


def main() -> int:
    if sys.version_info < MIN_PY:
        print(f"[FAIL] 需要 Python >= {MIN_PY[0]}.{MIN_PY[1]}, "
              f"当前 {sys.version.split()[0]}")
        return 1
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help", "tasks"):
        usage()
        return 0
    task, extra = argv[0], argv[1:]
    # `python run.py <task> -- --flag` 里的 `--` 是给 run.py 看的分隔符,
    # 不该透传给子脚本(argparse 会报 unrecognized arguments)。
    # 两种写法都支持: 带 `--` 和不带。
    if extra and extra[0] == "--":
        extra = extra[1:]
    if task == "doctor":
        return doctor()
    if task in CHAINS:
        for step in CHAINS[task][1]:
            rc = run_script(TASKS[step][1][0], [])
            if rc != 0:
                print(f"\n[FAIL] 步骤 '{step}' 失败 (exit={rc})，已中止")
                return rc
        print("\n[PASS] " + CHAINS[task][0] + " 完成")
        return 0
    if task in TASKS:
        return run_script(TASKS[task][1][0], extra)
    print(f"[FAIL] 未知任务: {task}\n")
    usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
