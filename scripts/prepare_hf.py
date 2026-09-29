"""
准备 HuggingFace Space 部署目录。

为什么不直接把主仓库推到 HF
---------------------------
  1. HF 靠 README.md 顶部的 YAML front-matter 识别 Space 配置(sdk/端口)。
     主仓库的 README 是给人看的, 加上那段 YAML 后 GitHub 会把它渲染成一张表, 很丑。
  2. HF 要求 Dockerfile 在仓库根目录, 而本项目放在 deploy/ 下。
  3. 评测脚本、报告、.venv 这些部署用不上, 白白撑大镜像。

所以生成一个干净的部署目录, 推那个。主仓库保持原样。

用法
----
    python run.py prepare-hf
    cd build/hf-space
    git init && git add -A && git commit -m "Deploy"
    git remote add origin https://huggingface.co/spaces/<用户名>/<space名>
    git push -u origin main
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, write_text   # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]

# 要带进 Space 的内容。刻意不含 .venv / .env / .git / 评测脚本
INCLUDE_DIRS = [
    "src",           # 核心代码
    "web",           # 演示页面
    "prompts",       # 运行时要读 answer prompt
    "models",        # 向量模型 23MB
    "data/synthetic",
    "data/eval",
    "data/index",
    "data/llm_cache",   # 离线回放的燃料
    "docs",             # 顺便让访客能读到文档
    "reports",          # 评测报告页
]
INCLUDE_FILES = [
    "scripts/serve.py",
    "scripts/verify_replay.py",
    "deploy/Dockerfile",        # -> 落到根目录
    "deploy/requirements.txt",
    ".dockerignore",
]

FRONT_MATTER = """---
title: GraphRAG Lab
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 可评测、可消融的 GraphRAG 检索增强系统（合成数据演示）
---
"""

INTRO = """
# GraphRAG Lab

一个**可评测、可消融**的 GraphRAG 检索增强系统。

> ⚠️ **全部语料为程序合成。** 机构名、基金名、人名、管辖区名及其风险等级
> 均属虚构，不指涉任何真实企业、个人或司法管辖区，不构成任何投资、法律或合规建议。

## 这个 Demo 要证明什么

> 纯向量 RAG 在什么情况下会失败？加上什么能把分数拉回来？每一步各值多少分？

检索 recall@10：**BM25 65.2% → +向量 70.5% → +图检索与路由 86.3%**

## 怎么玩

页面上有四个示例，按难度递增：

| # | 类型 | 看点 |
|---|---|---|
| ① | 单跳事实 | 点答案里的蓝色 `[n]` 可跳转回原文 |
| ② | 纯语义 | 问题里没有实体名，纯靠向量召回 |
| ③ | **四跳关系穿透** | 会画出关系链——**答案在任何单一文档里都不存在** |
| ④ | 幻觉陷阱 | 语料里没有，正确行为是拒答 |

**最值得试的**：问完 ③ 之后，把「图谱来源」从「抽取图」切到「标准图」：

```
抽取图：  基金 → 新港城                                （2 节点，链断了）
标准图：  基金 → 管理人 → 董事 → 关联公司 → 高风险辖区    （5 节点，完整）
```

这个差距就是 **LLM 抽取环节的损失**：检索层面差 4.8 分，端到端答对率差 **24.6 分**。

## 关于这个 Space

服务端**完全离线**：不调用任何外部 API、不持有任何凭据，
全部回答来自预先录制的缓存（2198 条）。因此：

- 只能回答评测集 train 划分里的 39 道题（点示例按钮即可）
- 自己输入的其他问题会显示"缓存未命中"，但**检索结果和关系图仍会正常展示**

完整代码、评测框架与文档见仓库内 `docs/`。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/hf-space")
    ap.add_argument("--clean", action="store_true", help="先清空目标目录")
    args = ap.parse_args()

    dest = ROOT / args.out
    if args.clean and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    missing = []
    for d in INCLUDE_DIRS:
        src = ROOT / d
        if not src.exists():
            missing.append(d)
            continue
        tgt = dest / d
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(src, tgt,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    for f in INCLUDE_FILES:
        src = ROOT / f
        if not src.exists():
            missing.append(f)
            continue
        # deploy/ 下的文件要落到根目录 —— HF 要求 Dockerfile 在根
        rel = Path(f).name if f.startswith("deploy/") else f
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)

    write_text(dest / "README.md", FRONT_MATTER + INTRO)
    write_text(dest / ".gitignore", "__pycache__/\n*.py[cod]\n.DS_Store\n.env\n")

    # ---- 体检 ----
    files = [p for p in dest.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    big = sorted(files, key=lambda p: -p.stat().st_size)[:3]

    print(f"[ok] 部署目录已生成 -> {dest}")
    print(f"     {len(files)} 个文件, 合计 {total/1024/1024:.1f} MB")
    for p in big:
        print(f"       {p.relative_to(dest)}  {p.stat().st_size/1024/1024:.1f} MB")
    if missing:
        print(f"[warn] 以下内容不存在，已跳过: {', '.join(missing)}")

    # 安全检查: 绝不能带上凭据
    leaked = [p for p in files if p.name in (".env",) or p.suffix in (".pem", ".key")]
    if leaked:
        print(f"[FAIL] 部署目录里出现凭据文件: {leaked}")
        return 1
    print("[ok] 未发现任何凭据文件")

    print(f"""
下一步（在 HuggingFace 网页上先建好 Space，SDK 选 Docker）：

  cd {dest.relative_to(ROOT)}
  git init -b main
  git add -A
  git commit -m "Deploy GraphRAG Lab"
  git remote add origin https://huggingface.co/spaces/<你的用户名>/<space名>
  git push -u origin main

详细步骤见 docs/05-部署说明.md""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
