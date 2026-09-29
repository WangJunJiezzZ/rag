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
short_description: 可评测、可消融的 GraphRAG（喜羊羊 × 铁甲小宝 角色关系）
---
"""

INTRO = """
# GraphRAG Lab

一个**可评测、可消融**的 GraphRAG 检索增强系统。题材是《喜羊羊与灰太狼》和《铁甲小宝》的角色关系。

> 事实取自维基百科等公开资料，译名采用中国大陆播出版；文档由程序按模板渲染，措辞不是原文引用。
> 角色及形象权利归原著作权方所有，本 Space 仅作检索技术演示。

## 这个 Demo 要证明什么

> 纯向量 RAG 在什么情况下会失败？加上什么能把分数拉回来？每一步各值多少分？

检索 recall@10：**BM25 80.9% → +向量 87.0% → +图检索与路由 94.4%**

## 怎么玩

页面上有四个示例，按难度递增：

| # | 类型 | 看点 |
|---|---|---|
| ① | 单跳事实 | 点答案里的蓝色 `[n]` 可跳转回原文 |
| ② | 纯语义 | 问题里没有角色名，纯靠向量召回 |
| ③ | **关系路径** | 会画出关系链——**两端角色从不出现在同一份文档里** |
| ④ | 幻觉陷阱 | 语料里没有，正确行为是拒答 |

**最值得试的**：问完 ③ 之后，把「图谱来源」从「抽取图」切到「标准图」——路径完全一样。
v4 抽取 prompt 的 recall 是 100%，抽取图与标准图的端到端答对率都是 89.7%。

## 关于这个 Space

服务端**完全离线**：不调用任何外部 API、不持有任何凭据，
全部回答来自预先录制的缓存（562 条）。因此：

- 只能回答评测集里的 109 道题（抽取图三版 prompt 都录了，标准图录了 v3），点示例按钮即可
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
