"""graphrag-lab: 一个可评测、可消融的 GraphRAG 参考实现。"""
from __future__ import annotations

__version__ = "0.1.0"

from .compat import load_dotenv, setup_console

setup_console()
# 导入包时自动加载仓库根目录的 .env。显式设置的环境变量优先级更高。
load_dotenv()
