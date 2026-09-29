"""启动 Web 演示。跨平台。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console   # noqa: E402

setup_console()


def main() -> int:
    ap = argparse.ArgumentParser()
    # PORT 环境变量优先于 --port。托管平台(HF Spaces / Render / Railway)
    # 都是通过 PORT 注入端口的, 本地默认仍是 8000, 行为不变。
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    try:
        import uvicorn
    except ImportError:
        print("[FAIL] 需要安装: pip install fastapi \"uvicorn[standard]\"")
        return 1
    print(f"演示地址: http://{args.host}:{args.port}")
    print("首次启动会建索引并向量化，约需十几秒。\n")
    uvicorn.run("graphrag.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
