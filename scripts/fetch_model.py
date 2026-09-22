"""下载 ONNX 向量模型（约 24MB）。跨平台，只用标准库。"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console   # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "bge-small-zh"

# 主源不可达时自动切镜像 —— 国内网络常见情况
MIRRORS = ["https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main",
           "https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main"]
FILES = [("onnx/model_quantized.onnx", "model_quantized.onnx"),
         ("tokenizer.json", "tokenizer.json"),
         ("config.json", "config.json")]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES:
        target = DEST / local
        if target.exists() and target.stat().st_size > 1000:
            print(f"  [skip] {local} 已存在 ({target.stat().st_size/1024:.0f} KB)")
            continue
        for base in MIRRORS:
            url = f"{base}/{remote}"
            try:
                print(f"  下载 {local} <- {base.split('/')[2]} ...",
                      end="", flush=True)
                with urllib.request.urlopen(url, timeout=120) as r, \
                     target.open("wb") as fh:
                    while chunk := r.read(1 << 16):
                        fh.write(chunk)
                print(f" 完成 ({target.stat().st_size/1024:.0f} KB)")
                break
            except Exception as e:
                print(f" 失败 ({type(e).__name__})")
        else:
            print(f"  [FAIL] {local} 全部源均不可达。"
                  f"可设 GRAPHRAG_EMBED=none 退化为纯 BM25 继续使用。")
            return 1
    print(f"\n[ok] 模型就位 -> {DEST}")
    print("     该目录随仓库一起拷贝即可，目标机器无需再联网下载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
