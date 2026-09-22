"""跨平台兼容层 (macOS / Linux / Windows)。

Windows 上最容易踩的三个坑, 这里一次性处理掉:

1. **控制台编码**。中文 Windows 的 cmd.exe 默认 code page 是 936 (GBK),
   直接 print 中文或 "✓" 这类符号会抛 UnicodeEncodeError 让脚本崩掉。
   这里把 stdout/stderr 重设为 utf-8 并用 errors="replace" 兜底。
2. **文件编码**。Windows 上 open() 的默认 encoding 跟随 locale 而非 utf-8,
   所以本项目所有读写一律显式传 encoding="utf-8"; read_text/write_text
   的包装放在这里, 避免遗漏。
3. **路径**。全程使用 pathlib.Path, 不出现硬编码的 "/" 拼接。
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

_CONSOLE_READY = False


def setup_console() -> None:
    """让 stdout/stderr 在任何平台上都能安全输出中文。幂等。"""
    global _CONSOLE_READY
    if _CONSOLE_READY:
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            # Python 3.7+: TextIOWrapper.reconfigure
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            buf = getattr(stream, "buffer", None)
            if buf is not None:
                setattr(sys, stream_name,
                        io.TextIOWrapper(buf, encoding="utf-8", errors="replace"))
    if os.name == "nt":
        # 让旧版 cmd.exe 也切到 UTF-8 code page; 失败不影响运行
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # type: ignore[attr-defined]
            ctypes.windll.kernel32.SetConsoleCP(65001)        # type: ignore[attr-defined]
        except Exception:
            pass
    _CONSOLE_READY = True


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" 避免 Windows 把 \n 写成 \r\n, 保证跨平台产出字节一致
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def read_jsonl(path: str | Path):
    import json
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows) -> int:
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_dotenv(path: str | Path | None = None, override: bool = False) -> int:
    """加载 .env 到环境变量。纯标准库, 不引入 python-dotenv。

    为什么需要: Windows 上设置环境变量比 Unix 麻烦得多(PowerShell 的 $env: 只对
    当前会话生效, 系统设置要重启终端)。一个 .env 文件是最不容易出错的方式。

    **默认不覆盖已存在的环境变量** —— 显式 export 的优先级高于文件,
    这是 dotenv 的通行语义, 也避免 .env 里的旧值静默盖掉命令行临时设置的值。

    返回实际注入的变量个数。
    """
    p = Path(path) if path else project_root() / ".env"
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if not v:                      # 空值视为未设置, 避免"设了但是空"的坑
            continue
        if override or k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def project_root() -> Path:
    """返回仓库根目录, 不依赖 cwd —— 双击运行 / 从任意目录调用都正确。"""
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path() -> None:
    """供 scripts/ 下的脚本在未安装包时也能 import graphrag。"""
    src = str(Path(__file__).resolve().parents[1])
    if src not in sys.path:
        sys.path.insert(0, src)
