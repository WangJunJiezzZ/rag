"""
Prompt 注册表 —— 把 prompt 当代码管。

为什么不把 prompt 内联在 Python 里
----------------------------------
内联的 prompt 有三个致命问题:
  1. 改了之后没法回滚, 也说不清"上一版是什么样"
  2. 没法 A/B —— 而 prompt 的效果只能靠 A/B 得知, 不能靠读
  3. diff 混在代码改动里, review 时看不出"这次到底改了哪句话"

这里每个 prompt 是一个带 front-matter 的 .md 文件:

    ---
    id: answer/v2_cited
    base: answer/v1_naive        <- 从哪一版演进而来
    changes: 强制每个事实标注引用编号
    hypothesis: 幻觉率下降, 但拒答率可能上升   <- 改之前先写下预期
    ---
    # system
    ...
    # user
    ...

`hypothesis` 这一栏是刻意设计的: **改 prompt 之前先写下你预期它会怎样变**,
跑完评测再对照。预期错了比分数没涨更有价值 —— 那说明你对失败模式的理解是错的。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .compat import project_root

PROMPT_DIR = project_root() / "prompts"

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^#\s+(system|user|schema_note)\s*$", re.MULTILINE)


@dataclass
class Prompt:
    id: str
    system: str
    user: str
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def family(self) -> str:
        return self.id.split("/")[0]

    @property
    def version(self) -> str:
        return self.id.split("/")[-1]

    def render(self, **kw) -> tuple[str, str]:
        """把 {占位符} 填上。缺变量会直接报错, 不做静默容错 ——
        prompt 少了一段上下文而不报错, 是最难查的一类 bug。"""
        try:
            return self.system.format(**kw), self.user.format(**kw)
        except KeyError as e:
            raise KeyError(f"prompt {self.id} 缺少变量 {e}") from e

    def describe(self) -> str:
        m = self.meta
        return (f"{self.id:<28} base={m.get('base', '-'):<22} "
                f"{m.get('changes', '')}")


def _parse(text: str, pid: str, path: Path) -> Prompt:
    meta: dict = {}
    m = _FM_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]

    parts = _SECTION_RE.split(text)
    # parts = [前导, 'system', 内容, 'user', 内容, ...]
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"{path} 缺少 '# system' 或 '# user' 段")
    return Prompt(id=meta.get("id", pid), system=sections["system"],
                  user=sections["user"], meta=meta, path=path)


class PromptRegistry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or PROMPT_DIR)
        self._cache: dict[str, Prompt] = {}

    def get(self, pid: str) -> Prompt:
        if pid not in self._cache:
            path = self.root / f"{pid}.md"
            if not path.exists():
                raise FileNotFoundError(
                    f"找不到 prompt {pid}\n可用: {', '.join(self.list_ids())}")
            self._cache[pid] = _parse(path.read_text(encoding="utf-8"), pid, path)
        return self._cache[pid]

    def list_ids(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)).replace("\\", "/")[:-3]
                      for p in self.root.rglob("*.md"))

    def family(self, name: str) -> list[Prompt]:
        """同一用途的全部版本, 按版本号排序 —— A/B 时直接遍历。"""
        return [self.get(i) for i in self.list_ids() if i.startswith(f"{name}/")]

    def lineage(self, pid: str) -> list[str]:
        """沿 base 字段回溯出这一版的演进路径。"""
        chain, cur = [pid], pid
        seen = {pid}
        while True:
            base = self.get(cur).meta.get("base", "")
            if not base or base in seen or base == "-":
                break
            chain.append(base)
            seen.add(base)
            cur = base
        return list(reversed(chain))


registry = PromptRegistry()
