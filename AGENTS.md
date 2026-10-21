# 项目命令执行规则

## Windows

- 自动命令使用 PowerShell 7 (`pwsh.exe`)；先通过 `Get-Command pwsh` 确认路径。不要用继承的 `SHELL` 判断实际 Shell，也不要默认调用 `powershell.exe`。
- 项目任务优先使用 `pwsh -NoProfile -File .\scripts\windows-run.ps1 <task> [args...]`。该入口只对子进程设置 `SHELL`，优先选择 `.venv\Scripts\python.exe`，其次 `python`，最后 `py -3.12`。
- Windows 不调用 `python3` 占位程序；Python 子进程继续使用 `sys.executable` 和参数数组。安装依赖使用目标解释器的 `-m pip`。
- 不要求激活虚拟环境；可以直接运行 `.\.venv\Scripts\python.exe run.py <task>`，避免 `Activate.ps1` 被执行策略拦截。
- PowerShell 直接调用 npm/npx 时使用 `npm.cmd` / `npx.cmd`。
- Node.js 的 `spawnSync`、`spawn`、`execFile` 调用 npm 时使用 `scripts/npm-process.cjs`，Windows 下通过当前 Node.js 直接执行 `npm-cli.js`，参数保留为数组。
- 必须执行 `.cmd` 时显式调用 `cmd.exe /d /s /c`；禁止把未经处理的动态参数拼接成命令字符串。复杂参数优先使用上述 npm 工具，避免 CMD 的 `%`、`!`、`&`、管道和引号展开。
- PowerShell 使用 `$env:NAME = 'value'` 设置当前进程环境变量；不复制 Bash 的 `export`、`source`、`NAME=value command` 或 heredoc 语法。包含空格的可执行路径使用调用运算符 `&`。
- `.sh` 脚本只在明确的 Git Bash 或 Linux 环境运行；Windows PATH 中的 `bash.exe` 可能是 WSL 启动器。Unix 文档中的 `python3` 与 shebang 保留。
- 本机专用 Codex 配置放在忽略提交的 `.codex/config.toml`；不要将 Windows 绝对路径设置提交成跨平台默认配置。
- 不修改全局执行策略，不删除旧版 PowerShell，不重装开发工具。修改用户或系统级环境变量前必须征得用户确认。

## 验证

命令基础设施变更后运行：

```powershell
python -B -m unittest discover -s tests -p 'test_windows_commands.py' -v
node --test tests/npm-process.test.cjs
pwsh -NoProfile -File .\scripts\windows-run.ps1 doctor
pwsh -NoProfile -File .\scripts\windows-run.ps1 verify
pwsh -NoProfile -File .\scripts\windows-run.ps1 verify-static
```

保持业务代码、数据、缓存和报告不变。不要为检查命令兼容性运行生成数据、在线评测、提交或发布任务。
