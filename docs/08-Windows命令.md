# Windows 命令兼容说明

## 项目任务

在项目根目录使用 PowerShell 7.3 或更高版本（入口使用现代原生参数传递）：

```powershell
pwsh -NoProfile -File .\scripts\windows-run.ps1 doctor
pwsh -NoProfile -File .\scripts\windows-run.ps1 verify
pwsh -NoProfile -File .\scripts\windows-run.ps1 verify-static
```

入口使用 `.venv\Scripts\python.exe`；若不存在，则依次尝试真正的 `python.exe`
和 `py -3.12`，跳过 WindowsApps 占位程序。参数通过数组传递，保留空格和引号，
并返回 Python 子进程的退出码。入口只在当前进程及其子进程内将 `SHELL` 指向
实际的 `pwsh.exe`，不写入用户或系统环境变量。

无需激活虚拟环境；可直接使用解释器：

```powershell
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe -m pip --version
```

Windows 不调用 `python3`；macOS / Linux 的 `python3` 和 Python shebang 保留。
现有 `run.py` 与 MCP 启动流程继续使用 `sys.executable`，无需修改业务代码。

## npm 子进程

PowerShell 命令行使用 `npm.cmd` 或 `npx.cmd`，避免选择 `npm.ps1` / `npx.ps1`：

```powershell
npm.cmd --version
npx.cmd --version
node scripts/npm-process.cjs --version
```

Node.js 自动化使用项目工具（从项目根目录执行以下 JavaScript）：

```javascript
const { npmSpawnSync, npmSpawn, npmExecFile } = require('./scripts/npm-process.cjs');

const result = npmSpawnSync(['--version'], { encoding: 'utf8' });
if (result.error) throw result.error;
if (result.status !== 0) throw new Error(result.stderr);

const child = npmSpawn(['--version'], { stdio: 'inherit' });
child.on('error', console.error);

npmExecFile(['--version'], { encoding: 'utf8' }, (error, stdout, stderr) => {
  if (error) console.error(error, stderr);
  else console.log(stdout.trim());
});
```

Windows 下工具通过当前 Node.js 执行其安装目录中的 `node_modules/npm/bin/npm-cli.js`。
不启动 Shell，也不拼接命令字符串，空格、中文、引号、`%`、`!`、`&`、管道等参数均按字面传递。
它使用当前 Node.js 安装附带的 npm，不跟随其他位置的全局 npm 包装脚本。
缺少该 CLI 文件时会明确报错，应使用带有 npm 的 Node.js 安装运行工具。
非 Windows 平台调用 PATH 中的 npm。

对于固定、无动态参数的批处理命令，可以显式启动 CMD：

```javascript
const { spawnSync } = require('node:child_process');
const result = spawnSync(process.env.ComSpec, ['/d', '/s', '/c', 'npm.cmd --version'], {
  encoding: 'utf8', windowsHide: true,
});
```

不要把动态参数直接拼接进 `/c` 后的字符串；需要传递任意参数时使用项目工具。
Node.js 对 Windows 批处理文件的限制见 [官方说明](https://nodejs.org/api/child_process.html#spawning-bat-and-cmd-files-on-windows)。

## Codex 自动命令

本机项目可在 `.codex/config.toml` 中设置子进程的 `SHELL`，示例：

```toml
[shell_environment_policy.set]
SHELL = 'C:\Program Files\PowerShell\7\pwsh.exe'
```

先用 `Get-Command pwsh` 验证本机路径。该文件已被 Git 忽略，不会给 macOS / Linux
用户注入 Windows 路径。Codex 只加载受信任项目的配置；已启动会话可能继续使用旧配置，
重新加载项目配置或开启新的会话后再验证 `$env:SHELL`。
当前会话可直接使用上述 PowerShell 7 入口，立即纠正任务子进程的 `SHELL`。
该设置控制子进程环境变量，不是选择实际命令 Shell 的配置；实际路径通过
`(Get-Process -Id $PID).Path` 核实。
参见 [OpenAI 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference) 和
[Shell 环境策略](https://learn.chatgpt.com/docs/config-file/config-advanced#shell-environment-policy)。

`.sh` 发布脚本需要 Git Bash 或 Linux，不直接交给 PowerShell。Windows 的 `bash`
可能解析到 WSL 启动器，请先确认路径；本次修复与验证不执行提交或发布。

## 回归验证

```powershell
python -B -m unittest discover -s tests -p 'test_windows_commands.py' -v
node --test tests/npm-process.test.cjs
pwsh -NoProfile -File .\scripts\windows-run.ps1 doctor
pwsh -NoProfile -File .\scripts\windows-run.ps1 verify
pwsh -NoProfile -File .\scripts\windows-run.ps1 verify-static
```

测试覆盖旧 `SHELL` 继承、带空格的项目目录、默认任务、特殊参数传递、失败退出码，
以及 npm 的三种子进程 API。测试临时文件在结束时清理，不修改系统配置。
