"""Regression checks for the Windows project launcher; no system settings change."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt" and shutil.which("pwsh"), "requires Windows and PowerShell 7")
class WindowsLauncherTests(unittest.TestCase):
    def launch(self, args: list[str], exit_code: int = 0) -> dict:
        with tempfile.TemporaryDirectory(prefix="rag-launch-") as tmp:
            root = Path(tmp) / "project with spaces"
            (root / "scripts").mkdir(parents=True)
            launcher = root / "scripts" / "windows-run.ps1"
            shutil.copyfile(ROOT / "scripts/windows-run.ps1", launcher)
            (root / "run.py").write_text(
                "import json, os, sys\n"
                "print(json.dumps({'argv': sys.argv[1:], 'shell': os.environ.get('SHELL'), "
                "'cwd': os.getcwd(), 'python': sys.executable}, ensure_ascii=True))\n"
                f"sys.exit({exit_code})\n", encoding="utf-8",
            )
            env = dict(os.environ)
            env["SHELL"] = str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
            result = subprocess.run(
                [shutil.which("pwsh"), "-NoLogo", "-NoProfile", "-NonInteractive",
                 "-File", str(launcher), *args], cwd=tmp, env=env,
                capture_output=True, encoding="utf-8", timeout=20,
            )
            self.assertEqual(result.returncode, exit_code, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(Path(data["cwd"]), root)
            self.assertEqual(Path(data["shell"]).name.lower(), "pwsh.exe")
            self.assertNotIn("windowsapps", data["python"].lower())
            return data

    def test_overrides_legacy_shell_and_preserves_arguments(self):
        args = ["verify", "--", "--flag", "空 格", 'a"b', "&|<>^%COMSPEC%!", "end\\"]
        self.assertEqual(self.launch(args)["argv"], args)

    def test_default_task_is_doctor(self):
        self.assertEqual(self.launch([])["argv"], ["doctor"])

    def test_python_failure_is_returned(self):
        self.assertEqual(self.launch(["bad-task"], exit_code=7)["argv"], ["bad-task"])


if __name__ == "__main__":
    unittest.main()
