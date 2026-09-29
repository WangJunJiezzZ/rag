#!/usr/bin/env bash
# 把 build/static/ 发布到 gh-pages 分支。
#
# 用 git worktree 而不是切分支 —— 主分支的工作区完全不受影响，
# 不会出现"发布一下把本地改动搞乱了"的情况。
set -e
cd "$(dirname "$0")/.."

[ -d build/static ] || { echo "[FAIL] 先跑: python run.py build-static"; exit 1; }
[ -f build/static/index.html ] || { echo "[FAIL] build/static/index.html 不存在"; exit 1; }

WT=.git/gh-pages-wt
rm -rf "$WT"
if git show-ref --verify --quiet refs/heads/gh-pages; then
  git worktree add -f "$WT" gh-pages >/dev/null
else
  git worktree add -f --orphan -b gh-pages "$WT" >/dev/null
fi

# 清空旧内容（保留 .git），拷入新站点
find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R build/static/. "$WT"/

cd "$WT"
git add -A
if git diff --cached --quiet; then
  echo "[ok] 内容无变化，无需发布"
else
  git commit -q -m "Publish static demo $(date +%Y-%m-%d\ %H:%M)"
  echo "[ok] 已提交到 gh-pages 分支"
fi
cd - >/dev/null
git worktree remove -f "$WT"

echo ""
echo "下一步："
echo "  git push origin gh-pages"
echo "  然后到 GitHub → Settings → Pages → Source 选 gh-pages 分支 / root"
