#!/usr/bin/env bash
# 部署本仓库源码到运行环境（官方唯一允许的生效方式，信息文档 §3.2）
# 用法:
#   bash scripts/install_dev.sh            # 只装 vllm-plugin-FL（常态）
#   bash scripts/install_dev.sh --flaggems # 连 FlagGems 一起装（改了算子库才需要）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== 部署前模块位置 =="
python3 - <<'EOF'
import importlib.util
for name in ("vllm_fl", "flag_gems"):
    spec = importlib.util.find_spec(name)
    print(f"  {name}: {spec.origin if spec else '(未安装)'}")
EOF

pip install --no-build-isolation "$REPO_ROOT/vllm-plugin-FL"

if [[ "${1:-}" == "--flaggems" ]]; then
    pip install --no-build-isolation "$REPO_ROOT/FlagGems"
fi

echo "== 部署后模块位置（应指向 site-packages 的全新安装）=="
python3 - <<'EOF'
import importlib.util
for name in ("vllm_fl", "flag_gems"):
    spec = importlib.util.find_spec(name)
    print(f"  {name}: {spec.origin if spec else '(未安装)'}")
EOF

echo "完成。重启 vllm serve 后生效（bash scripts/run_baseline.sh 会自动重启）。"
