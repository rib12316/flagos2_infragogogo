#!/usr/bin/env bash
# tier-0/1/2 快速 bench：直连 vllm bench serve 单轮（开发迭代用，非官方口径）
# 用法: bash scripts/tier_bench.sh <tier0|tier1|tier2> [label]
#   tier0: 4K 32 prompts (~3min)  tier1: 4K64 + 16K32 (~12min)  tier2: 4K256 + 16K128 (~35min)
set -uo pipefail
TIER="${1:?tier0|tier1|tier2}"
LABEL="${2:-dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$REPO_ROOT/results/${STAMP}_${LABEL}"
mkdir -p "$DIR"

PORT=9031
MODEL=/workspace/MiniCPM5-2B

run_case() {  # input_len output_len conc prompts
  local IL=$1 OL=$2 CONC=$3 NP=$4
  echo "--- ${IL}in/${OL}out/c${CONC}/n${NP} ---"
  vllm bench serve --backend vllm --model minicpm --tokenizer "$MODEL" \
    --endpoint /v1/completions --host localhost --port "$PORT" \
    --dataset-name random --ignore-eos \
    --random-input-len "$IL" --random-output-len "$OL" \
    --max-concurrency "$CONC" --num-prompts "$NP" \
    2>&1 | tee "$DIR/quick_${IL}.log" | grep -E "Successful requests|Total token throughput|Mean TTFT|request_latency"
}

case "$TIER" in
  tier0) run_case 4096 1024 64 32 ;;
  tier1) run_case 4096 1024 64 64; run_case 16384 1024 64 32 ;;
  tier2) run_case 4096 1024 64 256; run_case 16384 1024 64 128 ;;
esac
echo "结果: $DIR"
