#!/usr/bin/env bash
# 一键基线/回归流水线（沐曦 C500，官方口径，信息文档 §4）
# 流程: 预检 → 起服务(官方参数,不动) → 连通性 → 精度回归 → 性能benchmark → 收集CSV
#
# 用法:
#   bash scripts/run_baseline.sh                          # 全量: 精度 + 官方4轮×2场景 (提交级,耗时1.5~3h)
#   bash scripts/run_baseline.sh --skip-accuracy          # 只跑性能 (日常迭代)
#   bash scripts/run_baseline.sh --quick                  # 单轮直连 vllm bench (开发速测,几分钟级)
#   bash scripts/run_baseline.sh --label xxx --keep-server
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="$REPO_ROOT/results"

# ===== 官方固定参数（红线：永不修改）=====
PORT=9031
MODEL_PATH=/workspace/MiniCPM5-2B
SERVED_NAME=minicpm
TEST_CASES='[[4096,1024,64,256],[16384,1024,64,128]]'   # 4K / 16K 两个计分场景
WORK_DIR=/workspace                                       # benchmark_results 落点按官方从 /workspace 跑

# ===== 可选项 =====
SKIP_ACCURACY=0
SKIP_BENCH=0
QUICK=0
KEEP_SERVER=0
LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-accuracy) SKIP_ACCURACY=1 ;;
        --skip-bench)    SKIP_BENCH=1 ;;
        --quick)         QUICK=1; SKIP_ACCURACY=1 ;;
        --keep-server)   KEEP_SERVER=1 ;;
        --label)         LABEL="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

STAMP="$(date +%Y%m%d_%H%M%S)"
[[ -n "$LABEL" ]] && LABEL="_${LABEL}"
RESULT_DIR="$RESULTS_ROOT/${STAMP}${LABEL}"
mkdir -p "$RESULT_DIR"

log()  { echo "[run_baseline] $*"; }

# ===== 0. 平台探测（沐曦 C500 / 天数 BI-V150 双平台自适应）=====
PLATFORM="unknown"
if command -v mx-smi >/dev/null 2>&1 && ! mx-smi 2>/dev/null | grep -q "No available devices"; then
    PLATFORM="metax"
elif command -v ixsmi >/dev/null 2>&1 && ixsmi 2>/dev/null | grep -q "BI-V150"; then
    PLATFORM="iluvatar"
fi
if [[ "$PLATFORM" == "unknown" ]]; then
    log "❌ 未检测到可用 GPU（mx-smi/ixsmi 均无设备）。请先启动 GPU 再运行。"; exit 1
fi
log "检测到平台: $PLATFORM （metax=沐曦C500 / iluvatar=天数BI-V150）"

# ===== 1. 预检 =====
[[ -d "$MODEL_PATH" ]] || { log "❌ 模型不存在: $MODEL_PATH"; exit 1; }
log "预检通过。结果目录: $RESULT_DIR"

# ===== 2. 重启服务（官方启动参数，一字不改）=====
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
for i in $(seq 1 20); do
    pgrep -f "vllm serve" >/dev/null || break
    sleep 1
done
pgrep -f "vllm serve" >/dev/null && { log "❌ 旧 vllm serve 未退出，请手动处理"; exit 1; }

log "启动 vllm serve（官方参数，按平台附加天数 cudagraph 参数）..."
# 天数 BI-V150 官方基线要求 --compilation-config FULL_DECODE_ONLY（信息文档 §4.1），沐曦无需此参数
EXTRA_FLAGS=()
if [[ "$PLATFORM" == "iluvatar" ]]; then
    EXTRA_FLAGS+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}')
fi
VLLM_PLUGINS=fl nohup vllm serve "$MODEL_PATH" \
  --port "$PORT" \
  "${EXTRA_FLAGS[@]}" \
  --served-model-name "$SERVED_NAME" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  > "$RESULT_DIR/serve.log" 2>&1 &
SERVE_PID=$!
log "serve PID=$SERVE_PID，日志: $RESULT_DIR/serve.log"

log "等待服务就绪（最长 30 分钟）..."
READY=0
for i in $(seq 1 360); do
    if python3 - <<EOF 2>/dev/null
import urllib.request, sys
urllib.request.urlopen("http://127.0.0.1:${PORT}/v1/models", timeout=3)
EOF
    then READY=1; break; fi
    kill -0 "$SERVE_PID" 2>/dev/null || { log "❌ serve 进程退出，见 serve.log 尾部:"; tail -30 "$RESULT_DIR/serve.log"; exit 1; }
    sleep 5
done
[[ $READY -eq 1 ]] || { log "❌ 30 分钟未就绪，见 serve.log"; tail -30 "$RESULT_DIR/serve.log"; exit 1; }
log "服务就绪。"

# ===== 3. 连通性测试（信息文档 §4.2）=====
python3 - "$PORT" "$SERVED_NAME" > "$RESULT_DIR/connectivity.txt" <<'EOF' || { log "❌ 连通性失败"; cat "$RESULT_DIR/connectivity.txt"; exit 1; }
import json, sys, requests
port, name = sys.argv[1], sys.argv[2]
r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                  headers={"Content-Type": "application/json"},
                  json={"model": name, "messages": [{"role": "user", "content": "introduce llm"}], "max_tokens": 256},
                  timeout=120)
r.raise_for_status()
msg = r.json()["choices"][0]["message"]["content"]
print(f"HTTP OK, 回复前100字: {msg[:100]}")
EOF
cat "$RESULT_DIR/connectivity.txt"

# ===== 4. 精度回归（信息文档 §4.3，红线 ≥0.95）=====
if [[ $SKIP_ACCURACY -eq 0 ]]; then
    log "运行 evalscope math_500 Level 3 精度评测（耗时较长）..."
    evalscope eval \
      --model "$SERVED_NAME" \
      --api-url "http://127.0.0.1:${PORT}/v1/chat/completions" \
      --api-key EMPTY \
      --eval-type openai_api \
      --datasets math_500 \
      --dataset-args "{\"math_500\": {\"dataset_id\": \"/workspace/evalscope-datasets/math_500\", \"subset_list\": [\"Level 3\"]}}" \
      --eval-batch-size 8 \
      --timeout 3600 \
      --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
      --work-dir /workspace/evalscope-datasets/level3 \
      --ignore-errors \
      2>&1 | tee "$RESULT_DIR/evalscope.log" | grep -iE "accuracy|score" || true
    grep -iE "accuracy|score" "$RESULT_DIR/evalscope.log" | tail -5 > "$RESULT_DIR/accuracy.txt" || true
    log "精度摘要已存 $RESULT_DIR/accuracy.txt（人工核对 ≥0.95）"
fi

# ===== 5. 性能 Benchmark =====
if [[ $SKIP_BENCH -eq 0 ]]; then
    if [[ $QUICK -eq 1 ]]; then
        # 开发速测：每场景单轮直连 vllm bench（非官方口径，只看相对变化）
        log "quick 模式：单轮速测两场景..."
        for CASE in "[4096,1024,64,256]" "[16384,1024,64,128]"; do
            IFS=',' read -r IL OL CONC NP <<< "$(tr -d '[] ' <<< "$CASE")"
            log "  场景 input=$IL output=$OL conc=$CONC prompts=$NP"
            vllm bench serve --backend vllm \
              --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" \
              --endpoint /v1/completions --host localhost --port "$PORT" \
              --dataset-name random --ignore-eos \
              --random-input-len "$IL" --random-output-len "$OL" \
              --max-concurrency "$CONC" --num-prompts "$NP" \
              2>&1 | tee "$RESULT_DIR/quick_${IL}.log" | grep -E "Successful requests|Total token throughput|Mean TTFT" || true
        done
    else
        # 官方口径：4轮丢首轮取均值（计分脚本，pristine 不改）
        log "官方 benchmark（每场景 4 轮，耗时较长）..."
        (cd "$WORK_DIR" && python3 "$REPO_ROOT/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py" \
          --model "$MODEL_PATH" \
          --served-model-name "$SERVED_NAME" \
          --port "$PORT" \
          --test-cases "$TEST_CASES" \
          2>&1 | tee "$RESULT_DIR/benchmark.log" | tail -20)
        # 收集本轮新生成的 CSV（benchmark_results 按时间戳命名，取最新的 raw+summary）
        LATEST_SUMMARY=$(ls -t "$WORK_DIR"/benchmark_results/summary_*.csv 2>/dev/null | head -1 || true)
        if [[ -n "$LATEST_SUMMARY" ]]; then
            TS=$(basename "$LATEST_SUMMARY" .csv); TS=${TS#summary_}
            cp "$WORK_DIR/benchmark_results/raw_runs_${TS}.csv" "$RESULT_DIR/" 2>/dev/null || true
            cp "$LATEST_SUMMARY" "$RESULT_DIR/"
            log "CSV 已存档: $RESULT_DIR/$(basename "$LATEST_SUMMARY")"
        else
            log "⚠️ 未找到新生成的 summary CSV，检查 $RESULT_DIR/benchmark.log"
        fi
    fi
fi

# ===== 6. 收尾 =====
if [[ $KEEP_SERVER -eq 1 ]]; then
    log "保留服务运行 (PID=$SERVE_PID)。手工停止: pkill -f 'vllm serve'"
else
    log "停止服务（省资源纪律）..."
    pkill -f "vllm serve" 2>/dev/null || true
fi

echo
log "=========== 本轮完成 ==========="
log "结果目录: $RESULT_DIR"
ls -la "$RESULT_DIR"
log "对比历史数据: python3 scripts/compare_results.py"
[[ $KEEP_SERVER -eq 0 ]] && log "提醒：GPU 不用时记得关机（券额度有限，每36h开关机一次）"
