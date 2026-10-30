# 基线版本档案

> 本仓库内两棵源码树的来源与精确版本。评委复现 diff 即以本文件所记录 commit 为对照。
> 导入日期：2026-10-30 · 导入方式：`cp -a` 自镜像预置源码，剔除 `.git` 与构建产物（`build/`、`*.egg-info`）

| 源码树 | 上游 | 精确版本 | 验证方式 |
|---|---|---|---|
| `vllm-plugin-FL/` | https://github.com/flagos-ai/vllm-plugin-FL branch `flagos-2026-s2` | commit `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`（13eb9be "add vllm-plugin-fl for txda based on empty vllm==0.24.0. (#447)"） | `git status` 干净；`git diff origin/flagos-2026-s2 HEAD` 为空（镜像内 main 与该分支同 commit） |
| `FlagGems/` | https://github.com/flagos-ai/FlagGems tag `v5.3.5` | commit `a7620cc191a0b42e040194622c5758b22a7a25dc`（`git describe --tags` = v5.3.5） | `git status` 干净 |

## 运行环境（镜像）

### 沐曦侧（模力方舟实例）

- 镜像：模力方舟 `Flagos / vllm:0.24.0 / Python 3.12 / maca 3.7.0.38`
- Python 3.12.11；torch 2.8（metax3.7.0.7 构建的 flash_attn/flash_mla/flashinfer 等已预装）
- 关键预置路径：`/workspace/MiniCPM5-2B`、`/workspace/evalscope-datasets/math_500`
- 该机 `/workspace/` 下两棵官方源码**保留作纯净参照，永不修改**；开发只在本仓库

### 天数侧（BI-V150 机器，2026-10-30 实测）

- 硬件：Iluvatar **BI-V150 32GB**（容器内可见 1 卡）；宿主 16 卡，`/dev/iluvatar1`
- 驱动栈：`ixsmi` 报 IX-ML 4.4.0 / Driver 4.5.0 / CUDA 10.2；软件目录 `/usr/local/corex-4.5.0`
- Python 3.12.11；torch 2.10.0+corex.4.5.0；vllm 0.24.0+empty；vllm-plugin-fl 0.3.0rc0+g13eb9be69（即基线 commit 13eb9be）；FlagGems v5.3.5；evalscope 1.11.1
- `/workspace` 预置与沐曦镜像同构：MiniCPM5-2B（4.7GB）、evalscope-datasets/math_500、vllm-plugin-FL@13eb9be（干净）、FlagGems@v5.3.5
- torch.compile / AOT 编译缓存已预置（`/root/.cache/vllm/torch_compile_cache`），serve 启动 ~85s 即绪
- KV cache 实测：521,936 tokens（≈21GB），131072-token 请求最大并发 3.98x → 16K 场景 64 并发必然排队/抢占
- serve 参数差异：天数需 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`（run_baseline.sh 已按平台自适应）

## 基线数字（阶段 1 复现后填写）

| 指标 | 场景 | 基线值 | 来源 CSV |
|---|---|---|---|
| math_500 Level 3 精度 | — | **0.962（105 题，2026-10-31 天数侧实测，与官方参考值一致）**；沐曦侧 98.1% | `20261030_iluvatar_baseline/evalscope_report/` |
| total tokens/s | 4K [4096,1024,64,256] | 天数 **1879.68**（沐曦待节点修复后测） | `20261030_iluvatar_baseline/summary_*.csv` |
| total tokens/s | 16K [16384,1024,64,128] | 天数 **862.06**（沐曦待节点修复后测） | 同上 |
| ttft | 4K / 16K | 天数 12.3s / 636.5s（mean） | 同上 |

### 天数 BI-V150 基线数字（2026-10-30 实测，官方计分口径：4 轮丢首轮取均值）

CSV 存档：`results/20261030_iluvatar_baseline/summary_20261030_194838.csv`

| 场景 | Total tok/s | Output tok/s | Mean TTFT | Median TTFT | 单轮明细（tok/s） |
|---|---|---|---|---|---|
| 4K [4096,1024,64,256] | **1879.68** | 375.94 | 12.3s | 3.5s | 1700.65* / 1873.34 / 1846.66 / 1919.05 |
| 16K [16384,1024,64,128] | **862.06** | 50.71 | 636.5s | 687.3s | 845.32* / 862.15 / 862.25 / 861.79 |

*带 * 为丢弃轮（含 warmup）。两场景逐轮方差均 <0.1%，数据干净。

**关键观察（优化方向实锤）**：
- 4K 场景 KV 峰值仅 ~60%（引擎日志），零抢占零排队 —— 不是显存瓶颈，瓶颈在算力/访存（decode memory-bound）
- 16K 场景 KV 需求 64×17408×42KB≈46GB >> 21GB 预算 → admission control 排队（Waiting 34 条），**零抢占**但 TTFT 高达 636s，吞吐被排队拖到 862 —— D4（KV/调度）在 16K 场景收益上限极高
- 单流 decode TPOT 156ms（≈6.4 tok/s/流）→ decode kernel/融合优化直接决定 Output tok/s
- serve 启动 ~85s（compile 缓存已热），KV cache 521,936 tokens
