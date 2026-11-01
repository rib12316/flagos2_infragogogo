# Profile 热点笔记（M0，2026-10-31，天数 BI-V150）

> 抓取方式：`--enforce-eager`（图回放时 kineto 不展开图内 kernel，eager 才可见全貌）+ vllm profiler API（`/start_profile` `/stop_profile`）。
> 归因专用配置，**不用于计分对比**。trace 存档：`results_local/prof/rank0_4k_eager.json.gz`（+ graph 模式对照 `rank0_4k_graphmode.json.gz`）。

## 4K decode 稳态 kernel 分布（75 steps × 42 层窗口）

| 类别 | GPU 占比 | 明细（均次耗时） |
|---|---|---|
| **attention** | **80.2%** | `kernel_unified_attention` 2003.6µs/次 × 3150 次 = 84ms/step |
| GEMM | 11.8% | `linear_kernel` 73.6µs × 12675（42 层 × 5 GEMM × 75 step + lm_head） |
| sampler | 3.9% | `_topk_topp_kernel` 3956µs/step（图外，vocab 130560 × batch64） |
| silu_mul | 0.9% | 22.1µs/次 |
| rmsnorm | 0.7% | `fused_add_rms_norm_kernel` 8.2µs/次 × 6300（**D2a 已现役 ✓**，75step×42层×2） |
| rope | 0.4% | `apply_rotary_pos_emb_inplace` 10.2µs |
| memcpy/other | ~2% | |

eager step 时长 334.8ms（正式跑 graph 模式 156ms，差值为图收益）。

## 关键路径发现（改变 D3 实施方案）

1. **实际 attention 走 vllm 原生 TritonAttentionBackend**：`flaggems.py:attention_backend()` 默认返回 `AttentionBackendEnum.TRITON_ATTN.get_path()`；`VLLM_FL_USE_FLAGGEMS_ATTN=1` 才走 FlagGems FA2（AttentionFLBackend）。dispatch 日志 "attention_backend using default.flagos" 仅表示 dispatch 选中了 flagos impl，其返回值就是 vllm 原生路径。**修正规划期假设（当时以为走 FlagGems flash_varlen_fwd）**。
2. 采样参数：模型 `generation_config.json` 默认 `temperature=1.0, top_p=0.95` → bench 每请求走 top-p 采样 → `_topk_topp_kernel` 每 step 4.0ms（eager，被 `patch_sampler_compile_for_iluvatar` 解除编译后走 triton kernel）。
3. 图模式 trace 中图内 kernel 不可见（kineto+corex 限制），图外仅 sampler+lm_head+memcpy：`_topk_topp_kernel` 占可见 GPU 时间 64.5%——若图内时间被压缩，sampler 将成次级热点。

## 结论与方向修正

| 原计划 | 修正 |
|---|---|
| D3a：改 FlagGems `flash_api.py` split-KV | **改为：vendor/iluvatar 新建 attention backend**。参考同仓库 metax 侧现成实现 `vendor/metax/impl/attention/ops/triton_decode_attention.py`（SGLang/lightllm 血统的 **GQA FlashDecoding 两阶段 kernel**，split-KV+merge，paged KV，page≥1）+ `merge_attn_states.py`。合规路径：新算子开发 + `iluvatar.yaml` 把 attention_backend 路由到 vendor:iluvatar |
| D3b：FlagGems tune_configs attention 段 | 对象改为 vllm 原生 `triton_unified_attention.py` 的 tile（在 plugin 内做 iluvatar 专属 wrapper/config，不动 vllm 主库） |
| D2a fused_add_rms_norm 接线 | **已现役，取消**（trace 实证 6300 次） |
| sampler | 4.0ms/step，图外最大项。M2 候选（triton 优化或恢复编译） |

**16K 未单独抓 eager trace**：16K decode 与 4K 共用 kernel，长 KV 使 `kernel_unified_attention` 更慢（基线 Peak output 217 vs 768 tok/s），split-KV 收益只会更大；且 16K 吞吐主约束是 KV 容量（D4，暂缓）。split-KV 完成后可补 16K trace 验证。

## 下一步（M1）

1. `feat/d3a-splitkv`：metax `triton_decode_attention.py` 适配 iluvatar（GQA 16Q/2KV、head_dim 128、bf16、block 16）→ vendor/iluvatar backend 类 → iluvatar.yaml 路由 → 单测比对（vs vllm unified kernel）→ tier 验证
2. 注意图捕获约束：FULL_DECODE_ONLY 下 decode 走全图，新 kernel 必须可捕获（静态 shape、无 CPU 分支、num_splits 定长或按档位）
3. 预期：attention 2.0ms→0.6-1.0ms/层调用（16 SM 欠并行 → split 后 CTA×4-8），4K step 84→30-50ms attention

---

## 优化后复测（2026-11-01 晚，D3a+D3c 后，eager 4K 稳态 90 steps）

| 类别 | 占比 | 对比优化前 | 状态 |
|---|---|---|---|
| attention | 69.5%（kernel 均次 1124.7µs，混合 1.78×） | 80.2% | 仍主导 |
| GEMM | 18.3%（74.1µs/次） | 11.8% | **已达带宽理论下限**（12.5ms/step ≈ 4.5GB权重 ÷ 352GB/s），无空间 |
| sampler | 6.1% | 3.9% | 负结果已存档（图外时间与调度重叠） |

### launch 级扫描空间已穷尽（全部组内交替 A/B 验证）
- tile：16 最优（32 在 16K 慢 0.78×；8 违反 triton dot K≥16 约束不可编译）
- warps/stages：decode w1/s1、prefill w2/s1（已上线）
- segments：16 已最优（4K/16K 均确认）
- BLOCK_M：非单调反例（bs30 1.21× / bs24 0.94× / 4K 0.93×），动态 batch 下不可靠，放弃
- **结论：attention kernel 当前 267GB/s（76% 带宽利用率），进一步提升需 kernel 级开发（页表访存模式/L2 布局），预期收益 <10%、工作量 2-3 天**

### 方法论追加
- 微基准组间漂移可达 73%（GPU 时钟/缓存状态）——**所有对比必须组内交替 A/B（≥3 轮）**，单组对照会出 1.36× 假加速（本次 tile32 教训）

### KV 布局 HND 实验（2026-11-01 深夜，负结果）
- 假设：NHD 布局下 KV 读为 256B 连续+512B 跳（头间夹缝），HND（vllm 原生 `VLLM_KV_CACHE_LAYOUT=HND`，stride_order (0,1,3,2,4)）使固定头后 4KB 连续 → attention 读带宽应提升
- 实测：HND serve 正常（连通性/数学题正确，全链路兼容），但官方 4K 第 2 轮 2327 vs NHD 2562（**-9%**）
- 解释：decode attention 读收益 < KV 写入（reshape_and_cache_flash 交错写）/prefill 访存/其他 copy kernel 的损失
- 结论：放弃布局方向；HND env 不引入
