# D3a · decode attention launch 调优 · 可行性分析与实施记录

## 调研结论（2026-10-31）

### 瓶颈定位链
1. M0 profile：attention 占 decode step GPU 80.2%（`kernel_unified_attention` 2.0ms/层×42层），实际走 vllm 原生 TritonAttentionBackend（非 FlagGems FA2，见 profile_notes.md）
2. vllm 原生 kernel **已有 16 段 split**（`NUM_PAR_SOFTMAX_SEGMENTS=16`，3D grid=(q_blocks, 2 KV头, 16 seg)）→ **"欠并行"假设不成立**，规划期设想的"写 split-KV kernel"失去前提
3. 实测显存带宽：顺序读 352GB/s / copy 569GB/s；attention kernel 仅 150GB/s → **利用率 43%，差距在 launch 配置**
4. 微基准扫描（4K 真实形状 bs64/kv4600/GQA16:2/hd128/bf16/block16/seg16，微基准复现线上 2005.7µs vs 2043.9µs）：
   - **tile 扫描**：16（默认）已最优，32/64/128 依次恶化（tile↑反而慢，2170→3727→10310µs）
   - **seg 扫描**：默认 16 附近最优（8: -3%，32: +27% 慢）
   - **warps×stages 扫描**：**w1/s1 = 1290µs（1.58×）**！默认（triton auto w4/s3）在小 BLOCK_M=16 CTA 上 warp 划分与 stage 缓冲纯属开销
   - seg16+w1/s1 = 1197µs（**1.71×**）；16K 形状（bs30/kv17000）w1/s1 = 1741µs（300GB/s，约 2×）
5. **正确性**：w1/s1 输出与默认 bit-exact（max_abs=0）——launch 参数不改数值语义

### 落地方式（合规性）
- patch 位置：`vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py` 新增 `patch_unified_attention_launch_for_iluvatar()`（沿用本文件 patch 家族模式：模块导入即执行、幂等、try 保护）
- 机制：`_DecodeLaunchProxy` 包装 `kernel_unified_attention[grid]`，**仅 3D decode-split launch（len(grid)==3）注入 num_warps=1/num_stages=1**；2D prefill launch 保持上游默认；kernel 数学、constexpr 全部不动 → 属 launch 配置调优（"计算逻辑优化-内核调优"），非算子切换，不触红线
- 逃生门：`VLLM_FL_ATTN_DECODE_NUM_WARPS=0 VLLM_FL_ATTN_DECODE_NUM_STAGES=0` 恢复上游默认
- 注意：vllm 只在 `triton_attn.py` import wrapper 函数 `unified_attention`，kernel 本体仅模块内引用 → patch 模块属性即全面生效（已验证）

### 验证数据
| 层级 | 对照 | patch | 提升 |
|---|---|---|---|
| kernel 微基准（4K 形状） | 1970.7µs | 1164.7µs | **1.69×**（bit-exact） |
| **官方口径 4K**（4轮丢首，20261031_220848） | 1879.68（基线） | **2249.22** | **+19.7%** |
| tier-2 16K（128p 单轮） | 853.81 | **919.06** | **+7.6%**（TTFT -5.3%） |
| 精度 math_500 L3 全量 | 96.2% | 跑批中 | — |

官方 4K 四轮明细：2284.4(丢)/2360.04/2027.66/2359.95；TTFT 丢首轮后均值 13.0s（基线 12.3s，+5.7%——需在后续方向中关注，16K TTFT 为改善）。

### tier-0/1 的教训（记录给后续方向）
- tier-1 单轮噪声极大：同一对照组 4K 两轮 1164/2175（差 87%），TTFT 15~53s 波动——prefill 排队随机性主导单轮指标
- **结论：本环境（64 并发、长输入）下单轮小规模数据不可作数；方向门禁至少 tier-2（全 prompts 单轮）起，最好两轮取后一轮**
- tier-0（32p）仅当极端异常冒烟（数值偏低 ~50%，比例失真）
