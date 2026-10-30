# flagos2_infragogogo

FlagOS 开放计算全球大赛（赛季二）赛道二「MiniCPM 模型推理吞吐性能优化」· **infragogogo 队**（成员 xxinfra）工作仓库。

- 赛题：MiniCPM5-2B 在沐曦 C500 / 天数 BI-V150 上的推理吞吐优化（total tokens/s），精度红线 math_500 Level 3 ≥ 0.95
- 约束框架：vllm-plugin-FL `flagos-2026-s2` + FlagGems `v5.3.5`，serve 参数与基线一致
- 截止：2026-11-20 23:59 (UTC+8)

## 仓库导航

| 路径 | 内容 |
|---|---|
| [开发记录.md](开发记录.md) | **路线图 + 开发日志（先看这个）** |
| `vllm-plugin-FL/` | 比赛框架源码，基线 commit 之上的全部优化提交 |
| `FlagGems/` | 算子库（若涉及算子改动） |
| `scripts/` | 一键复现 / 部署 / 结果汇总脚本 |
| `docs/` | 基线档案、代码预研笔记、profile 笔记 |
| `results/` | 各轮 benchmark summary CSV 存档 |

## 快速开始（沐曦 C500 官方镜像内）

```bash
# 1. 部署本仓库源码到运行环境（等价 pip install --no-build-isolation）
bash scripts/install_dev.sh

# 2. 一键基线/回归：起服务 → 连通性 → 精度 → 4K/16K 性能 → 收集 CSV
bash scripts/run_baseline.sh

# 3. 汇总历史结果对比
python3 scripts/compare_results.py
```

基线环境与精确版本见 [docs/BASELINE.md](docs/BASELINE.md)。

> 本 README 将在提交阶段（阶段 4）改写为官方要求的复现说明（软硬件环境、依赖版本、部署步骤、结果校验方法）。
