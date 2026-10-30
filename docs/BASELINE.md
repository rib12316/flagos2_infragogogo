# 基线版本档案

> 本仓库内两棵源码树的来源与精确版本。评委复现 diff 即以本文件所记录 commit 为对照。
> 导入日期：2026-10-30 · 导入方式：`cp -a` 自镜像预置源码，剔除 `.git` 与构建产物（`build/`、`*.egg-info`）

| 源码树 | 上游 | 精确版本 | 验证方式 |
|---|---|---|---|
| `vllm-plugin-FL/` | https://github.com/flagos-ai/vllm-plugin-FL branch `flagos-2026-s2` | commit `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`（13eb9be "add vllm-plugin-fl for txda based on empty vllm==0.24.0. (#447)"） | `git status` 干净；`git diff origin/flagos-2026-s2 HEAD` 为空（镜像内 main 与该分支同 commit） |
| `FlagGems/` | https://github.com/flagos-ai/FlagGems tag `v5.3.5` | commit `a7620cc191a0b42e040194622c5758b22a7a25dc`（`git describe --tags` = v5.3.5） | `git status` 干净 |

## 运行环境（镜像）

- 镜像：模力方舟 `Flagos / vllm:0.24.0 / Python 3.12 / maca 3.7.0.38`
- Python 3.12.11；torch 2.8（metax3.7.0.7 构建的 flash_attn/flash_mla/flashinfer 等已预装）
- 关键预置路径：`/workspace/MiniCPM5-2B`、`/workspace/evalscope-datasets/math_500`
- 本机 `/workspace/` 下两棵官方源码**保留作纯净参照，永不修改**；开发只在本仓库

## 基线数字（阶段 1 复现后填写）

| 指标 | 场景 | 基线值 | 来源 CSV |
|---|---|---|---|
| math_500 Level 3 精度 | — | 期望 ≈0.962（官方口径） | 待填 |
| total tokens/s | 4K [4096,1024,64,256] | 待实测 | 待填 |
| total tokens/s | 16K [16384,1024,64,128] | 待实测 | 待填 |
| ttft | 4K / 16K | 待实测 | 待填 |
