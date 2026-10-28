# 合入上游 LTX 视频修复管线与跨代显卡自适应变更记录

- **负责人**：Mengsk
- **助手**：Gemini（合并执行、算子预编译、跨代显卡自适应与文档整理）
- **日期**：2026-10-25
- **状态**：已完成

## 1. 当前依据

- [项目约束](../../../AGENTS.md)：分支管理、文档隐私与统一归档规范。
- [历史基线变更](./change_20260628_merge-upstream-baseline.md)：前期上游基线合并记录。
- [LTX 修复核心](../../../jasna/ltx/)：LTX-2.5 22B 多窗口扩散修复与原生算子实现。
- [容器入口脚本](../../../.drone/entrypoint.sh)：RTX 4090 / RTX 5090 单镜像自适应启动脚本。

## 2. 合并背景与核心变更

### 2.1 引入 LTX-2.5 22B 视频修复管线

1. **多窗口扩散修复核心（[jasna/ltx/](../../../jasna/ltx/)）**：
   - `transformer.py` / `sampler.py`：LTX-2.5 视频修复 Transformer、Euler + STG 采样器与多窗口时序重叠融合。
   - `blocks.py`：显存自适应 Block 驻留与锁页内存双缓冲异步流式预取管理器（`BlockStore`）。
   - `native/`：C++/CUDA 原生算子扩展（`block.cpp`、`glue.cu`、`aot.py`）与 SageAttention INT8 加速内核。
   - `restore.py` / `compose.py` / `plan.py` / `speed.py`：马赛克轨迹规划、分窗裁剪、色彩匹配、羽化合成及 GPU 自动测速档位缓存。
2. **3D 因果 VAE 与解码器（[jasna/models/ltx_vae/](../../../jasna/models/ltx_vae/)）**：
   - 集成 `CausalDiffusionVAE` 视频编码器、`NADiffusionDecoder` 解码器及 Neighborhood Attention Triton 算子。
   - 支持 `ltx`（默认 2-step）、`ltx-4step`、`ltx-8step`、`ltx-undistilled` 四种模型档位，以及 `--ltx-fast`（Blackwell SM 12.0 原生 NVFP4 加速）与 `--ltx-trial` 测速模式。
3. **公共源码免加密兼容层（[jasna/license_api.py](../../../jasna/license_api.py)）**：
   - 通过 `find_spec("jasna.protection")` 自动判断私有子模块是否存在，公共源码构建无需额外桩文件即可原生运行。
4. **媒体与编码模块重构**：
   - 拆分为 [jasna/media/probe.py](../../../jasna/media/probe.py) 与 [jasna/media/encoder_settings.py](../../../jasna/media/encoder_settings.py)，统一硬件编解码器接口 `VideoReader` / `VideoEncoder` 与 `resolve_encoder_options()`。

### 2.2 冲突解决与本地定制迁移对照

| 模块或文件 | 保留特性与处理方案 |
| --- | --- |
| [.dockerignore](../../../.dockerignore) 与 [.gitignore](../../../.gitignore) | 合并上游构建白名单，保留 `!.drone/**`、`!jasna/**` 及本地忽略项 |
| [jasna/__main__.py](../../../jasna/__main__.py) | 在 `_wants_cli()` 中保留对 `sglang` / `jasna-cli` 可执行文件名的 CLI 识别支持 |
| [jasna/main.py](../../../jasna/main.py) 与 [jasna/os_utils.py](../../../jasna/os_utils.py) | 保留 `--disable-ffmpeg-check` 参数与 `JASNA_DISABLE_FFMPEG_CHECK=1` 环境变量跳过机制 |
| [jasna/media/encoder_settings.py](../../../jasna/media/encoder_settings.py) | 迁移自定义 NVENC 别名与 `maxrate`/`bufsize` 纯数字自动补 `k` 逻辑 |
| [jasna/crop_buffer.py](../../../jasna/crop_buffer.py) | 保留本地 `BORDER_RATIO = 0.10`、`MIN_BORDER = 40` 参数 |
| [.drone/dockerfile](../../../.drone/dockerfile) 与 [.drone/dockerfile.build](../../../.drone/dockerfile.build) | 升级至 `torch==2.12.1+cu130`、`av==18.1.0`、`comfy-kitchen==0.2.38` 及 `python_vali==4.8.8` 容错分支 |

## 3. LTX 权重机制与跨代显卡自适应实测

### 3.1 模型文件索引

| 模型文件名 | 用途与硬件推荐 |
| --- | --- |
| `ltx-restore-alpha1-distill2-int8.safetensors(.enc)` | 默认 2-step 模型（INT8，13.37 GB，Ada SM 8.9 / RTX 4090 推荐） |
| `ltx-restore-alpha1-distill2-nvfp4.safetensors(.enc)` | 默认 2-step 快速模型（NVFP4，7.73 GB，Blackwell SM 12.0 / RTX 5090 专属加速） |
| `ltx-restore-alpha1-distill4-int8.safetensors(.enc)` | 4-step 模型（`ltx-4step` INT8） |
| `ltx-restore-alpha1-distill4-nvfp4.safetensors(.enc)` | 4-step 快速模型（`ltx-4step --ltx-fast` NVFP4） |
| `ltx-restore-alpha1-distill8-int8.safetensors(.enc)` | 8-step 模型（`ltx-8step` INT8） |
| `ltx-restore-alpha1-distill8-nvfp4.safetensors(.enc)` | 8-step 快速模型（`ltx-8step --ltx-fast` NVFP4） |
| `ltx-restore-alpha1-int8.safetensors(.enc)` | Undistilled 完整步数模型（INT8） |
| `ltx-restore-alpha1-nvfp4.safetensors(.enc)` | Undistilled 快速模型（NVFP4） |
| `vae.safetensors(.enc)` | 共享 3D 因果视频 VAE（1.47 GB） |
| `vae-decoder.safetensors(.enc)` | 共享微调 VAE Decoder（38.7 MB） |

### 3.2 RTX 4090 与 RTX 5090 统一镜像实测对比

在 Ada（RTX 4090 24GB，`SM 8.9`）与 Blackwell（RTX 5090 32GB，`SM 12.0`）测试节点上使用同一统一镜像完成热缓存（`torch-compile-cache`）全 4 阶段基准对比：

- **自动匹配模型与算子**：RTX 4090 自动加载 `distill2-int8`（`sm89-int8`，48/48 Blocks 100% 显存常驻，加载耗时 4.22s）；RTX 5090 自动开启 `--ltx-fast` 加载 `distill2-nvfp4`（`sm120-nvfp4`，权重体积缩减 42%）。
- **`LTX 1/4 scan`（RF-DETR v6 Large + NVDEC）**：RTX 4090 耗时 `0.58 s`（`41.24 frame/s`），RTX 5090 耗时 `0.30 s`（`80.09 frame/s`，加速比 `1.94x`）。
- **`LTX 2/4 encode`（3D Causal VAE Encode）**：RTX 4090 耗时 `21.0 s`（`1.12 frame/s`），RTX 5090 耗时 `8.9 s`（`2.69 frame/s`，加速比 `2.36x`）。
- **`LTX 3/4 denoise`（22B DiT 2-Step）**：RTX 4090 耗时 `1.68 s`（`1.19 step/s`），RTX 5090 耗时 `0.80 s`（`2.48 step/s`，加速比 `2.10x`）。
- **`LTX 4/4 compose`（VAE Decode + NVENC）**：RTX 4090 耗时 `6.0 s`（`3.76 frame/s`），RTX 5090 耗时 `2.6 s`（`9.08 frame/s`，加速比 `2.31x`）。
- **4 阶段端到端总耗时**：RTX 4090 约 `29.3 s`，RTX 5090 约 `12.6 s`（综合加速比 `2.33x`）。

## 4. 执行与验收记录

- 2026-10-25，Gemini：完成 `v0.11.3` 代码合并、C++/CUDA 与全架构 Triton AOT 内核预编译（`sm75`~`sm121`），并通过 RTX 4090 与 RTX 5090 双架构真机端到端修复验证。
- 2026-10-28，Gemini：按统一文档规范移除机器连接地址与主机名，采用标准文件名 `change_20261025_integrate-ltx-pipeline.md` 归入 `docs/archive/changes/`。

