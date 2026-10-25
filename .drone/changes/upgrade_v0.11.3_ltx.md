# 升级上游 v0.11.3 暨 LTX 模型支持变更说明

- **变更日期**：2026-10-25
- **目标分支**：`dev`
- **基线版本**：`v0.10.0+` (`d2c6627`) -> `v0.11.3` (`8265609`，合并提交 `7ddd708`)
- **上游仓库**：`git@github.com:Kruk2/jasna.git` (`upstream/main`)

---

## 一、合并背景与核心变更概览

本次升级将上游 `v0.10.0` 至 `v0.11.3` 期间的 450+ 个提交合入本地 `dev` 分支，核心收益包括：

1. **完整引入 LTX (LTX-2.5 22B) 视频修复管线**：
   - 新增 [`jasna/ltx/`](../../jasna/ltx/) 全套多窗口扩散修复核心：
     - `transformer.py` / `sampler.py`：28 层 LTX-2.5 视频修复 Transformer、Euler + STG 采样器、多窗口时序重叠融合（MultiDiffusion-style overlap fusion）。
     - `blocks.py`：显存自适应 Block 驻留与 Host Page-Locked Pinned Memory 双缓冲异步流式预取管理器（`BlockStore`）。
     - `native/`：C++/CUDA 原生算子扩展（`block.cpp`、`glue.cu`、`aot.py`）与 **SageAttention INT8** 加速内核（`sage_attention/`）。
     - `restore.py` / `compose.py` / `plan.py` / `speed.py`：马赛克轨迹规划、分窗裁剪、色彩匹配、羽化合成及 GPU 自动测速档位缓存。
   - 新增 [`jasna/models/ltx_vae/`](../../jasna/models/ltx_vae/)：
     - `CausalDiffusionVAE` 视频编码器、`NADiffusionDecoder` 解码器及 Neighborhood Attention Triton 算子（`na_triton.py`）。
   - 支持 4 种 LTX 模型档位与 2 种量化精度（INT8 / RTX 50 系专属 NVFP4）：
     - `ltx`（默认 `distilled2`，2-step 标准质量）
     - `ltx-4step`（`distilled4`，4-step）
     - `ltx-8step`（`distilled`，8-step）
     - `ltx-undistilled`（`undistilled`，完整步数）
     - `--ltx-fast`（启用 Blackwell SM 12.0 / RTX 5090 硬件原生 NVFP4 权重与算子）
     - `--ltx-trial`（免权重/免授权的完整算子与速度基准测试模式）
2. **引入公共源码免加密兼容层（[`jasna/license_api.py`](../../jasna/license_api.py)）**：
   - 上游在 `v0.11.x` 中新增 `license_api.py`，通过 `find_spec("jasna.protection")` 自动判断私有子模块是否存在。
   - 当 `jasna/protection` 为未初始化空目录时，公共源码构建可原生直接运行，无需 `v0.10.0` 时期的 `.drone/protection_stub.py` 桩文件。
3. **媒体与编码模块重构**：
   - 原 `jasna/media/__init__.py` 拆分为 [`jasna/media/probe.py`](../../jasna/media/probe.py)（元数据探测）与 [`jasna/media/encoder_settings.py`](../../jasna/media/encoder_settings.py)（编码参数校验与 CQ 规格）。
   - `NvidiaVideoReader` / `NvidiaVideoEncoder` 统一重命名为 [`VideoReader`](../../jasna/media/video_decoder.py) / [`VideoEncoder`](../../jasna/media/video_encoder.py)，NVENC/AMF 参数组装收敛为纯函数 [`resolve_encoder_options()`](../../jasna/media/video_encoder.py)。

---

## 二、冲突解决与本地定制迁移清单

本次合并共处理 8 处冲突文件及多项容器配置同步，所有历史定制能力均已完整保留：

| 模块 / 文件                                                                                                                                                  | 冲突原因                                                                                                         | 解决方案与保留特性                                                                                                                                                                                                    |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`.dockerignore`](../../.dockerignore) / [`.gitignore`](../../.gitignore)                                                                                    | 上游新增 `keytool` 构建白名单与 `jasna/ltx/native/prebuilt/` 忽略规则                                            | 合并双方规则，保留 `!.drone/**`、`!jasna/**`、`!patches/**` 及本地 `.tmp`、`.pip_cache` 忽略项                                                                                                                        |
| [`jasna/__main__.py`](../../jasna/__main__.py)                                                                                                               | 上游重构 `_wants_cli(argv)` 分发函数                                                                             | 在 `_wants_cli()` 中保留对 `sglang` / `jasna-cli` 可执行文件名的 CLI 识别支持                                                                                                                                         |
| [`jasna/main.py`](../../jasna/main.py) / [`jasna/os_utils.py`](../../jasna/os_utils.py) / [`jasna/benchmark/__init__.py`](../../jasna/benchmark/__init__.py) | 上游将系统检查拆分为 `_check_system(args)` 并新增 LTX CLI 参数                                                   | 保留 `--disable-ffmpeg-check` 参数与 `JASNA_DISABLE_FFMPEG_CHECK=1` 环境变量跳过机制；保留未指定输出文件名时默认 `.mp4` 扩展名行为                                                                                    |
| [`jasna/media/encoder_settings.py`](../../jasna/media/encoder_settings.py) / [`jasna/media/video_encoder.py`](../../jasna/media/video_encoder.py)            | 上游将编码设置校验从 `media/__init__.py` 迁移至 `encoder_settings.py`，并在 `resolve_encoder_options()` 统一处理 | 将自定义 NVENC 别名（`gop`、`maxbitrate`、`vbvbufsize`、`lookahead`、`temporalaq`、`nonrefp`、`tflevel`、`bitrate` 等）及 `maxrate`/`bufsize` 纯数字自动补 `k` 逻辑迁移至新函数                                       |
| [`jasna/crop_buffer.py`](../../jasna/crop_buffer.py)                                                                                                         | 无冲突自动合并                                                                                                   | 保留本地优化的 `BORDER_RATIO = 0.10`、`MIN_BORDER = 40`                                                                                                                                                               |
| [`pyproject.toml`](../../pyproject.toml)                                                                                                                     | 上游升级依赖至 `v0.11.3`                                                                                         | 保留注释 `# "python_vali==4.8.8"`（由 Builder 镜像源码编译安装 Kruk2/vali 分支）                                                                                                                                      |
| [`.drone/dockerfile`](../dockerfile)                                                                                                                         | `v0.11.3` 新增 `license_api.py` 检测机制                                                                         | 升级基础镜像标签至 `v0.11.3-build`；**移除 `protection_stub.py` 复制步骤**（避免 `find_spec("jasna.protection")` 误判已安装私有模块）                                                                                 |
| [`.drone/dockerfile.build`](../dockerfile.build) & [`.github/workflows/*`](../../.github/workflows/)                                                         | `v0.11.3` 新增 LTX 依赖与新版 PyTorch/VALI                                                                       | 升级 `torch==2.12.1+cu130`、`torchvision==0.27.1+cu130`、`av==18.1.0`、`ultralytics>=8.4.96,<8.5`，新增 `comfy-kitchen==0.2.38`、`scipy`、`certifi`，同步 `vali` commit 至 `0e5c01ee57222a8ef4f9c7591284c4324bb3fba5` |

---

## 三、LTX 模型权重机制与硬件适配分析（RTX 4090 / RTX 5090）

### 1. 明文权重直跑通道 vs 加密权重通道

在 [`jasna/ltx/model_files.py`](../../jasna/ltx/model_files.py) 与 [`jasna/ltx/inference.py`](../../jasna/ltx/inference.py) 中：

- **明文通道（源码/容器环境 `not is_frozen()` 原生支持）**：
  - `_bundle_paths()` 检查 `model_weights/ltx-restore/` 下是否存在同名 `.safetensors` 明文文件；若存在，则不追加 `.enc` 后缀。
  - `LtxModelFiles.from_dir()` 与 `open_tensors()` 仅在文件后缀为 `.enc` 时调用 `require_protection()`；明文 `.safetensors` 直接通过 `_PlaintextHandle` (`safetensors.safe_open`) 读取。
  - `ModelRuntime` 在明文模式下 `self.protected = False`，所有 Encoder / Transformer (`BlockStore`) / Decoder 均在当前进程内直接运行，不经过 `jasna.protection.ltx_worker`。
- **加密通道（上游 HuggingFace 发布包）**：
  - 上游在 `https://huggingface.co/Kruk2/ltx-restore/resolve/main`（及国内镜像 `https://hf-mirror.com/Kruk2/ltx-restore/resolve/main`）发布的 `.safetensors.enc` 文件经过加密，由 `jasna.protection.protected_tensors.open_ltx_tensors` 解密加载。

### 2. LTX 模型文件清单（`model_weights/ltx-restore/`）

每个 LTX 运行组合由 **1 个 Transformer 文件 + 2 个共享 VAE 文件** 组成：

| 文件名（明文）                                  |   大小 (Bytes) | SHA-256 (发布版基准) | 用途与硬件推荐                                                              |
| :---------------------------------------------- | -------------: | :------------------- | :-------------------------------------------------------------------------- |
| `ltx-restore-alpha1-distill2-int8.safetensors`  | 13,368,048,749 | `c23559f9...ea51ad`  | 默认 2-step 模型（INT8，**RTX 4090 / SM 8.9 推荐**）                        |
| `ltx-restore-alpha1-distill2-nvfp4.safetensors` |  7,733,294,997 | `4aa08c14...10693b`  | 默认 2-step 快速模型（`--ltx-fast` NVFP4，**RTX 5090 / SM 12.0 专属加速**） |
| `ltx-restore-alpha1-distill4-int8.safetensors`  | 13,368,048,733 | `1c0f1642...1f30965` | 4-step 模型（`ltx-4step` INT8）                                             |
| `ltx-restore-alpha1-distill4-nvfp4.safetensors` |  7,733,294,981 | `57098e62...06973`   | 4-step 快速模型（`ltx-4step --ltx-fast` NVFP4）                             |
| `ltx-restore-alpha1-distill8-int8.safetensors`  | 13,368,048,773 | `b006342a...36e057`  | 8-step 模型（`ltx-8step` INT8）                                             |
| `ltx-restore-alpha1-distill8-nvfp4.safetensors` |  7,733,295,021 | `54f2f4f6...73e62e2` | 8-step 快速模型（`ltx-8step --ltx-fast` NVFP4）                             |
| `ltx-restore-alpha1-int8.safetensors`           | 13,368,048,821 | `c72161b1...780f83`  | Undistilled 完整步数模型（INT8）                                            |
| `ltx-restore-alpha1-nvfp4.safetensors`          |  7,733,295,061 | `8301d5ba...a2ccb5`  | Undistilled 快速模型（NVFP4）                                               |
| `vae.safetensors`                               |  1,472,223_815 | `b9252d5a...08f5e25` | 共享视频 VAE（基于 Lightricks `ltx-2.5-video-vae-bf16.safetensors`）        |
| `vae-decoder.safetensors`                       |     38,718,609 | `c60c3572...caeea2d` | 共享微调 VAE Decoder                                                        |

### 3. RTX 4090 vs RTX 5090 算子与显存行为对比

- **RTX 4090 (24GB VRAM, Ada Lovelace SM 8.9)**：
  - 使用 **INT8** 权重（约 13.3 GB），自动启用 `cuBLAS INT8 MM` + `SageAttention INT8` + `Triton Epilogue` 原生算子。
  - 24GB 显存可常驻大部分 Transformer Blocks，剩余少量 Block 通过 Pinned Host Memory 双缓冲异步流式传输。
- **RTX 5090 (32GB VRAM, Blackwell SM 12.0)**：
  - 支持 `--ltx-fast` 启用硬件原生 **NVFP4** 权重（仅 7.7 GB，体积减半）与 `cuBLAS BlockWise1x16 NVFP4 GEMM` + `Triton FP4 Quantize` 算子。
  - 32GB 显存可将全部 28 个 Transformer Blocks **100% 驻留 GPU 显存**（`resident = 28, streamed = 0`），彻底消除 PCIe 流式拷贝开销，同时支持更大的窗口并发组（`group_size`）与 `--ltx-large-canvas`。
