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

### 3. 上游 `jasna.protection` 与 `.safetensors.enc` 加密架构解析

通过对上游 `v0.11.3` 官方 Linux 发行版二进制常量区与 `protection/_core.so` 的完整分析，其保护层由 18 个子模块构成，核心数据流如下：

1. **文件格式与清单校验（`jasna/protection/model_manifest.py` & `_pins.py`）**：
   - `.safetensors.enc` 文件头部为 29 字节 `JASNAENC1` + `>8sQI`（8 字节 base nonce、8 字节明文总长度、4 字节分块数 `total_chunks`），随后每个分块以 4 字节块长度开头，后接密文与 16 字节 AES-GCM 认证标签。
   - `.manifest.json`（及 `_pins.py` 内置固定清单）记录分块 SHA-256 哈希列表并使用发布者 Ed25519 私钥签名。
2. **密钥信封机制（`jasna/protection/envelope.py` & `license_store.py` & `activation.py`）**：
   - 模型的真实 AES-GCM 内容密钥（`content_key`）**并不存储在客户端代码或 `.enc` 文件中**。
   - 当用户通过 `License code`（`JL1-...`）向激活服务（`https://jasna-activation.kruk2.workers.dev/v1/activate` 或 `https://47.238.135.110/v1/activate`）请求激活时，服务端校验机器标识（Linux 为 `JM1-<sha256>`，基于 `/etc/machine-id`）并返回 `Activation key`（保存在 `license.json`）。
   - `Activation key` 中包含 `wraps["ltx-2.5-v2"]`，即用构建主密钥（`master_key`）通过 HKDF-SHA256 + AES-GCM 封装后的模型 `content_key`，由 `protection/_core.so` 在原生层解封并解密分块。
3. **自建容器与源码环境落地路径**：
   - **路径 A（明文 `.safetensors` 直跑，无需任何 `jasna.protection` 组件）**：只要将解密后或自训练的明文 `.safetensors` 放入 `model_weights/`（或指定 `--restoration-model-path`），`jasna/ltx/model_files.py` 会自动走 `_PlaintextHandle` 原生加载，并通过开源 `BlockStore` + `jasna_ltx_native` 全速运行。
   - **路径 B（官方发行版组件 + 授权激活）**：已将官方 `v0.11.3` 完整 Linux 发行版解压于 4090 服务器 `/data/jasna/.release-cache/v0.11.3/jasna-linux-nvidia-0.11.3/`，并通过 `JASNA_LICENSE` 完成在线激活（离线凭证持久化于 `/root/.config/jasna/license.json` 与 `/data/jasna/license.json`）。同时已将完整 LTX 权重包（`ltx-restore-alpha1-distill2-int8.safetensors.enc`、`vae.safetensors.enc`、`vae-decoder.safetensors.enc` 及 `.manifest.json`）下载校验至 `/data/jasna/model_weights/`。

### 4. RTX 4090 真机热更新与算子实测结果（`36.170.21.98:36226`）

已在 8× RTX 4090 服务器（Driver `595.84`，CUDA `13.2`）完成热更新、授权激活与真实模型算子验证：

- **热更新镜像**：`registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-jasna-v0.11.3`（位于 `/data/jasna/src`，已集成 `python_vali==4.8.8` 容错分支，彻底解决 `DecodeSingleSurfaceAsyncDetailed` 缺失告警）。
- **预编译算子缓存（`jasna/ltx/native/prebuilt/`）**：
  - `jasna_ltx_native.so`（3.42 MB，含 C++/CUDA `block.cpp`、`glue.cu` 及 SageAttention INT8 内核）。
  - `kernels/`：已完成全架构 Triton AOT 内核预编译，涵盖 `sm75-int8`、`sm86-int8`、`sm89-int8`（RTX 4090）、`sm120-int8`、`sm120-nvfp4`（RTX 5090）、`sm121-int8`、`sm121-nvfp4`。
- **RTX 4090 (GPU 3, 24GB VRAM) 实测指标**：
  - `T.require_kernels(cuda:0)`：**通过**（`triton` + `comfy_kitchen` INT8 Attention 就绪）。
  - `LtxTransformer`（`distilled2` INT8，48 层 Transformer Blocks）加载耗时：**4.22 秒**（预编译后免 JIT 编译）。
  - 后端与显存驻留：`backend = native`，**`resident blocks = 48 / 48`（100% 显存常驻，0 PCIe 流式回退）**。
  - 真实授权激活与全链路推理验证：`Jasna: installation verified.`，容器内通过 `/usr/local/bin/entrypoint.sh --restoration-model-name ltx` 端到端视频修复输出通过。

### 5. RTX 4090 vs RTX 5090 统一镜像真机实测对比

已在 **8× RTX 4090 (`36.170.21.98:36226`, SM 8.9)** 与 **2× RTX 5090 (`rtx5090-200.5090.gpu.beagle`, SM 12.0)** 两台服务器上部署同一个统一镜像 `registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-jasna-v0.11.3`，并完成热缓存（`torch-compile-cache`）全 4 阶段实测对比：

| 阶段 / 指标 | RTX 4090 (24GB, Ada SM 8.9) | RTX 5090 (32GB, Blackwell SM 12.0) | 5090 加速比 |
| :--- | :--- | :--- | :--- |
| **自动匹配模型与算子** | `distill2-int8` (13.37 GB, `sm89-int8`) | `distill2-nvfp4` (7.73 GB, `sm120-nvfp4`) | 权重体积 **-42%** |
| **`LTX 1/4 scan` (RF-DETR v6 Large + NVDEC)** | `0.58 s` (`41.24 frame/s`) | **`0.30 s` (`80.09 frame/s`)** | **1.94x** |
| **`LTX 2/4 encode` (3D Causal VAE Encode)** | `21.0 s` (`1.12 frame/s`) | **`8.9 s` (`2.69 frame/s`)** | **2.36x** |
| **`LTX 3/4 denoise` (22B DiT 2-Step)** | `1.68 s` (`1.19 step/s`) | **`0.80 s` (`2.48 step/s`)** | **2.10x** |
| **`LTX 4/4 compose` (VAE Decode + NVENC)** | `6.0 s` (`3.76 frame/s`) | **`2.6 s` (`9.08 frame/s`)** | **2.31x** |
| **4 阶段端到端总耗时（热缓存）** | `~29.3 s` | **`~12.6 s`** | **2.33x** |

- **单镜像跨代兼容要点**：
  1. `entrypoint.sh` 启动时通过 `nvidia-smi --query-gpu=compute_cap` 自动识别架构：`12.*` 且存在 `*nvfp4*.safetensors*` 时自动开启 `--ltx-fast`，`8.9` 自动使用 `INT8`。
  2. 多卡环境下通过 `--gpus all -e CUDA_VISIBLE_DEVICES=<id>` 挂载完整 `/dev/nvidia0` 控制节点，确保 `595.104.02` 驱动下 `hevc_nvenc` (`OpenEncodeSessionEx`) 正常初始化。
  3. 自动从 `/data/jasna/license.json` 恢复离线授权凭证并自动排除 `.release-cache` / `model_weights` 系统目录，实现零环境变量、免重复激活的一键启动。
