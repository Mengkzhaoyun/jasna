# jasna

## git

<https://github.com/Kruk2/jasna>

```bash
git remote add upstream git@github.com:Kruk2/jasna.git
git fetch upstream
git switch dev
git merge v0.11.3
```

## CI/CD

```powershell
# 触发 release 构建
git checkout release ;`
  git merge main --ff-only ;`
  git push origin release ;`
  git checkout main

# 触发 build 构建
git checkout build ;`
  git merge main --ff-only ;`
  git push origin build ;`
  git checkout main
```

## Local Debugging

```powershell
# 1. 本地测试构建 builder 环境镜像
docker build -f .drone/dockerfile.build -t ghcr.io/mengkzhaoyun/jasna:v0.11.3-build --build-arg BASE=nvidia/cuda:13.0.3-devel-ubuntu24.04 .

# 2. 本地测试运行镜像构建 (内置预编译 LTX C++/CUDA 算子与 Triton AOT 内核)
docker build -f .drone/dockerfile -t registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-jasna-v0.11.3 .
```

## Hot Update

```powershell
# 快速热更新源码和 entrypoint:
#   本地 jasna/              -> /data/jasna/src/jasna              -> 容器 /app/jasna/jasna & site-packages/jasna
#   本地 .drone/entrypoint.sh -> /data/jasna/src/entrypoint.sh      -> 容器 /usr/local/bin/entrypoint.sh
.\.tmp\hot-update.ps1
```

## Production Deployment (RTX 4090 / RTX 5090 统一镜像)

自 `v0.11.3` 起，容器已默认内置 `RESTORATION_MODEL_NAME=ltx`（LTX-2.5 22B 2-step 视频去马赛克修复），并具备全自动发现与硬件架构自适应能力：
- **模型路径自动探测**：自动发现 `/data/jasna/model_weights` 或 `/app/sglang/model_weights`。
- **离线授权自动恢复**：自动从 `/data/jasna/license.json` 恢复离线证书，无需每次激活或传递 `JASNA_LICENSE`。
- **4090 / 5090 单镜像自适应**：启动时自动探测 GPU Compute Capability，在 **RTX 4090 (`SM 8.9`)** 上自动使用 `INT8` 算子，在 **RTX 5090 (`SM 12.0`)** 且存在 `nvfp4` 权重时自动开启 `LTX_FAST=true`（`NVFP4` 硬件加速，全流程提速约 **2.3x**）。

### 1. LTX 模式生产部署（默认推荐，4090 / 5090 通用）

```bash
docker rm -f sglang || true && \
docker run --name sglang \
  -it --rm \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TARGET_BITRATE=5M \
  -v /etc/machine-id:/etc/machine-id:ro \
  -v /data/jasna:/data/jasna \
  -v /nas/sglang/ai:/data \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-jasna-v0.11.3
```

> **提示**：
>
> - 4090 与 5090 节点的离线激活凭证已持久化于 `/data/jasna/license.json`，挂载 `-v /data/jasna:/data/jasna` 即可免激活离线直跑。
> - 在双卡/多卡节点（如 5090 双卡或 4090 八卡）上，建议使用 `--gpus all -e CUDA_VISIBLE_DEVICES=<卡号>` 指定目标显卡，以保证 `libnvidia-encode` (NVENC) 正常识别设备节点。
> - 如需切换回传统极速模式（BasicVSR++），传入 `-e RESTORATION_MODEL_NAME=basicvsrpp` 即可。

### 2. BasicVSR++ 极速模式部署（无需官方 release 挂载与授权）

```bash
docker rm -f sglang || true && \
docker run --name sglang \
  -it --rm \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e RESTORATION_MODEL_NAME=basicvsrpp \
  -e TARGET_BITRATE=5M \
  -v /nas/sglang/model_weights:/app/sglang/model_weights \
  -v /nas/sglang/ai:/data \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-jasna-v0.11.3
```

### 核心参数优化说明

- `RESTORATION_MODEL_NAME=ltx`（内置默认）：支持 `ltx`（默认 2-step）、`ltx-4step`（4-step）、`ltx-8step`（8-step）、`ltx-undistilled` 及 `basicvsrpp`。
- `RESTORATION_MODEL_PATH`（内置自动探测）：容器启动时自动桥接 `/app/sglang/model_weights` 与 `/data/jasna/model_weights` 并定位 `vae.safetensors(.enc)` 所在目录。
- `LTX_FAST=false`：设为 `true` 时启用 RTX 50 系（Blackwell）专属 NVFP4 模型。
- `LTX_LARGE_CANVAS=false`：设为 `true` 时以 768px 画布（默认 512px）修复大面积马赛克。
- `--max-clip-size 150`（仅 `basicvsrpp`）：大幅增加 BasicVSR++ 的时序上下文窗口（默认90），让模型能参考更多连续帧，极大提升视频一致性和极限画质。4090 的 24GB 显存完美吃下此参数。
- `--temporal-overlap 16`：配合更长的 clip 增加重叠区，配合 crossfade 实现完美丝滑的拼接过渡。
- `--denoise low`：开启轻度空间降噪，抹除模型生成的极其细微的杂色噪点，且不损失 4090 跑出的高清纹理细节。
- `TARGET_BITRATE=5M`：直接把 5M 目标码率传给 Jasna 内部 VALI/NVENC，避免先输出高码率文件再用 FFmpeg 二次转码。
- `PRESERVE_LOW_BITRATE=true`：批量扫描时先读取输入视频码率，若输入视频码率已经小于等于 `TARGET_BITRATE`，仍正常处理视频，但改用不限码率上限的 NVENC 设置，避免重编码把小文件压到 5M。旧的 `SKIP_LOW_BITRATE` 变量仍兼容，但语义已变为低码率不压缩。
- `LOW_BITRATE_ENCODER_SETTINGS=...`：可覆盖低码率输入使用的 NVENC 设置；默认是 `rc=vbr,maxrate=0,bufsize=0,cq=20,g=60`。

### 编码流水线说明

当前镜像默认采用全硬件单阶段处理（已集成 `python_vali==4.8.8` 容错分支，支持 `DecodeSingleSurfaceAsyncDetailed`）：

```text
VALI/NVDEC 硬件解码 (PyAV 自动兜底) -> RF-DETR v6 检测 -> LTX 22B (INT8/NVFP4) 或 BasicVSR++ 修复 -> VALI/NVENC 5M 硬件编码 -> 音频/元数据 remux
```

之前的 `POST_COMPRESS_BITRATE=5M` 是二阶段方案：

```text
Jasna 输出 -> FFmpeg 重新解码整片 -> hevc_nvenc 二次压缩 -> 输出
```

这个二阶段会多一次完整视频读写和解码/编码调度，所以经常表现为总耗时很长、GPU 利用率低。现在默认关闭后压缩；只有显式设置 `POST_COMPRESS_BITRATE=5M` 时才会启用它作为兜底实验。

### 模型手动下载 (离线/网络受限环境)

Jasna 容器运行依赖挂载目录（如 `/nas/sglang/model_weights`）中的模型权重文件。如果生产部署环境无法联网下载，可提前在宿主机使用 `curl` 准备以下模型：

```bash
# 假设您的挂载目录是 /nas/sglang/model_weights
mkdir -p /nas/sglang/model_weights
cd /nas/sglang/model_weights

# 1. LTX-2.5 22B 去马赛克修复模型套件 (默认推荐，支持国内 hf-mirror 高速直连)
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/vae.safetensors.enc
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/vae.safetensors.enc.manifest.json
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/vae-decoder.safetensors.enc
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/vae-decoder.safetensors.enc.manifest.json
# RTX 4090 / SM 8.9 推荐 INT8 2-step 蒸馏权重 (13.3 GB):
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/ltx-restore-alpha1-distill2-int8.safetensors.enc
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/ltx-restore-alpha1-distill2-int8.safetensors.enc.manifest.json
# RTX 5090 / SM 12.0 专属 NVFP4 2-step 蒸馏权重 (7.7 GB，配合 LTX_FAST=true):
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/ltx-restore-alpha1-distill2-nvfp4.safetensors.enc
curl -L -O https://hf-mirror.com/Kruk2/ltx-restore/resolve/main/ltx-restore-alpha1-distill2-nvfp4.safetensors.enc.manifest.json

# 2. BasicVSR++ 传统去马赛克修复模型
curl -L -O https://hf-mirror.com/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth

# 3. 最新 RF-DETR v6 高清/4K 检测模型 (推荐，检出率更优)
# (如遇直连 GitHub 较慢，可在链接前添加 ghproxy 代理，例如: https://ghproxy.net/https://github.com/...)
curl -L -O https://ghproxy.net/https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx

# 4. 稳定版 RF-DETR v5 检测模型 (通用兼容，可选)
curl -L -O https://ghproxy.net/https://github.com/Kruk2/jasna/releases/download/v0.7.1/rfdetr-v5.onnx

# 5. YOLO v4 极速轻量检测模型 (二次元/低显存可选)
curl -L -O https://hf-mirror.com/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt
```

## Video2X (480p/720p 视频增强)

对于 480p 或 720p 的真人视频资源，可以使用 Video2X 增强并放大至 1080p。相关镜像与详细配置请参考：

```bash
# 运行 Video2X 使用 4090 显卡处理真人视频 (使用最新的 realesr-general-x4v3 模型，强制输出 1080p)
# 建议挂载模型目录，避免每次下载：-v /nas/video2x/models:/root/.local/share/video2x -v /nas/video2x/cache:/root/.cache/video2x
docker run --gpus all --privileged -it --rm \
  -v "$PWD":/host \
  registry.cn-qingdao.aliyuncs.com/wod/video2x:6.1.1 \
  -i /host/input.mp4 \
  -o /host/output_1080p.mp4 \
  -p realesrgan \
  -h 1080 \
  --realesrgan-model realesr-general-x4v3
```

### 模型手动下载 (离线/网络受限环境)

根据 [Real-ESRGAN v0.3.0 更新日志](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.3.0)，目前最新且最轻量的通用视频增强模型是 `realesr-general-x4v3`（以及带降噪功能的 `realesr-general-wdn-x4v3`）。

如果容器内的自动下载因网络问题失败，您可以提前使用 `curl` 下载模型文件，并放置在挂载的模型缓存目录中：

```bash
# 假设您的挂载目录是 /nas/video2x/models
# 1. 创建对应的存放路径
mkdir -p /nas/video2x/models/realesrgan/weights/
cd /nas/video2x/models/realesrgan/weights/

# 2. 下载 v0.3.0 最新通用模型 (realesr-general-x4v3)
# (如遇国内网络直连 GitHub 失败，可在链接前添加代理，例如：curl -L -O https://ghproxy.net/https://github.com/...)
curl -L -O https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

# [可选扩展] 下载带有更强降噪(Denoise)功能的通用版本
curl -L -O https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth
```
