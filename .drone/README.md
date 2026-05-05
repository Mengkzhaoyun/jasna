# jasna

## git

```bash
git remote add upstream git@github.com:Kruk2/jasna.git
git fetch upstream
git merge v0.6.0
```

## jasna

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
docker build -f .drone/dockerfile.build -t ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build --build-arg BASE=nvidia/cuda:13.0.3-devel-ubuntu24.04 .

# 2. 本地测试编译流程 (挂载当前代码并执行 build.sh)
docker pull ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build ; `
docker run --rm -it `
  -v "${PWD}:/app/jasna" `
  -w /app/jasna `
  ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build `
  bash .drone/build.sh
```

## Hot Update

```powershell
# 1. 本地编译 (产物在 dist_linux/jasna/)
docker pull ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build ; `
docker run --rm -it `
  -v "${PWD}:/app/jasna" `
  -w /app/jasna `
  ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build `
  bash .drone/build.sh

# 2. 推送到服务器
scp -r dist_linux/jasna/ root@SERVER:/tmp/sglang_hotfix/

# 3. 热替换进运行中的容器
docker cp /tmp/sglang_hotfix/. sglang:/app/sglang/
docker restart sglang
```

## Production Deployment (RTX 4090 Recommended)

对于配备 RTX 4090 (24GB VRAM) 的生产环境，建议使用以下启动命令以充分榨干算力并实现极限画质：

```bash
docker rm -f sglang || true && \
docker pull registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-sglang-v0.6.0-alpha5 && \
docker run --name sglang \
  -it --rm \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  --gpus all \
  -v /nas/sglang/model_weights:/app/sglang/model_weights \
  -v /nas/sglang/sglang:/app/sglang/sglang \
  -v /nas/sglang/ai:/data \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-sglang-v0.6.0-alpha5
```

### 核心参数优化说明：

- `--max-clip-size 150`: 大幅增加 BasicVSR++ 的时序上下文窗口（默认90），让模型能参考更多连续帧，极大提升视频一致性和极限画质。4090 的 24GB 显存完美吃下此参数。
- `--temporal-overlap 16`: 配合更长的 clip 增加重叠区，配合 crossfade 实现完美丝滑的拼接过渡。
- `--denoise low`: 开启轻度空间降噪，抹除模型生成的极其细微的杂色噪点，且不损失 4090 跑出的高清纹理细节。

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
