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
scp -r dist_linux/jasna/ root@SERVER:/tmp/jasna_hotfix/

# 3. 热替换进运行中的容器
docker cp /tmp/jasna_hotfix/. jasna:/app/jasna/
docker restart jasna
```

## Production Deployment (RTX 4090 Recommended)

对于配备 RTX 4090 (24GB VRAM) 的生产环境，建议使用以下启动命令以充分榨干算力并实现极限画质：

```bash
docker rm -f jasna || true && \
docker pull registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-jasna-v0.6.0-alpha5 && \
docker run --name jasna \
  -it --rm \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  --gpus all \
  -e EXTRA_ARGS="--batch-size 4 --max-clip-size 150 --temporal-overlap 16 --enable-crossfade --denoise low --detection-score-threshold 0.25 --fp16 --compile-basicvsrpp --log-level info" \
  -v /nas/jasna/model_weights:/app/jasna/model_weights \
  -v /nas/jasna/jasna:/app/jasna/jasna \
  -v /nas/jasna/ai:/data \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-jasna-v0.6.0-alpha5
```

### 核心参数优化说明：

- `--max-clip-size 150`: 大幅增加 BasicVSR++ 的时序上下文窗口（默认90），让模型能参考更多连续帧，极大提升视频一致性和极限画质。4090 的 24GB 显存完美吃下此参数。
- `--temporal-overlap 16`: 配合更长的 clip 增加重叠区，配合 crossfade 实现完美丝滑的拼接过渡。
- `--denoise low`: 开启轻度空间降噪，抹除模型生成的极其细微的杂色噪点，且不损失 4090 跑出的高清纹理细节。
