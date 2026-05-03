# Bugs: 正方形伪影 (Square Artifacts)

## 问题描述

CLI 输出的视频中存在严重的正方形方块伪影（crop boundary artifacts）。  
GUI 处理同一个视频时**完全没有这个问题**。

## 日志对比

### GUI (Windows)

```
BasicVSR++ using TRT sub-engines (fp16=True)
RestorationPipeline: secondary=none denoise=NONE denoise_step=AFTER_PRIMARY
RF-DETR detection model loaded: model_weights\rfdetr-v5.bs4.fp16.win.engine (batch_size=4)
VramOffloader: threshold=7437 MiB (total=8187 MiB, safetynet=750 MiB)
Processing MIDV-041.mp4: 267631 frames @ 29.97 fps, 1920x1080
```

### CLI (Docker Linux)

```
torch_tensorrt._utils ERROR: CUDA 13 is not currently supported for TRT-LLM plugins
[TRT] [W] Functionality provided through tensorrt.plugin module is experimental.
Processing video: 34% (3544f)
```

## 发现的差异

### 1. ⚠️ 不同的视频文件

- GUI: `MIDV-041.mp4`
- CLI: `FC2-2175534.mp4`

**不是同一个视频！** 不同视频有不同的马赛克模式和检测结果。  
需用**同一个视频**在 GUI 和 CLI 分别跑一遍才能公平对比。

### 2. ⚠️ TensorRT CUDA 13 兼容性警告

CLI 日志出现：
```
torch_tensorrt._utils ERROR: CUDA 13 is not currently supported for TRT-LLM plugins
```

这可能导致 TRT 引擎行为异常或回退到非优化路径。  
GUI 在 Windows + CUDA 上正常工作，没有此错误。

### 3. CLI 的 log-level 是 error

CLI 默认 `--log-level error`，所以看不到 INFO 级别的：
- `BasicVSR++ using TRT sub-engines` — **不知道 TRT 是否真正生效**
- `RestorationPipeline: secondary=none` — 无法确认配置
- `VramOffloader` 信息

**建议**: CLI 运行时加 `--log-level info` 查看完整初始化日志。

## 根因分析

### blend_mask 边界截断 (tracking/blending.py)

```python
BLEND_DILATION_RATIO = 0.028   # ~30px at 1080p
BLEND_FALLOFF_RATIO = 0.028    # ~30px at 1080p
```

`create_blend_mask()` 使用 box_blur 做 dilation + falloff。  
当 crop 区域紧贴检测框时，blend_mask 渐变区被 crop 边界硬截断，  
blend weight 从 1.0 突变到 0.0 → 正方形伪影。

### crop 扩展边距不足 (crop_buffer.py)

```python
BORDER_RATIO = 0.06   # 原始值
MIN_BORDER = 20       # 原始值
```

crop 扩展边距太小，导致 blend_mask 的渐变区溢出到 crop 外面被截断。

## 已尝试的修复

### Patch 1: Edge Feathering (blend_buffer.py)

在 blend_mask 边缘加 linear ramp，防止硬截断。

- 状态：**已写入 build.sh** — 需重新构建生产镜像才能生效

### Patch 2: Expand Crop Border (crop_buffer.py)

`BORDER_RATIO` 0.06→0.10，`MIN_BORDER` 20→40。

- 状态：**已写入 build.sh** — 需重新构建生产镜像才能生效

## 下一步

1. **用同一个视频对比** — GUI 和 CLI 跑同一个文件
2. **CLI 加 `--log-level info`** — 确认 TRT 引擎是否正常加载
3. **重新构建生产镜像** — 触发 release 分支让 patch 生效
4. 如果 patch 生效后仍有伪影：
   - `BLEND_DILATION_RATIO` → 0.06+
   - `BLEND_FALLOFF_RATIO` → 0.06+
   - `BORDER_RATIO` → 0.15+
