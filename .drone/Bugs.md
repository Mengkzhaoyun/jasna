# Bugs: 边缘闪烁 (Edge Flickering)

## 问题描述

CLI (Linux Docker 4090) 输出视频存在边缘闪烁。
GUI (Windows 4070) 处理**同一个视频**效果很好，没有闪烁。

## 反编译结论

通过 pyinstxtractor 提取 GUI 的 jasna-gui.exe 并反编译全部关键模块：

| 模块                 | GUI 二进制                          | 源码     | 结论 |
| -------------------- | ----------------------------------- | -------- | ---- |
| tracking/blending.py | BLEND_DILATION=0.028, FALLOFF=0.028 | 完全一致 | ✅   |
| crop_buffer.py       | BORDER_RATIO=0.06, MIN_BORDER=20    | 完全一致 | ✅   |
| blend_buffer.py      | 无 edge feathering                  | 完全一致 | ✅   |
| clip_tracker.py      | iou_threshold=0.3                   | 完全一致 | ✅   |
| pipeline.py          | 所有常量一致                        | 完全一致 | ✅   |
| pipeline_threads.py  | 所有常量一致                        | 完全一致 | ✅   |
| restorer/\*          | 所有常量一致                        | 完全一致 | ✅   |
| gui/processor.py     | 参数传递一致                        | 完全一致 | ✅   |

**结论：GUI 和源码在 Python 层面完全一致，没有任何隐藏参数或不同的 pipeline。**

## 排除的原因

- ❌ Python 代码差异 — 反编译证明完全一致
- ❌ 模型没加载 — CLI 日志确认 BasicVSR++ TRT + RF-DETR 都正常加载
- ❌ 参数不同 — 所有参数完全一致

## 最终确定的根本原因 (Root Cause)

**致命元凶：Linux 环境下硬解码器 (PyNvVideoCodec/python_vali) 的 B 帧乱序 Bug。**

通过代码审查和日志分析，我们发现：

1. **Windows GUI 环境**：底层可能回退使用了标准解码库（如 OpenCV 或 PyAV），这些库天生支持按**显示时间戳 (PTS)** 对解码出的 B 帧进行重新排序，保证追踪器收到的是时间连续的画面（0 -> 1 -> 2 -> 3）。
2. **Linux Docker (4090) 环境**：使用了极端压榨性能的 `NvidiaVideoReader` (`python_vali.PyDecoder`) 进行硬件解码。该接口在底层直接按照**解码顺序 (Decode Order)** 吐出帧（例如 0 -> 3 -> 1 -> 2），并且**没有在 Python 层进行 PTS 重新排序**。

**灾难性的链式反应**：
由于画面是乱序输入的，追踪器 (ClipTracker) 以为人物在画面中瞬间来回瞬移。这导致追踪框剧烈拉扯、IoU 匹配失败、甚至频繁断开重建轨迹。在最后融合 (Blend) 时，这些错乱的裁剪框和乱序的帧叠在一起，就形成了**“马赛克解码部分像是浮在画面上飘来飘去”**的灵异现象。

这解释了为什么**所有 Python 代码和网络参数完全一致**，却只有在 Linux Docker 里会出现极其严重的漂移。

## Lada 原版项目是如何避免这个问题的？

Lada / Lada-mosaic 原版项目没有使用这种激进的 NVDEC 硬件直通解码。它一般依赖 OpenCV (`cv2.VideoCapture`) 或 PyAV 进行视频读取，虽然牺牲了一定的解码速度，但框架底层会自动处理 B 帧的 PTS 重排，从根本上避免了时序错乱导致的追踪漂移。

## 解决方案

**方案 A（推荐，最稳定）**：放弃在 Linux 环境下强制使用有缺陷的 `NvidiaVideoReader`，修改 `video_decoder.py`，引入基于 PyAV 或 OpenCV 的回退机制，确保所有输入帧严格按照 PTS 排序。
**方案 B（保留极致性能）**：在 `jasna/pipeline_threads.py` 的解码循环中，增加一个优先队列（PriorityQueue）缓冲池。根据硬件解码器返回的 `pts_list`，手动将乱序的帧按 PTS 重新排序，然后再喂给下游的检测器和追踪器。

## 方案 C（终极真相：实验性补丁导致的视觉漂浮）

在深入对比 Windows GUI 和 Linux CLI 的代码后，我们发现了一个更直接的原因。
GUI 能够完美运行（无漂浮、无闪烁），是因为它使用了默认的较小边界参数且没有边缘融合特效。
而我们在 Linux 容器的打包流程中，为了消除马赛克边界的硬折角，曾打入了两个实验性补丁：

1. `fix_blend_ratio_increase.patch`: 添加了 `edge_feathering` 边缘羽化。
2. `fix_crop_border_expand.patch`: 将边界扩展从 `0.10` / `40` 扩大到了 `0.15` / `50`。

**漂浮感产生的原因：**
当追踪框因为物体的细微运动或网络检测产生即使是 1-2 像素的抖动时，被强行扩大的半透明羽化层就会在原视频背景上产生相对滑动。这种视觉差就是“马赛克像是一张皮浮在脸上”的根本原因。

**原计划解决方案（对齐 GUI）：**

1. 删除 `.drone/patches/fix_blend_ratio_increase.patch`，恢复硬边缘融合。
2. 修改 `.drone/patches/fix_crop_border_expand.patch`，严格对齐 GUI 反编译得出的参数：`BORDER_RATIO = 0.06`，`MIN_BORDER = 20`。
3. 原本计划保留 PTS 乱序修复补丁（方案 B）。

## 最终修复与验证总结 (Final Resolution)

经过实机测试验证，**漂浮感已彻底消失，边缘闪烁问题得到完美解决！** 最终在代码库中实际落实并经得起考验的 Patch 组合策略如下：

1. **彻底解决硬件解码 Bug (落实方案 A)**
   抛弃了不稳定的 `NvidiaVideoReader` 和方案 B 的复杂手动重排机制，新增了 `.drone/patches/use_pyav_video_decoder.patch`。通过 PyAV 实现纯净的 CPU 解码流，天然支持严谨的 PTS 排序。同时删除了多余的 `fix_pts_reordering.patch`。
2. **解决视觉漂浮感 (落实方案 C)**
   移除了早期的羽化实验补丁，并将 `.drone/patches/fix_crop_border_expand.patch` 严格对齐至 GUI 反编译参数 (`0.06` 和 `20`)。

**结论**：CLI (Linux Docker) 的表现已经与 GUI (Windows) 完全对齐，核心原因确认为：**底层时序崩坏 (NVDEC B-frame) + 渲染边距过大 (Patch 漂移)** 共同引发了早期的毁灭性伪影。遵循“保持源码纯净，走 Patch 工作流”的规范，所有修复均已固化在 `.drone/patches/` 构建流中，此恶性 Bug 宣告完结！
