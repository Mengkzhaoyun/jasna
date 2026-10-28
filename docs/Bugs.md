# 边缘闪烁与时序漂移缺陷分析归档

## 1. 问题描述

早期版本在 Linux Docker（RTX 4090）环境下通过 CLI 输出的视频存在马赛克边缘闪烁与漂移现象，而 Windows GUI 处理同一视频无闪烁。

## 2. 反编译对照结论

通过反编译 GUI 二进制（`jasna-gui.exe`）并与源码关键模块逐项比对：

| 模块 | 对照结论 |
| --- | --- |
| `tracking/blending.py` | `BLEND_DILATION=0.028`、`FALLOFF=0.028`，与源码完全一致 |
| `crop_buffer.py` | `BORDER_RATIO=0.06`、`MIN_BORDER=20`，与源码完全一致 |
| `blend_buffer.py` | 无额外 `edge_feathering`，与源码完全一致 |
| `clip_tracker.py` | `iou_threshold=0.3`，与源码完全一致 |
| `pipeline.py` 与 `pipeline_threads.py` | 所有常量与流水线调度逻辑完全一致 |

结论：GUI 与源码在 Python 层面的推理与融合参数一致。

## 3. 根本原因分析

1. **硬件解码器 B 帧时序问题**：早期 Linux 容器环境使用的 `python_vali.PyDecoder` 接口在部分含有复杂 B 帧的视频流上按解码顺序（DTS）而非显示顺序（PTS）输出帧，导致追踪器（`ClipTracker`）接收到乱序画面，造成追踪框拉扯与轨迹断连。
2. **实验性边缘羽化补丁放大视觉偏移**：早期构建流程曾额外打入 `fix_blend_ratio_increase.patch`（添加半透明边缘羽化）并将裁剪边距扩大至 `0.15 / 50`，当追踪框发生微幅抖动时，扩大的半透明羽化层在原视频背景上产生相对滑动感。

## 4. 最终修复与验证总结

1. **保障解码 PTS 单调有序**：在视频解码层引入基于 PTS 严格重排与 PyAV 回退保障机制（后续在 `v0.11.x` / `v0.12.0` 中由上游 `python_vali==4.8.8` `DecodeSingleSurfaceAsyncDetailed` 与 `VideoReader` 原生 PTS 排序机制彻底固化）。
2. **清理实验性羽化补丁**：移除全部 `.drone/patches/*` 实验性羽化补丁，保持与上游原生融合逻辑一致。
