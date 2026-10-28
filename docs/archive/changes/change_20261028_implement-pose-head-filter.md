# 实现基于姿态关键点与几何拓扑的人头马赛克剔除器

- **负责人**：Mengsk
- **助手**：Gemini（模块设计、代码实现与单元测试验证）
- **日期**：2026-10-28
- **状态**：已完成
- **所属任务**：T2

## 1. 当前依据

- [实施计划](../../plan/2026.10.multi-gpu-smart-pipeline.plan.md)：T2 任务目标与验收标准。
- [架构设计方案](../../MultiGpuSmartPipeline.md)：第 4.1 节男优与路人头像马赛克剔除机制。
- [人头剔除模块](../../../jasna/mosaic/head_filter.py)：`PoseHeadMosaicFilter` 与 `HeadFilteredDetector` 实现。
- [会话工厂模块](../../../jasna/session_factory.py)：检测器缓存包装与生命周期管理。

## 2. 执行记录

- 2026-10-28，Gemini：新建 [jasna/mosaic/head_filter.py](../../../jasna/mosaic/head_filter.py)，实现 `PoseHeadMosaicFilter`：
  - 按需加载轻量级 `yolov8n-pose.pt` 模型，仅对检出候选马赛克框的视频帧执行关键点推理；
  - 基于 COCO 17 关键点的头肩点（`0~6`）构建 `HeadBox`，基于左/右髋部点（`11~12`）构建 `HipBox`；
  - 当候选框被 `HeadBox` 覆盖比例 $\ge 0.45$、最大边长 $< 0.35H$ 且不与任何 `HipBox` 重叠时，判定为人头马赛克并剔除；
  - 提供同框双码几何拓扑兜底策略，当面部重度打码导致关键点缺失时，自动剔除画面上部（`cy < 0.36H`）小型近似方形人头框并保留中下部（`cy > 0.48H`）核心马赛克框。
- 2026-10-28，Gemini：实现 `HeadFilteredDetector` 透明包装器，同时接管 Stage 1 抽帧扫描接口 `scan_scores_masks()` 与 Stage 2 逐帧检测接口 `__call__()`，并接入 [jasna/session_factory.py](../../../jasna/session_factory.py) 的 `detection_model_for()` 与 `build_compiled_detection_model()`。

## 3. 验收记录

- 语法与字节码检查：`py -3.13 -m compileall -q jasna tests` 通过（退出码 `0`）。
- 单元测试验证：[tests/test_multi_gpu_head_filter.py](../../../tests/test_multi_gpu_head_filter.py) 中的 `test_pose_head_filter_removes_head_mosaic_via_keypoints`、`test_pose_head_filter_never_removes_box_overlapping_hip_region`、`test_geometric_dual_box_topology_fallback_removes_upper_head_mosaic` 与 `test_head_filtered_detector_scan_scores_masks_zeros_head_only_frame` 全部通过。

