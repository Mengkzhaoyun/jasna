# 编制单容器多 GPU 智能分段与人头马赛克剔除架构设计

- **负责人**：Mengsk
- **助手**：Gemini（架构设计、SVG 流程图绘制与审阅修订）
- **日期**：2026-10-28
- **状态**：已完成
- **所属任务**：T1

## 1. 当前依据

- [实施计划](../../plan/2026.10.multi-gpu-smart-pipeline.plan.md)：所属计划任务分解与审批记录。
- [架构设计方案](../../MultiGpuSmartPipeline.md)：经 Mengsk 审批通过的正式设计文档。
- [多卡流水线架构图](../../assets/multi_gpu_pipeline.svg)：端到端三级流水线 SVG 架构图。
- [人头剔除决策图](../../assets/head_mosaic_filter.svg)：基于姿态关键点与几何拓扑的人头马赛克剔除 SVG 流程图。

## 2. 执行记录

- 2026-10-28，Gemini：梳理 `jasna/mosaic_scan.py`、`jasna/media/splice.py` 与 `jasna/ltx/restore.py` 现有能力边界，设计“Stage 1 极速预扫与关键帧切分 + Stage 2 单容器多 GPU 双层并发（L1 Span 级 + L2 Track 级）+ Stage 3 FFmpeg 无损混流”三阶段架构。
- 2026-10-28，Gemini：在 `docs/assets/` 下绘制 `multi_gpu_pipeline.svg` 与 `head_mosaic_filter.svg`，并在 `docs/MultiGpuSmartPipeline.md` 中完成引用与排版。
- 2026-10-28，Gemini：根据审阅人 **Mengsk** 的评审指示，在设计文档第 6 节固化三项核心决议（保持 `5M` 目标码率、默认开启人头马赛克剔除、采用单容器多 GPU 方案）。

## 3. 验收记录

- 架构设计文档与两幅 SVG 流程图已通过审阅人 **Mengsk** 批准，文档标题编号、两列表格与相对链接均符合规范。

