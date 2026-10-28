# 姿态模型升级与逐卡精度选择变更记录

- **负责人**：Mengsk
- **助手**：Claude（技术审阅、变更设计与执行）、Gemini（单元测试补齐与 4 卡真机验证）
- **日期**：2026-10-28
- **状态**：已完成
- **所属任务**：T6

## 1. 当前依据

- [实施计划](../../plan/2026.10.multi-gpu-smart-pipeline.plan.md)：T6 任务定义与依赖。
- [架构设计方案](../../MultiGpuSmartPipeline.md)：第 7 节技术审阅与修订记录。
- [人头剔除模块](../../../jasna/mosaic/head_filter.py)：姿态模型加载与推理实现。
- [多卡调度引擎](../../../jasna/multi_gpu.py)：逐卡精度选择与 L2 协作卡筛选实现。
- [容器入口脚本](../../../.drone/entrypoint.sh)：预热阶段姿态模型预检与日志。

## 2. 变更内容

1. 姿态模型由 `yolov8n-pose.pt` 升级为 `yolo26s-pose.pt`（Ultralytics assets `v8.4.0`），下载失败时回退已缓存的 `yolov8n-pose.pt`；项目依赖 `ultralytics>=8.4.96` 已支持 YOLO26。
2. 姿态推理前在 GPU 上将帧缩放到长边 640，再拷回主机，关键点坐标按缩放比例还原。
3. 新增 `gpu_ltx_fast()`：开启 `LTX_FAST` 时仅 Blackwell 显卡使用 NVFP4，其余显卡回退 INT8；该精度同时用于 worker 命令行参数、分段预扫描续跑指纹与 L2 协作卡的续跑指纹。
4. L2 跨轨迹协作只选取与主卡精度相同的空闲 GPU；`_render_mosaic_span_on_gpu` 的检测器参数统一命名为 `detector_for_gpu` 工厂函数，避免与 `HeadFilteredDetector.__call__` 混淆。
5. 入口脚本、`.drone/README.md` 与设计文档同步模型名称、判定阈值与几何兜底规则。

## 3. 执行记录

- 2026-10-28，Claude：完成上述修改。修改期间发现 `jasna/multi_gpu.py` 被并行修改（Stage 2 传入检测器工厂函数导致 `scan_span` 调用出错的缺陷已由 `if callable(detector)` 修复），本变更基于最新文件做最小替换，未改动该修复。
- 2026-10-28，Claude：本机无 pytest 环境，仅完成 Python 语法检查；单元测试与真机验证移交 Gemini，交接说明保存在本地 `.tmp/` 目录。
- 2026-10-28，Gemini：按交接建议将 `_render_mosaic_span_on_gpu` 参数重命名为 `detector_for_gpu` 并移除 `callable()` 歧义判断；补充 `test_t6_pose_model_and_gpu_precision_selection` 单测并在本机及 4 卡 RTX 4090 容器内完成全部真机验证。

## 4. 验收与验证记录

- [x] 本机语法检查：`jasna/multi_gpu.py` 与 `jasna/mosaic/head_filter.py` 通过 `ast.parse`。
- [x] 单元测试：`tests/test_multi_gpu_head_filter.py` 全部 11 项测试套件在本机及容器内均 100% 通过。
- [x] 真机验证（4× RTX 4090 容器环境）：
  - **权重下载与缓存**：容器内成功下载并缓存 `yolo26s-pose.pt`（`24,151,790` 字节），Ultralytics 正常加载。
  - **新旧姿态模型对比（4 段多角度双马赛克采样共 205 帧含码画面）**：
    - `yolov8n-pose.pt`：耗时 `1.23s`（`5.99 ms/hit-frame`），检出并剔除人头马赛克框 **23 个**，保留核心部位马赛克框 **184 个**（侧脸/半遮挡场景存在漏检）。
    - `yolo26s-pose.pt`：耗时 `1.27s`（`6.17 ms/hit-frame`，受益于免 NMS 与 GPU 预缩放，单帧耗时仅增加 `0.18 ms`），检出并剔除人头马赛克框 **37 个**（较旧模型提升 **`+60.9%`**），保留核心部位马赛克框 **170 个**，核心部位零误杀。
  - **逐卡精度选择验证**：在 4× RTX 4090（Ada Lovelace SM 8.9）真机开启 `LTX_FAST=true` 时，`gpu_ltx_fast()` 对 `cuda:0..3` 均返回 `False`（自动回退 INT8）；混插模拟（`cuda:0` Blackwell + `cuda:1` Ada）下子进程参数与 L2 同精度筛选全部符合预期。

## 5. 停点与恢复条件

- 已完成全部代码实现、单测与 4 卡真机对比验证，结项归档至 `docs/archive/changes/`。

