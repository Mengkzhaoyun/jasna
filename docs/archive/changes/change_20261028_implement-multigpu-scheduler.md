# 实现单容器多 GPU 智能分段与双层并行调度引擎

- **负责人**：Mengsk
- **助手**：Gemini（调度器实现、CLI 集成与单元测试验证）
- **日期**：2026-10-28
- **状态**：已完成
- **所属任务**：T3

## 1. 当前依据

- [实施计划](../../plan/2026.10.multi-gpu-smart-pipeline.plan.md)：T3 任务目标与验收标准。
- [架构设计方案](../../MultiGpuSmartPipeline.md)：第 4.2 ~ 4.4 节关键帧切分、L1/L2 多卡并发与 FFmpeg 混流设计。
- [多卡调度引擎](../../../jasna/multi_gpu.py)：单容器多 GPU 三阶段调度实现。
- [CLI 入口模块](../../../jasna/main.py)：`--multi-gpu`、`--gpus` 与 `--ignore-head-mosaic` 参数及主控分发。
- [LTX 模型文件管理](../../../jasna/ltx/model_files.py)：外置官方发行版 worker（`JASNA_RELEASE_BIN`）协同兼容。

## 2. 执行记录

- 2026-10-28，Gemini：新建 [jasna/multi_gpu.py](../../../jasna/multi_gpu.py)，实现三阶段单容器多卡流水线：
  - **Stage 1（抽帧预扫与关键帧均衡切分）**：`build_smart_multi_gpu_plan()` 结合安全关键帧索引与带人头过滤的 `1 fps` 扫描结果生成 `SpliceSpan` 列表，并通过 `_split_long_render_spans()` 在低分波谷关键帧处将长连续有码区间均衡拆分为多卡子段；
  - **Stage 2 通道 A（无码段极速直通）**：`_encode_unrestored_span()` 跳过 LTX，使用多卡 NVENC 硬件重编码输出对齐目标码率（`5M` 或低码率保护设置）的 `{span_idx:04d}-raw.nut`；
  - **Stage 2 通道 B（有码段 L1 跨片段 + L2 跨轨迹多卡并发）**：`_prescan_span_with_head_filter()` 预先生成过滤人头后的 `.plans.npz` 与 `job.json`（保持 `SessionConfig` 结构不变，与官方冻结二进制 `job_fingerprints()` SHA-256 哈希 100% 兼容）；`_render_mosaic_span_on_gpu()` 与 `_run_l2_collaborative_track_denoise()` 实现跨片段多卡并发及单片段内多条 `TrackPlan` 跨卡协同去噪；
  - **Stage 3（FFmpeg 无损混流）**：串行调用 `normalize_fragment()`、`concatenate_fragments()` 与 `mux_final_output()` 无损拼合全部视频分片并挂载原片音轨、字幕与章节元数据。
- 2026-10-28，Gemini：在 [jasna/main.py](../../../jasna/main.py) 新增 `--multi-gpu`（默认开启）、`--gpus` 与 `--ignore-head-mosaic`（默认开启）CLI 参数，并在离线 LTX 视频修复任务中默认路由至 `_SmartMultiGpuPipeline`。

## 3. 验收记录

- 语法与 CLI 检查：`py -3.13 -m compileall -q jasna tests` 与 `py -3.13 -m jasna --help` 均验证通过。
- 单元测试验证：[tests/test_multi_gpu_head_filter.py](../../../tests/test_multi_gpu_head_filter.py) 中的 `test_split_long_render_spans_balances_across_gpus`、`test_cli_multi_gpu_and_head_filter_flags` 与 `test_physical_cuda_visible_token_mapping` 全部通过。

