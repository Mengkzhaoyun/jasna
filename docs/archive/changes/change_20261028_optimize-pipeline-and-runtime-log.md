# 输出容器统一 MP4/HEVC、多卡管线深度优化与结构化运行日志变更记录

- **负责人**：Mengsk
- **助手**：Gemini（方案评估、代码实现与单元测试验证）
- **日期**：2026-10-28
- **状态**：已完成
- **所属任务**：T9

## 1. 当前依据

- [实施计划](../../plan/2026.10.multi-gpu-smart-pipeline.plan.md)：T9 任务定义与审批记录。
- [架构设计方案](../../MultiGpuSmartPipeline.md)：多卡分段、无码间隙阈值、尾部空窗并发与运行日志设计规范。
- [多卡调度引擎](../../../jasna/multi_gpu.py)：分段阈值、尾部空窗并发、TS 直接归一化与输出后缀守卫实现。
- [运行日志模块](../../../jasna/runtime_log.py)：对齐批量运行日志规范的单文件结构化日志与四类核心指标摘要实现。
- [媒体探测模块](../../../jasna/media/probe.py)：视频流码率与容器总码率分离解析实现。
- [容器入口脚本](../../../.drone/entrypoint.sh)：`-U.mp4` 输出命名与单批次日志串联实现。

## 2. 变更内容

### 2.1 输出容器统一为 `.mp4`（`HEVC / hev1` + `faststart`）

1. **容器批处理入口（`.drone/entrypoint.sh`）**：将输出文件路径由 `output="${dir}/${name}-U.mkv"` 统一改为 `output="${dir}/${name}-U.mp4"`。
2. **目录模式输出路径（`jasna/media/media_files.py`）**：`folder_output_path()` 对所有视频输入的默认输出后缀统一设为 `.mp4`。
3. **多卡管线入口守卫（`jasna/multi_gpu.py`）**：`run_multi_gpu_video_pipeline()` 对传入的 `output_video` 强制归一为 `.mp4` 后缀，交由 `concatenate_fragments()` 封装为带 `-tag:v hev1 -movflags +faststart` 的标准 MP4 文件。

### 2.2 多卡管线三项深度性能优化

1. **优化 1：渲染块内部无码间隙上限由 `180.0s` 下调至 `35.0s`（`DEFAULT_MAX_INNER_GAP_SECONDS = 35.0`）**：
   - 仅将 `<= 35s` 的短无码过渡并入渲染块内部（由子进程 Pass 1 帧级扫描自动跳过 LTX 2/3/4），而将 `> 35s` 的长无码区间保留在通道 A（`82.6x` NVENC 硬件快编通道），避免长无码区间在子进程内执行逐帧解码与检测扫描。
   - Stage 1 计划缓存键升级为 `task-chunks-v3`。
2. **优化 2：通道 A 与通道 B 尾部空窗重叠执行（Tail-Idle-Window Overlap）**：
   - 将通道 A（无码段 NVENC 快编）并入 `_gpu_worker_loop()`：每张 GPU 优先执行分配给本卡的通道 B（LTX 渲染大块），当本卡率先完成最后一块 LTX 大块而其余 GPU 仍在收尾时，立即在同一张卡的尾部等待空窗内执行本卡分担的无码段 NVENC 快编，将通道 A 的独立墙钟耗时完全掩蔽。
3. **优化 3：消除 `mp4 -> nut` 冗余中转并启用 Stage 3 多线程并行 TS 归一化**：
   - `_render_mosaic_span_on_gpu()` 在子进程生成 `span_out_mp4` 后直接调用 `normalize_fragment()` 归一化为 `.jasna-resume/<stem>/norm/{idx:04d}.ts`，不再写冗余 `.nut` 中转文件。
   - Stage 3 对剩余分片的 `normalize_fragment()` 改为 `ThreadPoolExecutor(max_workers=8)` 并行执行。

### 2.3 单文件结构化运行日志（`jasna/runtime_log.py`）

1. **统一单行日志格式**：`<YYYY-MM-DD HH:MM:SS> Jasna <任务标识> [<序号>/<总数>] <阶段> <详情>`，同时写入终端与 `$LOG_DIR/<RUN_ID>/jasna.log`，自动拦截 Python `stdout/stderr`、`\r` 进度条、`logging` 及子进程输出，并对 `--license-code` 等敏感参数自动脱敏。
2. **流码率与容器码率严格分离（`jasna/media/probe.py`）**：新增 `parse_stream_and_container_bitrate()`，分别解析并记录 `视频码率`（及来源 `视频流bit_rate` / `视频流tag:BPS`）与 `文件总码率`（及来源 `容器bit_rate`），杜绝混填。
3. **四类核心统计摘要原生内嵌**：
   - **片源与目标基本信息**：在 `任务开始` 与 `处理目标` 阶段输出分辨率、帧率、总帧数、时长、视频流码率、文件总码率、十进制文件大小（`MB/GB`）及编码目标配置。
   - **Stage 1 分析时间与免处理占比**：输出 `[Stage 1 分析汇总] 分析时间=... 片源总时长=... 免LTX处理时长=... (占比=..., 含无码与纯人头码跳过) 需LTX修复时长=... (占比=...)`。
   - **Stage 2 每卡处理时长与实际耗时**：输出每张 GPU 的 `[Stage 2 修复明细]`、4 卡汇总 `[Stage 2 修复汇总]` 以及 `[Stage 2 快编汇总]`。
   - **处理完毕与批次结束汇总**：输出最终 `.mp4` 路径、分辨率、帧率、视频码率、文件总码率、十进制大小、`总分析时间`、`总处理时间`、`处理时长`、`处理速度` 与 `处理帧数`。

## 3. 执行记录

- 2026-10-28，Gemini：完成 `jasna/media/probe.py`、`jasna/runtime_log.py`、`jasna/multi_gpu.py`、`jasna/main.py`、`jasna/media/media_files.py` 与 `.drone/entrypoint.sh` 的代码实现。
- 2026-10-28，Gemini：编写并运行 `tests/test_runtime_log.py`（5 项单测）与 `tests/test_multi_gpu_head_filter.py`（9 项单测），全部通过。

## 4. 验收与验证记录

- [x] 输出容器统一 `.mp4`（`HEVC / hev1`）代码与入口脚本检查通过，远端既有产物已完成无损重封装验证。
- [x] `tests/test_runtime_log.py` 全套 5 项测试（含码率来源分离、单文件日志生命周期、进度条 `\r` 折行、四类摘要格式化、默认 `.mp4` 输出路径与 `35.0s` 内部间隙阈值）100% 通过。
- [x] `tests/test_multi_gpu_head_filter.py` 与 `python -m compileall jasna tests` 语法及回归测试 100% 通过。

## 5. 停点与恢复条件

- 停点：无（已完成全部本机实现与单测验证并结项归档）。
- 恢复条件：无。

