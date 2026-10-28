# 更新容器入口脚本与统一文档归档结构

- **负责人**：Mengsk
- **助手**：Gemini（容器脚本更新、文档迁移、隐私脱敏与归档整理）
- **日期**：2026-10-28
- **状态**：已完成
- **所属任务**：T4

## 1. 当前依据

- [实施计划](../../plan/2026.10.multi-gpu-smart-pipeline.plan.md)：T4 任务目标与验收标准。
- [项目约束](../../../AGENTS.md)：计划与变更目录规范、唯一归档目录（`docs/archive/`）、标准文件命名与隐私边界。
- [容器入口脚本](../../../.drone/entrypoint.sh)：`MULTI_GPU`、`GPUS`、`IGNORE_HEAD_MOSAIC` 与 `JASNA_RELEASE_BIN` 环境变量集成。
- [容器部署说明](../../../.drone/README.md)：生产环境单容器多卡与人头剔除配置说明。

## 2. 执行记录

- 2026-10-28，Gemini：更新 [.drone/entrypoint.sh](../../../.drone/entrypoint.sh) 与 [.drone/README.md](../../../.drone/README.md)，默认开启 `MULTI_GPU=true` 与 `IGNORE_HEAD_MOSAIC=true`，自动探测并导出 `JASNA_RELEASE_BIN` 供加密权重 worker 调用，并在 `warmup` 子命令中增加 `yolov8n-pose.pt` 姿态模型预检缓存步骤。
- 2026-10-28，Gemini：更新 [AGENTS.md](../../../AGENTS.md) 对齐统一文档规范，建立 `docs/plan/`、`docs/changes/` 与唯一归档根目录 `docs/archive/`（含 `docs/archive/changes/` 与 `docs/archive/plan/`）。
- 2026-10-28，Gemini：将历史变更文档按 `change_YYYYMMDD_<动作与对象>.md` 标准命名规范化并迁入 `docs/archive/changes/`，清理公开文档中的具体机器网络地址与主机名，修正全部相对链接与两列表格格式。

## 3. 验收记录

- `docs/plan/`、`docs/changes/` 与 `docs/archive/` 目录结构、文件命名、标题编号、两列表格及隐私边界已全部核对通过。

