# 合并上游基线与切换 Nuitka 容器构建变更记录

- **负责人**：Mengsk
- **助手**：Codex（历史合并分析与执行）、Gemini（标准命名迁移与格式规范化）
- **日期**：2026-06-28
- **状态**：已完成

## 1. 当前依据

- [项目约束](../../../AGENTS.md)：分支管理与变更归档规范。
- [容器构建说明](../../../.drone/README.md)：容器构建与运行工作流说明。
- [项目依赖配置](../../../pyproject.toml)：上游依赖版本与构建配置。

## 2. 合并背景与范围

将本地定制补丁（CI/CD 流水线、Docker 容器环境、编码参数及边缘融合默认项）迁移至上游 `v0.7.2` 基线，核心上游变化包括：

1. **依赖栈升级**：升级至 `torch 2.12.0+cu130`、`torchvision 0.27.0+cu130`、`tensorrt 10.16.1.11` 与 `torch-tensorrt 2.12.0`。
2. **新增图像修复与色彩特性**：新增 `jasna/image_restore.py`（SD 1.5 图像修复）、`jasna/media/lut.py`（`.cube` LUT 支持）与 `jasna/post_export_action.py`。
3. **打包体系演进**：开发构建依赖从 `pyinstaller` 转向 `nuitka>=2.4`，并引入 `.gitmodules` 中的 `jasna/protection` 子模块引用。

## 3. 执行记录

- 2026-06-27，Codex：基于上游 `v0.7.2` 标签创建 `dev` 分支，按顺序移植 23 个本地唯一非合并提交，保留 `.drone/*`、`.github/workflows/*`、`jasna/crop_buffer.py`、`jasna/blend_buffer.py` 与 `jasna/main.py` 定制能力。
- 2026-06-28，Codex：将 `.drone/build.sh` 切换为 Nuitka standalone 构建，同步更新 `.drone/dockerfile.build` 依赖版本，清理已失效的旧版 `.drone/patches/*`，并更新 `.dockerignore` 放行 `.drone/entrypoint.sh` 与构建产物。
- 2026-10-28，Gemini：按统一文档规范采用标准文件名 `change_20260628_merge-upstream-baseline.md` 归入 `docs/archive/changes/`。

## 4. 验收记录

- `dev` 分支相对上游基线的 20 个定制文件差异核对通过，Nuitka 容器构建脚本与依赖版本对齐完成。

