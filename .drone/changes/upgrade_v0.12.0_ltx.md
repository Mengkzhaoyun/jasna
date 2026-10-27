# 升级上游 v0.12.0 暨 LTX 算子与断点续跑变更说明

- 负责人：Mengsk
- 助手：Gemini（变更设计、执行）
- 变更日期：2026-10-27
- 目标分支：`dev`
- 基线版本：`v0.11.3`（`6f6e311`）至 `v0.12.0`（`79d9f11`）

## 1. 当前依据

- [upgrade_v0.11.3_ltx.md](./upgrade_v0.11.3_ltx.md)：上游 `v0.11.3` LTX 架构与本地定制基线说明。
- [jasna/ltx/model_files.py](../../jasna/ltx/model_files.py)：LTX 模型版本分支绑定（`0.12.0`）与文件大小校验逻辑。
- [jasna/media/video_encoder.py](../../jasna/media/video_encoder.py)：视频硬件编码器与本地 NVENC 参数兼容层。
- [.drone/entrypoint.sh](../entrypoint.sh)：容器启动与多版本运行时自动探测脚本。

## 2. 合并范围与核心变更

### 2.1 上游 v0.12.0 核心特性

1. **LTX 断点续跑（Job Resume）与自动分段**：新增 [jasna/job_resume.py](../../jasna/job_resume.py)，支持 LTX 视频修复任务中断后按相同输入与输出路径恢复已完成的跟踪段。
2. **LTX 模型分支版本化校验**：[jasna/ltx/model_files.py](../../jasna/ltx/model_files.py) 将 `0.12.0` 权重绑定至独立分支（`https://huggingface.co/Kruk2/ltx-restore/resolve/0.12.0` 及 `https://hf-mirror.com/Kruk2/ltx-restore/resolve/0.12.0`），并在加载前校验 `.enc` 文件字节大小以识别旧版权重；共用 `vae.safetensors(.enc)` 与 `vae-decoder.safetensors(.enc)` 保持不变。
3. **无 Triton 预编译内核加载与跨厂商加速**：[jasna/ltx/native/__init__.py](../../jasna/ltx/native/__init__.py) 与 [jasna/ltx/kernel_policy.py](../../jasna/ltx/kernel_policy.py) 支持直接加载预编译内核而无需运行时 Triton JIT，并完善了 NVIDIA / AMD 双栈算子调度。
4. **片段拼接与色彩/锐化一致性修复**：修复 `--segments` 模式下 LUT 与 CAS 锐化全片应用逻辑（移除旧版 `_lut_flags` 队列），以及 MP4 GOP 中间切点的智能重编码衔接。

### 2.2 冲突解决与本地定制保留

| 文件 | 处理说明 |
| --- | --- |
| [jasna/media/video_encoder.py](../../jasna/media/video_encoder.py) | 移除上游已废弃的 `self._lut_flags` 队列初始化，完整保留 `_LEGACY_NVENC_OPTION_RENAMES` 与 `maxrate`/`bufsize` 纯数字自动补 `k` 逻辑 |
| [jasna/media/encoder_settings.py](../../jasna/media/encoder_settings.py) | 自动合并，保留本地扩展的 NVENC 别名白名单 |
| [jasna/__main__.py](../../jasna/__main__.py) | 自动合并，保留 `sglang` / `jasna-cli` CLI 识别入口 |
| [jasna/main.py](../../jasna/main.py) | 自动合并，保留 `--disable-ffmpeg-check` 参数与默认 `--restoration-model-name ltx` |
| [jasna/os_utils.py](../../jasna/os_utils.py) | 自动合并，保留 `JASNA_DISABLE_FFMPEG_CHECK=1` 环境变量跳过机制 |
| [pyproject.toml](../../pyproject.toml) | 版本号升至 `0.12.0`，保留注释 `# "python_vali==4.8.8"` |
| [.drone/entrypoint.sh](../entrypoint.sh) | 在 `rel_cand` 探测列表中优先加入 `v0.12.0` 发行版路径，向下兼容 `v0.11.3` |
| [.drone/dockerfile](../dockerfile) | 基础构建镜像标签同步更新为 `ghcr.io/mengkzhaoyun/jasna:v0.12.0-build` |

## 3. 执行与验证记录

- **执行助手**：Gemini
- **语法与字节码检查**：执行 `py -3.13 -m compileall -q jasna`，全量 Python 模块编译通过（退出码 `0`）。
- **冲突状态检查**：`git diff --check` 无残留冲突标记或空白错误。

