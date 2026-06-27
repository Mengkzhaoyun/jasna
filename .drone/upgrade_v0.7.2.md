# 合并上游 v0.7.2 升级分析

分析时间：2026-06-27  
当前本地分支：`main` / `40a9275`  
目标上游标签：`v0.7.2` / `278ab09`  
上游远端：`upstream git@github.com:Kruk2/jasna.git`

## 结论

不要直接在当前 `main` 上执行 `git merge v0.7.2` 或 `git merge upstream/main`。

原因是本地历史和上游标签的共同祖先退到了很早的 `0.1`：

```powershell
git merge-base HEAD v0.7.2
# 8da023f... (tag: 0.1)
```

直接 merge 会把大量同名源码文件识别成 add/add 冲突。`git merge-tree --write-tree --name-only HEAD v0.7.2` 探测到大批冲突，包含 `README*`、`jasna/main.py`、`jasna/gui/*`、`jasna/restorer/*`、`pyproject.toml`、大量 tests 等。这个冲突量主要来自历史重写/重排，不是业务改动真的都需要手工三方合并。

推荐路线是：以 `v0.7.2` 为新的基线建分支，然后只移植本地真正唯一的补丁。

## 版本与分叉状态

已执行：

```powershell
git fetch upstream --tags --prune
```

拉到了新标签：

- `v0.7.0` -> `ea15ba3`
- `v0.7.1` -> `56fc4f0`
- `v0.7.2` -> `278ab09`

注意：`upstream/main` 当前在 `v0.7.2` 后面还有 2 个提交：

```text
b32fe4c add missing so on linux build
a89e416 bundle ffmpeg and mkv on linux
```

本次如果目标是严格升级到 `v0.7.2`，不要把这两个提交混进来。若目标变成“升级到上游最新 main”，可以在完成 v0.7.2 迁移后再单独评估。

原始提交计数：

```text
HEAD-only:    340 commits
v0.7.2-only: 406 commits
```

按 patch 去重后：

```text
local-only:    24 commits
upstream-only: 90 commits
```

其中本地真正需要移植的非 merge 提交是 23 个，主要集中在 Drone/CI、Docker、构建补丁、少量 CLI/编码参数和 crop/blend 行为。

## 推荐合并步骤

建议在新分支操作，不要直接改当前 `main`：

```powershell
git fetch upstream --tags --prune
git status --short --branch
git switch -c upgrade/v0.7.2 v0.7.2
git submodule update --init --recursive
```

然后移植本地唯一非 merge 提交。探针显示以下命令能完整移植 23 个提交：

```powershell
$commits = git log --reverse --no-merges --cherry-pick --left-only --format=%H main...v0.7.2
foreach ($c in $commits) {
  git cherry-pick -x -X theirs $c
}
```

说明：在 cherry-pick 场景里，`-X theirs` 表示冲突时偏向“正在移植的本地提交”。它不是最终免审开关，只是探针验证过的低冲突移植方式。执行后仍要审查最终 diff，尤其是 `.drone/patches` 和构建脚本。

如果想更保守，可以不用 `-X theirs` 逐个 cherry-pick。探针结果是前 10 个提交可以直接应用，第 11 个 `630db7b` 会在 `.gitignore` 冲突。该冲突可以保留上游内容，并追加本地需要的：

```text
/.tmp
/.pip_cache
*.orig
*.rej
```

## 本地唯一补丁列表

按移植顺序：

```text
a238b09 feat: add CI/CD pipelines and runtime environment configurations for jasna image building
33edaf3 feat: add GitHub Actions workflows for building and deploying Jasna container images
22787fb ci: add GitHub Actions workflows for building jasna runtime and builder images
e80d88d docs: add README.md to .drone directory detailing git and build workflows
d69bbd2 docs: add README.md to .drone directory outlining git workflows and build triggers
3be3162 docs: add README instructions for drone CI build management
7058b4f feat: add GitHub Actions workflows for building builder and runtime container images
f6ce07f feat: add GitHub Actions workflow for automated release builds and include build documentation
3162ad5 feat: add shell script to automate drone build process
13a49b1 feat: add crop processing utilities and CI/CD pipeline configuration
630db7b feat: add automated build pipeline with artifact extraction and source patching scripts
5f45a9e feat: implement automated CI/CD pipeline using Drone with build scripts, custom patches, and environment documentation.
33d7f85 feat: add PTS reordering logic, adjust crop border parameters, and update gitignore exclude rules
7d96eef refactor: replace nvidia-video-codec-sdk with PyAV for video decoding in NvidiaVideoReader
880971c refactor: replace nvidia-video-codec with PyAV for video decoding in NvidiaVideoReader
4972e57 docs: add documentation for drone pipeline configurations in .drone directory
80b45c0 docs: add README for .drone configuration directory
570951c feat: add docker support with custom runtime environment and batch processing entrypoint
6db63b2 feat: add Docker entrypoint script for automated video processing and TensorRT warmup
ec2d493 docs: add README.md to .drone directory
3709e55 feat: enhance video encoding settings with TARGET_BITRATE and SKIP_LOW_BITRATE options
405bc0b feat: add container entrypoint script and documentation for automated batch video processing
40a9275 feat: add command-line interface for video restoration and streaming configuration
```

## 探针结果

我在临时 worktree 里做过两轮验证，没有改当前工作区。

直接 cherry-pick：

```text
APPLIED_COUNT=10
FAILED_COMMIT=630db7b feat: add automated build pipeline with artifact extraction and source patching scripts
CONFLICTS:
.gitignore
```

使用 `git cherry-pick -X theirs`：

```text
APPLIED_WITH_X_THEIRS_COUNT=23
FAILED_COMMIT=
```

最终相对 `v0.7.2` 只剩 20 个文件有差异：

```text
A .drone/Bugs.md
A .drone/README.md
A .drone/build.sh
A .drone/dockerfile
A .drone/dockerfile.build
A .drone/entrypoint.sh
A .drone/patches/fix_crop_border_expand.patch
A .drone/patches/fix_pyinstaller_noconfirm.patch
A .drone/patches/fix_pyinstaller_spec.patch
A .drone/patches/skip_ffmpeg_version_check.patch
A .drone/patches/use_pyav_video_decoder.patch
A .github/workflows/jasna-build.yml
A .github/workflows/jasna.yml
M .gitignore
M jasna/blend_buffer.py
M jasna/crop_buffer.py
M jasna/main.py
M jasna/media/__init__.py
M tests/test_main.py
M tests/test_media_init.py
```

这个结果比较理想：`v0.7.2` 新增的 `jasna/image_restore.py`、`jasna/media/lut.py`、`jasna/restorer/sd15_*`、`jasna/gui/interactive_image_restore.py`、`jasna/post_export_action.py` 等没有被本地分支删除。

## 上游 v0.7.2 主要变化

从现有提交和 diff 看，上游从本地旧基线到 `v0.7.2` 的重点是：

- 版本号升到 `0.7.2`。
- 依赖升级：`torch 2.12.0+cu130`、`torchvision 0.27.0+cu130`、`tensorrt 10.16.1.11`、`torch-tensorrt 2.12.0`。
- 新增依赖：`diffusers`、`accelerate`、`huggingface-hub`、`Pillow`、`onnx`、`cryptography>=42`。
- dev 构建依赖从 `pyinstaller>=6.0` 转向 `nuitka>=2.4`。
- 新增图像修复 CLI/GUI：`jasna/image_restore.py`、`jasna/media/image_io.py`、`jasna/gui/interactive_image_restore.py`。
- 新增 SD 1.5 inpaint restoration：`jasna/restorer/sd15_download.py`、`jasna/restorer/sd15_inpaint_restorer.py`、`jasna/sd15_crop_utils.py`。
- 新增 `.cube` LUT 支持：`jasna/media/lut.py`。
- 新增 `post_export_action`、`startup_timing`、`pipeline_timing` 等运行辅助功能。
- GUI、locales、folder processing、输出命名、打包/bundle 元数据都有较多更新。
- 新增 `.gitmodules` 和 `jasna/protection` gitlink/submodule。

## 需要重点处理的风险

### 1. `.drone/build.sh` 仍依赖旧 PyInstaller 入口

当前 `.drone/build.sh` 里有：

```bash
python3.13 build_exe.py
```

但 `v0.7.2` 的树里已经没有：

```text
build_exe.py
jasna.spec
```

同时 `.drone/patches/fix_pyinstaller_noconfirm.patch` 和 `.drone/patches/fix_pyinstaller_spec.patch` 都以这两个旧文件为目标。升级后继续运行当前 `.drone/build.sh` 会失败。

建议：

- 不要把旧的 `build_exe.py` / `jasna.spec` 从旧分支硬搬回来，除非确认还要维持旧 PyInstaller 流程。
- 优先把 `.drone/build.sh` 改成适配 `v0.7.2` 的新 bundle/Nuitka/protection 流程。
- 如果仍要 PyInstaller，单独新建“v0.7.2 PyInstaller 打包适配”任务，而不是在本次 merge 里顺手保留旧 spec。

### 2. `.drone/dockerfile.build` 依赖版本落后

当前 builder 镜像仍安装：

```text
torch==2.10.0+cu130
torchvision==0.25.0+cu130
tensorrt==10.14.1.48.post1
torch-tensorrt==2.10.0
transformers<5
```

但 `v0.7.2` 的 `pyproject.toml` 需要：

```text
torch==2.12.0+cu130
torchvision==0.27.0+cu130
tensorrt==10.16.1.11
torch-tensorrt==2.12.0
diffusers / accelerate / huggingface-hub / Pillow / onnx / cryptography
```

建议同步更新 `.drone/dockerfile.build`，否则容器内 `pip install .` 很可能重新下载/覆盖大包，甚至出现 ABI 或 torch-tensorrt 版本不匹配。

### 3. `.drone/patches` 需要清理

在升级探针树上 dry-run 结果：

```text
OK   fix_crop_border_expand.patch
FAIL fix_pyinstaller_noconfirm.patch
FAIL fix_pyinstaller_spec.patch
OK   skip_ffmpeg_version_check.patch
FAIL use_pyav_video_decoder.patch
```

具体判断：

- `fix_pyinstaller_*`：目标文件已不存在，应删除或重写。
- `use_pyav_video_decoder.patch`：上游 `video_decoder.py` 已变化，旧 patch 不能应用；如果仍要在构建阶段切到 PyAV，需要按 v0.7.2 的 `NvidiaVideoReader` 重新写 patch。
- `fix_crop_border_expand.patch`：名字容易误导。它现在会把 `BORDER_RATIO` 从 `0.10` 改回 `0.06`，把 `MIN_BORDER` 从 `40` 改回 `20`。但 cherry-pick 后源码本身已经改成 `0.10/40`；继续套这个 patch 会抵消本地源码改动。建议删除或反向整理为明确的源码改动。
- `skip_ffmpeg_version_check.patch`：仍可应用，如果 Docker 环境确实不能满足 ffmpeg major version 8，可暂时保留；更好的方式是给 CLI/CI 提供显式开关。

### 4. 子模块/保护文件

`v0.7.2` 新增：

```text
.gitmodules
jasna/protection  -> gitlink 1506724...
```

合并后需要执行：

```powershell
git submodule update --init --recursive
```

容器构建也要确认 `.dockerignore` 和 build context 没有把 `jasna/protection/keytool/wheels` 排除掉。上游 `.dockerignore` 是 default-deny：

```text
*
!pyproject.toml
!README.md
!patches
!jasna/protection/keytool/wheels
```

如果 `.drone/dockerfile*` 还需要源码全量上下文，必须调整 Docker build context 或 Dockerfile COPY 策略。

## 推荐升级后的检查项

完成 cherry-pick 和构建脚本调整后，建议至少跑：

```powershell
python -m pytest tests/test_main.py tests/test_media_init.py tests/test_crop_buffer.py tests/test_blend_buffer.py
python -m pytest tests/test_image_restore.py tests/test_lut.py tests/test_sd15_inpaint_restorer.py
python -m pytest tests/test_pipeline_run.py tests/test_video_encoder_unit.py tests/test_windows_dll_path_sanitization.py
```

如果依赖环境太重，至少先跑静态导入检查：

```powershell
python -c "import jasna.main; import jasna.media; import jasna.restorer; import jasna.gui.app"
```

容器侧建议单独验证：

```powershell
docker build -f .drone/dockerfile.build -t jasna-builder:v0.7.2 .drone
docker run --rm --gpus all -v ${PWD}:/app/jasna jasna-builder:v0.7.2 bash .drone/build.sh
```

上面的 Docker 命令可能需要根据最终 Dockerfile 的 build context 调整。

## 建议的提交拆分

为了以后升级容易追踪，建议不要把所有内容合成一个大提交：

1. `chore: base on upstream v0.7.2`
   - 从 `v0.7.2` 建分支，保留上游原样。
2. `ci: restore local drone and workflow files`
   - 移植 `.drone/*`、`.github/workflows/*`。
3. `build: adapt drone builder for v0.7.2 packaging`
   - 更新 `.drone/build.sh`、`.drone/dockerfile.build`、清理过期 patches。
4. `feat: restore local processing defaults`
   - `crop_buffer.py`、`blend_buffer.py`、`main.py`、`media/__init__.py` 等本地行为。
5. `test: update local upgrade expectations`
   - 对应测试更新。

## 最短可执行路线

```powershell
git fetch upstream --tags --prune
git switch -c upgrade/v0.7.2 v0.7.2
git submodule update --init --recursive

$commits = git log --reverse --no-merges --cherry-pick --left-only --format=%H main...v0.7.2
foreach ($c in $commits) {
  git cherry-pick -x -X theirs $c
}

git diff --stat v0.7.2..HEAD
git diff -- .drone/build.sh .drone/dockerfile.build .drone/patches pyproject.toml
```

然后先修 `.drone` 构建链路，再跑测试和容器构建。

## dev 分支执行记录

执行时间：2026-06-27

已按本报告方案创建本地 `dev` 分支：

```powershell
git switch -c dev v0.7.2
```

随后按顺序 cherry-pick 本地唯一非 merge 提交：

```powershell
$commits = git log --reverse --no-merges --cherry-pick --left-only --format=%H main...v0.7.2
foreach ($c in $commits) {
  git cherry-pick -x -X theirs $c
}
```

执行结果：

```text
APPLIED_COUNT=23
HEAD=1b6d10f feat: add command-line interface for video restoration and streaming configuration
```

最终 `dev` 相对 `v0.7.2` 的差异仍是预期的 20 个文件：

```text
A .drone/Bugs.md
A .drone/README.md
A .drone/build.sh
A .drone/dockerfile
A .drone/dockerfile.build
A .drone/entrypoint.sh
A .drone/patches/fix_crop_border_expand.patch
A .drone/patches/fix_pyinstaller_noconfirm.patch
A .drone/patches/fix_pyinstaller_spec.patch
A .drone/patches/skip_ffmpeg_version_check.patch
A .drone/patches/use_pyav_video_decoder.patch
A .github/workflows/jasna-build.yml
A .github/workflows/jasna.yml
M .gitignore
M jasna/blend_buffer.py
M jasna/crop_buffer.py
M jasna/main.py
M jasna/media/__init__.py
M tests/test_main.py
M tests/test_media_init.py
```

验证结果：

```text
git status --short --branch
## dev
?? .drone/upgrade_v0.7.2.md
```

`git diff --check v0.7.2..HEAD` 当前未通过，问题来自旧 CI/patch 文件中的 trailing whitespace 和 `.drone/dockerfile.build` 末尾空行。它们不是 cherry-pick 冲突，但后续清理 `.drone` 构建链路时应一并处理。

`git submodule update --init --recursive` 当前失败：

```text
Submodule 'jasna/protection' (git@github.com:Mengkzhaoyun/jasna.git/jasna/protection) registered for path 'jasna/protection'
fatal: remote error:
 Mengkzhaoyun/jasna.git/jasna/protection is not a valid repository name
```

原因是 `v0.7.2` 的 `.gitmodules` 使用相对 URL：

```text
[submodule "jasna/protection"]
    path = jasna/protection
    url = ./jasna/protection
```

在当前 fork 下会解析成无效远端。后续需要确认 upstream 的真实 protection 子模块来源，或者在 CI/build 中改成可访问的 URL/构建上下文。

patch dry-run 结果与分析一致：

```text
OK   fix_crop_border_expand.patch
FAIL fix_pyinstaller_noconfirm.patch
FAIL fix_pyinstaller_spec.patch
OK   skip_ffmpeg_version_check.patch
FAIL use_pyav_video_decoder.patch
```

下一步建议直接在 `dev` 上处理：

1. 修正 `jasna/protection` 子模块来源或构建策略。
2. 更新 `.drone/dockerfile.build` 到 `v0.7.2` 的依赖版本。
3. 重写或删除失效的 `.drone/patches/fix_pyinstaller_*` 和 `use_pyav_video_decoder.patch`。
4. 决定 `fix_crop_border_expand.patch` 是否还需要保留，避免它反向抵消源码里的 `BORDER_RATIO=0.10` / `MIN_BORDER=40`。
5. 清理 whitespace 后再跑 `git diff --check v0.7.2..HEAD`。
