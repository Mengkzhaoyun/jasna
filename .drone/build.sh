#!/bin/bash
set -ex

# ============================================================
# Jasna 自动化构建脚本 (纯编译阶段)
# 假定基础运行环境（PyTorch, TensorRT, mmengine 等）已由 Dockerfile 提前构建好
# ============================================================

# 模拟 Dockerfile 环境变量
export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Shanghai
export LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/lib/python3.13/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

echo ">>> 1. 拷贝 Jasna 源码进行编译..."
JASNA_SRC="/tmp/jasna_src"
rm -rf "$JASNA_SRC"
cp -a /app/jasna "$JASNA_SRC"
sed -i 's/if wrong_version:/if False:/g' "$JASNA_SRC/jasna/os_utils.py"

echo ">>> 2. 应用修复补丁..."
# 修复 blend_mask 在 crop 边界硬截断导致的正方形伪影
patch -p1 -d "$JASNA_SRC" < /app/jasna/.drone/patches/fix_blend_edge_feather.patch
# 增大 crop 边界使 blend_mask 渐变区完全包含在 crop 内
patch -p1 -d "$JASNA_SRC" < /app/jasna/.drone/patches/fix_crop_border_expand.patch

cd "$JASNA_SRC"
WORKDIR=$(pwd)
echo ">>> 工作目录已切换至: $WORKDIR"

echo ">>> 3. 修复 PyInstaller 打包问题..."
sed -i 's/"PyNvVideoCodec", "python_vali", //g' jasna.spec || true
sed -i '/hiddenimports += h/a\    hiddenimports += ["python_vali", "PyNvVideoCodec"]' jasna.spec || true
python3.13 -c '
import pathlib
try:
    p = pathlib.Path("jasna.spec")
    txt = p.read_text()
    old = "trt_ext = find_spec(\"torch_tensorrt._C\")\nif trt_ext is not None and trt_ext.origin:\n    binaries += [(trt_ext.origin, \"torch_tensorrt\")]\n    hiddenimports += [\"torch_tensorrt._C\"]"
    new = "try:\n    trt_ext = find_spec(\"torch_tensorrt._C\")\n    if trt_ext is not None and trt_ext.origin:\n        binaries += [(trt_ext.origin, \"torch_tensorrt\")]\n        hiddenimports += [\"torch_tensorrt._C\"]\nexcept (ImportError, Exception):\n    pass"
    if old in txt:
        p.write_text(txt.replace(old, new))
except Exception:
    pass
' || true

echo ">>> 4. 安装 Jasna 自身依赖..."
pip install --no-cache-dir --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com \
    .

echo ">>> 5. 生成独立二进制文件..."
# 使用 LD_PRELOAD 强制预加载系统 NCCL 库（双保险）
NCCL_LIB=$(find /usr/lib -name "libnccl.so.2" 2>/dev/null | head -1)
if [ -z "$NCCL_LIB" ]; then
    NCCL_LIB=$(find /usr/local/lib/python3.13/site-packages/nvidia/nccl/lib -name "libnccl.so.2" 2>/dev/null | head -1)
fi
if [ -n "$NCCL_LIB" ]; then
    echo ">>> 使用 NCCL 库: $NCCL_LIB"
    export LD_PRELOAD="$NCCL_LIB"
fi

export CUDA_VISIBLE_DEVICES=""

# 创建模型文件和资源占位符（build_exe.py 会尝试复制它们）
mkdir -p model_weights assets
touch model_weights/lada_mosaic_restoration_model_generic_v1.2.pth 2>/dev/null || true
touch model_weights/rfdetr-v5.onnx 2>/dev/null || true
touch model_weights/lada_mosaic_detection_model_v4_fast.pt 2>/dev/null || true
touch assets/test_clip1_1080p.mp4 2>/dev/null || true
touch assets/test_clip1_2160p.mp4 2>/dev/null || true

python3.13 build_exe.py
mkdir -p dist_linux/jasna/_internal/PyNvVideoCodec
find /usr/local/lib/python3.13 -name "PyNvVideoCodec_*.so" -exec cp {} dist_linux/jasna/_internal/PyNvVideoCodec/ \;
mkdir -p dist_linux/jasna/_internal/python_vali
find /usr/local/lib/python3.13 -name "python_vali*.so" -exec cp {} dist_linux/jasna/_internal/python_vali/ \;

# 将生成的二进制复制回挂载的主机目录
HOST_DIR="/app/jasna"
rm -rf "$HOST_DIR/dist_linux"
mkdir -p dist_linux/jasna/model_weights
cp -r dist_linux "$HOST_DIR/"

echo "============================================================"
echo "✅ 容器内编译与提取完成，结果已拷贝到宿主机的 jasna/dist_linux 目录！"
echo "============================================================"
