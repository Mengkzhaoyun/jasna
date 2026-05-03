#!/bin/bash
set -ex

# ============================================================
# Jasna 自动化构建脚本 (纯编译阶段)
# 假定基础运行环境（PyTorch, TensorRT, mmengine 等）已由 Dockerfile 提前构建好
# ============================================================

export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Shanghai
export LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/lib/python3.13/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

SRC_DIR="/app/jasna"
PATCH_DIR="$SRC_DIR/.drone/patches"
cd "$SRC_DIR"

echo ">>> 1. 应用补丁..."
find "$PATCH_DIR" -name "*.patch" -exec sed -i 's/\r$//' {} \;
find "$SRC_DIR" \( -name "*.py" -o -name "*.spec" \) -exec sed -i 's/\r$//' {} \;
for p in "$PATCH_DIR"/*.patch; do
    echo ">>> 应用: $(basename $p)"
    patch --batch -p1 < "$p"
done

echo ">>> 2. 安装依赖..."
VENV_DIR="$SRC_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3.13 -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --cache-dir "$SRC_DIR/.pip_cache" --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com \
    .

echo ">>> 3. 生成二进制..."
NCCL_LIB=$(find /usr/lib -name "libnccl.so.2" 2>/dev/null | head -1)
[ -z "$NCCL_LIB" ] && NCCL_LIB=$(find /usr/local/lib/python3.13/site-packages/nvidia/nccl/lib -name "libnccl.so.2" 2>/dev/null | head -1)
[ -n "$NCCL_LIB" ] && export LD_PRELOAD="$NCCL_LIB"
export CUDA_VISIBLE_DEVICES=""

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

echo ">>> 4. 还原补丁..."
for p in "$PATCH_DIR"/*.patch; do
    patch --batch -R -p1 < "$p"
done

echo "============================================================"
echo "✅ 编译完成，产物在 dist_linux/jasna/"
echo "============================================================"
