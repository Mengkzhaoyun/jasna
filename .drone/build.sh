#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# Jasna v0.7.2 自动化构建脚本 (Nuitka standalone)
# 运行环境中的 PyTorch / TensorRT / Vali 等大依赖由 dockerfile.build 预装。
# ============================================================

export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Shanghai
export LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:/usr/local/lib/python3.13/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

SRC_DIR="${SRC_DIR:-/app/jasna}"
PATCH_DIR="$SRC_DIR/.drone/patches"
VENV_DIR="$SRC_DIR/.venv"
DIST_ROOT="$SRC_DIR/dist_linux"
DIST_DIR="$DIST_ROOT/jasna"

cd "$SRC_DIR"
git config --global --add safe.directory "$SRC_DIR" || true

APPLIED_PATCHES=()
PROTECTION_STUB_CREATED=0

cleanup() {
    local p
    for ((idx=${#APPLIED_PATCHES[@]} - 1; idx >= 0; idx--)); do
        p="${APPLIED_PATCHES[$idx]}"
        patch --batch -R -p1 < "$p" || true
    done
    if [ "$PROTECTION_STUB_CREATED" = "1" ]; then
        rm -f "$SRC_DIR/jasna/protection/__init__.py"
        rmdir "$SRC_DIR/jasna/protection" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo ">>> 1. 准备可选 protection 模块..."
if [ ! -f "$SRC_DIR/jasna/protection/__init__.py" ]; then
    if git submodule update --init --recursive jasna/protection; then
        echo ">>> protection 子模块已初始化。"
    elif [ "${REQUIRE_PROTECTION_SUBMODULE:-0}" = "1" ]; then
        echo "ERROR: REQUIRE_PROTECTION_SUBMODULE=1，但 jasna/protection 无法初始化。" >&2
        exit 1
    else
        echo ">>> protection 子模块不可用，创建临时公共构建 stub。"
        mkdir -p "$SRC_DIR/jasna/protection"
        cat > "$SRC_DIR/jasna/protection/__init__.py" <<'PY'
class ProtectionError(RuntimeError):
    pass


class _LicenseStore:
    def load_license(self):
        return None

    def is_licensed(self):
        return False

    def set_license(self, email, key):
        raise ProtectionError("The protection module is not available in this public build.")


class _ProtectedModel:
    def _missing(self, *args, **kwargs):
        raise ProtectionError("The protection module is not available in this public build.")

    decrypt_model_to_buffer = _missing
    decrypt_model_bytes = _missing
    decrypt_engine_bytes = _missing
    encrypt_engine_bytes = _missing


license_store = _LicenseStore()
protected_model = _ProtectedModel()
PY
        PROTECTION_STUB_CREATED=1
    fi
fi

echo ">>> 2. 应用可选补丁..."
if [ -d "$PATCH_DIR" ]; then
    find "$PATCH_DIR" -name "*.patch" -exec sed -i 's/\r$//' {} \;
    shopt -s nullglob
    for p in "$PATCH_DIR"/*.patch; do
        echo ">>> 应用: $(basename "$p")"
        patch --batch -p1 < "$p"
        APPLIED_PATCHES+=("$p")
    done
    shopt -u nullglob
else
    echo ">>> 无 .drone/patches 目录，跳过。"
fi

echo ">>> 3. 安装/刷新 Python 依赖..."
if [ ! -d "$VENV_DIR" ]; then
    python3.13 -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --cache-dir "$SRC_DIR/.pip_cache" --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com \
    ".[dev]"

echo ">>> 4. 生成 Nuitka standalone 二进制..."
NCCL_LIB=$(find /usr/lib /usr/local/lib/python3.13/site-packages/nvidia/nccl/lib -name "libnccl.so.2" 2>/dev/null | head -1 || true)
[ -n "$NCCL_LIB" ] && export LD_PRELOAD="$NCCL_LIB"
export CUDA_VISIBLE_DEVICES=""

mkdir -p model_weights assets
touch model_weights/lada_mosaic_restoration_model_generic_v1.2.pth 2>/dev/null || true
touch model_weights/rfdetr-v5.onnx 2>/dev/null || true
touch model_weights/lada_mosaic_detection_model_v4_fast.pt 2>/dev/null || true
touch assets/test_clip1_1080p.mp4 2>/dev/null || true
touch assets/test_clip1_2160p.mp4 2>/dev/null || true

rm -rf "$DIST_ROOT"
python3.13 -m nuitka \
    --standalone \
    --assume-yes-for-downloads \
    --remove-output \
    --output-dir="$DIST_ROOT" \
    --output-filename=jasna \
    --python-flag=-m \
    --enable-plugin=tk-inter \
    --follow-imports \
    --include-package=jasna \
    --include-package-data=customtkinter \
    --include-package-data=tkinterdnd2 \
    --include-package-data=ultralytics \
    --include-distribution-metadata=torch \
    --include-distribution-metadata=torchvision \
    --include-distribution-metadata=tensorrt \
    --include-distribution-metadata=torch-tensorrt \
    --include-distribution-metadata=diffusers \
    --include-distribution-metadata=transformers \
    --include-distribution-metadata=huggingface-hub \
    --nofollow-import-to=pytest \
    --nofollow-import-to=tests \
    jasna

NUITKA_DIST="$(find "$DIST_ROOT" -maxdepth 1 -type d -name "*.dist" | head -1)"
if [ -z "$NUITKA_DIST" ]; then
    echo "ERROR: Nuitka dist directory not found under $DIST_ROOT" >&2
    exit 1
fi
mv "$NUITKA_DIST" "$DIST_DIR"
if [ ! -f "$DIST_DIR/jasna" ]; then
    NUITKA_BIN="$(find "$DIST_DIR" -maxdepth 1 -type f -perm -111 | head -1)"
    if [ -n "${NUITKA_BIN:-}" ]; then
        mv "$NUITKA_BIN" "$DIST_DIR/jasna"
    fi
fi
chmod +x "$DIST_DIR/jasna"

echo "============================================================"
echo "编译完成，产物在 dist_linux/jasna/"
echo "============================================================"
