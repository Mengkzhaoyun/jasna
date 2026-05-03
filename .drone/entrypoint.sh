#!/bin/bash
set -e

# ============================================================
# Jasna Docker Entrypoint
#
# 用法:
#   jasna                              # 自动扫描 /data 批量处理
#   jasna --input ... --output ...     # 正常处理视频
#   jasna --stream                     # 流媒体模式
#   jasna warmup [选项]                # 预编译 TensorRT 引擎
#
# 环境变量 (批量模式):
#   SCAN_DIR       扫描目录 (默认: /data)
#   CODEC          编码器 (默认: hevc, 可选: av1, h264)
#   EXTRA_ARGS     传给 jasna 的额外参数
# ============================================================

# 支持的视频扩展名
VIDEO_EXTS="mp4 mkv avi mov wmv flv webm ts"

# ---- warmup 子命令 ----
if [ "$1" = "warmup" ]; then
    shift

    CLIP_SIZE="${CLIP_SIZE:-90}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    DETECTION_MODEL="${DETECTION_MODEL:-rfdetr-v5}"
    RESTORATION_MODEL="${RESTORATION_MODEL:-model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
    DETECTION_MODEL_PATH="${DETECTION_MODEL_PATH:-model_weights/${DETECTION_MODEL}.onnx}"

    echo "============================================"
    echo " Jasna TensorRT Engine Warmup"
    echo "============================================"
    echo " Clip Size:         ${CLIP_SIZE}"
    echo " Batch Size:        ${BATCH_SIZE}"
    echo " Detection Model:   ${DETECTION_MODEL}"
    echo " Restoration Model: ${RESTORATION_MODEL}"
    echo "============================================"
    echo ""
    echo "编译 TensorRT 引擎中, 首次需要 15-60 分钟..."
    echo "引擎会缓存到 model_weights/ 目录, 后续启动无需重新编译。"
    echo ""

    JSON_DATA=$(cat <<EOF
{
    "device": "cuda:0",
    "fp16": true,
    "basicvsrpp": true,
    "basicvsrpp_model_path": "${RESTORATION_MODEL}",
    "basicvsrpp_max_clip_size": ${CLIP_SIZE},
    "detection": true,
    "detection_model_name": "${DETECTION_MODEL}",
    "detection_model_path": "${DETECTION_MODEL_PATH}",
    "detection_batch_size": ${BATCH_SIZE},
    "unet4x": false
}
EOF
)
    /app/jasna/jasna --compile-engines "$JSON_DATA"

    echo ""
    echo "============================================"
    echo " Warmup 完成! 引擎已缓存。"
    echo "============================================"
    exit 0
fi

# ---- 带参数时直接透传给 jasna ----
if [ $# -gt 0 ]; then
    exec /app/jasna/jasna --disable-ffmpeg-check --encoder-settings "cq=${ENCODER_CQ:-22}" "$@"
fi

# ---- 无参数: 批量扫描模式 ----
SCAN_DIR="${SCAN_DIR:-/data}"
CODEC="${CODEC:-hevc}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo " Jasna 批量处理模式"
echo "============================================"
echo " 扫描目录: ${SCAN_DIR}"
echo " 编码器:   ${CODEC}"
if [ -n "$EXTRA_ARGS" ]; then
    echo " 额外参数: ${EXTRA_ARGS}"
fi
echo "============================================"
echo ""

# 构建 find 的 -name 表达式
FIND_ARGS=()
first=true
for ext in $VIDEO_EXTS; do
    if [ "$first" = true ]; then
        first=false
    else
        FIND_ARGS+=("-o")
    fi
    FIND_ARGS+=("-iname" "*.${ext}")
done

# 收集待处理任务
TASKS=()
while IFS= read -r -d '' filepath; do
    dir="$(dirname "$filepath")"
    filename="$(basename "$filepath")"
    ext="${filename##*.}"
    name="${filename%.*}"

    # 跳过已经是 -U 结尾的输出文件
    if [[ "$name" == *-U ]]; then
        continue
    fi

    # 输出文件路径
    output="${dir}/${name}-U.${ext}"

    # 跳过已存在输出的文件
    if [ -f "$output" ]; then
        echo "[跳过] 输出已存在: ${output}"
        continue
    fi

    TASKS+=("${filepath}|${output}")
done < <(find "$SCAN_DIR" -type f \( "${FIND_ARGS[@]}" \) -print0 | sort -z)

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "未找到需要处理的视频文件。"
    exit 0
fi

echo "发现 ${#TASKS[@]} 个待处理视频:"
echo ""
for task in "${TASKS[@]}"; do
    input="${task%%|*}"
    output="${task##*|}"
    echo "  ${input}"
    echo "    → ${output}"
done
echo ""

# 逐个处理
SUCCESS=0
FAIL=0
TOTAL=${#TASKS[@]}

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    input="${task%%|*}"
    output="${task##*|}"
    idx=$((i + 1))

    echo "============================================"
    echo " [${idx}/${TOTAL}] 处理中..."
    echo " 输入: ${input}"
    echo " 输出: ${output}"
    echo " 编码: ${CODEC}"
    if [ -n "$EXTRA_ARGS" ]; then
        echo " 额外参数: ${EXTRA_ARGS}"
    fi
    echo "============================================"

    if /app/jasna/jasna \
        --disable-ffmpeg-check \
        --codec "$CODEC" \
        --encoder-settings "cq=${ENCODER_CQ:-20}" \
        --input "$input" \
        --output "$output" \
        $EXTRA_ARGS; then
        echo ""
        echo " ✅ [${idx}/${TOTAL}] 完成: ${output}"
        echo ""
        SUCCESS=$((SUCCESS + 1))
    else
        echo ""
        echo " ❌ [${idx}/${TOTAL}] 失败: ${input}"
        echo ""
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "============================================"
echo " 批量处理完成"
echo " 成功: ${SUCCESS} / ${TOTAL}"
if [ $FAIL -gt 0 ]; then
    echo " 失败: ${FAIL} / ${TOTAL}"
fi
echo "============================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
