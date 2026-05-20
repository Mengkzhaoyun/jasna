#!/bin/bash
set -e

# ============================================================
# Jasna Docker Entrypoint
#
# 用法:
#   sglang                              # 自动扫描 /data 批量处理
#   sglang --input ... --output ...     # 正常处理视频
#   sglang --stream                     # 流媒体模式
#   sglang warmup [选项]                # 预编译 TensorRT 引擎
#
# 环境变量 (批量模式功能及优化):
#   BATCH_SIZE                 (默认: 4)
#   MAX_CLIP_SIZE              (默认: 150)
#   TEMPORAL_OVERLAP           (默认: 16)
#   ENABLE_CROSSFADE           (默认: true)
#   DENOISE                    (默认: low)
#   DETECTION_SCORE_THRESHOLD  (默认: 0.25)
#   FP16                       (默认: true)
#   COMPILE_BASICVSRPP         (默认: true)
#   LOG_LEVEL                  (默认: info)
#   TARGET_BITRATE             (默认: 5M) 直接传给 Jasna NVENC，一阶段输出目标码率
#   SKIP_LOW_BITRATE           (默认: true) 输入视频码率 <= TARGET_BITRATE 时跳过
#   POST_COMPRESS_BITRATE      (默认: 空) 若设置则额外使用 FFmpeg 二次压缩
#   SCAN_DIR                   扫描目录 (默认: /data)
#   CODEC                      编码器 (默认: hevc, 可选: av1, h264)
#   EXTRA_ARGS                 传给 sglang 的额外参数
# ============================================================

# 支持的视频扩展名
VIDEO_EXTS="mp4 mkv avi mov wmv flv webm ts"

bitrate_to_kbps() {
	local value="${1:-}"
	value="${value// /}"
	if [ -z "$value" ]; then
		return 1
	fi

	case "$value" in
		*[Mm])
			echo "$(( ${value%[Mm]} * 1000 ))"
			;;
		*[Kk])
			echo "${value%[Kk]}"
			;;
		*)
			echo "$value"
			;;
	esac
}

build_encoder_settings() {
	local target_bitrate="${TARGET_BITRATE:-5M}"
	local target_kbps
	target_kbps="$(bitrate_to_kbps "$target_bitrate")"

	if [ -n "${ENCODER_SETTINGS:-}" ]; then
		echo "$ENCODER_SETTINGS"
	elif [ -n "$target_kbps" ]; then
		echo "rc=vbr,maxbitrate=${target_kbps},vbvbufsize=${target_kbps},cq=${ENCODER_CQ:-20},gop=${ENCODER_GOP:-60}"
	else
		echo "cq=${ENCODER_CQ:-20}"
	fi
}

probe_video_bitrate_kbps() {
	local file="$1"
	local bitrate=""

	bitrate="$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null | head -n 1 || true)"
	if [ -z "$bitrate" ] || [ "$bitrate" = "N/A" ]; then
		bitrate="$(ffprobe -v error -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null | head -n 1 || true)"
	fi

	if [ -z "$bitrate" ] || [ "$bitrate" = "N/A" ]; then
		return 1
	fi

	echo "$(( (bitrate + 999) / 1000 ))"
}

# ---- warmup 子命令 ----
if [ "$1" = "warmup" ]; then
	shift

	CLIP_SIZE="${MAX_CLIP_SIZE:-150}"
	BATCH_SIZE="${BATCH_SIZE:-4}"
	DETECTION_MODEL="${DETECTION_MODEL:-rfdetr-v5}"
	RESTORATION_MODEL="${RESTORATION_MODEL:-model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
	DETECTION_MODEL_PATH="${DETECTION_MODEL_PATH:-model_weights/${DETECTION_MODEL}.onnx}"

	echo "============================================"
	echo " sglang TensorRT Engine Warmup"
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

	JSON_DATA=$(
		cat <<EOF
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
	/app/sglang/sglang --compile-engines "$JSON_DATA"

	echo ""
	echo "============================================"
	echo " Warmup 完成! 引擎已缓存。"
	echo "============================================"
	exit 0
fi

# ---- 带参数时直接透传给 sglang ----
if [ $# -gt 0 ]; then
	exec /app/sglang/sglang --disable-ffmpeg-check --encoder-settings "$(build_encoder_settings)" "$@"
fi

# ---- 无参数: 批量扫描模式 ----
SCAN_DIR="${SCAN_DIR:-/data}"
CODEC="${CODEC:-hevc}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_CLIP_SIZE="${MAX_CLIP_SIZE:-150}"
TEMPORAL_OVERLAP="${TEMPORAL_OVERLAP:-16}"
ENABLE_CROSSFADE="${ENABLE_CROSSFADE:-true}"
DENOISE="${DENOISE:-low}"
DETECTION_SCORE_THRESHOLD="${DETECTION_SCORE_THRESHOLD:-0.25}"
FP16="${FP16:-true}"
COMPILE_BASICVSRPP="${COMPILE_BASICVSRPP:-true}"
LOG_LEVEL="${LOG_LEVEL:-info}"
TARGET_BITRATE="${TARGET_BITRATE:-5M}"
TARGET_BITRATE_KBPS="$(bitrate_to_kbps "$TARGET_BITRATE")"
SKIP_LOW_BITRATE="${SKIP_LOW_BITRATE:-true}"
ENCODER_SETTINGS="$(build_encoder_settings)"
POST_COMPRESS_BITRATE="${POST_COMPRESS_BITRATE:-}"

JASNA_OPTS=""
[ -n "$BATCH_SIZE" ] && JASNA_OPTS="$JASNA_OPTS --batch-size $BATCH_SIZE"
[ -n "$MAX_CLIP_SIZE" ] && JASNA_OPTS="$JASNA_OPTS --max-clip-size $MAX_CLIP_SIZE"
[ -n "$TEMPORAL_OVERLAP" ] && JASNA_OPTS="$JASNA_OPTS --temporal-overlap $TEMPORAL_OVERLAP"
[ "$ENABLE_CROSSFADE" = "true" ] && JASNA_OPTS="$JASNA_OPTS --enable-crossfade"
[ -n "$DENOISE" ] && JASNA_OPTS="$JASNA_OPTS --denoise $DENOISE"
[ -n "$DETECTION_SCORE_THRESHOLD" ] && JASNA_OPTS="$JASNA_OPTS --detection-score-threshold $DETECTION_SCORE_THRESHOLD"
[ "$FP16" = "true" ] && JASNA_OPTS="$JASNA_OPTS --fp16"
[ "$COMPILE_BASICVSRPP" = "true" ] && JASNA_OPTS="$JASNA_OPTS --compile-basicvsrpp"
[ -n "$LOG_LEVEL" ] && JASNA_OPTS="$JASNA_OPTS --log-level $LOG_LEVEL"

echo "============================================"
echo " sglang 批量处理模式"
echo "============================================"
echo " 扫描目录: ${SCAN_DIR}"
echo " 编码器:   ${CODEC}"
echo " NVENC:    ${ENCODER_SETTINGS}"
if [ "$SKIP_LOW_BITRATE" = "true" ] && [ -n "$TARGET_BITRATE_KBPS" ]; then
	echo " 低码率跳过: <= ${TARGET_BITRATE_KBPS} kbps"
fi
if [ -n "$POST_COMPRESS_BITRATE" ]; then
	echo " 后压缩:   ${POST_COMPRESS_BITRATE}"
fi
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

	if [ "$SKIP_LOW_BITRATE" = "true" ] && [ -n "$TARGET_BITRATE_KBPS" ]; then
		input_bitrate_kbps="$(probe_video_bitrate_kbps "$filepath" || true)"
		if [ -n "$input_bitrate_kbps" ] && [ "$input_bitrate_kbps" -le "$TARGET_BITRATE_KBPS" ]; then
			echo "[跳过] 输入码率 ${input_bitrate_kbps} kbps <= 目标 ${TARGET_BITRATE_KBPS} kbps: ${filepath}"
			continue
		fi
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
	echo " NVENC: ${ENCODER_SETTINGS}"
	if [ -n "$EXTRA_ARGS" ]; then
		echo " 额外参数: ${EXTRA_ARGS}"
	fi
	echo "============================================"

	if /app/sglang/sglang \
		--disable-ffmpeg-check \
		--codec "$CODEC" \
		--encoder-settings "$ENCODER_SETTINGS" \
		--input "$input" \
		--output "$output" \
		$JASNA_OPTS \
		$EXTRA_ARGS; then

		if [ -n "$POST_COMPRESS_BITRATE" ]; then
			echo ""
			echo " [后压缩] 正在使用 hevc_nvenc 二次压缩，目标码率: ${POST_COMPRESS_BITRATE}..."
			mv "$output" "${output}.raw.mp4"
			if ffmpeg -y -i "${output}.raw.mp4" -c:v hevc_nvenc -preset p6 -tune hq -b:v "$POST_COMPRESS_BITRATE" -maxrate "$POST_COMPRESS_BITRATE" -bufsize "$POST_COMPRESS_BITRATE" -c:a copy "$output" </dev/null; then
				rm "${output}.raw.mp4"
				echo " [后压缩] 压缩完成！"
			else
				echo " [后压缩] 压缩失败，已恢复高码率原始文件。"
				mv "${output}.raw.mp4" "$output"
			fi
		fi

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
