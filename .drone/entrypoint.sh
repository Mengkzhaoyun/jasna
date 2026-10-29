#!/bin/bash
set -e

# ============================================================
# Jasna Docker Entrypoint
#
# 用法:
#   sglang                              # 自动扫描 /data/input 批量处理
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
#   RESTORATION_MODEL_NAME     (默认: ltx; 可选: basicvsrpp, ltx, ltx-4step, ltx-8step, ltx-undistilled)
#   RESTORATION_MODEL_PATH     (默认: 自动探测 model_weights 或 /data/jasna/model_weights)
#   LTX_LARGE_CANVAS           (默认: false)
#   LTX_FAST                   (默认: false, 仅 RTX 50 系支持 NVFP4)
#   LTX_TRIAL                  (默认: false, LTX 免费测速/算子验证模式)
#   MULTI_GPU                  (默认: true, 开启 1fps 智能预扫 + 无码段跳过 + 单容器多卡 L1/L2 协同调度)
#   GPUS                       (默认: 空/自动探测容器内全部可见 GPU, 可指定如 "0,1" 或 "all")
#   IGNORE_HEAD_MOSAIC         (默认: true, 自动识别并剔除男优/路人头部人脸马赛克)
#   JASNA_LICENSE              (默认: 空, 激活码 JL1-...，提供后自动完成激活)
#   LOG_LEVEL                  (默认: info)
#   TARGET_BITRATE             (默认: 5M) 直接传给 Jasna NVENC，一阶段输出目标码率
#   PRESERVE_LOW_BITRATE       (默认: true) 输入视频码率 <= TARGET_BITRATE 时仍处理，但不压缩
#   LOW_BITRATE_ENCODER_SETTINGS 低码率输入使用的 NVENC 设置；默认禁用码率上限，仅保留 CQ/GOP
#   POST_COMPRESS_BITRATE      (默认: 空) 若设置则额外使用 FFmpeg 二次压缩
#   SCAN_DIR                   扫描目录 (默认: /data/input)
#   CODEC                      编码器 (默认: hevc, 可选: av1, h264)
#   EXTRA_ARGS                 传给 sglang 的额外参数
# ============================================================

# 支持的视频扩展名
VIDEO_EXTS="mp4 mkv avi mov wmv flv webm ts"

export JASNA_DISABLE_FFMPEG_CHECK=1
export DISABLE_FFMPEG_CHECK=1

if [ -d "/app/sglang" ]; then
	cd /app/sglang
fi

# 兼容容器内外 model_weights 双向软链接路径 (/app/sglang/model_weights <-> /data/jasna/model_weights)
if [ -d "/data/jasna/model_weights" ] && [ -d "/app/sglang/model_weights" ] && [ ! -L "/app/sglang/model_weights" ]; then
	if [ -z "$(ls -A /app/sglang/model_weights 2>/dev/null)" ]; then
		rmdir /app/sglang/model_weights 2>/dev/null && ln -sfn /data/jasna/model_weights /app/sglang/model_weights || true
	fi
elif [ -d "/app/sglang/model_weights" ] && [ ! -e "/data/jasna/model_weights" ]; then
	mkdir -p /data/jasna
	ln -sfn /app/sglang/model_weights /data/jasna/model_weights
fi
if [ -d "/data/jasna/model_weights" ] && [ ! -e "/data/jasna/model_weights/ltx-restore" ]; then
	ln -sfn . /data/jasna/model_weights/ltx-restore || true
fi

bitrate_to_kbps() {
	local value="${1:-}"
	value="${value// /}"
	if [ -z "$value" ]; then
		return 1
	fi

	case "$value" in
	*[Mm])
		echo "$((${value%[Mm]} * 1000))"
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
		echo "rc=vbr,maxrate=${target_kbps}k,bufsize=${target_kbps}k,cq=${ENCODER_CQ:-20},g=${ENCODER_GOP:-60}"
	else
		echo "cq=${ENCODER_CQ:-20}"
	fi
}

build_low_bitrate_encoder_settings() {
	if [ -n "${LOW_BITRATE_ENCODER_SETTINGS:-}" ]; then
		echo "$LOW_BITRATE_ENCODER_SETTINGS"
	else
		echo "rc=vbr,maxrate=0,bufsize=0,cq=${ENCODER_CQ:-20},g=${ENCODER_GOP:-60}"
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

	echo "$(((bitrate + 999) / 1000))"
}

# ---- 默认修复模型与路径自动探测 ----
RESTORATION_MODEL_NAME="${RESTORATION_MODEL_NAME:-ltx}"
if [ -z "${RESTORATION_MODEL_PATH:-}" ] && [[ "${RESTORATION_MODEL_NAME}" == ltx* ]]; then
	for cand in "/data/jasna/model_weights" "/data/model_weights" "/app/sglang/model_weights" "model_weights"; do
		if [ -f "${cand}/vae.safetensors.enc" ] || [ -f "${cand}/vae.safetensors" ]; then
			RESTORATION_MODEL_PATH="${cand}"
			break
		elif [ -f "${cand}/ltx-restore/vae.safetensors.enc" ] || [ -f "${cand}/ltx-restore/vae.safetensors" ]; then
			RESTORATION_MODEL_PATH="${cand}/ltx-restore"
			break
		fi
	done
fi

# 自动识别 GPU 架构 (RTX 5090 Blackwell SM 12.0+ 且存在 nvfp4 权重时自动开启 LTX_FAST=true，4090 自动使用 INT8)
if [ -z "${LTX_FAST:-}" ] && [[ "${RESTORATION_MODEL_NAME}" == ltx* ]]; then
	GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)"
	if [[ "${GPU_CC}" == 12.* ]] && [ -n "${RESTORATION_MODEL_PATH:-}" ]; then
		if ls "${RESTORATION_MODEL_PATH}"/*nvfp4*.safetensors* >/dev/null 2>&1; then
			LTX_FAST="true"
		fi
	fi
fi
LTX_FAST="${LTX_FAST:-false}"

# ---- 自动恢复持久化离线证书与探测 Release 运行时 ----
for mid_cand in "/data/jasna/.machine-id" "/data/.machine-id"; do
	if [ -f "${mid_cand}" ]; then
		cp "${mid_cand}" /etc/machine-id 2>/dev/null || true
		break
	fi
done
if [ ! -f "/root/.config/jasna/license.json" ]; then
	for lic_cand in "/data/jasna/license.json" "/data/license.json" "/app/sglang/model_weights/license.json"; do
		if [ -f "${lic_cand}" ]; then
			mkdir -p /root/.config/jasna
			cp "${lic_cand}" /root/.config/jasna/license.json
			break
		fi
	done
fi

JASNA_BIN="/app/sglang/sglang"
JASNA_RELEASE_BIN=""
FFMPEG_CHECK_ARG="--disable-ffmpeg-check"
MULTI_GPU="${MULTI_GPU:-true}"
IGNORE_HEAD_MOSAIC="${IGNORE_HEAD_MOSAIC:-true}"
export IGNORE_HEAD_MOSAIC
export JASNA_SMART_PIPELINE="$([ "${MULTI_GPU}" = "true" ] && echo "1" || echo "0")"

for rel_cand in "/app/jasna-release/jasna" "/data/jasna/.release-cache/v0.12.0/jasna-linux-nvidia-0.12.0/jasna" "/data/.release-cache/v0.12.0/jasna-linux-nvidia-0.12.0/jasna" "/data/jasna/.release-cache/v0.11.3/jasna-linux-nvidia-0.11.3/jasna" "/data/.release-cache/v0.11.3/jasna-linux-nvidia-0.11.3/jasna"; do
	if [ -x "${rel_cand}" ]; then
		rel_dir="$(dirname "${rel_cand}")"
		if [ -d "/data/jasna/model_weights" ] && [ ! -e "${rel_dir}/model_weights" ]; then
			ln -sfn /data/jasna/model_weights "${rel_dir}/model_weights" || true
		fi
		export JASNA_RELEASE_BIN="${rel_cand}"
		break
	fi
done

if [ -n "${JASNA_LICENSE:-}" ] && [ ! -f "/root/.config/jasna/license.json" ] && [ ! -f "/root/.local/share/jasna/license.json" ]; then
	echo "检测到 JASNA_LICENSE，正在执行激活..."
	${JASNA_RELEASE_BIN:-$JASNA_BIN} --license-code "${JASNA_LICENSE}" || true
fi

# ---- warmup 子命令 ----
if [ "$1" = "warmup" ]; then
	shift

	BATCH_SIZE="${BATCH_SIZE:-4}"
	DETECTION_MODEL="${DETECTION_MODEL:-rfdetr-v6-large}"
	WEIGHTS_DIR="/data/jasna/model_weights"
	[ ! -d "${WEIGHTS_DIR}" ] && WEIGHTS_DIR="model_weights"

	if [ -z "${DETECTION_MODEL_PATH:-}" ]; then
		if [ -f "${WEIGHTS_DIR}/${DETECTION_MODEL}.onnx" ]; then
			DETECTION_MODEL_PATH="${WEIGHTS_DIR}/${DETECTION_MODEL}.onnx"
		elif [ -f "${WEIGHTS_DIR}/${DETECTION_MODEL}.pt" ]; then
			DETECTION_MODEL_PATH="${WEIGHTS_DIR}/${DETECTION_MODEL}.pt"
		else
			DETECTION_MODEL_PATH="model_weights/${DETECTION_MODEL}.onnx"
		fi
	fi

	if [[ "${RESTORATION_MODEL_NAME}" == ltx* ]]; then
		echo "============================================"
		echo " Jasna LTX-2.5 22B & RF-DETR Warmup"
		echo "============================================"
		echo " Orchestrator:      ${JASNA_BIN}"
		echo " Release Worker:    ${JASNA_RELEASE_BIN:-local}"
		echo " Restoration Model: ${RESTORATION_MODEL_NAME}$([ "${LTX_FAST}" = "true" ] && echo " [NVFP4 Fast]" || echo " [INT8]")"
		echo " Restoration Path:  ${RESTORATION_MODEL_PATH:-${WEIGHTS_DIR}}"
		echo " Detection Model:   ${DETECTION_MODEL} (batch=${BATCH_SIZE})"
		echo " Detection Path:    ${DETECTION_MODEL_PATH}"
		echo " Head Filter:       ${IGNORE_HEAD_MOSAIC} (yolo26s-pose.pt)"
		echo "============================================"
		echo ""
		echo "[1/3] 预编译/校验 RF-DETR TensorRT 检测引擎..."

		JSON_DATA=$(
			cat <<EOF
{
    "device": "cuda:0",
    "fp16": true,
    "basicvsrpp": false,
    "detection": true,
    "detection_model_name": "${DETECTION_MODEL}",
    "detection_model_path": "${DETECTION_MODEL_PATH}",
    "detection_batch_size": ${BATCH_SIZE},
    "unet4x": false
}
EOF
		)
		${JASNA_RELEASE_BIN:-$JASNA_BIN} --compile-engines "$JSON_DATA"

		echo ""
		echo "[2/3] 预检/缓存人体姿态关键点模型 (yolo26s-pose.pt)..."
		python3.13 -c "from jasna.mosaic.head_filter import resolve_pose_model_path; p = resolve_pose_model_path(); print('Pose model ready:', p)" || true

		echo ""
		echo "[3/3] 预热 LTX 3D Causal VAE / 22B DiT / Decoder (torch-compile-cache & ltx-speed-profile.json)..."
		if [ "${FORCE_WARMUP:-false}" = "true" ] || [ ! -d "${WEIGHTS_DIR}/torch-compile-cache" ] || [ -z "$(ls -A "${WEIGHTS_DIR}/torch-compile-cache" 2>/dev/null)" ]; then
			rm -f "${WEIGHTS_DIR}/ltx-speed-profile.json"
		fi

		WARMUP_IN="/tmp/jasna_ltx_warmup_in.mp4"
		WARMUP_OUT="/tmp/jasna_ltx_warmup_out.mp4"
		rm -rf "${WARMUP_IN}" "${WARMUP_OUT}" /tmp/.jasna-resume-* 2>/dev/null || true
		ffmpeg -y -f lavfi -i color=c=black:s=1920x1080:r=30 -frames:v 225 -c:v libx264 -pix_fmt yuv420p "${WARMUP_IN}" >/dev/null 2>&1

		WARMUP_ARGS=(--detection-model "${DETECTION_MODEL}" --batch-size "${BATCH_SIZE}" --restoration-model-name "${RESTORATION_MODEL_NAME}" --fp16)
		[ -n "${RESTORATION_MODEL_PATH:-}" ] && WARMUP_ARGS+=(--restoration-model-path "${RESTORATION_MODEL_PATH}")
		[ "${LTX_FAST}" = "true" ] && WARMUP_ARGS+=(--ltx-fast)
		[ "${LTX_LARGE_CANVAS:-false}" = "true" ] && WARMUP_ARGS+=(--ltx-large-canvas)
		if [ "${LTX_TRIAL:-false}" = "true" ] || [ -z "${RESTORATION_MODEL_PATH:-}" ]; then
			WARMUP_ARGS+=(--ltx-trial)
		fi

		if [ -n "${JASNA_RELEASE_BIN:-}" ]; then
			$JASNA_RELEASE_BIN "${WARMUP_ARGS[@]}" --codec "${CODEC:-hevc}" --encoder-settings "$(build_encoder_settings)" --input "${WARMUP_IN}" --output "${WARMUP_OUT}" "$@"
		else
			$JASNA_BIN $FFMPEG_CHECK_ARG --no-multi-gpu "${WARMUP_ARGS[@]}" --codec "${CODEC:-hevc}" --encoder-settings "$(build_encoder_settings)" --input "${WARMUP_IN}" --output "${WARMUP_OUT}" "$@"
		fi
		rm -rf "${WARMUP_IN}" "${WARMUP_OUT}" "${WARMUP_OUT}.jasna-resume" 2>/dev/null || true

		echo ""
		echo "============================================"
		echo " LTX Warmup 完成! 检测引擎、torch-compile-cache 与 ltx-speed-profile.json 已就绪。"
		echo "============================================"
		exit 0
	else
		RESTORATION_MODEL="${RESTORATION_MODEL:-${WEIGHTS_DIR}/lada_mosaic_restoration_model_generic_v1.2.pth}"
		echo "============================================"
		echo " BasicVSR++ & RF-DETR TensorRT Engine Warmup"
		echo "============================================"
		echo " Batch Size:        ${BATCH_SIZE}"
		echo " Detection Model:   ${DETECTION_MODEL}"
		echo " Detection Path:    ${DETECTION_MODEL_PATH}"
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
    "detection": true,
    "detection_model_name": "${DETECTION_MODEL}",
    "detection_model_path": "${DETECTION_MODEL_PATH}",
    "detection_batch_size": ${BATCH_SIZE},
    "unet4x": false
}
EOF
		)
		$JASNA_BIN --compile-engines "$JSON_DATA"

		echo ""
		echo "============================================"
		echo " BasicVSR++ Warmup 完成! 引擎已缓存。"
		echo "============================================"
		exit 0
	fi
fi

if [ $# -gt 0 ]; then
	if [[ " $* " == *"--machine-key"* ]] || [[ " $* " == *"--license-code"* ]] || [[ " $* " == *"--activation-key"* ]]; then
		exec ${JASNA_RELEASE_BIN:-$JASNA_BIN} "$@"
	fi
	if [[ " $* " == *"--help"* ]] || [[ " $* " == *"--version"* ]]; then
		exec $JASNA_BIN "$@"
	fi
	PASSTHROUGH_MODEL_ARGS=()
	if [[ " $* " != *"--restoration-model-name"* ]] && [ -n "${RESTORATION_MODEL_NAME:-}" ]; then
		PASSTHROUGH_MODEL_ARGS+=(--restoration-model-name "${RESTORATION_MODEL_NAME}")
	fi
	if [[ " $* " != *"--restoration-model-path"* ]] && [ -n "${RESTORATION_MODEL_PATH:-}" ]; then
		PASSTHROUGH_MODEL_ARGS+=(--restoration-model-path "${RESTORATION_MODEL_PATH}")
	fi
	if [[ " $* " != *"--ltx-fast"* ]] && [[ " $* " != *"--no-ltx-fast"* ]] && [ "${LTX_FAST}" = "true" ]; then
		PASSTHROUGH_MODEL_ARGS+=(--ltx-fast)
	fi
	if [[ " $* " != *"--multi-gpu"* ]] && [[ " $* " != *"--no-multi-gpu"* ]]; then
		if [ "${MULTI_GPU}" = "true" ]; then
			PASSTHROUGH_MODEL_ARGS+=(--multi-gpu)
		else
			PASSTHROUGH_MODEL_ARGS+=(--no-multi-gpu)
		fi
	fi
	if [[ " $* " != *"--gpus"* ]] && [ -n "${GPUS:-}" ]; then
		PASSTHROUGH_MODEL_ARGS+=(--gpus "${GPUS}")
	fi
	if [[ " $* " != *"--ignore-head-mosaic"* ]] && [[ " $* " != *"--no-ignore-head-mosaic"* ]]; then
		if [ "${IGNORE_HEAD_MOSAIC}" = "true" ]; then
			PASSTHROUGH_MODEL_ARGS+=(--ignore-head-mosaic)
		else
			PASSTHROUGH_MODEL_ARGS+=(--no-ignore-head-mosaic)
		fi
	fi
	exec $JASNA_BIN $FFMPEG_CHECK_ARG --detection-model "${DETECTION_MODEL:-rfdetr-v6-large}" "${PASSTHROUGH_MODEL_ARGS[@]}" --encoder-settings "$(build_encoder_settings)" "$@"
fi

# ---- 无参数: 批量扫描模式 ----
SCAN_DIR="${SCAN_DIR:-/data/input}"
CODEC="${CODEC:-hevc}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BATCH_SIZE="${BATCH_SIZE:-4}"
DETECTION_MODEL="${DETECTION_MODEL:-rfdetr-v6-large}"
MAX_CLIP_SIZE="${MAX_CLIP_SIZE:-150}"
TEMPORAL_OVERLAP="${TEMPORAL_OVERLAP:-16}"
ENABLE_CROSSFADE="${ENABLE_CROSSFADE:-true}"
DENOISE="${DENOISE:-low}"
DETECTION_SCORE_THRESHOLD="${DETECTION_SCORE_THRESHOLD:-}"
FP16="${FP16:-true}"
COMPILE_BASICVSRPP="${COMPILE_BASICVSRPP:-true}"
LTX_LARGE_CANVAS="${LTX_LARGE_CANVAS:-false}"
LTX_FAST="${LTX_FAST:-false}"
LTX_TRIAL="${LTX_TRIAL:-false}"
LOG_LEVEL="${LOG_LEVEL:-info}"
TARGET_BITRATE="${TARGET_BITRATE:-5M}"
TARGET_BITRATE_KBPS="$(bitrate_to_kbps "$TARGET_BITRATE")"
PRESERVE_LOW_BITRATE="${PRESERVE_LOW_BITRATE:-${SKIP_LOW_BITRATE:-true}}"
ENCODER_SETTINGS="$(build_encoder_settings)"
LOW_BITRATE_ENCODER_SETTINGS="$(build_low_bitrate_encoder_settings)"
POST_COMPRESS_BITRATE="${POST_COMPRESS_BITRATE:-}"

JASNA_OPTS=""
[ -n "$BATCH_SIZE" ] && JASNA_OPTS="$JASNA_OPTS --batch-size $BATCH_SIZE"
[ -n "$DETECTION_MODEL" ] && JASNA_OPTS="$JASNA_OPTS --detection-model $DETECTION_MODEL"
[ -n "$RESTORATION_MODEL_NAME" ] && JASNA_OPTS="$JASNA_OPTS --restoration-model-name $RESTORATION_MODEL_NAME"
[ -n "$RESTORATION_MODEL_PATH" ] && JASNA_OPTS="$JASNA_OPTS --restoration-model-path $RESTORATION_MODEL_PATH"
[ -n "$DETECTION_SCORE_THRESHOLD" ] && JASNA_OPTS="$JASNA_OPTS --detection-score-threshold $DETECTION_SCORE_THRESHOLD"
[ "$FP16" = "true" ] && JASNA_OPTS="$JASNA_OPTS --fp16"
if [[ "${RESTORATION_MODEL_NAME}" == ltx* ]]; then
	[ "$LTX_LARGE_CANVAS" = "true" ] && JASNA_OPTS="$JASNA_OPTS --ltx-large-canvas"
	[ "$LTX_FAST" = "true" ] && JASNA_OPTS="$JASNA_OPTS --ltx-fast"
	[ "$LTX_TRIAL" = "true" ] && JASNA_OPTS="$JASNA_OPTS --ltx-trial"
else
	[ -n "$MAX_CLIP_SIZE" ] && JASNA_OPTS="$JASNA_OPTS --max-clip-size $MAX_CLIP_SIZE"
	[ -n "$TEMPORAL_OVERLAP" ] && JASNA_OPTS="$JASNA_OPTS --temporal-overlap $TEMPORAL_OVERLAP"
	[ "$ENABLE_CROSSFADE" = "true" ] && JASNA_OPTS="$JASNA_OPTS --enable-crossfade"
	[ -n "$DENOISE" ] && JASNA_OPTS="$JASNA_OPTS --denoise $DENOISE"
	[ "$COMPILE_BASICVSRPP" = "true" ] && JASNA_OPTS="$JASNA_OPTS --compile-basicvsrpp"
fi
[ -n "$LOG_LEVEL" ] && JASNA_OPTS="$JASNA_OPTS --log-level $LOG_LEVEL"
if [ "${MULTI_GPU}" = "true" ]; then
	JASNA_OPTS="$JASNA_OPTS --multi-gpu"
else
	JASNA_OPTS="$JASNA_OPTS --no-multi-gpu"
fi
[ -n "${GPUS:-}" ] && JASNA_OPTS="$JASNA_OPTS --gpus $GPUS"
if [ "${IGNORE_HEAD_MOSAIC}" = "true" ]; then
	JASNA_OPTS="$JASNA_OPTS --ignore-head-mosaic"
else
	JASNA_OPTS="$JASNA_OPTS --no-ignore-head-mosaic"
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "all" ]; then
	GPU_COUNT="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
else
	GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || echo "1")"
fi

echo "============================================"
echo " sglang 批量处理模式"
echo "============================================"
echo " 扫描目录: ${SCAN_DIR}"
echo " 修复模型: ${RESTORATION_MODEL_NAME}$([ "$LTX_FAST" = "true" ] && echo " [NVFP4 Fast]" || true)${RESTORATION_MODEL_PATH:+ (${RESTORATION_MODEL_PATH})}"
echo " 检测模型: ${DETECTION_MODEL}"
echo " 人头剔除: ${IGNORE_HEAD_MOSAIC} (Pose + 几何双框)"
echo " 多卡管线: ${MULTI_GPU} (可见 GPU=${GPU_COUNT}${GPUS:+, 指定 GPUS=${GPUS}})"
echo " 编码器:   ${CODEC}"
echo " NVENC:    ${ENCODER_SETTINGS}"
if [ "$PRESERVE_LOW_BITRATE" = "true" ] && [ -n "$TARGET_BITRATE_KBPS" ]; then
	echo " 低码率不压缩: <= ${TARGET_BITRATE_KBPS} kbps"
	echo " 低码率 NVENC: ${LOW_BITRATE_ENCODER_SETTINGS}"
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

# 收集候选队列（包含已有输出需记录“任务跳过”的条目，统一写入同一份 jasna.log）
CANDIDATES=()
while IFS= read -r -d '' filepath; do
	dir="$(dirname "$filepath")"
	filename="$(basename "$filepath")"
	name="${filename%.*}"

	# 跳过已经是 -U 结尾的输出文件
	if [[ "$name" == *-U ]]; then
		continue
	fi

	# 输出文件路径（统一转为 MP4 容器，搭配默认 HEVC/H.265 编码）
	output="${dir}/${name}-U.mp4"
	rel_name="${filepath#${SCAN_DIR}/}"
	[ "$rel_name" = "$filepath" ] && rel_name="$filename"

	if [ -f "$output" ]; then
		CANDIDATES+=("SKIP|${filepath}|${output}|${rel_name}||")
		continue
	fi

	task_encoder_settings="$ENCODER_SETTINGS"
	task_note=""
	if [ "$PRESERVE_LOW_BITRATE" = "true" ] && [ -n "$TARGET_BITRATE_KBPS" ]; then
		input_bitrate_kbps="$(probe_video_bitrate_kbps "$filepath" || true)"
		if [ -n "$input_bitrate_kbps" ] && [ "$input_bitrate_kbps" -le "$TARGET_BITRATE_KBPS" ]; then
			task_encoder_settings="$LOW_BITRATE_ENCODER_SETTINGS"
			task_note="输入码率 ${input_bitrate_kbps} kbps <= 目标 ${TARGET_BITRATE_KBPS} kbps，处理但不压缩"
		fi
	fi

	CANDIDATES+=("RUN|${filepath}|${output}|${rel_name}|${task_encoder_settings}|${task_note}")
done < <(find "$SCAN_DIR" -not -path "*/.*" -not -path "*/.release-cache/*" -not -path "*/jasna-release/*" -not -path "*/model_weights/*" -not -path "*/models/*" -not -path "*/src/*" -type f \( "${FIND_ARGS[@]}" \) -print0 | sort -z)

TOTAL=${#CANDIDATES[@]}
if [ "$TOTAL" -eq 0 ]; then
	echo "未找到需要处理的视频文件。"
	exit 0
fi

# 初始化批次唯一日志文件 (.tmp/restoration-order/private/runtime-logs/<run-id>/jasna.log)
export JASNA_RUNTIME_LOG_DIR="${JASNA_RUNTIME_LOG_DIR:-${SCAN_DIR}/.tmp/restoration-order/private/runtime-logs}"
LOG_PATH_TMP="/tmp/jasna_batch_log_path_$$.txt"
export JASNA_INIT_LOG_PATH_OUT="${LOG_PATH_TMP}"
python3.13 -m jasna.runtime_log init-batch \
	--total "$TOTAL" \
	--failure-policy "继续下一视频" \
	--params "--codec ${CODEC} --encoder-settings ${ENCODER_SETTINGS}${JASNA_OPTS}${EXTRA_ARGS:+ ${EXTRA_ARGS}}" || true
if [ -f "$LOG_PATH_TMP" ]; then
	export JASNA_BATCH_LOG_FILE="$(cat "$LOG_PATH_TMP")"
	rm -f "$LOG_PATH_TMP"
fi
unset JASNA_INIT_LOG_PATH_OUT

# 逐个串行处理并写入同一份 jasna.log
SUCCESS=0
FAIL=0
SKIPPED=0
BATCH_START_TS="$(date +%s)"

for i in "${!CANDIDATES[@]}"; do
	entry="${CANDIDATES[$i]}"
	mode="${entry%%|*}"
	rest="${entry#*|}"
	input="${rest%%|*}"
	rest="${rest#*|}"
	output="${rest%%|*}"
	rest="${rest#*|}"
	rel_name="${rest%%|*}"
	rest="${rest#*|}"
	task_encoder_settings="${rest%%|*}"
	task_note="${rest#*|}"
	idx=$((i + 1))

	if [ "$mode" = "SKIP" ]; then
		if [ -n "${JASNA_BATCH_LOG_FILE:-}" ]; then
			python3.13 -m jasna.runtime_log skip \
				--log-file "$JASNA_BATCH_LOG_FILE" \
				--name "$rel_name" \
				--index "$idx" \
				--total "$TOTAL" \
				--reason "输出已存在" \
				--output "$output" || true
		else
			echo "[跳过] 输出已存在: ${output}"
		fi
		SKIPPED=$((SKIPPED + 1))
		continue
	fi

	echo "============================================"
	echo " [${idx}/${TOTAL}] 处理中..."
	echo " 输入: ${input}"
	echo " 输出: ${output}"
	echo " 编码: ${CODEC}"
	echo " NVENC: ${task_encoder_settings}"
	if [ -n "$task_note" ]; then
		echo " 说明: ${task_note}"
	fi
	if [ -n "$EXTRA_ARGS" ]; then
		echo " 额外参数: ${EXTRA_ARGS}"
	fi
	echo "============================================"

	export JASNA_TASK_INDEX="$idx"
	export JASNA_TASK_TOTAL="$TOTAL"
	export JASNA_TASK_DISPLAY_NAME="$rel_name"

	if $JASNA_BIN \
		$FFMPEG_CHECK_ARG \
		--codec "$CODEC" \
		--encoder-settings "$task_encoder_settings" \
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

BATCH_ELAPSED="$(( $(date +%s) - BATCH_START_TS ))"
EXIT_CODE=0
if [ "$FAIL" -gt 0 ]; then
	EXIT_CODE=1
fi

if [ -n "${JASNA_BATCH_LOG_FILE:-}" ]; then
	python3.13 -m jasna.runtime_log end-batch \
		--log-file "$JASNA_BATCH_LOG_FILE" \
		--total "$TOTAL" \
		--success "$SUCCESS" \
		--failed "$FAIL" \
		--skipped "$SKIPPED" \
		--elapsed "$BATCH_ELAPSED" \
		--exit-code "$EXIT_CODE" || true
fi

echo ""
echo "============================================"
echo " 批量处理完成"
echo " 成功: ${SUCCESS} / ${TOTAL} (跳过: ${SKIPPED}, 失败: ${FAIL})"
if [ -n "${JASNA_BATCH_LOG_FILE:-}" ]; then
	echo " 日志: ${JASNA_BATCH_LOG_FILE}"
fi
echo "============================================"

exit "$EXIT_CODE"
