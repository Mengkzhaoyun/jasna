#!/bin/bash
# decrypt_acceptance.sh —— .enc 真解密验收闸门
#
# 用法： decrypt_acceptance.sh [GPU序号] [输入视频]
# 断言：退出码 0、输出视频存在且非空、日志出现 48/48 blocks resident、
#       日志点名了被解密的 .enc 对应的模型文件。
#
# 前置：该 GPU 上已预热（首次运行需 10–15 分钟编译 TensorRT 引擎，之后约 70 s）。
set -u

GPU="${1:-2}"
INPUT="${2:-/data/jasna/ais/.vela_work/MKZ-033/seg_0000.mp4}"
R=/data/jasna/.release-cache/v0.11.3/jasna-linux-nvidia-0.11.3
MODELS=/data/jasna/model_weights
OUT_DIR=$(mktemp -d /tmp/decacc.XXXXXX)
LOG="$OUT_DIR/run.log"
REPORT="$MODELS/decrypt_acceptance.json"

cd "$R" || { echo "release 目录不存在"; exit 3; }

START=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 timeout 600 ./jasna \
  --input "$INPUT" \
  --output "$OUT_DIR/out.mp4" \
  --working-directory "$OUT_DIR" \
  --device cuda:0 --batch-size 1 \
  --detection-model rfdetr-v6 --detection-score-threshold 0.4 \
  --restoration-model-name ltx \
  --restoration-model-path "$MODELS" \
  --max-clip-size 32 --temporal-overlap 2 \
  --ltx-seed 20260923 --log-level info --fp16 --no-progress \
  > "$LOG" 2>&1
RC=$?
END=$(date +%s)
ELAPSED=$((END - START))

fail=0
check() { # name, condition(0=ok), detail
  local name="$1" cond="$2" detail="$3"
  if [ "$cond" -eq 0 ]; then
    echo "[PASS] $name — $detail"
  else
    echo "[FAIL] $name — $detail"
    fail=1
  fi
}

check "process.exit" "$([ "$RC" -eq 0 ] && echo 0 || echo 1)" "exit_code=$RC, elapsed=${ELAPSED}s"

SIZE=$(stat -c %s "$OUT_DIR/out.mp4" 2>/dev/null || echo 0)
check "output.produced" "$([ "$SIZE" -gt 1000000 ] && echo 0 || echo 1)" "out.mp4=${SIZE}B"

grep -q "blocks 48/48 resident" "$LOG"
check "weights.decrypted" "$?" "日志: $(grep -o 'blocks [0-9]*/[0-9]* resident' "$LOG" | tail -1)"

grep -q "ltx-restore-alpha1-distill2-int8.safetensors" "$LOG"
check "weights.identified" "$?" "$(grep -o 'ltx-restore-[a-z0-9-]*\.safetensors' "$LOG" | tail -1)"

grep -q "Spawning GPU model compilation" "$LOG"
COMPILED=$((1 - $?))
echo "[INFO] 本次是否触发引擎编译: $([ "$COMPILED" -eq 1 ] && echo 是 || echo 否)"

cat > "$REPORT" <<JSON
{
  "schema": "jasna.decrypt.acceptance/1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "gpu": "$GPU",
  "input": "$INPUT",
  "exit_code": $RC,
  "elapsed_seconds": $ELAPSED,
  "output_bytes": $SIZE,
  "engine_compiled": $COMPILED,
  "blocks_resident": "$(grep -o 'blocks [0-9]*/[0-9]* resident' "$LOG" | tail -1)",
  "model_file": "$(grep -o 'ltx-restore-[a-z0-9-]*\.safetensors' "$LOG" | tail -1)",
  "verdict": "$([ "$fail" -eq 0 ] && echo PASS || echo FAIL)"
}
JSON

echo "报告: $REPORT"
echo "日志: $LOG"
exit "$fail"
