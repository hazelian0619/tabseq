#!/usr/bin/env bash
set -euo pipefail

# 用法说明
usage() {
  cat <<'USAGE'
用法:
  scripts/run_eval_suite.sh --ckpt <run_dir_or_ckpt> [options]

参数:
  --ckpt <path>              必填。run 目录或 checkpoint.pt
  --eval-batch-size <int>    默认: 256
  --prefix-batch-size <int>  默认: 128
  --confidence <float>       默认: 0.9
  --temperature <float>      默认: 1.0
  --random-state <int>       默认: 0
  --export-samples <int>     默认: 200 (0 表示导出全部样本)
  --masks <csv>              默认: 1.0,0.5,0.3,0.1
  --beams <csv>              默认: 16,32
  --full-beam <int>          默认: 8
  --skip-greedy              跳过 greedy + mask 扫描
  --skip-teacher             跳过 teacher_forcing
  --skip-prefix              跳过 prefix_search（beam/full）

说明:
  - export-samples 只影响导出的 .npz 样本数，指标仍在全量验证集上计算。
  - 建议使用单行命令，避免换行导致参数被 shell 误解析。
USAGE
}

CKPT=""
EVAL_BATCH_SIZE=256
PREFIX_BATCH_SIZE=128
CONFIDENCE=0.9
TEMPERATURE=1.0
RANDOM_STATE=0
EXPORT_SAMPLES=200
MASKS="1.0,0.5,0.3,0.1"
BEAMS="16,32"
FULL_BEAM=8
RUN_GREEDY=1
RUN_TEACHER=1
RUN_PREFIX=1

# 参数解析
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --prefix-batch-size)
      PREFIX_BATCH_SIZE="$2"
      shift 2
      ;;
    --confidence)
      CONFIDENCE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --random-state)
      RANDOM_STATE="$2"
      shift 2
      ;;
    --export-samples)
      EXPORT_SAMPLES="$2"
      shift 2
      ;;
    --masks)
      MASKS="$2"
      shift 2
      ;;
    --beams)
      BEAMS="$2"
      shift 2
      ;;
    --full-beam)
      FULL_BEAM="$2"
      shift 2
      ;;
    --skip-greedy)
      RUN_GREEDY=0
      shift 1
      ;;
    --skip-teacher)
      RUN_TEACHER=0
      shift 1
      ;;
    --skip-prefix)
      RUN_PREFIX=0
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CKPT" ]]; then
  echo "缺少 --ckpt" >&2
  usage
  exit 1
fi

if [[ ! -e "$CKPT" ]]; then
  echo "路径不存在: $CKPT" >&2
  exit 1
fi

IFS=',' read -r -a MASK_ARR <<< "$MASKS"
IFS=',' read -r -a BEAM_ARR <<< "$BEAMS"

if [[ "$RUN_GREEDY" -eq 1 ]]; then
  echo "[greedy] masks: ${MASK_ARR[*]}"
  for mask in "${MASK_ARR[@]}"; do
    python3 scripts/eval.py --ckpt "$CKPT" --mode greedy --batch-size "$EVAL_BATCH_SIZE" \
      --random-state "$RANDOM_STATE" --confidence "$CONFIDENCE" --temperature "$TEMPERATURE" \
      --mask-outside "$mask" --export-leaf-probs --export-samples "$EXPORT_SAMPLES"
  done
fi

if [[ "$RUN_TEACHER" -eq 1 ]]; then
  echo "[teacher_forcing]"
  python3 scripts/eval.py --ckpt "$CKPT" --mode teacher_forcing --batch-size "$EVAL_BATCH_SIZE" \
    --random-state "$RANDOM_STATE" --confidence "$CONFIDENCE" --temperature "$TEMPERATURE"
fi

if [[ "$RUN_PREFIX" -eq 1 ]]; then
  echo "[prefix_search] beam sizes: ${BEAM_ARR[*]}"
  for beam in "${BEAM_ARR[@]}"; do
    python3 scripts/eval_prefix_search.py --ckpt "$CKPT" --mode beam --beam-size "$beam" \
      --batch-size "$PREFIX_BATCH_SIZE" --random-state "$RANDOM_STATE" --confidence "$CONFIDENCE" \
      --temperature "$TEMPERATURE" --export-leaf-probs --export-samples "$EXPORT_SAMPLES"
  done

  echo "[prefix_search] full (beam size $FULL_BEAM)"
  python3 scripts/eval_prefix_search.py --ckpt "$CKPT" --mode full --beam-size "$FULL_BEAM" \
    --batch-size "$PREFIX_BATCH_SIZE" --random-state "$RANDOM_STATE" --confidence "$CONFIDENCE" \
    --temperature "$TEMPERATURE" --export-leaf-probs --export-samples "$EXPORT_SAMPLES"
fi

echo "Done."
