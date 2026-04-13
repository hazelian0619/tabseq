#!/usr/bin/env bash
set -u -o pipefail

usage() {
  cat <<'USAGE'
用法:
  scripts/run_catboost_4metrics_all.sh [options] [-- 透传给 test_catboost_4metrics.py 的参数]

说明:
  - 默认只跑 diamonds
  - 除脚本自身参数外，其余参数会透传给 scripts/test_catboost_4metrics.py

脚本自身参数:
  --datasets <csv>     只跑指定数据集，例如: abalone,diamonds
  --python-bin <path>  指定 python，可选，默认: python
  --stop-on-error      遇到第一个失败就停止
  -h, --help           显示帮助

示例:
  bash scripts/run_catboost_4metrics_all.sh
  bash scripts/run_catboost_4metrics_all.sh --iterations 1000 --model-depth 8 --learning-rate 0.03
  bash scripts/run_catboost_4metrics_all.sh --datasets abalone,diamonds --confidence 0.9
USAGE
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python"
STOP_ON_ERROR=0
DATASETS_CSV=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      DATASETS_CSV="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --stop-on-error)
      STOP_ON_ERROR=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARD_ARGS+=("$@")
      break
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if [[ -n "$DATASETS_CSV" ]]; then
  IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
else
  DATASETS=("diamonds")
fi

FAILED=()
TOTAL="${#DATASETS[@]}"

run_one() {
  local ds="$1"
  if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    (cd "$ROOT" && "$PYTHON_BIN" scripts/test_catboost_4metrics.py --dataset "$ds" "${FORWARD_ARGS[@]}")
  else
    (cd "$ROOT" && "$PYTHON_BIN" scripts/test_catboost_4metrics.py --dataset "$ds")
  fi
}

check_dependency() {
  if ! "$PYTHON_BIN" -c "import catboost" >/dev/null 2>&1; then
    echo "[ERROR] missing dependency: catboost" >&2
    echo "[ERROR] install it first, for example:" >&2
    echo "  pip install catboost" >&2
    return 1
  fi
}

echo "[RUN] catboost four-metrics batch"
echo "[RUN] python=${PYTHON_BIN}"
echo "[RUN] datasets=${DATASETS[*]}"
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
  echo "[RUN] extra_args=${FORWARD_ARGS[*]}"
fi

if ! check_dependency; then
  exit 1
fi

for ((i=0; i<TOTAL; i++)); do
  ds="${DATASETS[$i]}"
  echo "[RUN] ($((i+1))/${TOTAL}) ${ds}"
  if ! run_one "$ds"; then
    echo "[FAIL] ${ds}"
    FAILED+=("$ds")
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      break
    fi
  else
    echo "[DONE] ${ds}"
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[SUMMARY] failed=${#FAILED[@]} dataset(s): ${FAILED[*]}"
  exit 1
fi

echo "[SUMMARY] all ${TOTAL} dataset(s) finished successfully"
