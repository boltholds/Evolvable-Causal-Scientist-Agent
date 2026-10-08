#!/usr/bin/env bash
# Three pinned 7B checkpoints, evaluated sequentially on one CUDA GPU.
# bash scripts/benchmarks/run_three_stage_world_model.sh smoke|full
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-smoke}"
if [[ $# -gt 1 || ( "$MODE" != "smoke" && "$MODE" != "full" ) ]]; then
  echo "Usage: bash scripts/benchmarks/run_three_stage_world_model.sh [smoke|full]" >&2
  exit 2
fi

BASE_REV="d149729398750b98c0af14eb82c78cfe92750796"
SFT_REV="b052a201ae867c3058efba17c9af9cb1635d1f09"
BEHR_REV="6a4326a60540cc33ffa42ee7a16fc2bad8f6f613"
COMMIT="$(git rev-parse HEAD)"
OUT_DIR="results/three_stage_world_model/$MODE"
mkdir -p "$OUT_DIR"
EXPECTED_ROWS=4
EXPECTED_PROBES=2
if [[ "$MODE" == "full" ]]; then
  EXPECTED_ROWS=96
  EXPECTED_PROBES=6
fi
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$OUT_DIR/gpu-before.txt"
fi
python - <<'PY'
import importlib.metadata
import torch
assert torch.cuda.is_available(), "CUDA unavailable in this WSL Python environment"
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("bitsandbytes:", importlib.metadata.version("bitsandbytes"))
PY
python -m pip freeze > "$OUT_DIR/pip-freeze.txt"
printf 'scale=%s\nsource_revision=%s\nbase_revision=%s\nsft_revision=%s\nbehr_revision=%s\n' \
  "$MODE" "$COMMIT" "$BASE_REV" "$SFT_REV" "$BEHR_REV" > "$OUT_DIR/manifest.txt"

run_stage() {
  local stage="$1"
  local model="$2"
  local revision="$3"
  local result="$OUT_DIR/$stage.json"
  local log="$OUT_DIR/$stage.log"
  if [[ -s "$result" ]] && python - "$result" "$stage" "$model" "$revision" \
      "$MODE" "$COMMIT" "$EXPECTED_ROWS" "$EXPECTED_PROBES" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    valid=(d["stage"]==sys.argv[2] and d["model"]==sys.argv[3] and
           d["revision"]==sys.argv[4] and d["scale"]==sys.argv[5] and
           d["source_revision"]==sys.argv[6] and
           len(d["rows"])==int(sys.argv[7]) and
           len(d["likelihood"])==int(sys.argv[8]) and
           d["no_kan_to_encoder_gradient"] is True and
           all(row["negative_training"]=="witnessed" for row in d["rows"]))
except (OSError,KeyError,ValueError,TypeError):
    valid=False
raise SystemExit(0 if valid else 1)
PY
  then
    echo "[skip] $stage: matching pinned checkpoint, source commit and complete report"
    return
  fi
  echo "[run] $stage @ $revision"
  python -m ecsa.experimental.three_stage_world_model study \
    --stage "$stage" --scale "$MODE" --device cuda --4bit --max-length 1024 \
    --source-revision "$COMMIT" --output "$result" 2>&1 | tee "$log"
}

# Sequential processes guarantee that only one quantized checkpoint is resident.
run_stage base "Qwen/Qwen2.5-7B" "$BASE_REV"
run_stage textworld_sft "X1AOX1A/WorldModel-Textworld-Qwen2.5-7B" "$SFT_REV"
run_stage behr "Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B" "$BEHR_REV"

python -m ecsa.experimental.three_stage_world_model control \
  --directory "$OUT_DIR" --scale "$MODE" --source-revision "$COMMIT" \
  2>&1 | tee "$OUT_DIR/structured_control.log"
python -m ecsa.experimental.three_stage_world_model compare \
  --directory "$OUT_DIR" 2>&1 | tee "$OUT_DIR/comparison.log"
echo "Reports: $OUT_DIR/{base,textworld_sft,behr,structured_control,comparison}.json"
