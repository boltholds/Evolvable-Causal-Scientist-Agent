#!/usr/bin/env bash
# Reproducible two-checkpoint comparison on one CUDA GPU (WSL2/Linux).
# Usage: bash scripts/benchmarks/run_behr_vs_qwen25_7b.sh smoke
#        bash scripts/benchmarks/run_behr_vs_qwen25_7b.sh full
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-smoke}"
if [[ $# -gt 1 || ( "$MODE" != "smoke" && "$MODE" != "full" ) ]]; then
  echo "Usage: bash scripts/benchmarks/run_behr_vs_qwen25_7b.sh [smoke|full]" >&2
  exit 2
fi

BEHR_REV="6a4326a60540cc33ffa42ee7a16fc2bad8f6f613"
BASE_REV="d149729398750b98c0af14eb82c78cfe92750796"
OUT_DIR="results/behr_vs_qwen25_7b/$MODE"
mkdir -p "$OUT_DIR"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ "$MODE" == "smoke" ]]; then
  # Cheap contract/equality run. Separate output from the full experiment.
  EXPECTED_ROWS=32
  COMMON=(
    --device cuda --4bit --max-length 512
    --laws precision operator --seeds 0
    --pool sentence_full token_mean
    --projection random train_only_pca
    --dimensions 2 8
    --pairing witnessed shuffled_y_unsafe_control
    --train 24 --calibration 8 --heldout 12
    --replays 3 --steps 25
  )
else
  # Matched, multi-seed 7B study. Width changes KAN parameter count.
  EXPECTED_ROWS=192
  COMMON=(
    --device cuda --4bit --max-length 1024
    --laws precision operator --seeds 0 1 2
    --pool sentence_full token_mean
    --projection random train_only_pca
    --dimensions 2 8 16 32
    --pairing witnessed shuffled_y_unsafe_control
    --train 48 --calibration 16 --heldout 24
    --replays 3 --steps 100
  )
fi

echo "ECSA benchmark mode: $MODE"
echo "Outputs: $OUT_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$OUT_DIR/gpu-before.txt"
fi
python - <<'PY'
import importlib.metadata
import torch
assert torch.cuda.is_available(), "CUDA unavailable in this Python venv/WSL distro"
print("PyTorch:", torch.__version__, "CUDA wheel:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
print("bitsandbytes:", importlib.metadata.version("bitsandbytes"))
PY

git rev-parse HEAD > "$OUT_DIR/ecsa-commit.txt"
python -m pip freeze > "$OUT_DIR/pip-freeze.txt"
printf 'mode=%s\nbehr_revision=%s\nbase_revision=%s\n' \
  "$MODE" "$BEHR_REV" "$BASE_REV" > "$OUT_DIR/manifest.txt"

run_one() {
  local label="$1"
  local candidate="$2"
  local revision="$3"
  local model_id="$4"
  local result="$OUT_DIR/$label.json"
  local log="$OUT_DIR/$label.log"
  if [[ -s "$result" ]] && python - "$result" "$model_id" "$revision" "$EXPECTED_ROWS" <<'PY'
import json,sys
try:
    data=json.load(open(sys.argv[1]))
    valid=(data.get("model")==sys.argv[2] and
           data.get("revision")==sys.argv[3] and
           len(data.get("rows", []))==int(sys.argv[4]) and
           data.get("no_kan_to_encoder_gradient") is True)
except (ValueError,OSError,KeyError,TypeError):
    valid=False
raise SystemExit(0 if valid else 1)
PY
  then
    echo "[skip] $label: existing validated $result"
    return
  fi
  echo "[run] $label ($model_id@$revision)"
  python -m ecsa.experimental.verified_relation_kan \
    --candidate "$candidate" --revision "$revision" \
    "${COMMON[@]}" --output "$result" 2>&1 | tee "$log"
  python - "$result" "$model_id" "$revision" "$EXPECTED_ROWS" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data["model"]==sys.argv[2], data["model"]
assert data["revision"]==sys.argv[3], data["revision"]
assert len(data["rows"])==int(sys.argv[4]), len(data["rows"])
assert all(row["identity_overlap"]==0 for row in data["rows"])
assert data["no_kan_to_encoder_gradient"] is True
print("Validated:",sys.argv[1],len(data["rows"]),"measurements")
PY
}

# Sequential loading: do NOT keep two quantized 7B models in GPU memory.
run_one "behr" "behr_textworld" "$BEHR_REV" \
  "Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B"
run_one "qwen_base" "base_qwen25" "$BASE_REV" \
  "Qwen/Qwen2.5-7B"

python - "$OUT_DIR" <<'PY'
import json,statistics,sys
from collections import defaultdict
from pathlib import Path

directory=Path(sys.argv[1])
behr=json.loads((directory/"behr.json").read_text())
base=json.loads((directory/"qwen_base.json").read_text())
assert behr["config"]==base["config"]
assert behr["replays_are_deterministic_fixture_only"]
assert base["replays_are_deterministic_fixture_only"]
assert behr["no_kan_to_encoder_gradient"] and base["no_kan_to_encoder_gradient"]

def key(r):
    return (r["law"],r["seed"],r["pool"],r["projection"],
            r["negative_training"],r["dimension"])
a={key(r):r for r in behr["rows"]}
b={key(r):r for r in base["rows"]}
assert len(a)==len(behr["rows"]) and len(b)==len(base["rows"])
assert set(a)==set(b), "Model runs were not performed on identical benchmark cells"
for k in a:
    for dimension in ("train_pairs","calibration_pairs","heldout_pairs",
                      "kan_parameters","identity_overlap"):
        assert a[k][dimension]==b[k][dimension], (k,dimension)

groups=defaultdict(list)
for k in sorted(a):
    law,seed,pool,projection,training,dim=k
    groups[(law,pool,training)].append((
        a[k]["heldout_auroc"],b[k]["heldout_auroc"]
    ))

print()
print("MATCHED CHECKPOINT COMPARISON (same X/Y cohort, KAN seeds and budgets)")
print("law | pool | negative labels | BehR AUROC | base AUROC | delta | cells")
results=[]
for (law,pool,training),scores in sorted(groups.items()):
    ave_a=statistics.mean(x[0] for x in scores)
    ave_b=statistics.mean(x[1] for x in scores)
    delta=ave_a-ave_b
    print(f"{law} | {pool} | {training} | {ave_a:.4f} | "
          f"{ave_b:.4f} | {delta:+.4f} | {len(scores)}")
    results.append({"law":law,"pool":pool,"negative_training":training,
                    "behr_mean_auroc":ave_a,"qwen_base_mean_auroc":ave_b,
                    "delta":delta,"cells":len(scores)})

output={
    "models":{"behr":{"id":behr["model"],"revision":behr["revision"]},
              "qwen_base":{"id":base["model"],"revision":base["revision"]}},
    "config":behr["config"],"rows_per_model":len(a),
    "grouped_results":results,
    "cautions":[
        "Means aggregate architecture cells, not independent seeds.",
        "Both checkpoints evaluated on the same deterministic synthetic fixtures.",
        "The shuffled-Y arm contains false negatives by design and is not a valid training protocol.",
        "BehR has an intermediate TextWorld SFT ancestor; base comparison does not isolate the additional BehR RL stage.",
        "Encoder/feature projections are frozen; only KAN heads are trained.",
    ],
}
(directory/"comparison.json").write_text(json.dumps(output,indent=2)+"\n")
print("Written:",directory/"comparison.json")
PY

echo "Finished. Send these files: $OUT_DIR/behr.json,"
echo "  $OUT_DIR/qwen_base.json and $OUT_DIR/comparison.json"
