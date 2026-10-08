# Three-stage TextWorld model comparison (ECSA)

This is a diagnostic **frozen-representation** study, not a test of causal identification or TextWorld task success. Checkpoints are evaluated **sequentially** in 4-bit NF4 on one CUDA GPU. Each checkpoint is pinned to a particular Hugging Face revision.

| Stage | Model | Revision |
| --- | --- | --- |
| Base | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` |
| TextWorld SFT | `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` | `b052a201ae867c3058efba17c9af9cb1635d1f09` |
| BehR | `Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B` | `6a4326a60540cc33ffa42ee7a16fc2bad8f6f613` |

The SFT revision is the parent checkpoint specified by BehR's model card. Model weights are subject to their upstream licensing terms.

## Run in the existing WSL environment

```bash
cd /mnt/c/Users/bolthold/Documents/Code/Evolvable-Causal-Scientist-Agent
git switch main
git pull --ff-only origin main
source ~/.venvs/ecsa-behr/bin/activate
bash scripts/benchmarks/run_three_stage_world_model.sh smoke
bash scripts/benchmarks/run_three_stage_world_model.sh full
```

Ensure `torch.cuda.is_available()` and that enough disk space is available for **three separate 7B checkpoint downloads**. Do not attempt to fit all checkpoints simultaneously in 12 GB of GPU VRAM. The launcher validates CUDA, records installed packages/GPU/commit, resumes verified per-stage reports for the same source commit and runs one checkpoint per process.

`full` uses 3 seeds, 48 train / 16 calibration / 24 heldout witnessed pairs per law, two laws, two frozen pooling methods, two train-only projections, four KAN widths (2/8/16/32), 100 optimizer updates. There are **96 witnessed KAN cells per checkpoint**, without the intentionally contaminated shuffled-Y training arm. `smoke` uses 1 seed, 24/8/12 pairs, 2 widths, 20 updates and **4 KAN cells** per checkpoint.

A direct likelihood test uses the exact same witnessed heldout positive/negative pairs, teacher-forced `conditional_nll(X,Y)` and pair accuracy/margins without KAN. In `full` it evaluates up to 24 heldout pairs per law and seed; in `smoke`, 3. Identical negative Y outcomes are never labeled negative. The synthetic fixture uses hidden laws **only to evolve the environment**, not as learner labels.

A CPU-only structured control trains the **same KAN family** on observed numeric `u`, `v`, `u-v`, `u*v`, and one-hot `operator` plus observed Y measurement; neither the law type nor output predictions are provided as inputs. Normalization is fit on train positives only. This control intentionally has a more favorable inductive bias and should be treated as a KAN learnability sanity check.

## Files and interpretation

Outputs go to `results/three_stage_world_model/{smoke,full}/`:

- `base.json`, `textworld_sft.json`, `behr.json` — per-stage frozen-encoder KAN and direct NLL results
- `structured_control.json` — numeric/operator-only KAN control
- `comparison.json` — matched AUROC by law / width / seed, staged deltas, direct NLL metrics, structured control
- `*.log`, `gpu-before.txt`, `pip-freeze.txt`, `manifest.txt` — reproducibility

The key deltas are SFT − base (added TextWorld SFT), BehR − SFT (additional training), and BehR − base (total pipeline). These are **checkpoint differences**, not a randomized causal intervention on training algorithms. The same seeds share observations, while the many KAN widths/pools/projections reuse those observations and **must not be treated as 96 independent trials**. Follow-up work should add native TextWorld evaluation and prompt controls before generalizing to world-model skill.

Run offline tests without checkpoints using:

```bash
python -m pytest tests/test_three_stage_world_model.py tests/test_verified_relation_kan.py tests/test_world_model_encoders.py -q
```
