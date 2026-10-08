# Local BehR-7B vs base Qwen2.5-7B on RTX 4070 Ti (WSL2)

Run the matched, **frozen-encoder / KAN-only** benchmark on your Windows host
through a normal Ubuntu WSL2 distro. Do NOT use the broken
\`NVIDIA-SDKM-Ubuntu-24.04\` installation; use the other working \`Ubuntu\` distro
if it exposes the NVIDIA GPU.

All pretrained model weights are loaded in **4-bit NF4**, and the two
checkpoints are run **sequentially**, never at the same time. A 12 GB RTX
4070 Ti is a plausible target but free VRAM and compatible NVIDIA drivers
still matter. Close LM Studio or other GPU-heavy programs before starting.

The \`Qwen/Qwen2.5-7B\` base checkpoint is ~15.2 GB on disk before 4-bit
runtime quantization. BehR is another full 7B checkpoint. Reserve at least
~40 GB of free disk space for both HF weight caches, Python packages and
results. 4-bit runtime quantization does *not* make the source download 4-bit.

## 1. Enter working Ubuntu from Windows PowerShell

\`\`\`powershell
wsl -l -v
wsl -d Ubuntu
\`\`\`

All subsequent commands run **inside Ubuntu**, not PowerShell. Do not use
a Windows \`.venv\` from the shared checkout.

## 2. Open the existing ECSA checkout and create an isolated Python 3.12 venv

\`\`\`bash
cd /mnt/c/Users/bolthold/Documents/Code/Evolvable-Causal-Scientist-Agent
git status --short
git switch main
git pull --ff-only origin main
python3.12 -m venv ~/.venvs/ecsa-behr
source ~/.venvs/ecsa-behr/bin/activate
python -m pip install --upgrade pip setuptools wheel
\`\`\`

If \`python3.12\` or \`venv\` is missing, install a Python 3.12 interpreter for
this Ubuntu distro first. Do not reuse the existing Python 3.13 Conda base
environment for this experiment.

If \`git status --short\` shows edits, preserve or commit them before pulling.
If WSL networking/DNS fails, check whether AmneziaVPN is disrupting WSL
DNS as it did on this machine previously. Do not modify the system resolver
blindly.

## 3. Install matching CUDA packages

For systems whose \`nvidia-smi\` driver reports **CUDA 12.4**, PyTorch's
official CUDA 12.4 build is available:

\`\`\`bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[test,pretrained-encoder]' bitsandbytes
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'
df -h "$HOME"
\`\`\`

The benchmark checks \`torch.cuda.is_available()\` and the bitsandbytes
package before downloading the model checkpoints. Install a matching PyTorch
wheel if the machine's current driver does not support cu124.
Official instructions: https://pytorch.org/get-started/previous-versions/
and https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx.

## 4. Smoke test, then matched multi-seed comparison

\`\`\`bash
bash scripts/benchmarks/run_behr_vs_qwen25_7b.sh smoke
bash scripts/benchmarks/run_behr_vs_qwen25_7b.sh full
\`\`\`

The first mode evaluates 32 cells per checkpoint on narrower latent widths.
The full mode evaluates 192 cells per checkpoint across:
- precision and operator synthetic deterministic mechanisms;
- seeds 0, 1, 2;
- frozen last-token state and masked contextual-token mean;
- random and **training-only PCA** projections;
- latent widths 2, 8, 16 and 32;
- repeat-supported negatives and the historical shuffled-Y diagnostic.

Each run has exactly the same fixture seeds and splits, KAN hypothesis
architecture for a given latent dimension, KAN optimizer budget, and
heldout replay-supported evaluation. The script verifies model ID,
checkpoint SHA and matched row keys.

The script reuses **complete validated** JSON results on restart. If one
model fails (GPU OOM, network interruption), rerun the same command after
addressing the issue; it won't redo the successfully saved other model.
It records the repository revision, Python packages and GPU hardware.

Models are pinned to immutable Hugging Face commits:
- BehR: \`Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B\` at
  \`6a4326a60540cc33ffa42ee7a16fc2bad8f6f613\`.
- Base: \`Qwen/Qwen2.5-7B\` at
  \`d149729398750b98c0af14eb82c78cfe92750796\`.

## 5. Send back results

Outputs for the full run:

\`\`\`text
results/behr_vs_qwen25_7b/full/behr.json
results/behr_vs_qwen25_7b/full/qwen_base.json
results/behr_vs_qwen25_7b/full/comparison.json
results/behr_vs_qwen25_7b/full/pip-freeze.txt
results/behr_vs_qwen25_7b/full/gpu-before.txt
\`\`\`

The three JSONs contain the scientific measurements and paired comparison;
package and GPU reports are useful only for debugging/reproduction.
Do not upload HF tokens or private cache credentials.

## Interpretation limits

This test evaluates representation utility on synthetic deterministic
replay-observed X/action -> Y transitions. It does not prove generalizable
causal reasoning, counterfactual identification, or correct stochastic
world models. It does **not** reintroduce KAN-to-encoder feedback. PCA uses
training data only; latent width affects KAN parameter count, so compare
same-width variants separately.

A base-model-versus-BehR comparison checks the **combined effect** of
TextWorld world-model SFT and BehR optimization, not BehR's incremental
training alone. To isolate the latter, a third optional experiment should
compare against BehR's parent:
\`X1AOX1A/WorldModel-Textworld-Qwen2.5-7B\` at
\`b052a201ae867c3058efba17c9af9cb1635d1f09\` (if still available).
