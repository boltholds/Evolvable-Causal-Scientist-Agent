# Paired DiscoveryWorld study: Learning Progress Controller

**Status:** External behavioral evaluation of existing domain-independent ECSA components. Do not interpret passing interface tests as better task performance.

## Question and causal comparison

Does opt-in progress-based experiment ranking improve ECSA's exploration behavior and official benchmark success relative to the *same* Arena B scientist without it?

Each paired seed independently initializes the same DiscoveryWorld Reactor Lab / Normal world, `WorldModelAcquisitionKernel(Locm2Learner)`, public observation decoder, numerical/text relation learners, applicability and lifted selectors, and fixed candidate/action budgets. The **sole policy switch** is `ContractExperimentCoordinator(use_progress_scoring=False/True)`. No learned world-model memory, fitted representation, evaluation scorecard, or previous-seed observations are shared between arms.

**Do not use Arena A's `AutonomousScientistPolicy` for this comparison**: it wraps `LegacyAutonomousScientist`, a different policy altogether. Our A/B experiment compares two configurations of Arena B.

The paired comparison controls starting worlds and budgets, not the *subsequent* experienced observation sequence: policies influence states they visit, so logged transitions cannot be matched action-by-action. Post-episode score measures whole-policy utility, not proof of a specific causal mechanism.

## Pre-registered metrics

**Primary:** per-seed official `scoreNormalized` and `completedSuccessfully`, retrieved **after** the episode ends and never used by the policy. Primary summary is the mean within-seed score difference: progress minus baseline, with every raw seed result retained. A small number of fixed seeds does not justify generalization beyond these instances.

**Secondary objective exploration metrics:**
- `total_actions`, `distinct_schemas`, `action_counts`, `max_schema_fraction`, `schema_entropy_bits`.
- `longest_exact_action_streak` (same publicly grounded action schema and arguments).
- `low_information_failed_repeats`: action has already been executed **at least twice in the same public structural context**, latest execution has `success=False`, and action-schema Beta-Bernoulli variance reduction is <= **0.002** (fixed a priori).
- Prequential Brier scores already emitted by Arena B; report actual number of model-ready predictions and attempted actions, not only aggregate means.
- `independently_confirmed_causal_hypotheses`: **null/unmeasured**, not 0, until a prospective independently registered treatment/control protocol is run in the actual agent loop. World-model proposal counts and raw evidence IDs are not substitutes.

This low-information retry statistic is a conservative *action-outcome diagnostic*. A failed action could still be scientifically informative; a small Beta-variance gain does not show that no other prediction changed. No action name or sensor field is classified in advance.

A 5-seed mean and its per-seed variance are descriptive only. A policy can lower repetition yet fail to improve the task score or scientifically confirmed discoveries. Report this as a negative or mixed result rather than cherry-picking.

## Full study on the user's WSL machine

```bash
cd /mnt/c/Users/bolthold/Documents/Code/Evolvable-Causal-Scientist-Agent
source .venv-wsl/bin/activate

git fetch origin
git switch feat/paired-learning-progress-discoveryworld-20261009
python -m pip install -e '.[test,discoveryworld,mlmd,world-model-planning]'

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
python -m ecsa.benchmarks.discoveryworld.paired_progress_arena_b \
  --seeds 0,1,2,3,4 \
  --max-steps 1000 \
  --max-ground-actions 64 \
  --output results/paired_progress_full
```

The output directory must be new or empty. Running the command intentionally does not overwrite existing experimental results. Each seed writes to `seed-N/baseline` and `seed-N/progress`. Raw transition traces contain no oracle score. `final_evaluation.json` is written only after termination, in the evaluator branch. `paired_summary.json` contains all per-seed and aggregate results.

**CI smoke** is intentionally tiny (two seeds, 12 steps, 16 candidate actions) to verify that real paired episodes can run. It is **not** a substitute for the five-seed, 1000-step study. Neither CUDA nor JEPA is needed: this A/B study isolates the experiment-selection policy, so the GPU would introduce a confound without adding relevant information.

## Interpretation

A result is favorable only if the paired official score does not regress materially and the exploration diagnostics improve consistently. More action diversity alone is not causality, and fewer low-information failed retries alone are not successful autonomous science.

Following this study, an independent experiment must **actually execute** preregistered matched interventions, and then compare the number of independently supported claims, as required by the approved universal effect-attribution design.
