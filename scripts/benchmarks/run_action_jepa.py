#!/usr/bin/env python3
"""Offline CLI for JEPA causal-commutator experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecsa.experimental.jepa_commutator_arena import run_arena


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=0.16)
    parser.add_argument("--output", type=Path, default=Path("results/action_jepa/report.json"))
    args = parser.parse_args()
    report = run_arena(mode=args.mode, seed=args.seed, device=args.device, steps_override=args.steps, noise_std=args.noise_std)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}; {len(report['mechanisms'])} worlds, {args.mode=}")


if __name__ == "__main__":
    main()
