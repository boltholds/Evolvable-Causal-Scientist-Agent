"""Run the ECSA hidden-mechanism-shift causal scientist benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from ecsa.experimental.jepa_shift_arena import run_shift_arena


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("smoke","full"),default="smoke")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--noise-std",type=float,default=.16)
    parser.add_argument("--steps",type=int)
    parser.add_argument("--output",type=Path,default=Path("results/jepa_hidden_shift.json"))
    args=parser.parse_args()
    report=run_shift_arena(mode=args.mode,seed=args.seed,device=args.device,
                           steps_override=args.steps,noise_std=args.noise_std)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    summary={name:{key:value for key,value in row.items()
                   if key in ("alarm_index","delay_actions","null_false_alarm",
                              "selected_pair","confirmed_order_effect")}
             for name,row in report["controls"].items()}
    print(json.dumps({"protocol":report["protocol"],"seed":args.seed,
                      "controls":summary},indent=2))


if __name__=="__main__":
    main()
