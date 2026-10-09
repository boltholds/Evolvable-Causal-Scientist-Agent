"""Reproducible online JEPA mechanism-router benchmark.

The runner never passes mechanism labels or change indices to the router.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecsa.experimental.jepa_router_arena import run_router_arena


def main() -> None:
    parser=argparse.ArgumentParser(description='Online ECSA JEPA mechanism hypothesis router')
    parser.add_argument('--mode',choices=('smoke','full'),default='smoke')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--seed',type=int,default=14)
    parser.add_argument('--noise-std',type=float,default=.16)
    parser.add_argument('--bocpd',action='store_true',help='Enable optional existing bocd change-point diagnostic')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run_router_arena(mode=args.mode,device=args.device,seed=args.seed,
                            noise_std=args.noise_std,use_bocpd=args.bocpd)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    stages=result['phases']
    print(json.dumps({
        'protocol':result['protocol'],'mode':result['mode'],
        'seed':result['seed'],'device':result['device'],
        'known_regime_reuse':[row['correct_known_selections'] for row in stages],
        'unknown_proposed':result['novelty']['proposed'],
        'unknown_admitted':result['novelty']['admitted'],
        'false_proposal_attempts':result['novelty']['false_proposal_attempts'],
        'false_admissions':result['novelty']['false_admissions'],
        'active_actions':result['interventions']['total_active_actions'],
    },indent=2))


if __name__=='__main__':
    main()
