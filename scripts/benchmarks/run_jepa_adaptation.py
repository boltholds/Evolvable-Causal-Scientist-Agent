"""CLI for blinded recurrent Action-JEPA adaptation and forgetting research."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecsa.experimental.jepa_adaptation_arena import run_adaptation_arena


def main() -> None:
    parser=argparse.ArgumentParser(description='Unannounced additive-gated-additive action JEPA adaptation study')
    parser.add_argument('--mode',choices=('smoke','full'),default='smoke')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=None)
    parser.add_argument('--adapt-steps',type=int,default=None)
    parser.add_argument('--budget',type=int,default=None)
    parser.add_argument('--noise-std',type=float,default=.16)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run_adaptation_arena(
        mode=args.mode,seed=args.seed,device=args.device,
        steps_override=args.steps,adaptation_steps=args.adapt_steps,
        acquisition_budget=args.budget,noise_std=args.noise_std,
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    detection=report['detection']
    adaptation=report['adaptation']
    print(json.dumps({
        'protocol':report['protocol'],'seed':report['seed'],'device':report['device'],
        'alarm_index':detection['alarm_index'],
        'change_confirmed':detection['change_confirmed'],
        'adaptation_triggered':adaptation['triggered'],
        'new_transition_count':adaptation['new_transition_count'],
        'intervention_steps':adaptation['intervention_steps'],
        'variants':{
            key:{'forgetting_delta':value.get('forgetting_delta'),
                 'gated_gain':value.get('gated_gain')}
            for key,value in adaptation['variants'].items()
            if key in ('naive','replay')
        },
        'checkpoint_router':adaptation['checkpoint_router'],
        'output':str(args.output),
    },indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
