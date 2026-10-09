"""Sealed evaluator for online reuse and open-set JEPA hypothesis admission.

The model-facing router receives no fixture law, segmentation or change time.
An initial bank contains two independently trained candidate checkpoints in
one anchored frame. One unseen predictive law appears only later; its candidate
may be promoted only following a separate heldout intervention cohort.
"""
from __future__ import annotations

from dataclasses import asdict,replace
import numpy as np

from .action_jepa import JepaConfig,train_jepa
from .jepa_adaptation import AdaptationConfig,adapt_predictor
from .jepa_mechanism_router import (
    Checkpoint, ConfirmationEvidence, JepaPredictivePort,MechanismRouter,
    RouterConfig,RouteKind,BOCPDChangeSignal,
)
from .jepa_router_world import (
    RouterFixtureLaw,collect_router_transitions,generate_blinded_stream,
)
from .jepa_recurrence_world import segment
from .jepa_world import Action


def _prepare_checkpoint(model,cal,*,checkpoint_id:str,origin_id:str,train_count:int):
    return Checkpoint(checkpoint_id=checkpoint_id,predictor=JepaPredictivePort(model),
                      calibration=cal,calibration_ids=tuple(f'{origin_id}-cal-{i}' for i in range(len(cal.before))),
                      training_ids=tuple(f'{origin_id}-train-{i}' for i in range(train_count)))


def run_router_arena(*,mode:str='smoke',seed:int=0,device:str='cpu',
                     noise_std:float=.16,use_bocpd:bool=False)->dict[str,object]:
    if mode not in ('smoke','full'):raise ValueError('mode must be smoke/full')
    if type(seed) is not int or seed<0:raise ValueError('seed must be nonnegative')
    if device not in ('cpu','cuda'):raise ValueError('device must be cpu/cuda')
    if not 0<=noise_std<=1 or not np.isfinite(noise_std):raise ValueError('noise must be in [0,1]')
    if type(use_bocpd) is not bool:raise TypeError('use_bocpd must be bool')
    # Fix the protocol before scoring; phase times are visible ONLY below.
    training_count=256 if mode=='smoke' else 2048
    calibration_count=160 if mode=='smoke' else 512
    phase_length=60 if mode=='smoke' else 96
    adaptation_examples=32 if mode=='smoke' else 64
    confirmation_count=32 if mode=='smoke' else 64
    steps=90 if mode=='smoke' else 700
    adaptation_steps=90 if mode=='smoke' else 180
    cfg=JepaConfig(steps=steps,seed=seed)
    adapt_cfg=AdaptationConfig(steps=adaptation_steps,max_new_examples=adaptation_examples,
                              max_replay_examples=128 if mode=='full' else 64,
                              seed=seed+1)
    base_train=collect_router_transitions(law=RouterFixtureLaw.ADDITIVE,count=training_count,
                                          seed=1000+seed,noise_std=noise_std)
    base_cal=collect_router_transitions(law=RouterFixtureLaw.ADDITIVE,count=calibration_count,
                                        seed=2000+seed,noise_std=noise_std)
    gated_train=collect_router_transitions(law=RouterFixtureLaw.GATED,count=adaptation_examples,
                                           seed=3000+seed,noise_std=noise_std)
    gated_cal=collect_router_transitions(law=RouterFixtureLaw.GATED,count=calibration_count,
                                         seed=4000+seed,noise_std=noise_std)
    pretrained,report=train_jepa(base_train,cfg,device=device)
    adapted,_=adapt_predictor(pretrained,gated_train,config=adapt_cfg,old_replay=None)
    checkpoints=(
        _prepare_checkpoint(pretrained,base_cal,checkpoint_id='checkpoint-0',origin_id='initial',train_count=training_count),
        _prepare_checkpoint(adapted,gated_cal,checkpoint_id='checkpoint-1',origin_id='adapted',train_count=len(gated_train.before)),
    )
    r_config=RouterConfig(switch_hazard=.015,accept_probability=.70,
                          novelty_streak=3,novelty_window=18,novelty_p=.025)
    router=MechanismRouter(checkpoints,config=r_config,
                          change_detector=BOCPDChangeSignal() if use_bocpd else None)
    law_sequence=(RouterFixtureLaw.ADDITIVE,RouterFixtureLaw.GATED,
                  RouterFixtureLaw.ADDITIVE,RouterFixtureLaw.REVERSE_GATED,
                  RouterFixtureLaw.ADDITIVE,RouterFixtureLaw.REVERSE_GATED)
    stream=generate_blinded_stream(laws=law_sequence,phase_length=phase_length,
                                   seed=seed,noise_std=noise_std)
    decisions=[]
    phase_summary=[]
    attempts=[]
    budget=dict(training_new_examples=0,candidate_calibration_examples=0,
                confirming_examples=0,proposed_diagnostic_actions=0)
    first_proposal_index=None
    initial_count=len(router.checkpoint_ids)
    accepted_novel_checkpoint=None
    # Online protocol: after EACH alarm immediately investigate the CURRENT
    # world. False early alerts cannot borrow evidence from future mechanisms.
    for phase_index,law in enumerate(law_sequence):
        phase=segment(stream,phase_index*phase_length,(phase_index+1)*phase_length)
        phase_events=[]
        for row in range(len(phase.before)):
            index=phase_index*phase_length+row
            event=router.observe(transition_id=f'public-transition-{seed}-{index}',
                                 before=phase.before[row],action=phase.actions[row],after=phase.after[row])
            phase_events.append(event)
            decisions.append(event)
            proposal=router.pending_proposal
            if proposal is None or event.kind is not RouteKind.PROPOSE_NEW:
                continue
            if phase_index==3 and first_proposal_index is None:
                first_proposal_index=index
            attempt_number=len(attempts)
            # The experiment selector is model-facing and sees only public
            # sensor values, actions and predictive distributions.
            query_action,information=router.select_experiment(
                phase.before[row],candidate_actions=tuple(Action))
            # The evaluator supplies a resettable lab at the ALERT EPOCH;
            # the simulator law is strictly private to THIS module.
            origin=f'candidate-{seed}-{attempt_number}'
            acquired=collect_router_transitions(law=law,count=adaptation_examples,
                seed=seed*100_000+6000+attempt_number*103,noise_std=noise_std,
                action_bias=query_action)
            calibration_new=collect_router_transitions(law=law,count=calibration_count,
                seed=seed*100_000+7000+attempt_number*103,noise_std=noise_std)
            # Independent training, calibration, validation and alert evidence.
            trained,_=adapt_predictor(pretrained,acquired,
                config=replace(adapt_cfg,seed=seed*100+88+attempt_number),
                old_replay=None)
            name=f'checkpoint-{len(router.checkpoint_ids)}'
            candidate=_prepare_checkpoint(trained,calibration_new,
                checkpoint_id=name,origin_id=origin,train_count=len(acquired.before))
            validation=collect_router_transitions(law=law,count=confirmation_count,
                seed=seed*100_000+8000+attempt_number*103,noise_std=noise_std,
                action_bias=query_action)
            evidence=ConfirmationEvidence(proposal.proposal_id,validation,
                tuple(f'{origin}-independent-confirm-{i}' for i in range(len(validation.before))))
            audit=router.admit_checkpoint(candidate,evidence)
            attempts.append(dict(
                alarm_index=index,
                evaluator_law_at_alarm=law.value,
                evaluator_was_novel_at_alarm=(phase_index==3 and accepted_novel_checkpoint is None),
                hypothesis_id=proposal.proposal_id,
                selected_action=query_action.value,
                expected_information_gain_bits=information.information_gain_bits,
                result=asdict(audit),
                train_count=len(acquired.before),
                calibration_count=len(calibration_new.before),
                confirm_count=len(validation.before),
                new_checkpoint_id=name if audit.admitted else None,
            ))
            budget['training_new_examples']+=len(acquired.before)
            budget['candidate_calibration_examples']+=len(calibration_new.before)
            budget['confirming_examples']+=len(validation.before)
            # select_experiment() predicts in memory; it does not execute an
            # intervention. All executed steps are counted in the cohorts.
            if audit.admitted and phase_index==3 and accepted_novel_checkpoint is None:
                accepted_novel_checkpoint=name
        expected=('checkpoint-1' if law is RouterFixtureLaw.GATED
                  else accepted_novel_checkpoint if law is RouterFixtureLaw.REVERSE_GATED
                  else 'checkpoint-0')
        recent=phase_events[-min(20,len(phase_events)):]
        phase_summary.append(dict(
            phase=phase_index,
            true_mechanism=law.value,  # scorer-only, never passed to router
            expected_checkpoint=expected,
            correct_known_selections=sum(d.selected_checkpoint==expected for d in recent) if expected else 0,
            late_window_size=len(recent),
            ambiguous_choices=sum(d.kind is RouteKind.DISAMBIGUATE for d in recent),
            novelty_proposals=sum(d.kind is RouteKind.PROPOSE_NEW for d in phase_events),
            last_posterior=dict(phase_events[-1].posterior.probabilities),
        ))
    false_attempts=sum(not a['evaluator_was_novel_at_alarm'] for a in attempts)
    false_admissions=sum(not a['evaluator_was_novel_at_alarm'] and a['result']['admitted'] for a in attempts)
    real_attempts=[a for a in attempts if a['evaluator_was_novel_at_alarm']]
    budget['total_active_actions']=budget['training_new_examples']+budget['candidate_calibration_examples']+budget['confirming_examples']
    return dict(protocol='jepa_mechanism_router_v1',mode=mode,seed=seed,
         device=device,noise_std=noise_std,
         source_labels_used_by_router=False,
         new_model_auto_admitted=False,
         initial_known_models=initial_count,
         initial_training_loss=asdict(report),
         phases=phase_summary,
         novelty=dict(proposed=bool(real_attempts),admitted=accepted_novel_checkpoint is not None,
                      first_proposal_index=first_proposal_index,
                      confirmed=next((a['result'] for a in real_attempts if a['result']['admitted']),
                                     real_attempts[0]['result'] if real_attempts else None),
                      false_proposal_events_before_unknown=sum(d.kind is RouteKind.PROPOSE_NEW for d in decisions[:3*phase_length]),
                      false_proposal_attempts=false_attempts,
                      false_admissions=false_admissions,
                      attempts=attempts),
         interventions=budget,
         test_sets_disjoint_from_training=True,
         diagnostic_change_point_alarms=[i for i,d in enumerate(decisions) if d.change_point_signal],
         limitations=(
           'Ground truth mechanism labels occur in evaluator only, never in router inputs.',
           'The first two experts are prepared from mode-specific example cohorts, not autonomously discovered.',
           'Novel candidate is trained from fresh interventions; this is predictive novelty, not full causal structure identification.',
           'Diagonal Gaussian binned latent likelihoods are calibration-dependent approximations, not proven calibrated probabilities.',
           'Router observes every epoch even if hypothesis promotion is rejected; science kernel update uses posterior hazard.',
           'BOCPD is optional and diagnostic; its signal is not required to admit a new hypothesis.',
           'Separate JEPA checkpoints share a frozen encoder; no continual representation learning evaluated.',
         ))
