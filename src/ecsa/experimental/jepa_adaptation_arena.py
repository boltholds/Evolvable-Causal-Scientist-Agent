"""Blinded recurrent dynamics arena: detect, intervene, adapt, recall.

Only this evaluator imports simulator `Mechanism` and the two regime boundaries.
The JEPA adapter and checkpoint router see observations/actions, never laws.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .action_jepa import (
    JepaConfig, RawObservationEncoder, evaluate_prediction, train_jepa,
)
from .jepa_adaptation import (
    AdaptationConfig, adapt_predictor, predict_mse, route_by_prediction,
)
from .jepa_recurrence_world import (
    RecurrenceWorld, observe_recurrence_stream, segment,
)
from .jepa_shift_detection import CalibratedShiftMonitor, FrozenJEPAResiduals, RidgeDynamics
from .jepa_shift_scientist import discover_order_hypothesis, score_order_effect
from .jepa_causal_scientist import collect_commutator_observations
from .jepa_adaptation_scientist import compare_order_effect
from .jepa_shift_world import HiddenShiftWorld, observe_stream
from .jepa_world import Mechanism, collect_transitions


def _evaluate(model, additive, gated) -> dict[str,object]:
    a=evaluate_prediction(model,additive)
    g=evaluate_prediction(model,gated)
    return {
        'additive':asdict(a),'gated':asdict(g),
        'anchored_encoder':True,
    }


def _extract_discovery(report) -> dict[str,object]:
    confirm=report.confirmation
    return {
        'hypotheses':list(report.hypotheses),
        'selected_pair':[a.value for a in report.selected_pair],
        'pilot_ids':list(report.pilot_evidence_ids),
        'confirm_ids':list(confirm.evidence_ids),
        'p_value':confirm.p_value,
        'positive_effect':confirm.positive_effect,
        'supports_order_effect':confirm.supports_order_effect,
        'independent_pilot_confirmation':not bool(
            set(report.pilot_evidence_ids)&set(confirm.evidence_ids)),
    }


def _monitor_for(
    port, calibration, main_stream, null_stream, *,
    horizon,window,bootstrap,alpha,seed,
) -> dict[str,object]:
    calibration_errors=port.errors(calibration)
    train_actions=main_stream.actions[:horizon]
    monitor=CalibratedShiftMonitor(
        calibration_errors,calibration.actions,train_actions,
        horizon=horizon,window=window,bootstrap=bootstrap,
        alpha=alpha,seed=seed,
    )
    monitored=segment(main_stream,0,horizon)
    placebo=segment(null_stream,0,horizon)
    change=monitor.detect(port.errors(monitored),monitored.actions)
    unchanged=monitor.detect(port.errors(placebo),placebo.actions)
    return {
        'alarm_index':change.alarm_index,
        'null_alarm_index':unchanged.alarm_index,
        'false_alarm_stationary':unchanged.alarm_index is not None,
        'threshold':change.threshold,
        'stop_p':change.global_null_p,
        'calibration_count':len(calibration.before),
        'pre_stream_mse':float(port.errors(segment(main_stream,0,horizon//2)).mean()),
        'post_stream_mse':float(port.errors(segment(main_stream,horizon//2,horizon)).mean()),
    }


def _first_return_vote(previous, adapted, returned, *,window:int) -> int | None:
    for stop in range(window,len(returned.before)+1):
        vote=route_by_prediction(previous,adapted,segment(returned,0,stop),window=window)
        if vote.selected_checkpoint=='prechange':
            return stop
    return None


def run_adaptation_arena(
    *, mode: str = 'smoke', seed: int = 0, device: str = 'cpu',
    steps_override: int | None = None, noise_std: float = .16,
    adaptation_steps: int | None = None, acquisition_budget: int | None = None,
) -> dict[str,object]:
    """One unannounced double-switch; adaptation only after confirmed alarm.

    A cloned epoch laboratory is an *evaluator-only* source of post-alarm
    observations. No future regime metadata enters learner/scientist contracts.
    The return phase is never used for weight updates.
    """
    if mode not in ('smoke','full'):
        raise ValueError('mode must be smoke or full')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be nonnegative')
    if device not in ('cpu','cuda'):
        raise ValueError('device must be cpu or cuda')
    if not 0 <= noise_std <= 1 or not np.isfinite(noise_std):
        raise ValueError('noise_std must be finite in [0,1]')
    for name,value in (('steps',steps_override),('adaptation_steps',adaptation_steps),('acquisition_budget',acquisition_budget)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'{name} must be positive')

    steps=steps_override or (80 if mode=='smoke' else 700)
    adapt_steps=adaptation_steps or (64 if mode=='smoke' else 180)
    budget=acquisition_budget or (24 if mode=='smoke' else 64)
    train_count,cal_count,eval_count=(256,192,96) if mode=='smoke' else (2048,1024,256)
    phase=48 if mode=='smoke' else 80
    total=3*phase
    window=12 if mode=='smoke' else 16
    bootstrap=256 if mode=='smoke' else 1024
    pilot_trials=12 if mode=='smoke' else 16
    confirm_trials=32 if mode=='smoke' else 64
    permutations=1023 if mode=='smoke' else 4095
    alpha=.01
    cfg=JepaConfig(steps=steps,seed=seed)

    # Sealed, independently seeded pretraining, calibration, validation and
    # final heldout populations. These are evaluator datasets, not law labels.
    train=collect_transitions(mechanism=Mechanism.ADDITIVE,
                              count=train_count,seed=200+seed,noise_std=noise_std)
    calibration=collect_transitions(mechanism=Mechanism.ADDITIVE,
                                    count=cal_count,seed=2000+seed,noise_std=noise_std)
    additive_test=collect_transitions(mechanism=Mechanism.ADDITIVE,
                                      count=eval_count,seed=20_000+seed,noise_std=noise_std)
    gated_test=collect_transitions(mechanism=Mechanism.GATED,
                                   count=eval_count,seed=30_000+seed,noise_std=noise_std)
    pretrained,train_info=train_jepa(train,cfg,device=device)
    shuffled,_=train_jepa(train,cfg,device=device,shuffle_actions=True)
    pre_eval=_evaluate(pretrained,additive_test,gated_test)

    # Identical sensor map, initial states, noise seeds and action schedule.
    observed=observe_recurrence_stream(
        RecurrenceWorld(switch_at=(phase,2*phase),seed=5000+seed,noise_std=noise_std),
        count=total,seed=40_000+seed)
    # A stationary control with both switches beyond the observed horizon.
    stationary=observe_recurrence_stream(
        RecurrenceWorld(switch_at=(total+1,total+2),seed=5000+seed,noise_std=noise_std),
        count=total,seed=40_000+seed)
    if (not np.array_equal(observed.actions,stationary.actions)
        or not np.array_equal(observed.before,stationary.before)
        or not np.array_equal(observed.after[:phase],stationary.after[:phase])):
        raise AssertionError('paired worlds must use identical observation schedules')

    monitor_args=dict(horizon=2*phase,window=window,bootstrap=bootstrap,alpha=alpha)
    jepa_detector=_monitor_for(FrozenJEPAResiduals(pretrained),calibration,observed,stationary,
                                **monitor_args,seed=seed)
    raw_detector=_monitor_for(RidgeDynamics.fit(train,RawObservationEncoder()),
                              calibration,observed,stationary,**monitor_args,seed=seed+113)
    shuffled_detector=_monitor_for(FrozenJEPAResiduals(shuffled),calibration,observed,stationary,
                                   **monitor_args,seed=seed+226)
    alarm=jepa_detector['alarm_index']
    detection={
        **jepa_detector,
        'alarm_after_first_change':alarm is not None and phase<=alarm<2*phase,
        'early_alarm':alarm is not None and alarm<phase,
        'change_confirmed':False,
        'pilot_hypotheses':[],
        'causal_probe':None,
    }
    adaptation={
        'triggered':False,'new_transition_count':0,'replay_memory_examples':0,
        'prechange_training_count':train_count,'intervention_steps':0,
        'variants':{},'checkpoint_router':None,'return_evidence':None,
    }
    if alarm is not None:
        # Alert-epoch snapshot; early alarm cannot access gated physics.
        epoch_lab=HiddenShiftWorld(switch_after=0 if alarm>=phase else None,
                                   seed=65_000+seed,noise_std=noise_std)
        discovery=discover_order_hypothesis(
            pretrained,epoch_lab,pilot_trials=pilot_trials,
            confirmation_trials=confirm_trials,permutations=permutations,
            alpha=alpha,seed=19_000+seed,
        )
        evidence=_extract_discovery(discovery)
        detection['pilot_hypotheses']=evidence['hypotheses']
        detection['causal_probe']=evidence
        adaptation['intervention_steps']=6*(3*pilot_trials+confirm_trials)
        confirmed=bool(alarm>=phase and evidence['supports_order_effect'])
        detection['change_confirmed']=confirmed

        if confirmed:
            # Fresh independent anchor for the SAME action pair that the
            # pilot/confirmation selected at the first confirmed change.
            # Keep it in evidence memory; do not retrain on these trials.
            selected_pair=discovery.selected_pair
            # Separate physical/sensor RNG from the data-acquisition lab:
            # adding this audit cohort cannot perturb adaptation training.
            anchor_lab=HiddenShiftWorld(
                switch_after=0,seed=75_000+seed,noise_std=noise_std,
            )
            anchor_batch=collect_commutator_observations(
                anchor_lab,*selected_pair,trials=confirm_trials,
                seed=95_000+seed,
            )
            anchor_score=score_order_effect(
                pretrained,anchor_batch,selected_pair,
                permutations=permutations,alpha=alpha,seed=96_000+seed,
            )
            anchor_summary={
                'selected_pair':[action.value for action in selected_pair],
                'positive_effect':anchor_score.positive_effect,
                'supports_order_effect':anchor_score.supports_order_effect,
                'fresh_cohort':True,
                'evidence_ids':list(anchor_score.evidence_ids),
            }
            if set(anchor_score.evidence_ids)&set(evidence['confirm_ids']):
                raise AssertionError('fresh anchor may not reuse initial confirmation')
            # Acquire new transitions *after* confirmed intervention at alarm
            # time. Original JS mechanism label never travels with data.
            new=observe_stream(epoch_lab,count=budget,seed=70_000+seed)
            # This independent cohort comes from the *public* changed-epoch
            # laboratory after confirmation. It is NOT an oracle-selected
            # GATED dataset, and never enters gradient updates.
            gated_validation=observe_stream(
                epoch_lab,count=eval_count,seed=90_000+seed,
            )
            adapt_cfg=AdaptationConfig(
                steps=adapt_steps,max_new_examples=budget,
                max_replay_examples=min(train_count,64 if mode=='smoke' else 128),
                seed=seed+10_000,
            )
            naive,trace_naive=adapt_predictor(
                pretrained,new,config=adapt_cfg,validation=gated_validation)
            replay,trace_replay=adapt_predictor(
                pretrained,new,config=adapt_cfg,old_replay=train,
                validation=gated_validation,require_replay=True)
            fresh_cfg=JepaConfig(steps=adapt_steps,seed=seed+20_000)
            scratch,scratch_info=train_jepa(new,fresh_cfg,device=device)
            ridge_new=RidgeDynamics.fit(new,RawObservationEncoder())
            frozen_add=predict_mse(pretrained,additive_test)
            frozen_gate=predict_mse(pretrained,gated_test)

            def variant(model,trace):
                values=_evaluate(model,additive_test,gated_test)
                values.update({
                    'updates':trace.updates,
                    'new_examples':trace.new_examples,
                    'rehearsal_examples':trace.old_examples,
                    'forgetting_delta':values['additive']['prediction_mse']-frozen_add,
                    'gated_gain':frozen_gate-values['gated']['prediction_mse'],
                    'recovery_update_20pct':trace.recovery_update_20pct,
                    'validation_mse_by_update':[list(x) for x in trace.validation_mse_by_update],
                    'training_loss_initial':trace.training_loss_initial,
                    'training_loss_final':trace.training_loss_final,
                    'encoder_frozen':trace.encoder_frozen,
                })
                return values

            sealed_return=segment(observed,2*phase,total)
            gated_route=route_by_prediction(pretrained,naive,gated_test,window=window)
            return_route=route_by_prediction(pretrained,naive,sealed_return,window=window)
            first_return=_first_return_vote(pretrained,naive,sealed_return,window=window)
            replay_gated_route=route_by_prediction(pretrained,replay,gated_test,window=window)
            replay_return_route=route_by_prediction(pretrained,replay,sealed_return,window=window)
            replay_first_return=_first_return_vote(pretrained,replay,sealed_return,window=window)

            # A second detector starts only after the verified changed-epoch
            # experiment. It uses *gated* calibration residuals, not a known
            # return time. The first crossing is frozen at its stopping time.
            next_start=alarm+1
            next_trace=segment(observed,next_start,total)
            # Matched gated-world placebo: same public schedule and noise draws,
            # but it NEVER returns to additive. The evaluator alone builds it.
            gated_placebo=observe_stream(
                HiddenShiftWorld(switch_after=0,seed=5000+seed,
                                 noise_std=noise_std),
                count=total,seed=40_000+seed,
            )
            next_placebo=segment(gated_placebo,next_start,total)
            if not np.array_equal(next_trace.actions,next_placebo.actions):
                raise AssertionError('return and gated placebo must match action schedule')
            if not np.array_equal(observed.after[phase:2*phase],gated_placebo.after[phase:2*phase]):
                raise AssertionError('before return, real and gated-placebo physics must match')
            adapted_port=FrozenJEPAResiduals(naive)
            new_monitor=CalibratedShiftMonitor(
                adapted_port.errors(gated_validation),gated_validation.actions,
                next_trace.actions,horizon=len(next_trace.before),
                window=window,bootstrap=bootstrap,alpha=alpha,seed=seed+719,
            )
            return_alert=new_monitor.detect(adapted_port.errors(next_trace),next_trace.actions)
            placebo_alert=new_monitor.detect(adapted_port.errors(next_placebo),next_placebo.actions)
            absolute_alarm=(next_start+return_alert.alarm_index
                            if return_alert.alarm_index is not None else None)
            return_evidence=None
            return_intervention_steps=0
            if absolute_alarm is not None:
                # An early alarm investigates the STILL gated world; it cannot
                # borrow evidence from the eventual additive epoch.
                alarm_epoch_lab=HiddenShiftWorld(
                    switch_after=None if absolute_alarm>=2*phase else 0,
                    seed=80_000+seed,noise_std=noise_std,
                )
                # Test exactly the previously *confirmed* action pair,
                # not whichever arbitrary pair has the highest noise on return.
                returned_batch=collect_commutator_observations(
                    alarm_epoch_lab,*selected_pair,trials=confirm_trials,
                    seed=97_000+seed,
                )
                returned_effect=score_order_effect(
                    pretrained,returned_batch,selected_pair,
                    permutations=permutations,alpha=alpha,seed=98_000+seed,
                )
                change_contrast=compare_order_effect(
                    pretrained,anchor_batch,returned_batch,
                    pair=selected_pair,permutations=permutations,
                    alpha=alpha,seed=99_000+seed,
                )
                return_evidence={
                    'selected_pair':[action.value for action in selected_pair],
                    'reference_evidence_ids':list(change_contrast.anchor_evidence_ids),
                    'confirmation_evidence_ids':list(change_contrast.return_evidence_ids),
                    'independent_anchor_and_return':True,
                    'reference_epoch_effect_supported':anchor_score.supports_order_effect,
                    'supports_order_effect':returned_effect.supports_order_effect,
                    'return_effect_p':returned_effect.p_value,
                    'reference_effect':change_contrast.anchor_effect,
                    'return_effect':change_contrast.return_effect,
                    'decline':change_contrast.decline,
                    'decline_p':change_contrast.decline_p,
                    'effect_decrease_confirmed':change_contrast.effect_decrease_confirmed,
                    'return_to_old_mechanism_proven':False,
                    'intervention_at_alarm_epoch':True,
                    'interpretation':('decline_supported' if change_contrast.effect_decrease_confirmed
                                      else 'decline_not_established'),
                }
                return_intervention_steps=6*confirm_trials
                if set(evidence['pilot_ids'])&set(return_evidence['confirmation_evidence_ids']):
                    raise AssertionError('different experimental epochs may not reuse evidence IDs')
            return_recognized=bool(
                absolute_alarm is not None and absolute_alarm>=2*phase
                and return_evidence is not None
                and anchor_score.supports_order_effect
                and return_evidence['effect_decrease_confirmed']
                and not return_evidence['supports_order_effect']
            )
            # Rehearsal is a separate candidate live model. Its own
            # residual return monitor runs on the same public actions but
            # has an independent gated calibration distribution. No new
            # confirmatory interventions are spent unless a separate
            # researcher later uses this monitor as the operational arm.
            replay_port=FrozenJEPAResiduals(replay)
            replay_monitor=CalibratedShiftMonitor(
                replay_port.errors(gated_validation),gated_validation.actions,
                next_trace.actions,horizon=len(next_trace.before),
                window=window,bootstrap=bootstrap,alpha=alpha,seed=seed+931,
            )
            replay_alert=replay_monitor.detect(
                replay_port.errors(next_trace),next_trace.actions)
            replay_placebo=replay_monitor.detect(
                replay_port.errors(next_placebo),next_placebo.actions)
            replay_absolute=(next_start+replay_alert.alarm_index
                             if replay_alert.alarm_index is not None else None)
            replay_return_monitor={
                'alarm_index_global':replay_absolute,
                'early_alarm_before_return':replay_absolute is not None and replay_absolute<2*phase,
                'delay_actions':(replay_absolute-2*phase+1
                                 if replay_absolute is not None and replay_absolute>=2*phase else None),
                'false_alarm_gated_placebo':replay_placebo.alarm_index is not None,
                'calibration_count':len(gated_validation.before),
                'calibration_source':'observed_transitions_at_confirmed_epoch',
                'requires_separate_causal_confirmation':replay_absolute is not None,
                'causal_confirmation_performed':False,
            }
            return_monitor={
                'alarm_index_global':absolute_alarm,
                'early_alarm_before_return':absolute_alarm is not None and absolute_alarm<2*phase,
                'delay_actions':(absolute_alarm-2*phase+1
                                 if absolute_alarm is not None and absolute_alarm>=2*phase else None),
                'stop_p':return_alert.global_null_p,
                'false_alarm_gated_placebo':placebo_alert.alarm_index is not None,
                'monitor_started_after_confirmation':True,
                'calibration_count':len(gated_validation.before),
                'calibration_source':'observed_transitions_at_confirmed_epoch',
                'return_recognized':return_recognized,
                'return_intervention_steps':return_intervention_steps,
            }
            adaptation.update({
                'triggered':True,
                'new_transition_count':len(new.before),
                'replay_memory_examples':trace_replay.old_examples,
                'adaptation_validation_actions':len(gated_validation.before),
                'intervention_steps':adaptation['intervention_steps']+len(new.before)
                    +len(gated_validation.before)+6*confirm_trials+return_intervention_steps,
                'variants':{
                    'frozen':{**pre_eval,'updates':0,'forgetting_delta':0.,'gated_gain':0.},
                    'naive':variant(naive,trace_naive),
                    'replay':variant(replay,trace_replay),
                    'scratch':{
                        **_evaluate(scratch,additive_test,gated_test),
                        'updates':adapt_steps,
                        'cross_model_mse_comparable':False,
                        'remark':'scratch learns a different latent frame; compare action ranking only',
                        'training':asdict(scratch_info),
                    },
                    'raw_ridge':{
                        'additive_observation_mse':float(RidgeDynamics.fit(train,RawObservationEncoder()).errors(additive_test).mean()),
                        'adapted_gated_observation_mse':float(ridge_new.errors(gated_test).mean()),
                        'adapted_return_observation_mse':float(ridge_new.errors(sealed_return).mean()),
                        'latent_mse_comparable':False,
                    },
                },
                'checkpoint_router':{
                    'gated':asdict(gated_route),
                    'return':asdict(return_route),
                    'first_prechange_vote_after_return_actions':first_return,
                    'replay_gated':asdict(replay_gated_route),
                    'replay_return':asdict(replay_return_route),
                    'replay_first_prechange_vote_after_return_actions':replay_first_return,
                    'two_checkpoint_memory':True,
                    'retrospective_phase_scoring':True,
                    'router_receives_mechanism_labels':False,
                },
                'reference_order_effect':anchor_summary,
                'return_monitor':return_monitor,
                'replay_return_monitor':replay_return_monitor,
                'return_evidence':return_evidence,
            })

    return {
        'protocol':'action_jepa_recurrent_adaptation_v1',
        'mode':mode,'seed':seed,'device':device,'noise_std':noise_std,
        'config':asdict(cfg),
        'phases':['additive','gated','additive'],  # evaluator only
        'phase_lengths':[phase,phase,phase],
        'boundaries_evaluator_only':[phase,2*phase],
        'training':asdict(train_info),
        'measurement':{'pretrain':pre_eval},
        'detection':detection,
        'detection_controls':{
            'raw_ridge':raw_detector,
            'shuffled_actions':shuffled_detector,
        },
        'adaptation':adaptation,
        'limitations':[
            'Predictor-only repair: encoder and target coordinates remain frozen; joint JEPA representation plasticity is not tested.',
            'Synthetic resettable world: cloning at an alarm and at return is not available for arbitrary real environments.',
            'The adaptation trigger requires a statistically confirmed alert-time order effect; no oracle-triggered repair.',
            'The second return monitor is rearmed after confirmation, calibrated on separate changed-epoch data and evaluated against a gated no-return placebo.',
            'Naive and rehearsal predictors have separately calibrated return monitors. Only the naive return alarm triggers a new causal intervention in this v1 arena; replay alarms are observational diagnostics.',
            'Single adapted checkpoint forgetting is measured separately from two-checkpoint memory routing.',
            'The checkpoint router compares heldout losses; source of each checkpoint is known but simulator mechanism labels are not inputs.',
            'Small independent seed cohorts and action-dependent residual bootstrap are not distribution-free causal proofs.',
            'A return claim requires significant decline on the SAME previously confirmed action pair, using disjoint anchor and return cohorts.',
            'Incomplete evidence against order dependence on return does not prove independence or exact recovery of the old law.',
            'Scratch model has its own latent coordinates; raw Ridge scores have different units. Do not compare their MSE with JEPA.',
            'Validation and return-detector calibration are collected as separate observed post-confirmation transitions; their acquisition cost is counted.',
            'Evaluation, adaptation validation, old rehearsal memory and intervention confirmations are distinct cohorts.',
            'Prediction MSE is relative to the current learned embedding scale; fixed encoder makes naive/replay/frozen comparable.',
            'Adaptation runs after an alarm with an epoch-cloned lab; historical gated-window evaluation is retrospective.',
        ],
    }
