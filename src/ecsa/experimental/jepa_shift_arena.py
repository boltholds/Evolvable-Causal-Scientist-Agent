"""Synthetic blinded mechanism-shift arena for one frozen action JEPA.

Only this evaluator imports simulator Mechanism. Learners and scientists see
observations, actions, residuals, and matched intervention evidence — never
hidden state, mechanism type, change-point or oracle labels.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import combinations

import numpy as np

from .action_jepa import (
    JepaConfig, RandomEncoder, RawObservationEncoder, evaluate_prediction,
    train_jepa,
)
from .jepa_world import Action, Mechanism, collect_transitions
from .jepa_causal_scientist import collect_commutator_observations
from .jepa_shift_world import HiddenShiftWorld, observe_stream
from .jepa_shift_detection import (
    FrozenJEPAResiduals, RidgeDynamics, ObservedActionDeltaTemplate,
    CalibratedShiftMonitor,
)
from .jepa_shift_scientist import discover_from_pilot_batches


def _laboratory_at_alarm(
    *, alarm_index: int, change_after: int, seed: int,
    noise_std: float,
) -> HiddenShiftWorld:
    """Evaluator-only cloned laboratory frozen at the *alert-time* epoch.

    Resettable counterfactual experiments are possible in this simulator.
    The caller passes no epoch, switch metadata or law labels to scientists:
    they receive only the resulting `reset`/`step` interface.
    """
    if type(alarm_index) is not int or alarm_index < 0:
        raise ValueError("a valid alarm index is required")
    if type(change_after) is not int or change_after < 0:
        raise ValueError("a nonnegative change boundary is required")
    return HiddenShiftWorld(
        switch_after=0 if alarm_index >= change_after else None,
        seed=seed,noise_std=noise_std,
    )



def periodic_symbolic_baseline(
    *, seed: int, noise_std: float, horizon: int,change_after: int,
    interval: int = 16, pilot_trials: int = 12,
    confirmation_trials: int = 64, permutations: int = 4095,
    alpha: float = .01,
) -> dict[str,object]:
    """Strong symbolic control: *actively* probe all action-order pairs.

    A resettable laboratory is snapshotted by the evaluator at each scheduled
    observation epoch; the symbolic scientist gets only public actions and
    raw observations. Independent trials at every poll, Bonferroni-corrected
    confirmatory p-values over scheduled polls. Intervention cost is reported.
    """
    if type(horizon) is not int or type(change_after) is not int or not 0 <= change_after < horizon:
        raise ValueError("invalid monitoring horizon/change schedule")
    if type(interval) is not int or not 4 <= interval <= horizon:
        raise ValueError("invalid fixed probe interval")
    polls=list(range(interval-1,horizon,interval))
    if not polls:
        raise ValueError("symbolic probe schedule is empty")
    corrected_alpha=alpha/len(polls)
    pairs=tuple(combinations(tuple(Action),2))
    encoded=RawObservationEncoder()
    cost_per_round=6*(len(pairs)*pilot_trials+confirmation_trials)

    def investigate(*, stationary: bool) -> tuple[int | None,list[dict[str,object]]]:
        ledger=[]
        first_detection=None
        for iteration,epoch in enumerate(polls):
            # Hidden physics is fixed at the start of this lab round. The
            # scientist is not passed `epoch` or any simulator mechanism ID.
            lab=_laboratory_at_alarm(
                alarm_index=0 if stationary else epoch,
                change_after=change_after,
                seed=88_000+seed*11+iteration*1009,
                noise_std=noise_std,
            )
            pilot={pair:collect_commutator_observations(
                lab,*pair,trials=pilot_trials,
                seed=6_000_000+seed*100_000+iteration*1000+i*17,
            ) for i,pair in enumerate(pairs)}
            confirm={pair:collect_commutator_observations(
                lab,*pair,trials=confirmation_trials,
                seed=7_000_000+seed*100_000+iteration*1000+i*17,
            ) for i,pair in enumerate(pairs)}
            discovery=discover_from_pilot_batches(
                encoded,pilot,confirm,permutations=permutations,
                alpha=corrected_alpha,seed=seed+iteration*31,
            )
            result=discovery.confirmation
            ledger.append({
                "poll_index":epoch,
                "selected_pair":[action.value for action in discovery.selected_pair],
                "p_value":result.p_value,
                "supports_order_effect":result.supports_order_effect,
                "pilot_evidence_ids":list(discovery.pilot_evidence_ids),
                "confirmation_evidence_ids":list(result.evidence_ids),
                "positive_effect":result.positive_effect,
                "intervention_steps":cost_per_round,
            })
            if result.supports_order_effect:
                first_detection=epoch
                break
        return first_detection,ledger

    alarm,probe_log=investigate(stationary=False)
    null_alarm,null_log=investigate(stationary=True)
    selected = next((entry for entry in probe_log if entry["supports_order_effect"]),None)
    return {
        "method":"periodic_raw_intervention_permutation",
        "alarm_index":alarm,
        "change_detected_after_switch":alarm is not None and alarm>=change_after,
        "early_false_alarm":alarm is not None and alarm<change_after,
        "delay_actions":None if alarm is None or alarm<change_after else alarm-change_after+1,
        "null_alarm_index":null_alarm,
        "null_false_alarm":null_alarm is not None,
        "null_confirmed_order_effect":null_alarm is not None,
        "confirmed_order_effect":selected is not None,
        "selected_pair":selected["selected_pair"] if selected else None,
        "pilot_evidence_ids":selected["pilot_evidence_ids"] if selected else [],
        "confirmation_evidence_ids":selected["confirmation_evidence_ids"] if selected else [],
        "confirmation_p":selected["p_value"] if selected else None,
        "calibration_count":0,
        "intervention_steps":len(probe_log)*cost_per_round,
        "null_intervention_steps":len(null_log)*cost_per_round,
        "poll_count":len(probe_log),
        "null_poll_count":len(null_log),
        "per_poll_alpha":corrected_alpha,
        "probe_log":probe_log,
        "null_probe_log":null_log,
        "world_snapshot_at_each_poll":True,
    }


def run_shift_arena(
    *, mode: str = "smoke", seed: int = 0, device: str = "cpu",
    steps_override: int | None = None, noise_std: float = .16,
) -> dict[str,object]:
    """Train additive-only; monitor hidden switch, propose and confirm.

    For honest matched controls, the evaluator pre-acquires the same paired
    pilot/confirmation observations for *all* methods after the stream.
    Each method is given only its pilot observations to select a pair; the
    confirmation batch for its selected pair remains independent.
    """
    if mode not in ("smoke","full"):
        raise ValueError("mode must be smoke or full")
    if type(seed) is not int or seed<0:
        raise ValueError("seed must be a nonnegative int")
    steps = steps_override if steps_override is not None else (70 if mode=="smoke" else 700)
    cfg = JepaConfig(steps=steps, seed=seed)
    train_count,cal_count,held_count = (256,192,96) if mode=="smoke" else (2048,1024,256)
    before_shift = 48 if mode=="smoke" else 80
    horizon=before_shift*2
    window = 12 if mode=="smoke" else 16
    bootstrap = 256 if mode=="smoke" else 1024
    pilot_trials = 10 if mode=="smoke" else 16
    confirm_trials = 32 if mode=="smoke" else 64
    permutations = 1023 if mode=="smoke" else 4095
    alpha=.01

    # Only observational train/calibration/heldout splits: no gated examples.
    train=collect_transitions(mechanism=Mechanism.ADDITIVE,count=train_count,
                              seed=200+seed,noise_std=noise_std)
    calibration=collect_transitions(mechanism=Mechanism.ADDITIVE,count=cal_count,
                                    seed=2_000+seed,noise_std=noise_std)
    heldout=collect_transitions(mechanism=Mechanism.ADDITIVE,count=held_count,
                                seed=20_000+seed,noise_std=noise_std)
    learned,training_report = train_jepa(train,cfg,device=device)
    shuffled, shuffled_training_report = train_jepa(
        train,cfg,device=device,shuffle_actions=True)
    learned_eval = evaluate_prediction(learned,heldout)
    shuffled_eval = evaluate_prediction(shuffled,heldout)
    raw_encoder=RawObservationEncoder()
    random_encoder=RandomEncoder(train.before.shape[1],cfg)
    ports={
        "jepa": (FrozenJEPAResiduals(learned),learned),
        "shuffled_actions": (FrozenJEPAResiduals(shuffled),shuffled),
        "random_ridge": (RidgeDynamics.fit(train,random_encoder),random_encoder),
        "raw_ridge": (RidgeDynamics.fit(train,raw_encoder),raw_encoder),
        "delta_template": (ObservedActionDeltaTemplate.fit(train),raw_encoder),
    }

    # Both worlds see the EXACT same public action schedule, initial trial
    # seeds, noise draws, and sensor mixing; the only difference is physics.
    stream_seed=40_000+seed
    sensor_seed=5_000+seed
    switched=HiddenShiftWorld(switch_after=before_shift,seed=sensor_seed,
                              noise_std=noise_std)
    stationary=HiddenShiftWorld(switch_after=None,seed=sensor_seed,
                                noise_std=noise_std)
    observed=observe_stream(switched,count=horizon,seed=stream_seed)
    null=observe_stream(stationary,count=horizon,seed=stream_seed)
    if not np.array_equal(observed.actions,null.actions):
        raise AssertionError("controls must see the same action schedule")
    if not np.array_equal(observed.before[:before_shift],null.before[:before_shift]):
        raise AssertionError("prechange observations should be matched")
    if not np.array_equal(observed.after[:before_shift],null.after[:before_shift]):
        raise AssertionError("prechange transitions should be matched")

    # Matching evidence banks are generated lazily at the first alarm of
    # each *epoch*. Both control models in the same epoch receive identical
    # pilot/confirm samples, while early alarms get the prechange laboratory.
    pairs=tuple(combinations(tuple(Action),2))
    bank_by_epoch={}

    def intervention_bank(epoch: str):
        if epoch not in bank_by_epoch:
            # The evaluator knows the regime boundary. The scientist never
            # receives it or calls this function; it only sees lab outcomes.
            index_for_epoch = 0 if epoch == "prechange" else before_shift
            lab=_laboratory_at_alarm(
                alarm_index=index_for_epoch,change_after=before_shift,
                seed=5_000+seed,noise_std=noise_std,
            )
            pilot={
                pair:collect_commutator_observations(
                    lab,*pair,trials=pilot_trials,
                    seed=1_000_000+seed*1_000+i*37)
                for i,pair in enumerate(pairs)
            }
            confirm={
                pair:collect_commutator_observations(
                    lab,*pair,trials=confirm_trials,
                    seed=2_000_000+seed*1_000+i*37)
                for i,pair in enumerate(pairs)
            }
            ids1=[e for batch in pilot.values() for e in batch.evidence_ids]
            ids2=[e for batch in confirm.values() for e in batch.evidence_ids]
            if (len(ids1)!=len(set(ids1)) or len(ids2)!=len(set(ids2))
                    or set(ids1)&set(ids2)):
                raise AssertionError("independent pilot/confirm evidence IDs required")
            bank_by_epoch[epoch]=(pilot,confirm)
        return bank_by_epoch[epoch]

    controls={}
    for index,(name,(port,encoder)) in enumerate(ports.items()):
        baseline_errors=port.errors(calibration)
        live_errors=port.errors(observed)
        placebo_errors=port.errors(null)
        monitor=CalibratedShiftMonitor(
            baseline_errors,calibration.actions,observed.actions,
            horizon=horizon,window=window,bootstrap=bootstrap,alpha=alpha,
            seed=seed+index*113,
        )
        changed=monitor.detect(live_errors,observed.actions)
        placebo=monitor.detect(placebo_errors,null.actions)
        alarm=changed.alarm_index
        outcome: dict[str,object]={
            "alarm_index":alarm,
            "change_detected_after_switch":alarm is not None and alarm>=before_shift,
            "early_false_alarm":alarm is not None and alarm<before_shift,
            "delay_actions":None if alarm is None or alarm<before_shift else alarm-before_shift+1,
            "null_alarm_index":placebo.alarm_index,
            "null_false_alarm":placebo.alarm_index is not None,
            "null_confirmed_order_effect":None,
            "null_confirmation_p":None,
            "null_intervention_steps":0,
            "bootstrap_max_p":changed.global_null_p,
            "monitor_threshold":changed.threshold,
            "monitor_peak":changed.peak_score,
            "calibration_count":len(calibration.before),
            "mean_prechange_residual":float(live_errors[:before_shift].mean()),
            "mean_postchange_residual":float(live_errors[before_shift:].mean()),
            "mean_null_residual":float(placebo_errors.mean()),
            "hypotheses":[],
            "selected_pair":None,
            "pilot_scores":[],
            "pilot_evidence_ids":[],
            "confirmation_evidence_ids":[],
            "confirmation_p":None,
            "confirmed_order_effect":None,
            "confirmation_positive_effect":None,
            "intervention_steps":0,
            "confirmation_epoch":None,
            "causal_shift_supported":False,
        }
        # Online first-crossing alert uses no future samples; confirmation
        # is performed on a simulator clone frozen at the *alert* epoch.
        if alarm is not None:
            epoch="prechange" if alarm < before_shift else "postchange"
            pilot_batches,confirm_batches=intervention_bank(epoch)
            discovery=discover_from_pilot_batches(
                encoder,pilot_batches,confirm_batches,permutations=permutations,
                alpha=alpha,seed=seed+1_000,
            )
            outcome.update({
                "hypotheses":list(discovery.hypotheses),
                "confirmation_epoch":epoch, # evaluator-only audit
                "causal_shift_supported":(
                    epoch=="postchange" and discovery.confirmation.supports_order_effect
                ),
                "selected_pair":[a.value for a in discovery.selected_pair],
                "pilot_scores":[{"pair":label,"excess":float(signal)}
                                for label,signal in discovery.pilot_scores],
                "pilot_evidence_ids":list(discovery.pilot_evidence_ids),
                "confirmation_evidence_ids":list(discovery.confirmation.evidence_ids),
                "confirmation_p":discovery.confirmation.p_value,
                "confirmed_order_effect":discovery.confirmation.supports_order_effect,
                "confirmation_positive_effect":discovery.confirmation.positive_effect,
                # Each paired trial runs AB, BA and repeat AB: six actions.
                "intervention_steps":6*(len(pairs)*pilot_trials+confirm_trials),
            })
        if placebo.alarm_index is not None:
            # No-switch placebo always remains in the additive epoch; this
            # secondary inquiry measures actual false causal discovery.
            placebo_pilot,placebo_confirm=intervention_bank("prechange")
            placebo_discovery=discover_from_pilot_batches(
                encoder,placebo_pilot,placebo_confirm,
                permutations=permutations,alpha=alpha,seed=seed+1_000,
            )
            outcome["null_confirmed_order_effect"]=(
                placebo_discovery.confirmation.supports_order_effect)
            outcome["null_confirmation_p"]=placebo_discovery.confirmation.p_value
            outcome["null_intervention_steps"]=6*(len(pairs)*pilot_trials+confirm_trials)
        controls[name]=outcome

    controls["symbolic"]=periodic_symbolic_baseline(
        seed=seed,noise_std=noise_std,horizon=horizon,
        change_after=before_shift,
        interval=12 if mode=="smoke" else 16,
        pilot_trials=pilot_trials,confirmation_trials=confirm_trials,
        permutations=permutations,alpha=alpha,
    )

    return {
        "protocol":"action_jepa_hidden_shift_v1",
        "mode":mode,"seed":seed,"device":device,"noise_std":noise_std,
        "config":asdict(cfg),
        "train_count":train_count,"calibration_count":cal_count,
        "heldout_count":held_count,
        "prechange_steps":before_shift,"monitor_count":horizon,
        "change_after":before_shift, # evaluator only, never passed to agents
        "window":window,"alpha":alpha,"bootstrap":bootstrap,
        "null_stream":{"mechanism_changes":0,"length":horizon},
        "jepa_training":asdict(training_report),
        "shuffled_action_training":asdict(shuffled_training_report),
        "jepa_baseline_prediction":asdict(learned_eval),
        "shuffled_baseline_prediction":asdict(shuffled_eval),
        "controls":controls,
        "intervention_protocol":{
            "pilot_trials_per_pair":pilot_trials,
            "confirmation_trials_per_pair":confirm_trials,
            "shared_pilot_across_controls_at_same_epoch":True,
            "shared_confirm_across_controls_at_same_epoch":True,
            "pilot_and_confirm_disjoint":True,
            "snapshot_at_alarm":True,
            "replica_epochs_acquired":list(bank_by_epoch),
        },
        "limitations":[
            "Synthetic additive->gated fixture with known action vocabulary; no general causal discovery.",
            "Only one prechange-trained JEPA, frozen across switch; no cross-world adaptation or online model repair.",
            "Monitor assumes action-conditional iid baseline residuals; bootstrap is a finite-sample null approximation, not an anytime guarantee.",
            "The switch point is reported by the evaluator after scoring; no detector or experiment selector receives it.",
            "Predetermined exploratory all-pairs scan precedes independent confirmation; no oracle action-pair supplied to the scientist.",
            "Reproducible counterfactual laboratory clones freeze the environment at the alert epoch; this ability is not assumed of real systems.",
            "First alert only: false alarms are investigated but the monitor does not yet re-arm after negative confirmation.",
            "Delta-template is a simple observable finite-difference rule; the separate symbolic baseline proactively tests action order using raw observations.",
            "Periodic symbolic testing uses far more simulator actions than passive monitoring, and treats each observation epoch as clonable for counterfactual intervention.",
            "Confirmation tests order dependence of one selected action pair, not an identifiable structural causal model.",
            "Sensor geometry remains fixed; test does not establish domain shift or transfer to real physical systems.",
            "Failure to reject order dependence is not proof of action-order independence.",
            "Action-shuffled training can still produce useful embeddings; report its performance as a real control.",
        ],
    }
