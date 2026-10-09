from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ecsa.benchmarks.discoveryworld.jepa_replay import (
    ReplayConfig, load_episode, load_splits, encode_public_json,
)


def _episode(root: Path, seed: int, *, arm: str = 'cold', n: int = 42, label: str = 'USE') -> Path:
    directory = root / f'seed-{seed}' / arm
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'run.json').write_text(json.dumps({
        'scenario': 'Reactor Lab', 'difficulty': 'Normal', 'seed': seed,
    }))
    with (directory / 'observations.jsonl').open('w') as obs, \
         (directory / 'actions.jsonl').open('w') as acts:
        for step in range(n):
            action = label if step % 2 == 0 else 'WAIT'
            pre = {'ui': {'accessibleEnvironmentObjects': [
                {'name': 'crystal', 'description': f'crystal {step % 4}', 'uuid': step+100},
            ], 'taskProgress': [{'description': 'measure crystal'}]},
                'agent': {'x': float(step % 7)}}
            post = {'ui': {'accessibleEnvironmentObjects': [
                {'name': 'crystal', 'description': f'crystal {(step+int(action=="USE")) % 4}', 'uuid': step+100},
            ], 'taskProgress': [{'description': 'measure crystal'}]},
                'agent': {'x': float((step+int(action=="USE")) % 7)}}
            obs.write(json.dumps({'step': step, 'phase': 'pre', 'observation': pre})+'\n')
            acts.write(json.dumps({'step': step, 'action': {'action': action, 'arg1': step % 5}})+'\n')
            obs.write(json.dumps({'step': step + 1, 'phase': 'post', 'observation': post})+'\n')
    return directory


def test_hashed_public_features_are_stable_and_dict_order_invariant() -> None:
    config = ReplayConfig(observation_dim=96, action_dim=32)
    a = {'ui': {'inventoryObjects': [{'name': 'sample', 'temperature': 12.3}],
                'lastActionMessage': 'sample measured'}}
    b = {'ui': {'lastActionMessage': 'sample measured',
                'inventoryObjects': [{'temperature': 12.3, 'name': 'sample'}]}}
    x = encode_public_json(a, width=config.observation_dim, kind='observation')
    y = encode_public_json(b, width=config.observation_dim, kind='observation')
    assert x.shape == (96,)
    assert np.array_equal(x, y)
    assert np.isfinite(x).all()
    assert np.linalg.norm(x) > 0


def test_public_feature_encoder_rejects_evaluator_oracle_keys() -> None:
    with pytest.raises(ValueError, match='oracle'):
        encode_public_json({'ui': {'scoringInfo': {'resonanceFreq': 12}}}, width=64, kind='observation')
    with pytest.raises(ValueError, match='oracle'):
        encode_public_json({'criticalHypotheses': []}, width=64, kind='observation')


def test_log_reader_aligns_pre_action_and_post_and_does_not_read_oracles(tmp_path: Path) -> None:
    folder = _episode(tmp_path, 0, n=8)
    (folder / 'final_scorecard.json').write_text('NOT VALID JSON OR SECRET')
    episode = load_episode(folder, ReplayConfig(observation_dim=96, action_dim=32))
    assert episode.before.shape == (8, 96)
    assert episode.after.shape == (8, 96)
    assert episode.actions.shape == (8, 32)
    assert episode.action_names == ('USE', 'WAIT') * 4
    assert episode.steps == tuple(range(8))
    assert not np.array_equal(episode.before, episode.after)


def test_bad_log_step_alignment_is_rejected(tmp_path: Path) -> None:
    folder = _episode(tmp_path, 0, n=6)
    path = folder / 'observations.jsonl'
    lines = path.read_text().splitlines()
    item = json.loads(lines[1]); item['step'] = 456
    lines[1] = json.dumps(item)
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='alignment'):
        load_episode(folder, ReplayConfig())


def test_study_holds_out_complete_seed_not_cold_vs_reuse(tmp_path: Path) -> None:
    for seed in range(5):
        _episode(tmp_path, seed, n=12)
        _episode(tmp_path, seed, arm='reuse', n=12)
    split = load_splits(tmp_path, ReplayConfig(observation_dim=64, action_dim=32), mode='study', arm='cold')
    assert split.train_seeds == (0,1,2)
    assert split.validation_seeds == (3,)
    assert split.test_seeds == (4,)
    assert split.train.before.shape[0] == 36
    assert split.validation.before.shape[0] == 12
    assert split.test.before.shape[0] == 12
    assert split.split_kind == 'independent_seed_holdout'


def test_study_refuses_single_seed_to_avoid_pseudoreplication(tmp_path: Path) -> None:
    _episode(tmp_path, 0, n=30)
    with pytest.raises(ValueError, match='three distinct seeds'):
        load_splits(tmp_path, ReplayConfig(), mode='study', arm='cold')
    split = load_splits(tmp_path, ReplayConfig(), mode='smoke', arm='cold')
    assert split.split_kind == 'single_episode_temporal_probe_only'
    assert len(split.train.before) > 0
    assert len(split.test.before) > 0


def test_dynamic_action_width_preserves_synthetic_default() -> None:
    torch = pytest.importorskip('torch')
    from ecsa.experimental.action_jepa import ActionJEPA, JepaConfig
    cfg=JepaConfig(steps=1,seed=0)
    original=ActionJEPA(12,cfg)
    assert original.action_dim == 3
    model=ActionJEPA(12,cfg,action_dim=32)
    predictions=model.predict_tensor(torch.zeros(2,8),torch.zeros(2,32))
    assert predictions.shape == (2,8)
    with pytest.raises(ValueError,match='actions'):
        model.predict_tensor(torch.zeros(2,8),torch.zeros(2,3))


def test_replay_training_and_controls_produce_finite_diagnostic_metrics(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    from ecsa.benchmarks.discoveryworld.jepa_probe import ProbeConfig, run_probe
    for seed in range(3): _episode(tmp_path,seed,n=38)
    splits=load_splits(tmp_path,ReplayConfig(observation_dim=48,action_dim=24),mode='study',arm='cold')
    output=run_probe(splits,ProbeConfig(steps=12,batch_size=16,latent_dim=6,hidden_dim=24,seed=4),device='cpu')
    assert output['split_kind']=='independent_seed_holdout'
    assert output['train_count']==38
    assert output['heldout_count']==38
    assert {'jepa','action_shuffled','no_action','raw_ridge'} <= set(output['arms'])
    for name,metrics in output['arms'].items():
        assert np.isfinite(metrics['heldout_mse'])
        assert np.isfinite(metrics['persistence_mse'])
        assert metrics['action_choice_pairs'] > 0
        assert 0.0 <= metrics['action_choice_accuracy'] <= 1.0
    assert output['arms']['no_action']['action_choice_accuracy']==.5


def test_smoke_cli_from_existing_discoveryworld_logs(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    from ecsa.benchmarks.discoveryworld.jepa_probe import run_probe_from_logs
    _episode(tmp_path,0,n=48)
    out=tmp_path/'result.json'
    result=run_probe_from_logs(tmp_path,output=out,mode='smoke',arm='cold',device='cpu',
        observation_dim=48,action_dim=24,steps=6,batch_size=12,latent_dim=6,hidden_dim=20)
    assert out.exists()
    assert json.loads(out.read_text())['protocol']=='discoveryworld_action_jepa_replay_v1'
    assert result['split_kind']=='single_episode_temporal_probe_only'