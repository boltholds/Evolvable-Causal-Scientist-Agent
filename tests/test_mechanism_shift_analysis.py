import pytest
pytest.importorskip("numpy")
pytest.importorskip("torch")
from ecsa.experimental.analyze_mechanism_shift import analyze, NOVEL


def _report(duplicates=False):
    results=[]
    for law in NOVEL:
        for seed in (0,1):
            for arm in ('retrain','structure'):
                results.append({
                    'mechanism':law,'seed':seed,'arm':arm,
                    'world_interventions':10,
                    'final_nll':2.0 if arm=='retrain' else 1.5,
                    'normalized_nll_auc':2.0,'heldout_brier':0.5,
                    'heldout_top_class_ece':0.1,'observed_80_coverage':0.8,
                    'gradient_steps':30,'parameter_steps':12000,
                    'topology_changes_accepted':3 if arm=='structure' else 0,
                    'basis_changes_accepted':0,'eig_choices':0,
                    'prequential_nll':1.0,'heldout_outside_grid':0,
                })
    if duplicates:
        results.append(results[0].copy())
    return {'results':results}


def test_cluster_bootstrap_and_complete_matched_matrix():
    output=analyze([_report()],draws=2000)
    assert output['novel_runs']==12
    assert output['seed_clusters']==[0,1]
    [comparison]=output['paired_bootstrap']
    assert comparison['mean_delta_nll']==-0.5
    assert comparison['seed_clusters_better']==2
    assert comparison['ci95_percentile']==[-0.5,-0.5]


def test_duplicate_and_unpaired_samples_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        analyze([_report(duplicates=True)])
    raw=_report()
    raw['results'].pop()
    with pytest.raises(ValueError,match='unpaired'):
        analyze([raw])
