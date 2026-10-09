"""Cross-model scoring must freeze both targets and denominators."""
from ecsa.world_model.shared_evaluation import (
    SharedEvaluation, SharedTarget, SharedForecast,
)


FINGERPRINT="shared-public-outcomes:fixed-test"
COORDINATES=("global:temperature","global:changed","action:success")


def truth(v=0.4, observed=(True,True,True), unseen=0):
    return SharedTarget((v,1.,1.),observed,unseen,COORDINATES)


def forecast(v=0.4, mask=(True,True,True), positive=1.):
    return SharedForecast((v,positive,positive),mask,COORDINATES,FINGERPRINT)


def test_identical_rows_and_target_mask_for_all_backends():
    examples=(truth(),truth(.5),truth(.1))
    inputs={"jepa":(forecast(),forecast(.5),forecast(.1)),
            "ridge":(forecast(.2),forecast(.2),forecast(.2))}
    score=SharedEvaluation.score(examples,inputs,actions=("A","B","A"))
    assert score["jepa"].row_count==score["ridge"].row_count==3
    assert score["jepa"].scored_numeric_count==score["ridge"].scored_numeric_count==3
    assert score["jepa"].numeric_normalized_mse<score["ridge"].numeric_normalized_mse


def test_abstention_does_not_change_common_denominator():
    examples=(truth(),truth(.1))
    score=SharedEvaluation.score(
        examples,{"full":(forecast(),forecast(.1)),
                  "partial":(forecast(),forecast(.1,(False,True,True)))},
        actions=("A","B"),
    )
    assert score["partial"].row_count==score["full"].row_count==2
    assert score["partial"].coverage<score["full"].coverage
    assert score["partial"].abstentions>=1
    assert score["partial"].numeric_normalized_mse is None


def test_macro_action_metric_prevents_majority_type_dominance():
    examples=tuple(truth(0.) for _ in range(9))+ (truth(.9),)
    predicted=tuple(forecast(0.) for _ in range(10))
    score=SharedEvaluation.score(examples,{"baseline":predicted},
                                 actions=("frequent",)*9+("rare",))["baseline"]
    assert "rare" in score.per_action
    assert score.per_action["rare"] > score.per_action["frequent"]
    assert score.numeric_normalized_mse is not None


def test_no_action_success_field_means_not_scored():
    examples=(truth(observed=(True,True,False)),)
    score=SharedEvaluation.score(examples,{"model":(forecast(),)},
                                 actions=("A",))["model"]
    assert score.action_success_brier is None
    assert score.scored_binary_count==1


def test_unseen_features_and_zero_variance_are_reported():
    examples=(truth(unseen=2),truth(unseen=3))
    scored=SharedEvaluation.score(
        examples,{"model":(forecast(),forecast())},actions=("A","B"))["model"]
    assert scored.unseen_feature_count==5
    assert scored.row_count==2


def test_interventional_and_observational_evidence_never_conflated():
    from ecsa.world_model.effect_attribution import (
        EffectClaim,EffectExplanation,EffectEvidenceStatus,
    )
    claims=(EffectClaim("f","A",EffectExplanation.ACTION_DEPENDENT,
                        EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED,("s",),(),(),()),)
    score=SharedEvaluation.score((truth(),),{"model":(forecast(),)},
                                 actions=("A",),intervention_records=claims)["model"]
    assert score.interventional_confirmation_count==0


def test_reject_mismatched_forecast_frame():
    import pytest
    with pytest.raises(ValueError,match="frame"):
        SharedEvaluation.score(
            (truth(),),{"a":(forecast(),),
                        "b":(SharedForecast((.1,1.,1.),(True,True,True),
                                          COORDINATES,
                                          "shared-public-outcomes:other"),)},
            actions=("A",),
        )
