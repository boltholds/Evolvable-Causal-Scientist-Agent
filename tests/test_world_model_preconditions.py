from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.perception.base import ObservedFeature, PerceptualObservation
from ecsa.world_model.preconditions import (
    ActivePreconditionLearner, PreconditionStatus,
)


def observation(value: int) -> PerceptualObservation:
    return PerceptualObservation(
        observation_id=str(value), entities=(),
        global_features=(
            ObservedFeature("flag", freeze_raw_value(value), "public"),
        ),
    )


def test_success_generates_candidate_and_new_success_falsifies_it():
    learner = ActivePreconditionLearner()
    action = GroundAction("opaque", ())
    learner.observe(action, success=True, before=observation(1))
    candidates = learner.hypotheses(action)
    assert len(candidates) == 1
    assert candidates[0].status is PreconditionStatus.CANDIDATE
    learner.observe(action, success=True, before=observation(0))
    updated = learner.hypotheses(action)
    assert len(updated) == 2
    assert all(item.status is PreconditionStatus.CONTRADICTED for item in updated)


def test_failure_does_not_create_required_precondition():
    learner = ActivePreconditionLearner()
    action = GroundAction("opaque", ())
    learner.observe(action, success=False, before=observation(1))
    assert learner.hypotheses(action) == ()


def test_failed_matching_state_does_not_prove_condition():
    learner = ActivePreconditionLearner()
    action = GroundAction("opaque", ())
    learner.observe(action, success=True, before=observation(1))
    learner.observe(action, success=False, before=observation(1))
    [hypothesis] = learner.hypotheses(action)
    assert hypothesis.failing_matches == 1
    assert learner.state_score(action, observation(0)) == 0.0
    assert learner.state_score(action, observation(1)) == 1.0


def test_distinct_ground_actions_have_separate_hypotheses():
    learner = ActivePreconditionLearner()
    first = GroundAction("opaque", (freeze_raw_value("x"),))
    second = GroundAction("opaque", (freeze_raw_value("y"),))
    learner.observe(first, success=True, before=observation(1))
    assert learner.hypotheses(second) == ()
