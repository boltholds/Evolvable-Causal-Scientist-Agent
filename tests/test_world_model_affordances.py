from ecsa.world_model.affordances import ActiveAffordanceLearner
from ecsa.world_model.contracts import GroundAction, freeze_raw_value


def action(*args: str) -> GroundAction:
    return GroundAction("opaque", tuple(freeze_raw_value(a) for a in args))


def test_argument_position_is_not_conflated():
    learner = ActiveAffordanceLearner()
    learner.observe(action("x", "y"), success=True)
    assert learner.evidence(action("x", "y"), 0).successes == 1
    assert learner.evidence(action("y", "x"), 0).successes == 0
    assert learner.evidence(action("y", "x"), 1).successes == 0


def test_single_change_same_state_produces_contrast():
    learner = ActiveAffordanceLearner()
    learner.observe(action("x", "good"), success=True, state_id="s0")
    learner.observe(action("x", "bad"), success=False, state_id="s0")
    assert len(learner.contrasts()) == 1
    contrast = learner.contrasts()[0]
    assert contrast.position == 1
    assert contrast.successful.thaw() == "good"
    assert contrast.unsuccessful.thaw() == "bad"


def test_different_states_do_not_prove_argument_contrast():
    learner = ActiveAffordanceLearner()
    learner.observe(action("x"), success=True, state_id="before")
    learner.observe(action("y"), success=False, state_id="after")
    assert learner.contrasts() == ()


def test_two_changed_arguments_are_not_a_controlled_contrast():
    learner = ActiveAffordanceLearner()
    learner.observe(action("a", "b"), success=True, state_id="same")
    learner.observe(action("c", "d"), success=False, state_id="same")
    assert learner.contrasts() == ()


def test_untried_combination_has_higher_exploration_score():
    learner = ActiveAffordanceLearner()
    tried = action("x", "a")
    untried = action("x", "b")
    learner.observe(tried, success=False)
    assert learner.score(untried)[0] > learner.score(tried)[0]


def test_outcome_validation():
    learner = ActiveAffordanceLearner()
    try:
        learner.observe(action("x"), success=1)
    except TypeError:
        pass
    else:
        raise AssertionError("boolean outcome must be enforced")
