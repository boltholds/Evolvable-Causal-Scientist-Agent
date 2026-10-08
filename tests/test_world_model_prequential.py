from ecsa.world_model.candidate_generation import ExperimentHistory
from ecsa.world_model.contracts import GroundAction, freeze_raw_value


def test_schema_beta_bernoulli_predictor_updates_before_next_trial():
    history = ExperimentHistory()
    action = GroundAction("opaque", (freeze_raw_value("x"),))
    assert history.schema_success_probability("opaque", 1) == 0.5
    history.record(action, success=True)
    assert history.schema_success_probability("opaque", 1) == 2 / 3
    history.record(action, success=False)
    assert history.schema_success_probability("opaque", 1) == 0.5
    assert history.schema_success_probability("unobserved", 1) == 0.5
