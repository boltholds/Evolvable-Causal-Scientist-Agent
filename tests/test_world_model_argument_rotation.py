from ecsa.world_model.candidate_generation import ExperimentHistory, StructuralCandidateGenerator
from ecsa.world_model.contracts import RawActionParameter, RawActionSchema, freeze_raw_value
from ecsa.world_model.perception.base import PerceptualObservation


def test_candidate_generator_rotates_after_outcome():
    history = ExperimentHistory()
    generator = StructuralCandidateGenerator(history=history)
    schema = RawActionSchema(
        schema_id="opaque",
        parameters=(RawActionParameter(
            "arg0",
            tuple(freeze_raw_value(v) for v in ("a", "b", "c")),
        ),),
        public_metadata=freeze_raw_value({}),
    )
    observation = PerceptualObservation("observation", (), ())
    first = generator.propose(action_schemas=(schema,), perception=observation, max_ground_actions=1)[0]
    history.record(first, success=False)
    second = generator.propose(action_schemas=(schema,), perception=observation, max_ground_actions=1)[0]
    history.record(second, success=False)
    third = generator.propose(action_schemas=(schema,), perception=observation, max_ground_actions=1)[0]
    assert tuple(a.arguments[0].thaw() for a in (first, second, third)) == ("a", "b", "c")
