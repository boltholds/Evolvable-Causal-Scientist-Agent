from ecsa.contracts import TheoryPosterior, TheoryRef
from ecsa.science import ScienceKernel
from ecsa.world_model.contracts import (
    RawActionParameter,
    RawActionSchema,
    freeze_raw_value,
)
from ecsa.world_model.experiments import (
    ContractExperimentCoordinator,
    ExperimentBudget,
)
from ecsa.world_model.hypotheses import (
    ActionSchemaHypothesis,
    AffordanceHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    WorldContractHypothesis,
)
from ecsa.world_model.perception.base import (
    EntityObservation,
    PerceptualObservation,
)


def _contract(
    contract_id: str,
    *,
    first_type: str,
    second_type: str,
    success_probability: float,
) -> WorldContractHypothesis:
    x_type = EntityTypeHypothesis(
        hypothesis_id=f"{contract_id}-type-x",
        member_refs=("x",),
        supporting_evidence_ids=(f"{contract_id}-e1",),
        contradicting_evidence_ids=(),
        confidence=0.8,
        status=HypothesisStatus.SUPPORTED,
    )
    y_type = EntityTypeHypothesis(
        hypothesis_id=f"{contract_id}-type-y",
        member_refs=("y",),
        supporting_evidence_ids=(f"{contract_id}-e2",),
        contradicting_evidence_ids=(),
        confidence=0.8,
        status=HypothesisStatus.SUPPORTED,
    )
    type_by_name = {
        "x": x_type.hypothesis_id,
        "y": y_type.hypothesis_id,
    }
    action = ActionSchemaHypothesis(
        hypothesis_id=f"{contract_id}-action",
        source_schema_id="A17",
        parameter_type_ids=(
            type_by_name[first_type],
            type_by_name[second_type],
        ),
        precondition_predicate_ids=(),
        add_effect_predicate_ids=(),
        delete_effect_predicate_ids=(),
        numeric_effect_ids=(),
        symmetric_parameter_groups=(),
        supporting_evidence_ids=(f"{contract_id}-e3",),
        contradicting_evidence_ids=(),
        confidence=0.8,
        status=HypothesisStatus.SUPPORTED,
    )
    affordance = AffordanceHypothesis(
        hypothesis_id=f"{contract_id}-affordance",
        action_schema_id=action.hypothesis_id,
        argument_type_ids=action.parameter_type_ids,
        success_probability=success_probability,
        supporting_evidence_ids=(f"{contract_id}-e4",),
        contradicting_evidence_ids=(),
        confidence=0.7,
        status=HypothesisStatus.SUPPORTED,
    )
    return WorldContractHypothesis(
        contract_id=contract_id,
        entity_types=(x_type, y_type),
        predicates=(),
        numeric_fluents=(),
        argument_roles=(),
        actions=(action,),
        affordances=(affordance,),
        supporting_evidence_ids=(f"{contract_id}-trace",),
        contradicting_evidence_ids=(),
        confidence=0.7,
        status=HypothesisStatus.SUPPORTED,
    )


def _schema() -> RawActionSchema:
    return RawActionSchema(
        schema_id="A17",
        parameters=(
            RawActionParameter(
                name="p",
                public_candidates=(
                    freeze_raw_value("x"),
                    freeze_raw_value("y"),
                ),
            ),
            RawActionParameter(
                name="q",
                public_candidates=(
                    freeze_raw_value("x"),
                    freeze_raw_value("y"),
                ),
            ),
        ),
        public_metadata=freeze_raw_value({}),
    )


def _empty_perception() -> PerceptualObservation:
    return PerceptualObservation(
        observation_id="obs",
        entities=(),
        global_features=(),
    )


def test_contract_experiment_distinguishes_argument_role_hypotheses() -> None:
    first = _contract(
        "h1",
        first_type="x",
        second_type="y",
        success_probability=0.95,
    )
    second = _contract(
        "h2",
        first_type="y",
        second_type="x",
        success_probability=0.95,
    )
    coordinator = ContractExperimentCoordinator()

    experiments = coordinator.propose(
        contracts=(first, second),
        action_schemas=(_schema(),),
        perception=_empty_perception(),
        budget=ExperimentBudget(max_ground_actions=8),
    )

    xy = next(
        experiment
        for experiment in experiments
        if tuple(
            argument.thaw()
            for argument in experiment.action.arguments
        ) == ("x", "y")
    )
    probabilities = {
        prediction.contract_id: prediction.success_probability
        for prediction in xy.predictions
    }
    assert probabilities["h1"] == 0.95
    assert probabilities["h2"] == 0.5


def test_failed_action_outcome_can_discriminate_precondition_hypotheses() -> None:
    permissive = _contract(
        "permissive",
        first_type="x",
        second_type="y",
        success_probability=0.9,
    )
    restrictive = _contract(
        "restrictive",
        first_type="x",
        second_type="y",
        success_probability=0.2,
    )
    coordinator = ContractExperimentCoordinator()
    [experiment] = [
        item
        for item in coordinator.propose(
            contracts=(permissive, restrictive),
            action_schemas=(_schema(),),
            perception=_empty_perception(),
            budget=ExperimentBudget(max_ground_actions=4),
        )
        if tuple(
            argument.thaw()
            for argument in item.action.arguments
        ) == ("x", "y")
    ]

    by_contract = {
        prediction.contract_id: prediction.as_predictive_distribution()
        for prediction in experiment.predictions
    }

    assert by_contract["permissive"].probability(("failure",)) == (
        0.1
    )
    assert by_contract["restrictive"].probability(("failure",)) == (
        0.8
    )


def test_contract_and_scientific_candidates_use_same_information_gain_selector() -> None:
    first = _contract(
        "h1",
        first_type="x",
        second_type="y",
        success_probability=0.95,
    )
    second = _contract(
        "h2",
        first_type="y",
        second_type="x",
        success_probability=0.95,
    )
    science = ScienceKernel()
    coordinator = ContractExperimentCoordinator(science=science)
    experiments = coordinator.propose(
        contracts=(first, second),
        action_schemas=(_schema(),),
        perception=_empty_perception(),
        budget=ExperimentBudget(max_ground_actions=4),
    )
    posterior = TheoryPosterior.uniform(
        (
            TheoryRef("h1", "world-contract:h1"),
            TheoryRef("h2", "world-contract:h2"),
        )
    )

    selected = coordinator.select(
        posterior=posterior,
        experiments=experiments,
    )
    direct = science.select_experiment(
        posterior,
        tuple(
            tuple(
                prediction.as_predictive_distribution()
                for prediction in experiment.predictions
            )
            for experiment in experiments
        ),
    )

    assert selected.experiment_id == direct.experiment_id


def test_budget_limits_ground_action_enumeration_without_semantic_action_roles() -> None:
    coordinator = ContractExperimentCoordinator()
    schema = _schema()

    experiments = coordinator.propose(
        contracts=(),
        action_schemas=(schema,),
        perception=_empty_perception(),
        budget=ExperimentBudget(max_ground_actions=2),
    )

    assert len(experiments) == 2
    assert all(
        experiment.action.schema_id == "A17"
        for experiment in experiments
    )
    assert all(
        len(experiment.action.arguments) == 2
        for experiment in experiments
    )


def test_entity_refs_fill_untyped_action_candidates() -> None:
    perception = PerceptualObservation(
        observation_id="obs",
        entities=(
            EntityObservation(
                local_ref="entity-a",
                source_identity=None,
                features=(),
                provenance_id="obs",
            ),
            EntityObservation(
                local_ref="entity-b",
                source_identity=None,
                features=(),
                provenance_id="obs",
            ),
        ),
        global_features=(),
    )
    schema = RawActionSchema(
        schema_id="opaque",
        parameters=(
            RawActionParameter(name="arg0"),
        ),
        public_metadata=freeze_raw_value({}),
    )

    experiments = ContractExperimentCoordinator().propose(
        contracts=(),
        action_schemas=(schema,),
        perception=perception,
        budget=ExperimentBudget(max_ground_actions=4),
    )

    assert {
        experiment.action.arguments[0].thaw()
        for experiment in experiments
    } == {"entity-a", "entity-b"}



def test_budget_covers_distinct_action_schemas_before_repeating_schema() -> None:
    many = RawActionSchema(
        schema_id="A",
        parameters=(
            RawActionParameter(
                name="arg",
                public_candidates=tuple(
                    freeze_raw_value(value)
                    for value in ("x1", "x2", "x3", "x4")
                ),
            ),
        ),
        public_metadata=freeze_raw_value({}),
    )
    schemas = (
        many,
        RawActionSchema(
            schema_id="B",
            parameters=(),
            public_metadata=freeze_raw_value({}),
        ),
        RawActionSchema(
            schema_id="C",
            parameters=(),
            public_metadata=freeze_raw_value({}),
        ),
    )

    experiments = ContractExperimentCoordinator().propose(
        contracts=(),
        action_schemas=schemas,
        perception=_empty_perception(),
        budget=ExperimentBudget(max_ground_actions=3),
    )

    assert tuple(
        experiment.action.schema_id
        for experiment in experiments
    ) == ("A", "B", "C")
