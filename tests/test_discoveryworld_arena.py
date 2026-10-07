import json

import pytest
from pathlib import Path

from ecsa.adapters.mlmd import MLMDMechanismRepository
from ecsa.benchmarks.discoveryworld.contracts import (
    DiscoveryWorldActionResult,
    DiscoveryWorldEpisodeConfig,
    DiscoveryWorldEvaluation,
    PolicyDecision,
)
from ecsa.benchmarks.discoveryworld.reactor_lab import (
    MeasurementKind,
    ReactorFrequencyPrediction,
    ReactorMechanismHypothesis,
    reactor_context,
)
from ecsa.benchmarks.discoveryworld import arena as arena_module
from ecsa.benchmarks.discoveryworld.arena import (
    load_policy_factory,
    policy_config_hash,
    run_episode,
    run_progressive_transfer,
)
from ecsa.benchmarks.discoveryworld.metrics import ReactorRunMetricsAccumulator, compute_transfer_gain, summarize_transfer
from ecsa.mechanisms import (
    EpistemicStatus,
    MechanismKind,
    MechanismRecord,
    MechanismScope,
    TransferStatus,
)


def observation(
    reactor_name: str = "crystal reactor (uncalibrated)",
) -> dict:
    dialog_box = {}
    if "(activated)" in reactor_name:
        dialog_box = {
            "dialogIn": (
                "Hello, I am Crystal Reactor #3.\n"
                "The current resonance frequence is: 1324.0 Hertz.\n"
                "The allowable range is 0 to 10,000 Hz."
            ),
            "dialogOptions": {},
        }
    return {
        "ui": {
            "taskProgress": [
                {
                    "description": "Tune the reactors",
                    "completed": False,
                }
            ],
            "lastActionMessage": "",
            "dialog_box": dialog_box,
            "inventoryObjects": [],
            "accessibleEnvironmentObjects": [
                {
                    "uuid": 404,
                    "name": reactor_name,
                    "description": reactor_name,
                }
            ],
            "nearbyObjects": {"objects": {}},
        }
    }


class FakeEnvironment:
    def __init__(
        self,
        events: list[str],
        *,
        activate_after_action: bool = False,
        finish_after_action: bool = True,
    ) -> None:
        self.events = events
        self._steps = 0
        self.activated = False
        self.activate_after_action = activate_after_action
        self.finish_after_action = finish_after_action

    def observe(self) -> dict:
        self.events.append("observe")
        return observation(
            "crystal reactor (activated)"
            if self.activated
            else "crystal reactor (uncalibrated)"
        )

    def available_actions(self) -> dict:
        self.events.append("available_actions")
        return {"PICKUP": {"args": ["arg1"]}}

    def teleport_locations(self) -> dict:
        self.events.append("teleport_locations")
        return {"lab": [1, 1]}

    def act(self, action: dict) -> DiscoveryWorldActionResult:
        self.events.append("act")
        self._steps += 1
        if self.activate_after_action:
            self.activated = True
        return DiscoveryWorldActionResult(True, ())

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def done(self) -> bool:
        return self.finish_after_action and self._steps >= 1

    def evaluate_after_run(self) -> DiscoveryWorldEvaluation:
        self.events.append("evaluate")
        return DiscoveryWorldEvaluation(
            completed_successfully=self.done,
            score_normalized=1.0 if self.done else 0.0,
            steps=self._steps,
            scorecard=[
                {
                    "completedSuccessfully": self.done,
                    "scoreNormalized": 1.0 if self.done else 0.0,
                    "criticalQuestions": ["ORACLE"],
                }
            ],
        )


class RecordingPolicy:
    def __init__(
        self,
        events: list[str],
        *,
        hypothesis: ReactorMechanismHypothesis | None = None,
        commit_validation: bool = True,
    ) -> None:
        self.events = events
        self.contexts = []
        self.observations = []
        self.hypothesis = hypothesis
        self.commit_validation = commit_validation

    def decide(
        self,
        observation,
        available_actions,
        teleport_locations,
        scientific_context,
    ) -> PolicyDecision:
        self.events.append("decide")
        self.observations.append(observation)
        self.contexts.append(scientific_context)
        hypotheses = (
            (self.hypothesis,)
            if self.hypothesis is not None
            else ()
        )
        validation_ids = (
            (self.hypothesis.hypothesis_id,)
            if self.hypothesis is not None and self.commit_validation
            else ()
        )
        return PolicyDecision(
            action={"action": "PICKUP", "arg1": 404},
            hypotheses=hypotheses,
            validation_hypothesis_ids=validation_ids,
        )


def patch_environment(monkeypatch, fake: FakeEnvironment) -> None:
    monkeypatch.setattr(
        arena_module.DiscoveryWorldEnvironmentAdapter,
        "reactor_lab_normal",
        classmethod(lambda cls, seed, max_steps=1000, thread_id=0: fake),
    )


def config(seed: int = 0, max_steps: int = 10) -> DiscoveryWorldEpisodeConfig:
    return DiscoveryWorldEpisodeConfig(
        scenario="Reactor Lab",
        difficulty="Normal",
        seed=seed,
        max_steps=max_steps,
    )


def source_mechanism() -> MechanismRecord:
    source = reactor_context(0)
    return MechanismRecord(
        mechanism_id="source-law",
        version=1,
        kind=MechanismKind.SYMBOLIC_RULE,
        epistemic_status=EpistemicStatus.ADMITTED,
        representation_artifact="reactor-rule:sha256:" + "a" * 64,
        scope=MechanismScope(
            context_ids=(source.context_id,),
            regime_ids=(source.regime_id,),
            domain_ids=(source.domain_id,),
            task_ids=(source.task_id,),
            required_assumptions=source.assumptions,
        ),
        transfer_status=TransferStatus.CONTEXT_SPECIALIZED,
    )


def test_episode_loop_keeps_oracle_outside_policy_surface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake = FakeEnvironment(events)
    patch_environment(monkeypatch, fake)
    policy = RecordingPolicy(events)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )

    result = run_episode(
        config=config(),
        policy=policy,
        repository=repository,
        output_dir=tmp_path / "run",
    )

    assert result.evaluation.completed_successfully is True
    assert events == [
        "observe",
        "available_actions",
        "teleport_locations",
        "decide",
        "act",
        "observe",
        "evaluate",
    ]
    assert "criticalQuestions" not in json.dumps(policy.observations)
    assert policy.contexts[0].context_id == (
        "discoveryworld:reactor-lab:normal:seed-0"
    )

    names = {
        "run.json",
        "actions.jsonl",
        "action_outcomes.jsonl",
        "observations.jsonl",
        "scientific_events.jsonl",
        "mechanism_events.jsonl",
        "metrics.json",
        "final_scorecard.json",
    }
    assert names.issubset(
        {path.name for path in (tmp_path / "run").iterdir()}
    )
    scorecard = json.loads(
        (tmp_path / "run" / "final_scorecard.json").read_text()
    )
    assert scorecard[0]["criticalQuestions"] == ["ORACLE"]
    observations_log = (
        tmp_path / "run" / "observations.jsonl"
    ).read_text()
    assert "criticalQuestions" not in observations_log


def test_validated_transfer_is_admitted_as_target_specialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake = FakeEnvironment(events, activate_after_action=True)
    patch_environment(monkeypatch, fake)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )
    source = source_mechanism()
    repository.admit(source)
    hypothesis = ReactorMechanismHypothesis(
        hypothesis_id="reuse-density-law",
        measurement_kind=MeasurementKind.DENSITY,
        slope=100.0,
        offset=90.0,
        source_evidence_ids=("source-evidence",),
        predictions=(
            ReactorFrequencyPrediction(
                target_crystal_uuid=202,
                target_reactor_uuid=404,
                predicted_frequency=1324.0,
                frozen_step=0,
            ),
        ),
        source_mechanism=source.ref,
    )
    policy = RecordingPolicy(events, hypothesis=hypothesis)

    result = run_episode(
        config=config(seed=1),
        policy=policy,
        repository=repository,
        output_dir=tmp_path / "run",
    )

    assert policy.contexts[0].transfer_candidates == (source,)
    assert len(result.admitted_mechanisms) == 1
    [target] = result.admitted_mechanisms
    assert target.scope.context_ids == (
        "discoveryworld:reactor-lab:normal:seed-1",
    )
    assert target.relations[0].target == source.ref
    assert repository.find_applicable(reactor_context(1)) == (target,)




def test_source_hypothesis_without_validation_commitment_is_not_false_transfer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake = FakeEnvironment(events)
    patch_environment(monkeypatch, fake)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )
    source = source_mechanism()
    repository.admit(source)
    h = ReactorMechanismHypothesis(
        hypothesis_id="candidate-only",
        measurement_kind=MeasurementKind.DENSITY,
        slope=100.0,
        offset=90.0,
        source_evidence_ids=("source-evidence",),
        predictions=(
            ReactorFrequencyPrediction(
                target_crystal_uuid=202,
                target_reactor_uuid=404,
                predicted_frequency=1324.0,
                frozen_step=0,
            ),
        ),
        source_mechanism=source.ref,
    )

    result = run_episode(
        config=config(seed=1),
        policy=RecordingPolicy(
            events,
            hypothesis=h,
            commit_validation=False,
        ),
        repository=repository,
        output_dir=tmp_path / "run",
    )

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert result.admitted_mechanisms == ()
    assert metrics["transfer_candidates_tested"] == 0
    assert metrics["false_transfer_events"] == 0


def test_committed_source_hypothesis_failure_counts_false_transfer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake = FakeEnvironment(events)
    patch_environment(monkeypatch, fake)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )
    source = source_mechanism()
    repository.admit(source)
    h = ReactorMechanismHypothesis(
        hypothesis_id="committed-transfer",
        measurement_kind=MeasurementKind.DENSITY,
        slope=100.0,
        offset=90.0,
        source_evidence_ids=("source-evidence",),
        predictions=(
            ReactorFrequencyPrediction(
                target_crystal_uuid=202,
                target_reactor_uuid=404,
                predicted_frequency=1324.0,
                frozen_step=0,
            ),
        ),
        source_mechanism=source.ref,
    )

    result = run_episode(
        config=config(seed=1),
        policy=RecordingPolicy(events, hypothesis=h),
        repository=repository,
        output_dir=tmp_path / "run",
    )

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert result.admitted_mechanisms == ()
    assert metrics["transfer_candidates_tested"] == 1
    assert metrics["transfer_candidates_rejected"] == 1
    assert metrics["false_transfer_events"] == 1

def test_episode_stops_at_max_steps_when_environment_is_not_done(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake = FakeEnvironment(events, finish_after_action=False)
    patch_environment(monkeypatch, fake)
    policy = RecordingPolicy(events)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )

    result = run_episode(
        config=config(max_steps=2),
        policy=policy,
        repository=repository,
        output_dir=tmp_path / "run",
    )

    assert result.evaluation.steps == 2
    assert len(policy.contexts) == 2



def test_progressive_transfer_has_no_future_seed_leakage(
    tmp_path: Path,
) -> None:
    calls = []

    def policy_factory(config_dict):
        return RecordingPolicy([])

    def episode_runner(*, config, policy, repository, output_dir):
        starting = len(repository.find_transfer_candidates(reactor_context(config.seed)))
        calls.append((config.seed, output_dir.name, starting))
        admitted = ()
        if output_dir.name == "reuse":
            mechanism = MechanismRecord(
                mechanism_id=f"learned-{config.seed}",
                version=1,
                kind=MechanismKind.SYMBOLIC_RULE,
                status=EpistemicStatus.ADMITTED,
                representation_artifact_id="reactor-rule:sha256:" + str(config.seed) * 64,
                scope=MechanismScope(
                    context_ids=(reactor_context(config.seed).context_id,),
                    regime_ids=("normal",),
                    domain_ids=("discoveryworld",),
                    task_ids=("reactor-lab",),
                    required_assumptions=("public-observation-only", "linear-family"),
                ),
                transfer=TransferStatus.CONTEXT_SPECIALIZED,
            )
            repository.admit(mechanism)
            admitted = (mechanism,)
        return arena_module.ArenaEpisodeResult(
            config=config,
            evaluation=DiscoveryWorldEvaluation(
                completed_successfully=False,
                score_normalized=0.0,
                steps=1,
                scorecard=[],
            ),
            admitted_mechanisms=admitted,
            measurement_count=0,
        )

    result = run_progressive_transfer(
        seeds=(0, 1, 2, 3, 4),
        policy_factory=policy_factory,
        policy_config={"model": "deterministic"},
        output_dir=tmp_path,
        max_steps=1,
        episode_runner=episode_runner,
    )

    assert [pair.seed for pair in result.pairs] == [0, 1, 2, 3, 4]
    assert result.pairs[0].cold.starting_mechanism_count == 0
    assert result.pairs[0].reuse.starting_mechanism_count == 0
    assert all(pair.cold.starting_mechanism_count == 0 for pair in result.pairs)
    assert [pair.reuse.starting_mechanism_count for pair in result.pairs] == [0, 1, 2, 3, 4]
    assert all(
        pair.cold.policy_config_hash == pair.reuse.policy_config_hash
        for pair in result.pairs
    )



def test_transfer_gain_and_measurement_accounting() -> None:
    assert compute_transfer_gain(cold=10, reuse=4) == pytest.approx(0.6)
    assert compute_transfer_gain(cold=0, reuse=0) is None

    metrics = ReactorRunMetricsAccumulator()
    metrics.record_measurement(instrument_uuid=10, crystal_uuid=20)
    metrics.record_measurement(instrument_uuid=10, crystal_uuid=20)
    metrics.record_measurement(instrument_uuid=11, crystal_uuid=20)

    snapshot = metrics.snapshot()
    assert snapshot.measurement_actions == 3
    assert snapshot.distinct_measurements == 2


def test_transfer_candidate_outcomes_are_separate_counters() -> None:
    metrics = ReactorRunMetricsAccumulator()
    metrics.record_candidate_retrieved("m1@1")
    metrics.record_candidate_retrieved("m2@1")
    metrics.record_candidate_tested("m1@1")
    metrics.record_candidate_accepted("m1@1")
    metrics.record_candidate_tested("m2@1")
    metrics.record_candidate_rejected("m2@1", false_transfer=True)

    snapshot = metrics.snapshot()
    assert snapshot.transfer_candidates_retrieved == 2
    assert snapshot.transfer_candidates_tested == 2
    assert snapshot.transfer_candidates_accepted == 1
    assert snapshot.transfer_candidates_rejected == 1
    assert snapshot.false_transfer_events == 1



def test_policy_factory_loader_and_config_hash() -> None:
    factory = load_policy_factory(
        "tests.support.discoveryworld_policy:create_policy"
    )
    policy = factory({"model": "deterministic-test"})
    assert isinstance(policy, arena_module.DiscoveryWorldActionPolicy)

    left = policy_config_hash({"b": 2, "a": 1})
    right = policy_config_hash({"a": 1, "b": 2})
    assert left == right


@pytest.mark.parametrize(
    "path,exception",
    (
        ("not-a-factory-path", ValueError),
        ("tests.support.discoveryworld_policy:missing", ValueError),
        ("tests.support.discoveryworld_policy:not_callable", TypeError),
    ),
)
def test_policy_factory_loader_rejects_invalid_factories(
    path: str,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        load_policy_factory(path)


def test_policy_factory_must_return_policy() -> None:
    factory = load_policy_factory(
        "tests.support.discoveryworld_policy:create_not_policy"
    )
    with pytest.raises(TypeError):
        arena_module._create_policy(
            factory,
            {"model": "deterministic-test"},
        )



def test_real_progressive_smoke_all_official_seeds(
    tmp_path: Path,
) -> None:
    policy_config = tmp_path / "policy.json"
    policy_config.write_text('{"model":"deterministic-test"}\n')
    output_dir = tmp_path / "real-smoke"

    exit_code = arena_module.main(
        [
            "--scenario",
            "Reactor Lab",
            "--difficulty",
            "Normal",
            "--seeds",
            "0,1,2,3,4",
            "--arms",
            "cold,reuse",
            "--max-steps",
            "1",
            "--output",
            str(output_dir),
            "--policy-factory",
            "tests.support.discoveryworld_policy:create_policy",
            "--policy-config",
            str(policy_config),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text())
    assert [pair["seed"] for pair in summary["pairs"]] == [0, 1, 2, 3, 4]
    assert len(summary["policy_config_hash"]) == 64

    for pair in summary["pairs"]:
        seed = pair["seed"]
        for arm in ("cold", "reuse"):
            assert pair[arm]["steps"] == 1
            run_dir = output_dir / f"seed-{seed}" / arm
            assert (run_dir / "final_scorecard.json").exists()
            assert (run_dir / "metrics.json").exists()

            run_meta = json.loads((run_dir / "run.json").read_text())
            assert run_meta["arm"] == arm
            assert (
                run_meta["policy_config_hash"]
                == summary["policy_config_hash"]
            )
            assert run_meta["discoveryworld_revision"] == (
                "fd591323920be0d3786ef350955de1945aa571e5"
            )
            assert len(run_meta["ecsa_revision"]) == 40
            assert run_meta["timestamp_utc"].endswith("Z")
            assert (
                run_meta["mechanism_repository"]
                == "MLMDMechanismRepository"
            )


def test_transfer_summary_reports_per_seed_and_seed1_to4_aggregate(
    tmp_path: Path,
) -> None:
    def policy_factory(config_dict):
        return RecordingPolicy([])

    def episode_runner(*, config, policy, repository, output_dir):
        is_reuse = output_dir.name == "reuse"
        measurements = (
            0 if config.seed == 0
            else (4 if is_reuse else 10)
        )
        steps = 1 if config.seed == 0 else (5 if is_reuse else 10)
        return arena_module.ArenaEpisodeResult(
            config=config,
            evaluation=DiscoveryWorldEvaluation(
                completed_successfully=False,
                score_normalized=0.0,
                steps=steps,
                scorecard=[],
            ),
            admitted_mechanisms=(),
            measurement_count=measurements,
        )

    result = run_progressive_transfer(
        seeds=(0, 1, 2, 3, 4),
        policy_factory=policy_factory,
        policy_config={"model": "deterministic"},
        output_dir=tmp_path,
        max_steps=10,
        episode_runner=episode_runner,
    )
    summary = summarize_transfer(result)

    assert [item.seed for item in summary.per_seed] == [0, 1, 2, 3, 4]
    assert summary.per_seed[0].measurement_transfer_gain is None
    assert summary.per_seed[1].measurement_transfer_gain == pytest.approx(0.6)
    assert summary.per_seed[1].step_transfer_gain == pytest.approx(0.5)
    assert summary.mean_measurement_transfer_gain == pytest.approx(0.6)
    assert summary.mean_step_transfer_gain == pytest.approx(0.5)



def test_progressive_transfer_refuses_existing_repository_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "reuse-mechanisms.sqlite").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="reuse"):
        run_progressive_transfer(
            seeds=(0,),
            policy_factory=lambda config: RecordingPolicy([]),
            policy_config={"model": "deterministic"},
            output_dir=tmp_path,
            max_steps=1,
            episode_runner=lambda **kwargs: None,
        )



def test_episode_rejects_source_mechanism_that_is_not_current_transfer_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake = FakeEnvironment(events, activate_after_action=True)
    patch_environment(monkeypatch, fake)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )
    source = MechanismRecord(
        mechanism_id="unrelated-law",
        version=1,
        kind=MechanismKind.SYMBOLIC_RULE,
        status=EpistemicStatus.ADMITTED,
        representation_artifact_id="reactor-rule:sha256:" + "e" * 64,
        scope=MechanismScope(
            context_ids=(reactor_context(0).context_id,),
            regime_ids=("normal",),
            domain_ids=("discoveryworld",),
            task_ids=("different-task",),
            required_assumptions=("public-observation-only", "linear-family"),
        ),
        transfer=TransferStatus.CONTEXT_SPECIALIZED,
    )
    repository.admit(source)
    hypothesis = ReactorMechanismHypothesis(
        hypothesis_id="bad-lineage",
        measurement_kind=MeasurementKind.DENSITY,
        slope=100.0,
        offset=90.0,
        source_evidence_ids=("source-evidence",),
        predictions=(
            ReactorFrequencyPrediction(
                target_crystal_uuid=202,
                target_reactor_uuid=404,
                predicted_frequency=1324.0,
                frozen_step=0,
            ),
        ),
        source_mechanism=source.ref,
    )

    with pytest.raises(ValueError, match="current transfer candidate"):
        run_episode(
            config=config(seed=1),
            policy=RecordingPolicy(events, hypothesis=hypothesis),
            repository=repository,
            output_dir=tmp_path / "run",
        )


def test_policy_factory_cannot_mutate_shared_nested_config_between_arms() -> None:
    seen: list[tuple[str, ...]] = []
    original = {"nested": {"labels": ["original"]}}

    def factory(config_dict):
        labels = config_dict["nested"]["labels"]
        seen.append(tuple(labels))
        labels.append("mutated")
        return RecordingPolicy([])

    arena_module._create_policy(factory, original)
    arena_module._create_policy(factory, original)

    assert seen == [("original",), ("original",)]
    assert original == {"nested": {"labels": ["original"]}}



def test_action_outcomes_log_records_failed_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake = FakeEnvironment(events)
    patch_environment(monkeypatch, fake)
    policy = RecordingPolicy(events)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )

    run_episode(
        config=config(max_steps=1),
        policy=policy,
        repository=repository,
        output_dir=tmp_path / "run",
    )

    [outcome] = [
        json.loads(line)
        for line in (tmp_path / "run" / "action_outcomes.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert outcome["step"] == 1
    assert isinstance(outcome["success"], bool)
    assert isinstance(outcome["errors"], list)
