from dataclasses import dataclass
from pathlib import Path

from ecsa.adapters.dreamcoder import QualifiedProgramMechanism
from ecsa.adapters.stitch import (
    StitchAbstractionEngine,
    StitchAbstractionProposal,
)
from ecsa.contracts import TheoryRef


@dataclass
class ArtifactProvider:
    programs: dict[str, str]

    def load_artifact(self, artifact_id: str) -> dict:
        return {
            "schema": "ecsa.dreamcoder-program.v1",
            "program": self.programs[artifact_id],
        }


def qualified(
    name: str,
    artifact_id: str,
) -> QualifiedProgramMechanism:
    return QualifiedProgramMechanism(
        theory=TheoryRef(
            theory_id=name,
            artifact_id="program-mechanism:sha256:" + name[-1] * 64,
        ),
        source_program_artifact_id=artifact_id,
        input_variables=("A", "B"),
        output_variable="Y",
        heldout_experiment_ids=(f"heldout-{name}",),
    )


def test_real_stitch_learns_abstraction_only_from_qualified_program_mechanisms(
    tmp_path: Path,
) -> None:
    mechanisms = (
        qualified("mechanism-1", "program-1"),
        qualified("mechanism-2", "program-2"),
        qualified("mechanism-3", "program-3"),
    )
    provider = ArtifactProvider(
        {
            "program-1": "(lambda (lambda (and (not (and $1 $0)) $1)))",
            "program-2": "(lambda (lambda (or (not (and $1 $0)) $0)))",
            "program-3": "(lambda (lambda (or (not (and $1 $0)) $1)))",
        }
    )
    engine = StitchAbstractionEngine(
        mechanisms,
        program_artifacts=provider,
        cache_root=tmp_path,
        max_arity=2,
        iterations=1,
        threads=1,
    )

    run = engine.compress()

    assert len(run.proposals) == 1
    proposal = run.proposals[0]
    assert isinstance(proposal, StitchAbstractionProposal)
    assert proposal.engine_id == "stitch"
    assert proposal.artifact_id.startswith("stitch-abstraction:sha256:")
    assert proposal.source_theory_ids == (
        "mechanism-1",
        "mechanism-2",
        "mechanism-3",
    )
    assert proposal.source_program_artifact_ids == (
        "program-1",
        "program-2",
        "program-3",
    )

    artifact = engine.load_artifact(proposal.artifact_id)
    assert artifact["schema"] == "ecsa.stitch-abstraction.v1"
    assert artifact["source_repository"] == "https://github.com/mlb2251/stitch.git"
    assert artifact["source_revision"] == "350804b7b35807c78bd21c313785ae5152ae2985"
    assert artifact["abstraction"]["body"]
    assert artifact["abstraction"]["utility"] > 0
    assert artifact["abstraction"]["num_uses"] >= 2
    assert artifact["compression_ratio"] > 1.0
    assert len(artifact["rewritten_programs"]) == 3


def test_stitch_does_not_invent_reusable_abstraction_from_one_mechanism(
    tmp_path: Path,
) -> None:
    mechanism = qualified("mechanism-1", "program-1")
    provider = ArtifactProvider(
        {
            "program-1": "(lambda (lambda (and (not (and $1 $0)) $1)))",
        }
    )
    engine = StitchAbstractionEngine(
        (mechanism,),
        program_artifacts=provider,
        cache_root=tmp_path,
    )

    run = engine.compress()

    assert run.proposals == ()
