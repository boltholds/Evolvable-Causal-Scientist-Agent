from pathlib import Path

import pytest

from ecsa.adapters.mlmd import MLMDMechanismRepository
from ecsa.benchmarks.discoveryworld.reactor_lab import reactor_context
from ecsa.mechanisms import (
    ApplicabilityContext,
    EpistemicStatus,
    MechanismKind,
    MechanismRecord,
    MechanismRelation,
    MechanismRelationKind,
    MechanismScope,
    TransferStatus,
)


def source_mechanism(
    *,
    mechanism_id: str = "source-reactor-law",
) -> MechanismRecord:
    source = reactor_context(0)
    return MechanismRecord(
        mechanism_id=mechanism_id,
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
        supporting_evidence=("validation-e1",),
    )


def test_unseen_seed_is_transfer_candidate_but_not_applicable(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    source = source_mechanism()
    repo.admit(source)
    target = reactor_context(1)

    assert repo.find_applicable(target) == ()
    assert repo.find_transfer_candidates(target) == (source,)




def test_current_context_mechanism_is_applicable_not_transfer_candidate(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    source = source_mechanism()
    repo.admit(source)
    current = reactor_context(0)

    assert repo.find_applicable(current) == (source,)
    assert repo.find_transfer_candidates(current) == ()



def test_split_mechanism_is_not_cross_context_transfer_candidate(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    source = source_mechanism()
    split = MechanismRecord(
        mechanism_id="split-reactor-law",
        version=1,
        kind=source.kind,
        epistemic_status=source.epistemic_status,
        representation_artifact="reactor-rule:sha256:" + "b" * 64,
        scope=source.scope,
        transfer_status=TransferStatus.SPLIT,
        supporting_evidence=source.supporting_evidence,
    )
    repo.admit(split)

    assert repo.find_applicable(reactor_context(0)) == (split,)
    assert repo.find_transfer_candidates(reactor_context(1)) == ()

def test_transfer_candidates_exclude_deprecated_latest_version(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    repo.admit(source_mechanism())
    repo.deprecate(
        "source-reactor-law",
        reason="failed target-context prospective validation",
    )

    assert repo.find_transfer_candidates(reactor_context(1)) == ()


@pytest.mark.parametrize(
    "context",
    (
        ApplicabilityContext(
            context_id="discoveryworld:reactor-lab:normal:seed-1",
            regime_id=None,
            domain_id="discoveryworld",
            task_id="reactor-lab",
            assumptions=("public-observation-only", "linear-family"),
        ),
        ApplicabilityContext(
            context_id="discoveryworld:reactor-lab:normal:seed-1",
            regime_id="normal",
            domain_id=None,
            task_id="reactor-lab",
            assumptions=("public-observation-only", "linear-family"),
        ),
        ApplicabilityContext(
            context_id="discoveryworld:reactor-lab:normal:seed-1",
            regime_id="normal",
            domain_id="discoveryworld",
            task_id=None,
            assumptions=("public-observation-only", "linear-family"),
        ),
        ApplicabilityContext(
            context_id="discoveryworld:reactor-lab:normal:seed-1",
            regime_id="normal",
            domain_id="discoveryworld",
            task_id="reactor-lab",
            assumptions=(),
        ),
    ),
)
def test_transfer_candidate_query_requires_explicit_semantics(
    tmp_path: Path,
    context: ApplicabilityContext,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    repo.admit(source_mechanism())

    with pytest.raises(ValueError, match="transfer"):
        repo.find_transfer_candidates(context)


def test_transfer_candidate_must_match_non_context_scope(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    repo.admit(source_mechanism())

    wrong_domain = ApplicabilityContext(
        context_id="another-context",
        regime_id="normal",
        domain_id="another-domain",
        task_id="reactor-lab",
        assumptions=("public-observation-only", "linear-family"),
    )
    missing_assumption = ApplicabilityContext(
        context_id="another-context",
        regime_id="normal",
        domain_id="discoveryworld",
        task_id="reactor-lab",
        assumptions=("public-observation-only",),
    )

    assert repo.find_transfer_candidates(wrong_domain) == ()
    assert repo.find_transfer_candidates(missing_assumption) == ()



def test_superseded_source_is_not_returned_when_replacement_is_target_applicable(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    source = source_mechanism()
    repo.admit(source)

    target = reactor_context(1)
    replacement = MechanismRecord(
        mechanism_id="target-reactor-law",
        version=1,
        kind=MechanismKind.SYMBOLIC_RULE,
        epistemic_status=EpistemicStatus.ADMITTED,
        representation_artifact="reactor-rule:sha256:" + "c" * 64,
        scope=MechanismScope(
            context_ids=(target.context_id,),
            regime_ids=(target.regime_id,),
            domain_ids=(target.domain_id,),
            task_ids=(target.task_id,),
            required_assumptions=target.assumptions,
        ),
        transfer_status=TransferStatus.CONTEXT_SPECIALIZED,
        relations=(
            MechanismRelation(
                MechanismRelationKind.SUPERSEDES,
                source.ref,
            ),
        ),
    )
    repo.admit(replacement)

    assert repo.find_applicable(target) == (replacement,)
    assert repo.find_transfer_candidates(target) == ()
