# SDD ledger — plan: docs/superpowers/plans/2026-10-07-world-model-acquisition-kernel.md

Execution mode: Native via GitHub branch because local Desktop Commander is offline.
Base: main at branch creation.
Pre-flight: Tasks 1-11 implementation already present on main before this execution pass.
Pre-flight: world-model CI green; autonomous-discovery CI green; full suite red with 7 failures.
Ruling: do not reimplement completed plan tasks; resume at verification/recovery because current main already contains their production deliverables. Cost if wrong: an incomplete task could be missed; mitigated by targeted/full CI and spec boundary checks.
Ruling: use isolated branch feat/world-model-acquisition-verify instead of direct main despite prior main-write permission. Cost if wrong: only extra integration step.

Verification recovery:
- Root cause: legacy DiscoveryWorld tests instantiated new ecsa.autonomy.AutonomousScientist after the migration; production legacy policy already uses LegacyAutonomousScientist.
- Root cause: full suite omits optional world-model-planning extra, but two MAcq integration tests asserted macq was installed.
Ruling: legacy helper tests target LegacyAutonomousScientist explicitly; obsolete real-seed generic-hypothesis test is removed because Arena B replaces that acceptance criterion. Cost if wrong: less coverage of the deprecated heuristic path; retained real public-loop legacy test and Arena B end-to-end tests cover supported behavior.
Ruling: real MAcq integration tests use pytest.importorskip("macq"); the missing-backend test remains mandatory. Cost if wrong: an environment without the optional extra could hide MAcq integration defects; mitigated by a separate with-MAcq CI job.
Fast verification: with macq -> 3 passed; without macq -> 14 passed, 2 skipped.
Static spec gate: no forbidden Arena B/core semantic-role imports; SlotAttentionFrontend is backend-neutral and does not import torch.
