from __future__ import annotations

import ast
from pathlib import Path


EXPLICIT_AUTHORITY_FILES = (
    "phasebatch/optimizer.py",
    "phasebatch/relation.py",
    "phasebatch/graph.py",
    "phasebatch/batcher.py",
    "phasebatch/batch_correctness.py",
    "phasebatch/batch_explorer.py",
    "phasebatch/batch_objective.py",
    "phasebatch/batch_reporting.py",
    "phasebatch/batch_validation_dag.py",
    "phasebatch/batch_validation_ladder.py",
    "phasebatch/certified_batch_refinement.py",
    "phasebatch/explorer.py",
    "phasebatch/staged_optimizer.py",
    "phasebatch/evidence_pack.py",
    "phasebatch/static_soundness_audit.py",
    "phasebatch/static_source_audit.py",
    "phasebatch/ei_idem_gate.py",
    "phasebatch/ei_idem_screen.py",
    "phasebatch/root_pair_screen.py",
    "phasebatch/pair_rule_mining.py",
    "phasebatch/pair_scheduling.py",
    "phasebatch/pair_tester.py",
    "phasebatch/pipeline_replay.py",
    "phasebatch/artifact_cleanup.py",
    "phasebatch/experiment_manifest.py",
)

AUTHORITY_NAME_MARKERS = (
    "batch",
    "search",
    "certificate",
    "certif",
    "ledger",
    "durability",
    "optimizer",
    "relation",
    "graph",
    "explorer",
)

FORBIDDEN_STUDY_MODULES = (
    "advisor_study",
    "advisor_pair_scale",
    "advisor_pair_matrix",
    "advisor_2n_direct_merge",
    "advisor_pass_universe",
)


def _imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
            modules.update(alias.name for alias in node.names)
    return modules


def _production_authority_files(repo: Path) -> tuple[Path, ...]:
    phasebatch = repo / "phasebatch"
    discovered = {
        path
        for path in phasebatch.rglob("*.py")
        if any(marker in path.stem.lower() for marker in AUTHORITY_NAME_MARKERS)
        and "advisor" not in path.stem.lower()
    }
    discovered.update(repo / relative for relative in EXPLICIT_AUTHORITY_FILES)
    return tuple(sorted(discovered))


def test_advisor_study_modules_are_report_only() -> None:
    repo = Path(__file__).resolve().parents[3]
    protected = _production_authority_files(repo)
    protected_relatives = {
        path.relative_to(repo).as_posix() for path in protected
    }
    assert set(EXPLICIT_AUTHORITY_FILES).issubset(protected_relatives)
    assert "phasebatch/evidence_pack.py" in protected_relatives
    assert any("batch" in path.stem for path in protected)
    for path in protected:
        relative = path.relative_to(repo).as_posix()
        assert path.is_file(), relative
        imported = _imported_modules(path.read_text(encoding="utf-8"))
        assert not any(
            forbidden in module
            for module in imported
            for forbidden in FORBIDDEN_STUDY_MODULES
        ), relative
