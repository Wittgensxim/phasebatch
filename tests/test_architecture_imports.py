from __future__ import annotations

from pathlib import Path


def test_core_modules_do_not_import_analyze_state_from_cli() -> None:
    repo = Path(__file__).resolve().parents[1]
    for relative in [
        "phasebatch/optimizer.py",
        "phasebatch/explorer.py",
        "phasebatch/batch_explorer.py",
    ]:
        source = (repo / relative).read_text(encoding="utf-8")
        assert "from .cli import analyze_state" not in source, relative
        assert "from .state_analysis import analyze_state" in source, relative


def test_batch_construction_core_is_pairwise_only() -> None:
    repo = Path(__file__).resolve().parents[1]
    for relative in [
        "phasebatch/optimizer.py",
        "phasebatch/batch_explorer.py",
        "phasebatch/state_analysis.py",
    ]:
        source = (repo / relative).read_text(encoding="utf-8")
        assert 'batch_construction_mode != "pairwise"' in source, relative
        assert "batch construction only supports pairwise" in source, relative
