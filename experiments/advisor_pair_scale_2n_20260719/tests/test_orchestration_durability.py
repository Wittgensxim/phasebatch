"""Task 15 orchestration durability regressions.

Initial RED (2026-07-18): ``7 failed in 0.57s``.  The failures covered
backup-only recovery, active-plus-backup recovery, staging-to-active replace
failure, delete-old-before-replace, live second-writer exclusion, exception
release, and crash/stale-lock recovery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import advisor_study.orchestration as orchestration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def _publish(
    stage: Path,
    isolation_root: Path,
    input_digest: str,
    value: str,
) -> tuple[dict[str, object], bool]:
    return orchestration._run_or_reuse_stage(
        stage,
        lambda _directory: {"value": value},
        expected_input_sha256=input_digest,
        isolation_root=isolation_root,
    )


def _backup(stage: Path) -> Path:
    return stage.parent / f".{stage.name}.publication-backup"


def test_stage_publication_recovers_backup_only_crash_without_recomputing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    stage = root / "raw" / "profiles"
    old_digest = "1" * 64
    _publish(stage, root, old_digest, "old")
    backup = _backup(stage)
    os.replace(stage, backup)
    calls: list[str] = []

    payload, reused = orchestration._run_or_reuse_stage(
        stage,
        lambda _directory: calls.append("unexpected") or {"value": "new"},
        expected_input_sha256=old_digest,
        isolation_root=root,
    )

    assert payload == {"value": "old"}
    assert reused is True
    assert calls == []
    assert stage.is_dir()
    assert not backup.exists()


def test_stage_publication_recovers_active_plus_backup_by_validating_active_first(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    stage = root / "raw" / "pairs"
    digest = "2" * 64
    _publish(stage, root, digest, "active")
    backup = _backup(stage)
    shutil.copytree(stage, backup)
    active_bytes = (stage / "result.json").read_bytes()

    payload, reused = _publish(stage, root, digest, "must-not-run")

    assert payload == {"value": "active"}
    assert reused is True
    assert (stage / "result.json").read_bytes() == active_bytes
    assert not backup.exists()


def test_stage_publication_restores_valid_backup_when_active_hash_drifted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated"
    stage = root / "raw" / "drifted"
    digest = "a" * 64
    _publish(stage, root, digest, "valid-old")
    backup = _backup(stage)
    shutil.copytree(stage, backup)
    (stage / "result.json").write_text(
        '{"value":"corrupt"}\n', encoding="utf-8", newline="\n"
    )
    calls: list[str] = []

    payload, reused = orchestration._run_or_reuse_stage(
        stage,
        lambda _directory: calls.append("unexpected") or {"value": "new"},
        expected_input_sha256=digest,
        isolation_root=root,
    )

    assert payload == {"value": "valid-old"}
    assert reused is True
    assert calls == []
    assert not backup.exists()
    assert list(stage.parent.glob(f".{stage.name}.invalid-active-*"))


def test_stage_publication_replace_failure_restores_old_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated"
    stage = root / "raw" / "two-n"
    old_digest = "3" * 64
    new_digest = "4" * 64
    _publish(stage, root, old_digest, "old")
    old_result = (stage / "result.json").read_bytes()
    real_replace = orchestration.os.replace

    def fail_new_active(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if ".stage-" in source_path.name and destination_path == stage:
            raise OSError("injected staging-to-active failure")
        real_replace(source, destination)

    monkeypatch.setattr(orchestration.os, "replace", fail_new_active)
    with pytest.raises(OSError, match="staging-to-active"):
        _publish(stage, root, new_digest, "new")

    assert stage.is_dir()
    assert (stage / "result.json").read_bytes() == old_result
    assert orchestration._load_complete_stage(stage, old_digest) == {"value": "old"}
    assert not _backup(stage).exists()


def test_stage_publication_never_deletes_active_before_new_stage_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "isolated"
    stage = root / "raw" / "view"
    _publish(stage, root, "5" * 64, "old")
    removed: list[Path] = []
    real_safe_rmtree = orchestration._safe_rmtree

    def observe_delete(path: Path, isolation_root: Path) -> None:
        removed.append(Path(path).resolve(strict=False))
        real_safe_rmtree(path, isolation_root)

    monkeypatch.setattr(orchestration, "_safe_rmtree", observe_delete)
    payload, reused = _publish(stage, root, "6" * 64, "new")

    assert payload == {"value": "new"}
    assert reused is False
    assert stage.resolve(strict=False) not in removed
    assert _backup(stage).resolve(strict=False) in removed


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT), str(REPOSITORY_ROOT))
    )
    return environment


def _run_lock_child(output: Path, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-c", script, str(output)),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
        env=_child_environment(),
    )


def test_orchestration_lock_rejects_second_process_without_touching_stage(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output" / "smoke"
    output.mkdir(parents=True)
    sentinel = output / "raw" / "sentinel.json"
    sentinel.parent.mkdir()
    sentinel.write_text('{"stable":true}\n', encoding="utf-8", newline="\n")
    before = sentinel.read_bytes()
    script = """
import sys
from pathlib import Path
from advisor_study.orchestration import run_study_orchestration
try:
    output = Path(sys.argv[1])
    run_study_orchestration(
        out_dir=output,
        isolation_root=output.parents[1],
        study_manifest_id="manifest",
        programs={},
        groups={},
        dependencies=None,
    )
except RuntimeError:
    raise SystemExit(42)
except Exception:
    raise SystemExit(43)
raise SystemExit(0)
"""

    with orchestration._exclusive_output_lock(output):
        child = _run_lock_child(output, script)

    assert child.returncode == 42, child.stderr
    assert sentinel.read_bytes() == before


def test_orchestration_lock_is_released_when_owner_raises(tmp_path: Path) -> None:
    output = tmp_path / "output" / "smoke"
    output.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="owner failed"):
        with orchestration._exclusive_output_lock(output):
            raise RuntimeError("owner failed")

    with orchestration._exclusive_output_lock(output):
        assert True


def test_orchestration_lock_recovers_after_process_crash(tmp_path: Path) -> None:
    output = tmp_path / "output" / "formal"
    output.mkdir(parents=True)
    crash_script = """
import os
import sys
from pathlib import Path
from advisor_study.orchestration import _exclusive_output_lock
with _exclusive_output_lock(Path(sys.argv[1])):
    os._exit(17)
"""

    crashed = _run_lock_child(output, crash_script)
    assert crashed.returncode == 17

    with orchestration._exclusive_output_lock(output):
        assert True
