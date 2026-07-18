from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from advisor_study import cli
from advisor_study.run_control import _control_id, ensure_run_control, load_run_control
from advisor_study import supervisor


def _frozen(tmp_path: Path) -> cli.FrozenPhase:
    out_dir = tmp_path / "output" / "formal"
    out_dir.mkdir(parents=True)
    manifest = out_dir / "study_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return cli.FrozenPhase(
        out_dir=out_dir,
        manifest_path=manifest,
        study_manifest_id="manifest-1",
        program_count=1,
        program_ids=("p1",),
        groups={"U14": ("A", "B"), "U30": ("A", "B"), "Uall": ("A", "B")},
        jobs=1,
        timeout_s=1,
    )


def _set_budget_to_one_second(frozen: cli.FrozenPhase) -> None:
    control = ensure_run_control(
        frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )
    payload = dict(control.raw_payload)
    payload["program_wall_time_budget_s"] = 1
    payload["control_id"] = _control_id(payload)
    control.path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_supervisor_times_out_program_records_skip_and_restarts_to_success(
    tmp_path: Path,
) -> None:
    frozen = _frozen(tmp_path)
    _set_budget_to_one_second(frozen)
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        """
import json
from pathlib import Path
import sys
import time

out_dir = Path(sys.argv[1])
control = json.loads((out_dir / "run_control.json").read_text(encoding="utf-8"))
if any(row["program_id"] == "p1" for row in control["skip_programs"]):
    raise SystemExit(0)
logs = out_dir / "logs"
logs.mkdir(parents=True, exist_ok=True)
event = {
    "schema_version": "advisor-pair-scale-2n/current-program-v1",
    "study_manifest_id": "manifest-1",
    "program_id": "p1",
    "status": "start",
    "program_wall_time_budget_s": 1,
    "utc": "2026-07-18T00:00:00Z",
    "authority_granted": False,
    "proved_commute": False,
}
(logs / "current_program.json").write_text(json.dumps(event), encoding="utf-8")
while True:
    time.sleep(0.05)
""".lstrip(),
        encoding="utf-8",
    )

    result = supervisor.supervise_run(
        frozen.manifest_path,
        frozen=frozen,
        command=(sys.executable, str(runner), str(frozen.out_dir)),
        poll_interval_s=0.02,
    )

    assert result == 0
    control = load_run_control(
        frozen.out_dir / "run_control.json",
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )
    decision = control.decision_for("p1")
    assert decision.decision == "skip"
    assert decision.observed_wall_time_s is not None
    assert decision.observed_wall_time_s >= 1
    audit_path = frozen.out_dir / "logs" / "supervisor_audit.jsonl"
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records].count("child_start") == 2
    assert any(record["event"] == "runtime_budget_exceeded" for record in records)
    assert records[-1]["event"] == "supervisor_success"
    assert all(record["authority_granted"] is False for record in records)
    assert all(record["record_sha256"] for record in records)


def test_supervisor_non_timeout_failure_is_fail_closed_without_restart(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    ensure_run_control(
        frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )

    result = supervisor.supervise_run(
        frozen.manifest_path,
        frozen=frozen,
        command=(sys.executable, "-c", "raise SystemExit(7)"),
        poll_interval_s=0.01,
    )

    assert result == 7
    control = load_run_control(
        frozen.out_dir / "run_control.json",
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )
    assert control.skip_programs == {}
    records = [
        json.loads(line)
        for line in (frozen.out_dir / "logs" / "supervisor_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event"] for record in records].count("child_start") == 1
    assert records[-1]["event"] == "supervisor_child_failure"


def test_supervisor_internal_validation_error_terminates_started_process_tree(
    tmp_path: Path, monkeypatch
) -> None:
    frozen = _frozen(tmp_path)
    ensure_run_control(
        frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )
    spawned: list[subprocess.Popen[bytes]] = []
    real_popen = supervisor._popen

    def tracking_popen(command: tuple[str, ...]) -> subprocess.Popen[bytes]:
        process = real_popen(command)
        spawned.append(process)
        return process

    monkeypatch.setattr(supervisor, "_popen", tracking_popen)
    monkeypatch.setattr(
        supervisor,
        "_append_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("injected audit validation failure")
        ),
    )
    with pytest.raises(ValueError, match="injected audit"):
        supervisor.supervise_run(
            frozen.manifest_path,
            frozen=frozen,
            command=(sys.executable, "-c", "import time; time.sleep(60)"),
            poll_interval_s=0.01,
        )
    assert len(spawned) == 1
    assert spawned[0].poll() is not None


def test_supervisor_rejects_second_instance_before_starting_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    spawned: list[tuple[str, ...]] = []

    def tracking_popen(command: tuple[str, ...]) -> subprocess.Popen[bytes]:
        spawned.append(command)
        raise AssertionError("second supervisor must not start a child")

    monkeypatch.setattr(supervisor, "_popen", tracking_popen)
    with supervisor._supervisor_instance_lock(frozen.out_dir):
        with pytest.raises(RuntimeError, match="supervisor.*already active"):
            supervisor.supervise_run(
                frozen.manifest_path,
                frozen=frozen,
                command=(sys.executable, "-c", "raise SystemExit(0)"),
                poll_interval_s=0.01,
            )
    assert spawned == []


def test_supervisor_lock_is_cross_process_and_crash_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    holder_script = tmp_path / "hold_supervisor_lock.py"
    holder_script.write_text(
        """
from pathlib import Path
import sys
import time
from advisor_study.supervisor import _supervisor_instance_lock

with _supervisor_instance_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    time.sleep(60)
""".lstrip(),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    experiment_root = str(Path(__file__).resolve().parents[1])
    repository_root = str(Path(__file__).resolve().parents[3])
    environment["PYTHONPATH"] = os.pathsep.join(
        (experiment_root, repository_root, environment.get("PYTHONPATH", ""))
    )
    holder = subprocess.Popen(
        (sys.executable, str(holder_script), str(frozen.out_dir)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        monkeypatch.setattr(
            supervisor,
            "_popen",
            lambda _command: (_ for _ in ()).throw(
                AssertionError("competing supervisor started a child")
            ),
        )
        with pytest.raises(RuntimeError, match="supervisor.*already active"):
            supervisor.supervise_run(
                frozen.manifest_path,
                frozen=frozen,
                command=(sys.executable, "-c", "raise SystemExit(0)"),
                poll_interval_s=0.01,
            )
    finally:
        holder.kill()
        holder.wait(timeout=10)

    with supervisor._supervisor_instance_lock(frozen.out_dir):
        pass


def test_supervisor_does_not_record_timeout_when_child_finishes_at_termination_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    _set_budget_to_one_second(frozen)
    runner = tmp_path / "boundary_runner.py"
    runner.write_text(
        """
import json
from pathlib import Path
import sys
import time

out_dir = Path(sys.argv[1])
logs = out_dir / "logs"
logs.mkdir(parents=True, exist_ok=True)
event = {
    "schema_version": "advisor-pair-scale-2n/current-program-v1",
    "study_manifest_id": "manifest-1",
    "program_id": "p1",
    "status": "start",
    "program_wall_time_budget_s": 1,
    "utc": "2026-07-18T00:00:00Z",
    "authority_granted": False,
    "proved_commute": False,
}
(logs / "current_program.json").write_text(json.dumps(event), encoding="utf-8")
while True:
    time.sleep(0.05)
""".lstrip(),
        encoding="utf-8",
    )

    def child_won_boundary_race(process: subprocess.Popen[bytes]) -> bool:
        process.terminate()
        process.wait(timeout=10)
        return False

    monkeypatch.setattr(
        supervisor, "_terminate_process_tree", child_won_boundary_race
    )
    result = supervisor.supervise_run(
        frozen.manifest_path,
        frozen=frozen,
        command=(sys.executable, str(runner), str(frozen.out_dir)),
        poll_interval_s=0.02,
    )

    assert result != 0
    control = load_run_control(
        frozen.out_dir / "run_control.json",
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )
    assert control.skip_programs == {}
    records = [
        json.loads(line)
        for line in (frozen.out_dir / "logs" / "supervisor_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(record["event"] == "runtime_budget_exceeded" for record in records)
    assert records[-1]["event"] == "supervisor_child_failure"
