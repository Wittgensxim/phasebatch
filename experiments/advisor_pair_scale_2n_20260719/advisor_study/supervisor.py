"""Crash-safe external wall-time supervisor for the report-only study run.

The experiment runner intentionally remains synchronous.  This module owns a
separate child process, watches the runner's atomic ``current_program.json``
publication, and records a typed program-boundary skip before restarting.  It
never edits evidence or makes an optimizer authority decision.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterator, Mapping, Sequence

from . import cli
from .run_control import (
    RUN_CONTROL_FILE,
    add_runtime_budget_skip,
    ensure_run_control,
    load_run_control,
)


SUPERVISOR_AUDIT_SCHEMA_VERSION = "advisor-pair-scale-2n/supervisor-audit-v1"
_CURRENT_PROGRAM_SCHEMA_VERSION = "advisor-pair-scale-2n/current-program-v1"
_SUPERVISOR_FORCED_EXIT_CODE = 0xE117
_SUPERVISOR_LOCK_FILE = ".supervisor-instance.lock"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_sha256(record: Mapping[str, object]) -> str:
    body = {str(key): value for key, value in record.items() if key != "record_sha256"}
    return _sha256_bytes(_canonical_json(body).encode("utf-8"))


def _load_audit_tail(path: Path) -> tuple[int, str]:
    if not path.exists():
        return 0, ""
    previous = ""
    count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"supervisor audit is unreadable: {path}") from error
    for line in lines:
        if not line:
            raise ValueError("supervisor audit contains an empty record")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("supervisor audit contains invalid JSON") from error
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != SUPERVISOR_AUDIT_SCHEMA_VERSION
            or record.get("authority_granted") is not False
            or record.get("proved_commute") is not False
            or record.get("sequence") != count + 1
            or record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != _record_sha256(record)
        ):
            raise ValueError("supervisor audit hash chain is invalid")
        previous = str(record["record_sha256"])
        count += 1
    return count, previous


def _append_audit(
    path: Path,
    *,
    study_manifest_id: str,
    event: str,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, previous = _load_audit_tail(path)
    record: dict[str, object] = {
        "schema_version": SUPERVISOR_AUDIT_SCHEMA_VERSION,
        "sequence": count + 1,
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "study_manifest_id": str(study_manifest_id),
        "event": str(event),
        "previous_record_sha256": previous,
        "details": dict(details or {}),
        "authority_granted": False,
        "proved_commute": False,
    }
    record["record_sha256"] = _record_sha256(record)
    data = (_canonical_json(record) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return record


@contextmanager
def _supervisor_instance_lock(out_dir: Path) -> Iterator[None]:
    """Hold one crash-released supervisor lock for the complete lifecycle."""

    root = Path(out_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / _SUPERVISOR_LOCK_FILE
    stream = lock_path.open("a+b")
    acquired = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised on non-Windows CI only.
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"supervisor is already active for output root: {root}"
            ) from error
        acquired = True
        yield
    finally:
        if acquired:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised on non-Windows CI only.
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        else:
            stream.close()


def _file_signature(path: Path) -> tuple[int, int, str] | None:
    try:
        data = path.read_bytes()
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, len(data), _sha256_bytes(data)


def _read_current_start(
    path: Path,
    *,
    frozen: cli.FrozenPhase,
    baseline: tuple[int, int, str] | None,
) -> tuple[dict[str, object], tuple[int, int, str]] | None:
    signature = _file_signature(path)
    if signature is None or signature == baseline:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if (
        raw.get("schema_version") != _CURRENT_PROGRAM_SCHEMA_VERSION
        or raw.get("study_manifest_id") != frozen.study_manifest_id
        or raw.get("status") != "start"
        or raw.get("authority_granted") is not False
        or raw.get("proved_commute") is not False
    ):
        return None
    program_id = str(raw.get("program_id", ""))
    if program_id not in frozen.program_ids:
        raise ValueError(
            f"current_program event is outside the frozen program set: {program_id}"
        )
    return raw, signature


def _popen(command: Sequence[str]) -> subprocess.Popen[bytes]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        # Start suspended so the child cannot create an unbound descendant in
        # the interval before it is assigned to the exact Job Object.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004
    else:  # pragma: no cover - exercised on non-Windows CI.
        kwargs["start_new_session"] = True
    process = subprocess.Popen(tuple(str(part) for part in command), **kwargs)
    if os.name == "nt":
        # A Windows Job Object is the only reliable way to bind descendants
        # created after launch to this exact supervised tree.  taskkill is kept
        # only as a defensive fallback because it can fail under system load.
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        job = kernel32.CreateJobObjectW(None, None)
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        configured = bool(
            job
            and kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        )
        assigned = bool(
            configured
            and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle)))
        )
        if not assigned:
            if job:
                kernel32.CloseHandle(job)
            process.kill()
            process.wait()
            raise OSError(ctypes.get_last_error(), "could not bind supervised child to Job Object")
        setattr(process, "_advisor_job_handle", int(job))
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)
        ntdll.NtResumeProcess.restype = ctypes.c_long
        resume_status = int(ntdll.NtResumeProcess(wintypes.HANDLE(int(process._handle))))
        if resume_status != 0:
            kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            kernel32.TerminateJobObject(job, 1)
            kernel32.CloseHandle(job)
            setattr(process, "_advisor_job_handle", 0)
            process.wait()
            raise OSError(resume_status, "could not resume supervised child process")
    return process


def _close_windows_job(process: subprocess.Popen[bytes], *, terminate: bool) -> bool:
    if os.name != "nt":
        return False
    handle = int(getattr(process, "_advisor_job_handle", 0) or 0)
    if not handle:
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    operation_succeeded = True
    if terminate:
        kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        operation_succeeded = bool(
            kernel32.TerminateJobObject(
                wintypes.HANDLE(handle), _SUPERVISOR_FORCED_EXIT_CODE
            )
        )
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(handle))
    setattr(process, "_advisor_job_handle", 0)
    return operation_succeeded


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> bool:
    """Terminate/reap the child tree; report whether this call forced its exit."""

    if process.poll() is not None:
        process.wait()
        _close_windows_job(process, terminate=False)
        return False
    forced = False
    job_terminated = False
    if os.name == "nt":
        job_terminated = _close_windows_job(process, terminate=True)
        forced = job_terminated
        if not job_terminated:
            taskkill = subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            forced = taskkill.returncode == 0
            if process.poll() is None:
                process.kill()
                forced = True
    else:  # pragma: no cover - exercised on non-Windows CI.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            forced = True
        except ProcessLookupError:
            forced = False
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                forced = True
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"supervised process tree did not terminate: pid={process.pid}"
        ) from error
    returncode = int(process.returncode or 0)
    if os.name == "nt":
        if job_terminated:
            return forced and returncode == _SUPERVISOR_FORCED_EXIT_CODE
        return forced
    return forced and returncode in {-signal.SIGTERM, -signal.SIGKILL}


def _wait_for_study_writer_release(out_dir: Path, *, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with cli._study_run_writer_lock(out_dir):
                return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise TimeoutError("study run writer lock was not released after child exit")
            time.sleep(0.02)


def _monitor_supervised_child(
    process: subprocess.Popen[bytes],
    *,
    phase: cli.FrozenPhase,
    child_command: Sequence[str],
    audit: Path,
    current_program_path: Path,
    baseline: tuple[int, int, str] | None,
    poll_interval_s: float,
) -> int | None:
    """Return ``None`` only after a typed timeout is durable and restart-ready."""

    _append_audit(
        audit,
        study_manifest_id=phase.study_manifest_id,
        event="child_start",
        details={"pid": process.pid, "command": list(child_command)},
    )
    active_program = ""
    active_since: float | None = None
    event_signature = baseline
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        observed = _read_current_start(
            current_program_path,
            frozen=phase,
            baseline=event_signature,
        )
        if observed is not None:
            event, event_signature = observed
            program_id = str(event["program_id"])
            if program_id != active_program:
                active_program = program_id
                active_since = time.monotonic()
                _append_audit(
                    audit,
                    study_manifest_id=phase.study_manifest_id,
                    event="program_observed",
                    details={"program_id": active_program, "pid": process.pid},
                )
        if active_program and active_since is not None:
            control = load_run_control(
                phase.out_dir / RUN_CONTROL_FILE,
                study_manifest_id=phase.study_manifest_id,
                program_ids=phase.program_ids,
            )
            budget = control.program_wall_time_budget_s
            elapsed = time.monotonic() - active_since
            if elapsed >= budget:
                if process.poll() is not None:
                    break
                if not _terminate_process_tree(process):
                    break
                observed_wall_time = max(float(budget), elapsed)
                reason = (
                    "external supervisor terminated the exact child process tree "
                    "after the frozen per-program wall-time budget was exceeded"
                )
                updated = add_runtime_budget_skip(
                    phase.out_dir,
                    study_manifest_id=phase.study_manifest_id,
                    program_ids=phase.program_ids,
                    program_id=active_program,
                    observed_wall_time_s=observed_wall_time,
                    reason=reason,
                )
                _append_audit(
                    audit,
                    study_manifest_id=phase.study_manifest_id,
                    event="runtime_budget_exceeded",
                    details={
                        "program_id": active_program,
                        "pid": process.pid,
                        "observed_wall_time_s": observed_wall_time,
                        "program_wall_time_budget_s": budget,
                        "control_id": updated.control_id,
                        "control_file_sha256": updated.control_file_sha256,
                    },
                )
                _wait_for_study_writer_release(phase.out_dir)
                _append_audit(
                    audit,
                    study_manifest_id=phase.study_manifest_id,
                    event="restart_ready",
                    details={"program_id": active_program},
                )
                return None
        time.sleep(poll_interval_s)

    returncode = process.wait()
    _close_windows_job(process, terminate=False)
    if returncode == 0:
        _append_audit(
            audit,
            study_manifest_id=phase.study_manifest_id,
            event="supervisor_success",
            details={"child_returncode": 0},
        )
        return 0
    _append_audit(
        audit,
        study_manifest_id=phase.study_manifest_id,
        event="supervisor_child_failure",
        details={"child_returncode": returncode, "pid": process.pid},
    )
    return int(returncode)


def supervise_run(
    manifest_path: Path,
    *,
    frozen: cli.FrozenPhase | None = None,
    command: Sequence[str] | None = None,
    poll_interval_s: float = 0.25,
) -> int:
    """Run, supervise, and restart the frozen study at program boundaries."""

    manifest = Path(manifest_path).resolve(strict=False)
    phase = frozen or cli.load_frozen_phase(manifest, phase="run")
    if manifest != Path(phase.manifest_path).resolve(strict=False):
        raise ValueError("supervisor manifest does not match the frozen phase")
    if poll_interval_s <= 0:
        raise ValueError("supervisor poll interval must be positive")
    child_command = tuple(command or (
        sys.executable,
        "-m",
        "advisor_study.cli",
        "run",
        "--manifest",
        str(manifest),
    ))
    if not child_command:
        raise ValueError("supervisor child command must be non-empty")

    with _supervisor_instance_lock(phase.out_dir):
        ensure_run_control(
            phase.out_dir,
            study_manifest_id=phase.study_manifest_id,
            program_ids=phase.program_ids,
        )
        audit = phase.out_dir / "logs" / "supervisor_audit.jsonl"
        current_program_path = phase.out_dir / "logs" / "current_program.json"

        while True:
            baseline = _file_signature(current_program_path)
            process = _popen(child_command)
            try:
                result = _monitor_supervised_child(
                    process,
                    phase=phase,
                    child_command=child_command,
                    audit=audit,
                    current_program_path=current_program_path,
                    baseline=baseline,
                    poll_interval_s=poll_interval_s,
                )
            except BaseException:
                if process.poll() is None:
                    _terminate_process_tree(process)
                else:
                    process.wait()
                    _close_windows_job(process, terminate=False)
                raise
            if result is None:
                continue
            return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m advisor_study.supervisor")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return supervise_run(args.manifest, poll_interval_s=args.poll_interval)


if __name__ == "__main__":  # pragma: no cover - exercised through Python -m.
    raise SystemExit(main())
