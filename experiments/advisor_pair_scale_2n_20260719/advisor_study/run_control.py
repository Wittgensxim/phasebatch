"""Auditable program-boundary controls for the isolated formal study.

The LLVM runners used by this experiment execute synchronously in the main
Python process.  Killing that process from an in-process timer would also kill
the Worker/helper ownership boundary and can leave an ambiguous current stage.
Consequently this module deliberately implements the safe protocol authorized
for long programs: an external monitor stops the process safely, records the
over-budget program in this self-hashed file, and resume observes that decision
at the next program boundary.  The orchestration layer then emits typed,
denominator-preserving limitation rows instead of silently dropping work.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Iterator, Mapping, Sequence


RUN_CONTROL_SCHEMA_VERSION = "advisor-pair-scale-2n/run-control-v1"
RUN_CONTROL_FILE = "run_control.json"
DEFAULT_PROGRAM_WALL_TIME_BUDGET_S = 600
ENFORCEMENT_MODE = "external_program_boundary_skip"
RUNTIME_LIMITATION_KIND = "runtime_budget_exceeded"
_WRITER_LOCK_FILE = ".run-control-writer.lock"
_WRITER_LOCK_TIMEOUT_S = 30.0


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _control_id(payload: Mapping[str, object]) -> str:
    canonical = {str(key): value for key, value in payload.items() if key != "control_id"}
    return _sha256_bytes(_canonical_json(canonical).encode("utf-8"))


@dataclass(frozen=True)
class ProgramDecision:
    program_id: str
    decision: str
    limitation_kind: str
    reason: str
    observed_wall_time_s: float | None
    program_wall_time_budget_s: int
    control_id: str
    control_file_sha256: str

    def provenance(self) -> dict[str, object]:
        return {
            "program_id": self.program_id,
            "decision": self.decision,
            "limitation_kind": self.limitation_kind,
            "reason": self.reason,
            "observed_wall_time_s": (
                self.observed_wall_time_s if self.observed_wall_time_s is not None else ""
            ),
            "program_wall_time_budget_s": self.program_wall_time_budget_s,
            "control_id": self.control_id,
            "control_file_sha256": self.control_file_sha256,
            "enforcement_mode": ENFORCEMENT_MODE,
        }


@dataclass(frozen=True)
class RunControl:
    path: Path
    study_manifest_id: str
    program_wall_time_budget_s: int
    enforcement_mode: str
    program_ids: tuple[str, ...]
    skip_programs: Mapping[str, Mapping[str, object]]
    control_id: str
    control_file_sha256: str
    raw_payload: Mapping[str, object]

    def decision_for(self, program_id: str) -> ProgramDecision:
        program = str(program_id)
        if program not in self.program_ids:
            raise ValueError(f"run control decision is outside the frozen program set: {program}")
        entry = self.skip_programs.get(program)
        if entry is None:
            return ProgramDecision(
                program_id=program,
                decision="execute",
                limitation_kind="",
                reason="",
                observed_wall_time_s=None,
                program_wall_time_budget_s=self.program_wall_time_budget_s,
                control_id=self.control_id,
                control_file_sha256=self.control_file_sha256,
            )
        return ProgramDecision(
            program_id=program,
            decision="skip",
            limitation_kind=RUNTIME_LIMITATION_KIND,
            reason=str(entry["reason"]),
            observed_wall_time_s=float(entry["observed_wall_time_s"]),
            program_wall_time_budget_s=self.program_wall_time_budget_s,
            control_id=self.control_id,
            control_file_sha256=self.control_file_sha256,
        )


def _default_payload(
    study_manifest_id: str, program_ids: Sequence[str]
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RUN_CONTROL_SCHEMA_VERSION,
        "study_manifest_id": str(study_manifest_id),
        "program_wall_time_budget_s": DEFAULT_PROGRAM_WALL_TIME_BUDGET_S,
        "enforcement_mode": ENFORCEMENT_MODE,
        "frozen_program_ids": sorted(str(value) for value in program_ids),
        "skip_programs": [],
        "authority_granted": False,
        "proved_commute": False,
    }
    payload["control_id"] = _control_id(payload)
    return payload


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp"
    data = (_canonical_json(payload) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _run_control_writer_lock(out_dir: Path) -> Iterator[None]:
    """Serialize read-modify-write updates across processes.

    ``os.replace`` makes each individual publication atomic, but without a
    kernel lock two supervisors can still read the same predecessor and lose
    one skip entry.  The one-byte advisory lock is deliberately kept outside
    the self-hashed control file so its bytes never affect evidence identity.
    """

    root = Path(out_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / _WRITER_LOCK_FILE
    stream = lock_path.open("a+b")
    acquired = False
    deadline = time.monotonic() + _WRITER_LOCK_TIMEOUT_S
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out acquiring run-control writer lock: {lock_path}"
                        )
                    time.sleep(0.01)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out acquiring run-control writer lock: {lock_path}"
                        )
                    time.sleep(0.01)
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _ensure_run_control_unlocked(
    out_dir: Path,
    *,
    study_manifest_id: str,
    program_ids: Sequence[str],
) -> RunControl:
    path = Path(out_dir).resolve(strict=False) / RUN_CONTROL_FILE
    if not path.exists():
        _atomic_write(path, _default_payload(study_manifest_id, program_ids))
    return load_run_control(
        path,
        study_manifest_id=study_manifest_id,
        program_ids=program_ids,
    )


def ensure_run_control(
    out_dir: Path,
    *,
    study_manifest_id: str,
    program_ids: Sequence[str],
) -> RunControl:
    """Create the explicit 600s default control once, or validate the existing one."""

    with _run_control_writer_lock(out_dir):
        return _ensure_run_control_unlocked(
            out_dir,
            study_manifest_id=study_manifest_id,
            program_ids=program_ids,
        )


def _validate_run_control_payload(
    raw: object,
    data: bytes,
    *,
    study_manifest_id: str,
    program_ids: Sequence[str],
) -> tuple[int, dict[str, Mapping[str, object]], str, str]:
    """Validate one serialized control for file and checkpoint consumers."""

    if not isinstance(raw, dict):
        raise ValueError("run control must be a JSON object")
    if raw.get("schema_version") != RUN_CONTROL_SCHEMA_VERSION:
        raise ValueError("run control schema_version mismatch")
    if str(raw.get("study_manifest_id", "")) != str(study_manifest_id):
        raise ValueError("run control study_manifest_id mismatch")
    if raw.get("authority_granted") is not False or raw.get("proved_commute") is not False:
        raise ValueError("run control must remain report-only")
    if raw.get("enforcement_mode") != ENFORCEMENT_MODE:
        raise ValueError("run control enforcement_mode mismatch")
    budget = raw.get("program_wall_time_budget_s")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("run control program_wall_time_budget_s must be a positive integer")
    supplied_id = str(raw.get("control_id", ""))
    if len(supplied_id) != 64 or supplied_id != _control_id(raw):
        raise ValueError("run control control_id self-hash mismatch")

    program_values = tuple(program_ids)
    frozen_programs = {str(value) for value in program_values}
    if len(frozen_programs) != len(program_values) or any(
        not value for value in frozen_programs
    ):
        raise ValueError("frozen program IDs must be unique and non-empty")
    persisted_programs = raw.get("frozen_program_ids")
    if (
        not isinstance(persisted_programs, list)
        or any(not isinstance(value, str) or not value for value in persisted_programs)
        or persisted_programs != sorted(frozen_programs)
    ):
        raise ValueError("run control frozen program set mismatch")
    entries = raw.get("skip_programs")
    if not isinstance(entries, list):
        raise ValueError("run control skip_programs must be a list")
    normalized: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("run control skip entry must be an object")
        program_id = str(entry.get("program_id", ""))
        if program_id not in frozen_programs:
            raise ValueError(f"run control skip is outside the frozen program set: {program_id}")
        if program_id in normalized:
            raise ValueError(f"run control contains duplicate skip for frozen program: {program_id}")
        if entry.get("limitation_kind") != RUNTIME_LIMITATION_KIND:
            raise ValueError("run control skip limitation_kind must be runtime_budget_exceeded")
        reason = str(entry.get("reason", "")).strip()
        observed = entry.get("observed_wall_time_s")
        if (
            not reason
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < float(budget)
        ):
            raise ValueError("runtime budget skip requires a reason and observed wall time at least the budget")
        normalized[program_id] = {
            "program_id": program_id,
            "limitation_kind": RUNTIME_LIMITATION_KIND,
            "reason": reason,
            "observed_wall_time_s": float(observed),
        }
    return (
        budget,
        dict(sorted(normalized.items())),
        supplied_id,
        _sha256_bytes(data),
    )


def load_run_control(
    path: Path,
    *,
    study_manifest_id: str,
    program_ids: Sequence[str],
) -> RunControl:
    """Load a self-hashed control and reject all ambiguous decisions."""

    control_path = Path(path).resolve(strict=False)
    try:
        data = control_path.read_bytes()
        raw = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"run control is not valid UTF-8 JSON: {control_path}") from error
    budget, normalized, supplied_id, control_file_sha256 = (
        _validate_run_control_payload(
            raw,
            data,
            study_manifest_id=study_manifest_id,
            program_ids=program_ids,
        )
    )
    return RunControl(
        path=control_path,
        study_manifest_id=str(study_manifest_id),
        program_wall_time_budget_s=budget,
        enforcement_mode=ENFORCEMENT_MODE,
        program_ids=tuple(sorted(str(value) for value in program_ids)),
        skip_programs=normalized,
        control_id=supplied_id,
        control_file_sha256=control_file_sha256,
        raw_payload=raw,
    )


def add_runtime_budget_skip(
    out_dir: Path,
    *,
    study_manifest_id: str,
    program_ids: Sequence[str],
    program_id: str,
    observed_wall_time_s: float,
    reason: str = "external monitor observed the program wall-time budget was exceeded",
) -> RunControl:
    """Atomically add one typed skip without removing or rewriting another entry."""

    with _run_control_writer_lock(out_dir):
        current = _ensure_run_control_unlocked(
            out_dir,
            study_manifest_id=study_manifest_id,
            program_ids=program_ids,
        )
        program = str(program_id)
        if program not in {str(value) for value in program_ids}:
            raise ValueError(f"runtime skip is outside the frozen program set: {program}")
        if program in current.skip_programs:
            existing = current.decision_for(program)
            if (
                existing.reason == str(reason).strip()
                and existing.observed_wall_time_s == float(observed_wall_time_s)
            ):
                return current
            raise ValueError(f"runtime skip already exists for frozen program: {program}")
        observed = float(observed_wall_time_s)
        if not math.isfinite(observed) or observed < current.program_wall_time_budget_s:
            raise ValueError("observed wall time must be finite and at least the configured budget")
        explanation = str(reason).strip()
        if not explanation:
            raise ValueError("runtime skip reason must be non-empty")
        entries = [dict(value) for value in current.skip_programs.values()]
        entries.append(
            {
                "program_id": program,
                "limitation_kind": RUNTIME_LIMITATION_KIND,
                "observed_wall_time_s": observed,
                "reason": explanation,
            }
        )
        entries.sort(key=lambda value: str(value["program_id"]))
        payload = dict(current.raw_payload)
        payload["skip_programs"] = entries
        payload["control_id"] = _control_id(payload)
        _atomic_write(current.path, payload)
        return load_run_control(
            current.path,
            study_manifest_id=study_manifest_id,
            program_ids=program_ids,
        )
