"""Standalone command boundary for the isolated pair/advisor-2N study.

The command boundary is intentionally small and fail-closed.  It owns no
production authority: ``prepare`` only freezes inputs, ``run`` receives one
already frozen manifest, and ``summarize`` receives only hash-validated frozen
evidence.  Concrete LLVM adapters are deliberately kept behind private
functions so the parser/gates can be tested without executing a compiler.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from typing import Callable, Iterator, Mapping, Sequence, TypeVar

import yaml

from .manifest import (
    DEFAULT_MAX_SOURCE_BYTES,
    DEFAULT_SELECTION_SEED,
    FORMAL_PROGRAM_TARGET,
    FORMAL_SAMPLING_FRAME_SCHEMA_VERSION,
    FORMAL_SELECTION_RULE_ID,
    FORMAL_SOURCE_INVENTORY_COUNT,
    FORMAL_SOURCE_POSITIONS,
    ProgramRecord,
    canonical_sha256,
    normalize_program_id,
    normalize_program_relative_path,
    require_study_manifest,
    stable_rank,
)
from .orchestration import (
    RAW_EXECUTION_SEMANTICS_REVISION,
    _terminal_error_fingerprint,
)
from .pass_universe import ActionRecord, load_frozen_policy, load_u14_actions
from .schema import TABLE_FIELDS
from .study import (
    PrepareDependencies,
    RunResult,
    _validate_candidate_identity_exclusion_document,
    prepare_study,
)


_GROUP_IDS = ("U14", "U30", "Uall")
_TOOL_IDS = ("opt", "clang", "worker", "merge_helper")
_PREPARE_SCHEMA = "advisor-pair-scale-2n/prepare-v1"
_PREPARE_COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "study_manifest_id",
        "authority_granted",
        "proved_commute",
        "program_count",
        "formal_program_count",
        "fixed_program_count",
        "formal_source_inventory_count",
        "formal_selection_rule_id",
        "formal_source_positions",
        "formal_sampling_frame_sha256",
        "candidate_reserve_count",
        "candidate_inventory_count",
        "candidate_identity_exclusion_count",
        "candidate_identity_exclusions_sha256",
        "selection_seed",
        "group_sizes",
        "scale_gate",
        "files_sha256",
        "document_sha256",
    }
)
_RAW_COMPLETION_SCHEMA = "advisor-pair-scale-2n/raw-v2"
_ACTIVE_RAW_HANDOFF_SCHEMA = "advisor-pair-scale-2n/raw-active-handoff-v2"
_ACTIVE_RAW_HANDOFF_FILE = "active_cleanup_handoff.json"
_PROGRAM_CHECKPOINT_INDEX_SCHEMA = "advisor-pair-scale-2n/program-checkpoint-index-v1"
_PROGRAM_CHECKPOINT_PROGRESS_FILE = "program_checkpoint_progress.json"
_HARD_STATE_COMPARATOR_VERSION = "phasebatch.ir_equivalence.v2"
_FORMAL_SAMPLING_FRAME_FIELDS = frozenset(
    {
        "schema_version",
        "source_inventory_count",
        "selection_rule_id",
        "source_positions",
        "source_programs",
        "source_programs_sha256",
        "selected_programs",
        "selected_programs_sha256",
        "document_sha256",
    }
)
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ReplayCallback = Callable[[dict[str, object], int, Path], Mapping[str, object]]
ReplayDependencyFactory = Callable[[Mapping[str, ReplayCallback]], Mapping[str, ReplayCallback]]
_CleanupResultT = TypeVar("_CleanupResultT")


def _build_replay_dependencies(
    defaults: Mapping[str, ReplayCallback],
    replay_dependency_factory: ReplayDependencyFactory | None = None,
) -> dict[str, ReplayCallback]:
    """Return the three fail-closed replay callbacks used by ``_run_frozen``.

    The optional factory is deliberately a test seam, rather than an alternate
    command-line mode.  Production therefore always constructs the real
    Worker/external-opt/2N callbacks below, while a focused test can prove that
    the CLI hands all three callback families to orchestration for both replay
    repetitions without launching LLVM.
    """

    required = ("worker", "external_opt", "two_n")
    if set(defaults) != set(required) or not all(callable(defaults[key]) for key in required):
        raise ValueError("CLI replay defaults must contain exactly worker, external_opt, and two_n")
    candidate = dict(defaults)
    if replay_dependency_factory is not None:
        candidate = replay_dependency_factory(dict(defaults))
    if not isinstance(candidate, Mapping) or set(candidate) != set(required):
        raise ValueError("CLI replay dependency factory returned an invalid callback set")
    callbacks: dict[str, ReplayCallback] = {}
    for key in required:
        callback = candidate[key]
        if not callable(callback):
            raise ValueError(f"CLI replay dependency {key} is not callable")
        callbacks[key] = callback
    return callbacks


@dataclass(frozen=True)
class FrozenPhase:
    """Validated immutable hand-off between the three study phases."""

    out_dir: Path
    manifest_path: Path
    study_manifest_id: str
    program_count: int
    program_ids: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    jobs: int
    timeout_s: float


class _WorkerRunner:
    """Narrow, read-only protocol adapter around the existing Worker binary.

    It is deliberately local to this isolated CLI.  The Worker is used solely
    to materialize AB/BA and 2N experiment observations; no response is ever
    mapped to a production authority decision.
    """

    def __init__(self, worker_path: Path, *, timeout_s: float) -> None:
        from phasebatch.opt_worker import WorkerProcess

        self._process = WorkerProcess((str(worker_path),), worker_id=0)
        self._timeout_s = timeout_s
        self._worker_path = worker_path

    def close(self) -> None:
        self._process.close()

    def apply(self, parent: Path, action: object, output: Path) -> dict[str, object]:
        """Apply exactly one pipeline to one materialized parent artifact."""

        from phasebatch.opt_worker import WorkerError, WorkerTimeoutError

        pipeline = str(getattr(action, "pipeline", "")).strip()
        if not pipeline:
            return self._failure(output, "error", "action pipeline is missing")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = (
            str(self._worker_path), "load", str(parent), "apply", pipeline,
            "materialize", str(output),
        )
        parent_handle = ""
        child_handle = ""
        try:
            loaded = self._request("load", path=str(parent))
            parent_handle = str(loaded["module_handle"])
            applied = self._request(
                "apply",
                parent_handle=parent_handle,
                pipeline=pipeline,
                verify_each=True,
                materialize_path=str(output),
            )
            child_handle = str(applied["module_handle"])
            worker_canonical_hash = str(applied.get("canonical_hash", ""))
            if len(worker_canonical_hash) != 64 or not output.is_file():
                return self._failure(output, "error", "worker did not materialize a canonical output", command)
            try:
                state_id = _phasebatch_hard_state_id(output)
            except (OSError, UnicodeError, ValueError) as error:
                return self._failure(
                    output,
                    "error",
                    f"phasebatch hard-state hash failed: {error}",
                    command,
                )
            return {
                "success": True,
                "execution_status": "success",
                "verifier_status": "success",
                "output_path": str(output),
                "hard_state_id": state_id,
                "worker_canonical_hash": worker_canonical_hash,
                "hard_state_source": "phasebatch_hard_state_policy",
                "hard_state_policy_verified": True,
                "worker_hash_verified": True,
                "physical_pass_invocations": 1,
                "command": command,
                # WorkerProcess exposes only process-lifetime accumulated
                # stderr, not stderr scoped to this successful request.  A
                # successful stage therefore has the canonical local value.
                "stderr": "",
            }
        except WorkerTimeoutError as error:
            return self._failure(output, "timeout", str(error), command)
        except WorkerError as error:
            return self._failure(output, "error", str(error), command)
        finally:
            for handle in (child_handle, parent_handle):
                if not handle:
                    continue
                try:
                    self._process.request("release", timeout=self._timeout_s, module_handle=handle)
                except Exception:
                    # Evidence is already fail-closed if the original request
                    # failed.  A release error must not redirect any output.
                    pass

    def _request(self, operation: str, **payload: object) -> Mapping[str, object]:
        reply = self._process.request(operation, timeout=self._timeout_s, **payload).payload
        if reply.get("status") != "ok":
            detail = str(reply.get("error_message", reply.get("error_kind", "worker operation failed")))
            from phasebatch.opt_worker import WorkerError

            raise WorkerError(detail)
        return reply

    def _failure(
        self, output: Path, status: str, stderr: str, command: Sequence[str] = ()
    ) -> dict[str, object]:
        return {
            "success": False,
            "execution_status": status,
            "timed_out": status == "timeout",
            "verifier_status": "not_run",
            "output_path": str(output),
            "hard_state_id": "",
            "worker_hash_verified": False,
            "physical_pass_invocations": 1,
            "command": list(command),
            "stderr": stderr or self._process.stderr_text,
        }

    def canonical_state(self, path: Path) -> str:
        """Return the Worker canonical hash for one already materialized IR."""

        from phasebatch.opt_worker import WorkerError

        handle = ""
        try:
            loaded = self._request("load", path=str(path))
            handle = str(loaded["module_handle"])
            digest = str(loaded.get("canonical_hash", ""))
            if len(digest) != 64:
                raise WorkerError("worker load did not return canonical_hash")
            return digest
        finally:
            if handle:
                try:
                    self._process.request("release", timeout=self._timeout_s, module_handle=handle)
                except Exception:
                    pass

    def hard_state(self, path: Path) -> str:
        """Hash materialized IR under the frozen root-study policy."""

        return _phasebatch_hard_state_id(path)


class _WorkerRunnerPool:
    """Bind each pair-executor thread to one exclusive Worker runner.

    The pool is used only by the bounded per-program Uall pair section.  A
    binding is retained for the lifetime of one executor and explicitly
    released after ``run_complete_pair_matrix`` has joined all worker threads.
    """

    def __init__(
        self,
        worker_path: Path,
        *,
        timeout_s: float,
        jobs: int,
        runner_factory: Callable[..., object] | None = None,
    ) -> None:
        if type(jobs) is not int or jobs < 1:
            raise ValueError("jobs must be a positive integer")
        factory = _WorkerRunner if runner_factory is None else runner_factory
        self._condition = threading.Condition()
        self._bindings: dict[int, object] = {}
        self._active_threads: set[int] = set()
        self._closed = False
        runners: list[object] = []
        try:
            for _index in range(jobs):
                runner = factory(Path(worker_path), timeout_s=timeout_s)
                if not callable(getattr(runner, "apply", None)) or not callable(
                    getattr(runner, "close", None)
                ):
                    close = getattr(runner, "close", None)
                    if callable(close):
                        close()
                    raise TypeError("runner_factory must return apply/close runners")
                runners.append(runner)
        except BaseException:
            for runner in runners:
                try:
                    runner.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            raise
        self._runners = tuple(runners)
        self._available = list(runners)

    @property
    def primary(self) -> object:
        """Return the first runner for serial profile/2N/replay stages."""

        return self._runners[0]

    def apply(self, parent: Path, action: object, output: Path) -> object:
        thread_id = threading.get_ident()
        with self._condition:
            if self._closed:
                raise RuntimeError("Worker runner pool is closed")
            if thread_id in self._active_threads:
                raise RuntimeError("Worker runner pool does not allow recursive apply")
            runner = self._bindings.get(thread_id)
            while runner is None and not self._available:
                self._condition.wait()
                if self._closed:
                    raise RuntimeError("Worker runner pool is closed")
                runner = self._bindings.get(thread_id)
            if runner is None:
                runner = self._available.pop(0)
                self._bindings[thread_id] = runner
            self._active_threads.add(thread_id)
        try:
            return runner.apply(parent, action, output)  # type: ignore[attr-defined]
        finally:
            with self._condition:
                self._active_threads.remove(thread_id)
                self._condition.notify_all()

    def release_thread_bindings(self) -> None:
        """Release affinity only after the pair executor has joined."""

        with self._condition:
            if self._active_threads:
                raise RuntimeError("cannot release active Worker runner bindings")
            if self._closed:
                raise RuntimeError("Worker runner pool is closed")
            self._bindings.clear()
            self._available = list(self._runners)
            self._condition.notify_all()

    def close(self) -> None:
        """Close every runner exactly once, even if one close raises."""

        with self._condition:
            if self._closed:
                return
            if self._active_threads:
                raise RuntimeError("cannot close an active Worker runner pool")
            self._closed = True
            self._bindings.clear()
            self._available.clear()
            self._condition.notify_all()
        first_error: BaseException | None = None
        for runner in self._runners:
            try:
                runner.close()  # type: ignore[attr-defined]
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _create_worker_runners(
    worker_path: Path,
    *,
    timeout_s: float,
    jobs: int,
    runner_factory: Callable[..., object] | None = None,
) -> tuple[object, _WorkerRunnerPool | None]:
    """Keep jobs=1 on the direct runner path; pool only parallel pair work."""

    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    factory = _WorkerRunner if runner_factory is None else runner_factory
    if jobs == 1:
        runner = factory(Path(worker_path), timeout_s=timeout_s)
        if not callable(getattr(runner, "apply", None)) or not callable(
            getattr(runner, "close", None)
        ):
            close = getattr(runner, "close", None)
            if callable(close):
                close()
            raise TypeError("runner_factory must return apply/close runners")
        return runner, None
    pool = _WorkerRunnerPool(
        worker_path,
        timeout_s=timeout_s,
        jobs=jobs,
        runner_factory=factory,
    )
    return pool.primary, pool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes | str) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def _phasebatch_hard_state_id(path: Path) -> str:
    """Return the authoritative root-study hard-state identity for one IR."""

    from phasebatch.ir_equivalence import DEFAULT_HARD_STATE_POLICY, hard_state_hash

    digest = hard_state_hash(Path(path), DEFAULT_HARD_STATE_POLICY)
    if len(digest) != 64:
        raise ValueError("phasebatch hard-state hash is malformed")
    return digest


def _run_replay_pair_stages(
    *,
    root: Path,
    left: object,
    right: object,
    directory: Path,
    runner: Callable[[Path, object, Path], Mapping[str, object]],
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    """Replay A/B/AB/BA without ever substituting ``S`` for a failed parent."""

    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        name: directory / f"{name}.ll"
        for name in ("S", "A", "B", "AB", "BA", "merged_input")
    }
    shutil.copyfile(root, paths["S"])
    raw_a = runner(paths["S"], left, paths["A"])
    raw_b = runner(paths["S"], right, paths["B"])
    stages = {
        "A": _replay_stage_evidence(raw_a, paths["A"]),
        "B": _replay_stage_evidence(raw_b, paths["B"]),
    }
    if stages["A"]["execution_status"] == "success":
        raw_ab = runner(paths["A"], right, paths["AB"])
        stages["AB"] = _replay_stage_evidence(raw_ab, paths["AB"])
    else:
        stages["AB"] = _replay_not_run_stage("A did not produce a successful parent")
    if stages["B"]["execution_status"] == "success":
        raw_ba = runner(paths["B"], left, paths["BA"])
        stages["BA"] = _replay_stage_evidence(raw_ba, paths["BA"])
    else:
        stages["BA"] = _replay_not_run_stage("B did not produce a successful parent")
    return paths, stages


def _build_pair_only_replay_record(
    *,
    root: Path,
    left: object,
    right: object,
    directory: Path,
    runner: Callable[[Path, object, Path], Mapping[str, object]],
    family: str,
    repetition: int,
) -> dict[str, object]:
    """Capture A/B/AB/BA evidence without making pair replay depend on merge."""

    if family not in {"worker", "external_opt", "two_n"}:
        raise ValueError(f"unknown replay family: {family}")
    paths, stages = _run_replay_pair_stages(
        root=root,
        left=left,
        right=right,
        directory=directory,
        runner=runner,
    )
    status = "success"
    stderr_parts = [
        str(stages[name]["stderr"])
        for name in ("A", "B", "AB", "BA")
        if str(stages[name]["stderr"])
    ]
    commands = [
        str(part)
        for name in ("A", "B", "AB", "BA")
        for part in stages[name]["command"]
    ]
    existing = {
        name: path
        for name, path in paths.items()
        if name != "merged_input" and path.is_file()
    }
    artifact_sha = {name: _sha256_file(path) for name, path in existing.items()}
    hard_hashes: dict[str, str] = {}
    for name, path in existing.items():
        try:
            hard_hashes[name] = _phasebatch_hard_state_id(path)
        except Exception as error:
            if name == "S" or any(
                stages[stage]["execution_status"] == "success" and stage == name
                for stage in ("A", "B", "AB", "BA")
            ):
                status = "error"
                stderr_parts.append(
                    "phasebatch hard-state replay hash failed:"
                    f"{name}:{type(error).__name__}"
                )
    command_kind = {
        "worker": "worker-replay",
        "external_opt": "external-opt-replay",
        "two_n": "two-n-replay",
    }[family]
    return {
        "status": status,
        "hard_state_hashes": hard_hashes,
        "artifact_sha256": artifact_sha,
        "artifacts": {name: str(path) for name, path in existing.items()},
        "stderr": "\n".join(stderr_parts),
        "command": tuple((command_kind, str(repetition), *commands)),
        "two_n_result": {},
        "stage_results": {
            name: {
                field: stages[name][field]
                for field in (
                    "execution_status",
                    "verifier_status",
                    "hard_state_id",
                    "output_sha256",
                    "command_sha256",
                    "stderr_sha256",
                    "error_fingerprint",
                )
            }
            for name in ("A", "B", "AB", "BA")
        },
        "merge_status": "not_applicable",
        "merge_error_fingerprint": "",
    }


def _external_replay_apply(
    opt: Path,
    timeout_s: float,
    parent: Path,
    action: object,
    output: Path,
) -> dict[str, object]:
    """Apply one action with external opt and independently verify its output."""

    pipeline = str(getattr(action, "pipeline", "")).strip()
    command = (str(opt), f"-passes={pipeline}", "-S", str(parent), "-o", str(output))
    if not pipeline:
        return {
            "success": False,
            "execution_status": "error",
            "verifier_status": "not_run",
            "output_path": str(output),
            "stderr": "action pipeline is missing",
            "command": command,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = _run_process(command, timeout_s=timeout_s)
    except TimeoutError:
        return {
            "success": False,
            "execution_status": "timeout",
            "verifier_status": "not_run",
            "output_path": str(output),
            "stderr": "opt timeout",
            "command": command,
        }
    except OSError as error:
        return {
            "success": False,
            "execution_status": "error",
            "verifier_status": "not_run",
            "output_path": str(output),
            "stderr": f"opt launch failed:{type(error).__name__}",
            "command": command,
        }
    if completed.returncode != 0 or not output.is_file():
        return {
            "success": False,
            "execution_status": "error",
            "verifier_status": "not_run",
            "output_path": str(output),
            "stderr": completed.stderr,
            "command": command,
        }
    if not _verify_with_opt(opt, timeout_s, output):
        return {
            "success": False,
            "execution_status": "invalid",
            "verifier_status": "invalid",
            "output_path": str(output),
            "stderr": completed.stderr,
            "command": command,
        }
    return {
        "success": True,
        "execution_status": "success",
        "verifier_status": "success",
        "output_path": str(output),
        "stderr": completed.stderr,
        "command": command,
    }


def _replay_stage_evidence(
    raw: Mapping[str, object], requested_output: Path
) -> dict[str, object]:
    status = str(raw.get("execution_status", raw.get("status", ""))).strip()
    if not status:
        status = "success" if bool(raw.get("success", False)) else "error"
    verifier = str(raw.get("verifier_status", "")).strip() or (
        "success" if status == "success" else "not_run"
    )
    output_path = Path(str(raw.get("output_path", requested_output))).resolve(
        strict=False
    )
    if output_path != requested_output.resolve(strict=False):
        status = "error"
    output_sha256 = _sha256_file(requested_output) if requested_output.is_file() else ""
    hard_state_id = str(raw.get("hard_state_id", ""))
    if status == "success" and requested_output.is_file() and len(hard_state_id) != 64:
        hard_state_id = _phasebatch_hard_state_id(requested_output)
    if status == "success" and (
        not requested_output.is_file() or verifier != "success" or len(hard_state_id) != 64
    ):
        status = "error"
    command_raw = raw.get("command", ())
    command = (
        tuple(str(part) for part in command_raw)
        if isinstance(command_raw, (list, tuple))
        else (str(command_raw),)
        if str(command_raw)
        else ()
    )
    stderr = str(raw.get("stderr", ""))
    command_sha256 = _sha256_bytes("\0".join(command))
    stderr_sha256 = _sha256_bytes(stderr)
    fingerprint = _terminal_error_fingerprint(status, verifier, stderr_sha256)
    return {
        "execution_status": status,
        "verifier_status": verifier,
        "hard_state_id": hard_state_id if status == "success" else "",
        "output_sha256": output_sha256,
        "output_path": str(requested_output.resolve(strict=False)),
        "command": command,
        "stderr": stderr,
        "command_sha256": command_sha256,
        "stderr_sha256": stderr_sha256,
        "error_fingerprint": fingerprint,
    }


def _replay_not_run_stage(reason: str) -> dict[str, object]:
    command_sha256 = _sha256_bytes("")
    stderr_sha256 = _sha256_bytes(reason)
    fingerprint = _terminal_error_fingerprint("not_run", "not_run", stderr_sha256)
    return {
        "execution_status": "not_run",
        "verifier_status": "not_run",
        "hard_state_id": "",
        "output_sha256": "",
        "output_path": "",
        "command": (),
        "stderr": reason,
        "command_sha256": command_sha256,
        "stderr_sha256": stderr_sha256,
        "error_fingerprint": fingerprint,
    }


def _bind_two_n_replay_artifacts(
    base: dict[str, object],
    directional: Mapping[str, object],
    directory: Path,
) -> dict[str, str] | None:
    """Bind the exact full-group 2N merge and second-round artifacts."""

    artifact_maps: list[dict[str, object]] = []
    for field in ("artifacts", "artifact_sha256", "hard_state_hashes"):
        value = base.get(field)
        if not isinstance(value, dict):
            _mark_two_n_replay_evidence_error(base, f"2N replay {field} mapping unavailable")
            return None
        value.pop("merged_input", None)
        value.pop("second_round_output", None)
        artifact_maps.append(value)
    artifacts, artifact_sha256, hard_state_hashes = artifact_maps
    root = directory.resolve(strict=False)
    merged_source = Path(str(directional.get("merged_input_path", ""))).resolve(
        strict=False
    )
    second_source = Path(str(directional.get("second_output_path", ""))).resolve(
        strict=False
    )
    try:
        merged_source.relative_to(root)
        second_source.relative_to(root)
    except ValueError:
        _mark_two_n_replay_evidence_error(base, "2N replay directional artifact escapes repeat directory")
        return None
    if not merged_source.is_file():
        _mark_two_n_replay_evidence_error(base, "2N replay full-group merged input unavailable")
        return None
    if not second_source.is_file():
        _mark_two_n_replay_evidence_error(base, "2N replay second-round output unavailable")
        return None
    merged_sha256 = _sha256_file(merged_source)
    second_sha256 = _sha256_file(second_source)
    try:
        merged_hard_state_id = _phasebatch_hard_state_id(merged_source)
        second_hard_state_id = _phasebatch_hard_state_id(second_source)
    except (OSError, UnicodeError, ValueError):
        _mark_two_n_replay_evidence_error(base, "2N replay hard-state identity unavailable")
        return None
    if (
        not _sha256_shaped(directional.get("merged_input_sha256", ""))
        or str(directional.get("merged_input_sha256", "")) != merged_sha256
        or not _sha256_shaped(directional.get("merged_input_hard_state_id", ""))
        or str(directional.get("merged_input_hard_state_id", ""))
        != merged_hard_state_id
        or not _sha256_shaped(directional.get("second_output_sha256", ""))
        or str(directional.get("second_output_sha256", "")) != second_sha256
    ):
        _mark_two_n_replay_evidence_error(base, "2N replay directional artifact identity mismatch")
        return None
    merged_artifact = directory / "merged_input.ll"
    second_artifact = directory / "second_round_output.ll"
    if merged_source != merged_artifact.resolve(strict=False):
        shutil.copyfile(merged_source, merged_artifact)
    if second_source != second_artifact.resolve(strict=False):
        shutil.copyfile(second_source, second_artifact)
    artifacts["merged_input"] = str(merged_artifact.resolve())
    artifacts["second_round_output"] = str(second_artifact.resolve())
    artifact_sha256["merged_input"] = merged_sha256
    artifact_sha256["second_round_output"] = second_sha256
    hard_state_hashes["merged_input"] = merged_hard_state_id
    hard_state_hashes["second_round_output"] = second_hard_state_id
    base["merge_status"] = "complete"
    base["merge_error_fingerprint"] = ""
    return {
        "merged_input_sha256": merged_sha256,
        "merged_input_hard_state_id": merged_hard_state_id,
        "second_output_sha256": second_sha256,
        "second_output_hard_state_id": second_hard_state_id,
    }


def _mark_two_n_replay_evidence_error(base: dict[str, object], detail: str) -> None:
    base["status"] = "error"
    base["merge_status"] = "error"
    base["merge_error_fingerprint"] = _sha256_bytes(detail)
    stderr = str(base.get("stderr", ""))
    base["stderr"] = "\n".join(part for part in (stderr, detail) if part)


def _run_output_path(value: object) -> Path:
    raw = value.get("output_path", "") if isinstance(value, Mapping) else getattr(value, "output_path", "")
    return Path(str(raw)) if str(raw) else Path()


def _compare_phasebatch_hard_states(
    left: object,
    right: object,
    *,
    opt: Path,
    timeout_s: float,
) -> object | None:
    """Adapt the approved Phasebatch comparator to the report-only oracle."""

    from phasebatch.ir_equivalence import (
        DEFAULT_HARD_STATE_POLICY,
        compare_hard_states,
    )
    from .pair_matrix import HardStateEquality

    left_path, right_path = _run_output_path(left), _run_output_path(right)
    if not left_path.is_file() or not right_path.is_file():
        return None
    equality = compare_hard_states(
        left_path,
        right_path,
        tools={"opt": {"path": str(Path(opt).resolve())}},
        timeout=max(1, int(timeout_s)),
        policy=DEFAULT_HARD_STATE_POLICY,
    )
    if equality.policy_id != DEFAULT_HARD_STATE_POLICY.policy_id or equality.tier == "failed":
        return None
    return HardStateEquality(
        can_hard_fold=bool(equality.can_hard_fold),
        tier="hard_state",
        reason=str(equality.reason),
        trusted_hard_comparator=True,
    )


def _json_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _require_output_kind(out_dir: Path) -> str:
    """Accept only an ``output/smoke`` or ``output/formal`` study root.

    ``load_frozen_phase`` is deliberately portable for offline audit copies;
    prepare-time path handling additionally pins new output to this experiment
    root.  No command ever writes through a path supplied by raw evidence.
    """

    normalized = out_dir.resolve()
    parts = normalized.parts
    try:
        output_index = [part.casefold() for part in parts].index("output")
    except ValueError as error:
        raise ValueError("manifest output must be inside output/smoke or output/formal") from error
    if output_index + 1 != len(parts) - 1 or parts[output_index + 1] not in {"smoke", "formal"}:
        raise ValueError("manifest output must be the output/smoke or output/formal root")
    return parts[output_index + 1]


def validate_prepare_output(out_dir: Path, *, require_experiment_root: bool = False) -> str:
    """Reject a collision before a prepare adapter can execute anything."""

    target = Path(out_dir).resolve()
    kind = _require_output_kind(target)
    expected_root = EXPERIMENT_ROOT.resolve() / "output" / kind
    if require_experiment_root and target != expected_root:
        raise ValueError("prepare output must be exactly the isolated output/smoke or output/formal root")
    if target.exists():
        if target.is_dir() and not any(target.iterdir()):
            return kind
        # A pre-existing directory is acceptable only if Task 4's exact
        # self-hash protocol recognizes it.  The implementation lives in the
        # isolated study module and contains no production authority imports.
        from .study import _validate_existing_prepare_state

        _validate_existing_prepare_state(target)
    return kind


def _group_ids_from_sidecar(out_dir: Path, manifest: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    sidecar = out_dir / "pass_groups.json"
    groups = _json_mapping(sidecar, label="pass_groups")
    if set(groups) != set(_GROUP_IDS):
        raise ValueError("pass_groups must contain exactly U14, U30, Uall")
    normalized: dict[str, tuple[str, ...]] = {}
    for group_id in _GROUP_IDS:
        raw = groups[group_id]
        if not isinstance(raw, Mapping):
            raise ValueError(f"pass_groups.{group_id} must be a mapping")
        if raw.get("group_id") != group_id:
            raise ValueError(f"pass_groups.{group_id}.group_id mismatch")
        action_ids = raw.get("action_ids")
        size = raw.get("group_size")
        if not isinstance(action_ids, list) or not all(isinstance(item, str) and item for item in action_ids):
            raise ValueError(f"pass_groups.{group_id}.action_ids must be a non-empty string list")
        ids = tuple(action_ids)
        if len(ids) != len(set(ids)) or size != len(ids):
            raise ValueError(f"pass_groups.{group_id} has duplicate or mismatched actions")
        normalized[group_id] = ids
    if len(normalized["U14"]) != 14:
        raise ValueError("U14 requires exactly 14 frozen actions")
    u14, u30, uall = (normalized[group] for group in _GROUP_IDS)
    if len(u30) != 30 or len(set(u30) - set(u14)) != 16 or not set(u14).issubset(u30):
        raise ValueError("U30 requires exactly 16 eligible non-U14 additions")
    if not set(u30).issubset(uall):
        raise ValueError("pass groups must be nested U14 subset U30 subset Uall")

    frozen = manifest.get("frozen_inputs")
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get("pass_groups"), Mapping):
        raise ValueError("study manifest is missing frozen pass_groups identity")
    identity = frozen["pass_groups"]
    if identity.get("kind") != "canonical_object" or identity.get("sha256") != canonical_sha256(groups):
        raise ValueError("pass_groups does not match frozen manifest identity")
    return normalized


def _validate_completion(out_dir: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    completion = _json_mapping(out_dir / "prepare_complete.json", label="prepare_complete")
    if set(completion) != _PREPARE_COMPLETION_FIELDS:
        raise ValueError("prepare completion field set mismatch")
    document_sha256 = completion.get("document_sha256")
    unsigned_completion = {
        str(key): value
        for key, value in completion.items()
        if key != "document_sha256"
    }
    if (
        not isinstance(document_sha256, str)
        or len(document_sha256) != 64
        or canonical_sha256(unsigned_completion) != document_sha256
    ):
        raise ValueError("prepare completion document hash mismatch")
    if completion.get("schema_version") != _PREPARE_SCHEMA:
        raise ValueError("prepare completion schema mismatch")
    if completion.get("study_manifest_id") != manifest.get("study_manifest_id"):
        raise ValueError("prepare completion manifest mismatch")
    if completion.get("authority_granted") is not False or completion.get("proved_commute") is not False:
        raise ValueError("prepare completion must keep authority_granted=false and proved_commute=false")
    hashes = completion.get("files_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("prepare completion is not hash-validated")
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("prepare completion is not hash-validated")
        path = (out_dir / relative).resolve()
        try:
            path.relative_to(out_dir.resolve())
        except ValueError as error:
            raise ValueError("prepare completion hash path escapes output") from error
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("prepare completion is not hash-validated")
    return completion


def _validate_tools(manifest: Mapping[str, object]) -> None:
    tools = manifest.get("tools")
    if not isinstance(tools, Mapping) or set(tools) != set(_TOOL_IDS):
        raise ValueError("study manifest tool set mismatch")
    for name in _TOOL_IDS:
        record = tools[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"{name} tool record must be a mapping")
        path = Path(str(record.get("path", "")))
        expected = record.get("sha256")
        if not path.is_file():
            raise ValueError(f"{name} tool is missing: {path}")
        if not isinstance(expected, str) or _sha256_file(path) != expected:
            raise ValueError(f"{name} tool hash mismatch")


def _canonical_manifest_value(
    manifest: Mapping[str, object], field: str
) -> Mapping[str, object]:
    record = manifest.get(field)
    if (
        not isinstance(record, Mapping)
        or record.get("kind") != "canonical_object"
        or not isinstance(record.get("value"), Mapping)
        or record.get("sha256") != canonical_sha256(record["value"])
    ):
        raise ValueError(f"study manifest {field} identity mismatch")
    return record["value"]  # type: ignore[return-value]


def _validate_execution_semantics(manifest: Mapping[str, object]) -> None:
    """Reject manifests produced by the invalid pre-Task-15 execution path."""

    from phasebatch.ir_equivalence import DEFAULT_HARD_STATE_POLICY

    hard_state = _canonical_manifest_value(manifest, "hard_state_policy")
    comparator = _canonical_manifest_value(manifest, "comparator")
    artifact = _canonical_manifest_value(manifest, "artifact_policy")
    if (
        hard_state.get("policy_id") != DEFAULT_HARD_STATE_POLICY.policy_id
        or hard_state.get("implementation")
        != "phasebatch.ir_equivalence.hard_state_hash"
        or hard_state.get("raw_execution_semantics_revision")
        != RAW_EXECUTION_SEMANTICS_REVISION
    ):
        raise ValueError("frozen hard-state execution semantics mismatch")
    expected_comparator_id = (
        f"{DEFAULT_HARD_STATE_POLICY.policy_id}@{_HARD_STATE_COMPARATOR_VERSION}"
    )
    llvm_diff = comparator.get("llvm_diff")
    if (
        comparator.get("comparator_id") != expected_comparator_id
        or comparator.get("implementation")
        != "phasebatch.ir_equivalence.compare_hard_states"
        or comparator.get("comparator_version") != _HARD_STATE_COMPARATOR_VERSION
        or comparator.get("raw_execution_semantics_revision")
        != RAW_EXECUTION_SEMANTICS_REVISION
        or not isinstance(llvm_diff, Mapping)
    ):
        raise ValueError("frozen hard-state comparator semantics mismatch")
    llvm_diff_path = Path(str(llvm_diff.get("path", "")))
    llvm_diff_sha = str(llvm_diff.get("sha256", ""))
    if (
        not llvm_diff_path.is_file()
        or len(llvm_diff_sha) != 64
        or _sha256_file(llvm_diff_path) != llvm_diff_sha
    ):
        raise ValueError("frozen llvm-diff identity mismatch")
    if (
        artifact.get("raw_execution_semantics_revision")
        != RAW_EXECUTION_SEMANTICS_REVISION
    ):
        raise ValueError("frozen raw execution semantics revision mismatch")


def _canonical_program_rows(value: object, *, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        command = raw.get("compile_command")
        if not isinstance(command, list):
            raise ValueError(f"{label}[{index}] compile_command must be a list")
        try:
            record = ProgramRecord(
                program_id=raw.get("program_id"),
                source_path=raw.get("source_path"),
                relative_path=raw.get("relative_path"),
                program_family=raw.get("program_family"),
                source_sha256=raw.get("source_sha256"),
                source_size_bytes=raw.get("source_size_bytes"),
                compile_command=tuple(command),
                compile_status=raw.get("compile_status"),
                compile_stderr_sha256=raw.get("compile_stderr_sha256"),
                root_ir_path=raw.get("root_ir_path"),
                root_ir_sha256=raw.get("root_ir_sha256"),
                root_hard_state_id=raw.get("root_hard_state_id"),
                target=raw.get("target"),
                data_layout=raw.get("data_layout"),
                preflight_status=raw.get("preflight_status"),
                selection_class=raw.get("selection_class"),
                selection_order=raw.get("selection_order"),
                reserve_rank=raw.get("reserve_rank"),
                replacement_for_program_id=raw.get("replacement_for_program_id"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}[{index}] is invalid: {error}") from error
        canonical = record.as_manifest_record()
        if canonical != dict(raw):
            raise ValueError(f"{label}[{index}] is not canonical")
        rows.append(canonical)
    return rows


def _read_formal_sampling_frame_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _validate_formal_sampling_frame(
    *,
    scope: Mapping[str, object],
    manifest_programs: object,
    program_sidecar: Mapping[str, object],
    out_dir: Path,
) -> None:
    frame_path = Path(out_dir) / "formal_sampling_frame.json"
    expected_file_sha256 = scope.get("formal_sampling_frame_sha256")
    try:
        frame_bytes = _read_formal_sampling_frame_bytes(frame_path)
    except OSError as error:
        raise ValueError("formal sampling frame is not readable") from error
    if (
        not isinstance(expected_file_sha256, str)
        or len(expected_file_sha256) != 64
        or _sha256_bytes(frame_bytes) != expected_file_sha256
    ):
        raise ValueError("formal sampling frame artifact hash mismatch")
    try:
        frame_value = json.loads(frame_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal sampling frame is not valid JSON") from error
    if not isinstance(frame_value, dict):
        raise ValueError("formal sampling frame must be a JSON object")
    frame = {str(key): value for key, value in frame_value.items()}
    if set(frame) != _FORMAL_SAMPLING_FRAME_FIELDS:
        raise ValueError("formal sampling frame fields mismatch")
    document_sha256 = frame.get("document_sha256")
    unsigned_frame = {
        str(key): value for key, value in frame.items() if key != "document_sha256"
    }
    if (
        not isinstance(document_sha256, str)
        or len(document_sha256) != 64
        or canonical_sha256(unsigned_frame) != document_sha256
    ):
        raise ValueError("formal sampling frame document hash mismatch")
    if frame.get("schema_version") != FORMAL_SAMPLING_FRAME_SCHEMA_VERSION:
        raise ValueError("formal sampling frame schema mismatch")
    if (
        type(frame.get("source_inventory_count")) is not int
        or frame.get("source_inventory_count") != FORMAL_SOURCE_INVENTORY_COUNT
        or frame.get("selection_rule_id") != FORMAL_SELECTION_RULE_ID
        or type(frame.get("source_positions")) is not list
        or frame.get("source_positions") != list(FORMAL_SOURCE_POSITIONS)
    ):
        raise ValueError("formal sampling frame selection metadata mismatch")

    source_rows = _canonical_program_rows(
        frame.get("source_programs"),
        label="formal sampling frame source_programs",
    )
    if (
        len(source_rows) != FORMAL_SOURCE_INVENTORY_COUNT
        or [row["selection_order"] for row in source_rows]
        != list(range(1, FORMAL_SOURCE_INVENTORY_COUNT + 1))
        or any(row["selection_class"] != "fixed" for row in source_rows)
        or frame.get("source_programs_sha256") != canonical_sha256(source_rows)
    ):
        raise ValueError("formal sampling frame source inventory mismatch")
    selected_rows = _canonical_program_rows(
        frame.get("selected_programs"),
        label="formal sampling frame selected_programs",
    )
    if (
        len(selected_rows) != FORMAL_PROGRAM_TARGET
        or frame.get("selected_programs_sha256") != canonical_sha256(selected_rows)
    ):
        raise ValueError("formal sampling frame selected programs mismatch")
    selected_by_order = {row["selection_order"]: row for row in selected_rows}
    if set(selected_by_order) != set(range(1, FORMAL_PROGRAM_TARGET + 1)):
        raise ValueError("formal sampling frame selected order mismatch")
    for selection_order, source_position in enumerate(FORMAL_SOURCE_POSITIONS, start=1):
        source_row = source_rows[source_position - 1]
        selected_row = selected_by_order[selection_order]
        for field, expected in source_row.items():
            if field in {"root_ir_path", "selection_order"}:
                continue
            if selected_row.get(field) != expected:
                raise ValueError("formal sampling frame selected program identity mismatch")

    manifest_rows = _canonical_program_rows(
        manifest_programs,
        label="study manifest program_manifest",
    )
    sidecar_rows = _canonical_program_rows(
        program_sidecar.get("programs"),
        label="program manifest sidecar programs",
    )
    if selected_rows != manifest_rows or selected_rows != sidecar_rows:
        raise ValueError("formal sampling frame program binding mismatch")


def _validate_formal_scope_metadata(
    manifest: Mapping[str, object],
    completion: Mapping[str, object],
    out_dir: Path,
) -> None:
    """Require the self-hashed formal-10 sampling boundary and exact bindings."""

    artifact = manifest.get("artifact_policy")
    if not isinstance(artifact, Mapping) or artifact.get("kind") != "canonical_object":
        raise ValueError("formal study is missing sampling scope metadata")
    scope = artifact.get("value")
    if not isinstance(scope, Mapping):
        raise ValueError("formal study is missing sampling scope metadata")
    if (
        type(scope.get("formal_program_count")) is not int
        or scope.get("formal_program_count") != FORMAL_PROGRAM_TARGET
    ):
        raise ValueError("formal study requires formal_program_count=10")
    if (
        type(scope.get("fixed_program_count")) is not int
        or scope.get("fixed_program_count") != FORMAL_PROGRAM_TARGET
    ):
        raise ValueError("formal study requires fixed_program_count=10")
    if (
        type(scope.get("formal_source_inventory_count")) is not int
        or scope.get("formal_source_inventory_count")
        != FORMAL_SOURCE_INVENTORY_COUNT
    ):
        raise ValueError("formal study source inventory must contain exactly 50 programs")
    if scope.get("formal_selection_rule_id") != FORMAL_SELECTION_RULE_ID:
        raise ValueError("formal study selection rule mismatch")
    if (
        type(scope.get("formal_source_positions")) is not list
        or scope.get("formal_source_positions") != list(FORMAL_SOURCE_POSITIONS)
    ):
        raise ValueError("formal study source positions mismatch")
    candidate_count = scope.get("candidate_reserve_count")
    if type(candidate_count) is not int or candidate_count != 0:
        raise ValueError("formal study requires candidate_reserve_count=0")
    inventory_count = scope.get("candidate_inventory_count")
    exclusion_count = scope.get("candidate_identity_exclusion_count")
    if (
        type(inventory_count) is not int
        or inventory_count != 0
        or type(exclusion_count) is not int
        or exclusion_count != 0
    ):
        raise ValueError("formal study forbids candidate inventory and exclusions")
    selection_seed = scope.get("selection_seed")
    if type(selection_seed) is not int or selection_seed != DEFAULT_SELECTION_SEED:
        raise ValueError("formal study requires selection_seed=0 as an integer")

    completion_bindings = {
        "formal_program_count": FORMAL_PROGRAM_TARGET,
        "fixed_program_count": FORMAL_PROGRAM_TARGET,
        "formal_source_inventory_count": FORMAL_SOURCE_INVENTORY_COUNT,
        "formal_selection_rule_id": FORMAL_SELECTION_RULE_ID,
        "formal_source_positions": list(FORMAL_SOURCE_POSITIONS),
        "formal_sampling_frame_sha256": scope.get("formal_sampling_frame_sha256"),
        "candidate_reserve_count": 0,
        "candidate_inventory_count": 0,
        "candidate_identity_exclusion_count": 0,
        "selection_seed": DEFAULT_SELECTION_SEED,
    }
    for field, expected in completion_bindings.items():
        actual = completion.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"prepare completion formal scope mismatch: {field}"
            )

    sidecar = Path(out_dir) / "candidate_identity_exclusions.json"
    expected_sidecar_sha = scope.get("candidate_identity_exclusions_sha256")
    if (
        not sidecar.is_file()
        or not isinstance(expected_sidecar_sha, str)
        or _sha256_file(sidecar) != expected_sidecar_sha
    ):
        raise ValueError("candidate identity exclusions artifact hash mismatch")
    try:
        exclusion_document = _validate_candidate_identity_exclusion_document(
            _json_mapping(sidecar, label="candidate identity exclusions"),
            verify_sources=True,
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"candidate identity exclusions are invalid: {error}") from error
    if (
        exclusion_document["candidate_inventory_count"] != inventory_count
        or exclusion_document["candidate_reserve_count"] != candidate_count
        or exclusion_document["exclusion_count"] != exclusion_count
    ):
        raise ValueError("candidate identity exclusions metadata mismatch")
    if exclusion_document["exclusions"]:
        raise ValueError("formal midpoint scope forbids candidate exclusions")
    program_sidecar = _json_mapping(
        Path(out_dir) / "program_manifest.json", label="program manifest sidecar"
    )
    if program_sidecar.get("reserve_order") != []:
        raise ValueError("formal midpoint scope requires an empty reserve order")
    programs = manifest.get("program_manifest")
    if not isinstance(programs, list) or len(programs) != FORMAL_PROGRAM_TARGET:
        raise ValueError("formal study requires exactly 10 frozen programs")
    if any(
        not isinstance(row, Mapping) or row.get("selection_class") != "fixed"
        for row in programs
    ):
        raise ValueError("formal study requires only existing fixed programs")
    selection_orders = [row.get("selection_order") for row in programs]
    if (
        any(type(order) is not int for order in selection_orders)
        or sorted(selection_orders) != list(range(1, FORMAL_PROGRAM_TARGET + 1))
    ):
        raise ValueError("formal study requires the complete selected program order")
    if len({str(row.get("program_family", "")) for row in programs}) != FORMAL_PROGRAM_TARGET:
        raise ValueError("formal study requires 10 distinct program_family values")
    _validate_formal_sampling_frame(
        scope=scope,
        manifest_programs=programs,
        program_sidecar=program_sidecar,
        out_dir=out_dir,
    )


def load_frozen_phase(manifest_path: Path, *, phase: str) -> FrozenPhase:
    """Load one self-validating frozen study, without examining pair results."""

    if phase not in {"run", "summarize"}:
        raise ValueError("phase must be run or summarize")
    path = Path(manifest_path).resolve()
    if path.name != "study_manifest.json" or not path.is_file():
        raise ValueError("--manifest must name an existing study_manifest.json")
    out_dir = path.parent
    kind = _require_output_kind(out_dir)
    if out_dir != (EXPERIMENT_ROOT.resolve() / "output" / kind):
        raise ValueError("manifest must remain at the isolated experiment output root")
    manifest = _json_mapping(path, label="study manifest")
    try:
        require_study_manifest(manifest, manifest)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"study manifest mismatch: {error}") from error
    completion = _validate_completion(out_dir, manifest)
    groups = _group_ids_from_sidecar(out_dir, manifest)
    _validate_tools(manifest)
    _validate_execution_semantics(manifest)
    programs = manifest.get("program_manifest")
    if not isinstance(programs, list):
        raise ValueError("study manifest program_manifest must be a list")
    program_count = len(programs)
    if completion.get("program_count") != program_count:
        raise ValueError("prepare completion program count mismatch")
    if kind == "formal" and program_count != FORMAL_PROGRAM_TARGET:
        raise ValueError("formal study requires exactly 10 frozen programs")
    if kind == "formal":
        _validate_formal_scope_metadata(manifest, completion, out_dir)
    if kind == "smoke" and program_count != 3:
        raise ValueError("smoke study requires exactly 3 frozen programs")
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("study manifest execution section is missing")
    jobs, timeout_s = execution.get("jobs"), execution.get("timeout_s")
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError("frozen jobs must be a positive integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("frozen timeout must be positive")
    program_ids = tuple(sorted(str(row.get("program_id", "")) for row in programs))
    if len(program_ids) != program_count or any(not value for value in program_ids) or len(set(program_ids)) != len(program_ids):
        raise ValueError("frozen program IDs must be unique and complete")
    return FrozenPhase(
        out_dir=out_dir,
        manifest_path=path,
        study_manifest_id=str(manifest["study_manifest_id"]),
        program_count=program_count,
        program_ids=program_ids,
        groups=groups,
        jobs=jobs,
        timeout_s=float(timeout_s),
    )


def _manifest_for_frozen(frozen: FrozenPhase) -> dict[str, object]:
    manifest = _json_mapping(frozen.manifest_path, label="study manifest")
    if str(manifest.get("study_manifest_id", "")) != frozen.study_manifest_id:
        raise ValueError("frozen manifest changed after phase gate")
    return manifest


def _load_actions(out_dir: Path) -> dict[str, ActionRecord]:
    path = out_dir / "pass_inventory.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pass_inventory is not valid JSON") from error
    if not isinstance(raw, list):
        raise ValueError("pass_inventory must be a JSON list")
    actions: dict[str, ActionRecord] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("pass_inventory rows must be mappings")
        action_record = item.get("action")
        if action_record is None:
            continue
        if not isinstance(action_record, Mapping):
            raise ValueError("pass_inventory action must be a mapping or null")
        action = ActionRecord.from_manifest_record(action_record)
        if action.action_id in actions:
            raise ValueError(f"duplicate action in pass_inventory: {action.action_id}")
        actions[action.action_id] = action
    if not actions:
        raise ValueError("pass_inventory contains no eligible Function actions")
    return dict(sorted(actions.items()))


def _verify_with_opt(opt: Path, timeout_s: float, path: Path) -> bool:
    """Run the external verifier only; it never chooses or authorizes a pass."""

    try:
        result = subprocess.run(
            (str(opt), "-disable-output", "-passes=verify", str(path)),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _project_row(table_name: str, row: Mapping[str, object], *, program_family: str = "") -> dict[str, object]:
    """Project rich raw runner evidence into the frozen public CSV schema."""

    fields = TABLE_FIELDS[table_name]
    numeric = {
        field for field in fields
        if field.endswith(("_n", "_count", "_calls", "_applications", "_invocations", "_ms"))
        or field in {"selection_order", "selection_rank", "group_size", "source_size_bytes", "reserve_rank", "repetition"}
    }
    projected: dict[str, object] = {
        field: 0 if field in numeric else "" for field in fields
    }
    projected.update({field: row[field] for field in fields if field in row})
    if "program_family" in projected and not projected["program_family"]:
        projected["program_family"] = program_family
    if "cache_reused" in projected and not projected["cache_reused"]:
        projected["cache_reused"] = "false"
    if "authority_granted" in projected:
        projected["authority_granted"] = "false"
    if "proved_commute" in projected:
        projected["proved_commute"] = "false"
    return projected


def _write_raw_completion(
    out_dir: Path,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    manifest_id: str,
    *,
    cleanup_ledger: Mapping[str, object] | None = None,
) -> None:
    """Publish a self-hashed aggregation hand-off without replacing old state.

    Legacy callers retain their compact root-level hand-off.  Cleanup callers
    receive a versioned hand-off directory and one atomic active-pointer swap:
    a ledger or completion write failure therefore leaves the prior active
    completion usable and, crucially, occurs before any cleanup is attempted.
    """

    if cleanup_ledger is not None:
        _publish_cleanup_handoff(out_dir, rows, manifest_id, cleanup_ledger)
        return

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows_path = raw_dir / "study_rows.json"
    payload = {
        "study_manifest_id": manifest_id,
        "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
        "authority_granted": False,
        "proved_commute": False,
        "tables": {name: list(value) for name, value in sorted(rows.items())},
    }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    staging = raw_dir / ".study_rows.staging.json"
    staging.write_text(content, encoding="utf-8", newline="\n")
    staging.replace(rows_path)
    files_sha256 = {"study_rows.json": _sha256_file(rows_path)}
    if cleanup_ledger is not None:
        cleanup_path = raw_dir / "cleanup_ledger.json"
        cleanup_staging = raw_dir / ".cleanup_ledger.staging.json"
        cleanup_staging.write_text(
            json.dumps(dict(cleanup_ledger), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        cleanup_staging.replace(cleanup_path)
        files_sha256["cleanup_ledger.json"] = _sha256_file(cleanup_path)
    completion = {
        "schema_version": _RAW_COMPLETION_SCHEMA,
        "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
        "study_manifest_id": manifest_id,
        "authority_granted": False,
        "proved_commute": False,
        "files_sha256": files_sha256,
    }
    completion_path = raw_dir / "complete.json"
    stage_completion = raw_dir / ".complete.staging.json"
    stage_completion.write_text(
        json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    stage_completion.replace(completion_path)


def _handoff_payload(
    rows: Mapping[str, Sequence[Mapping[str, object]]], manifest_id: str,
) -> str:
    return json.dumps(
        {
            "study_manifest_id": manifest_id,
            "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
            "authority_granted": False,
            "proved_commute": False,
            "tables": {name: list(value) for name, value in sorted(rows.items())},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _write_handoff_file(path: Path, content: str) -> None:
    """Small test seam for injected ledger/completion publication failures."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _sha256_shaped(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_program_checkpoint_index(
    frozen: FrozenPhase,
    checkpoint_index: Mapping[str, object],
    *,
    require_complete: bool = True,
) -> dict[str, dict[str, object]]:
    """Re-open every pointer-selected complete checkpoint bound by the index."""

    from .program_runtime import (
        PROGRAM_RUNTIME_IMPLEMENTATION_VERSION,
        load_program_checkpoint,
        program_checkpoint_input_sha256,
        program_checkpoint_path,
    )

    entries = checkpoint_index.get("entries")
    if (
        checkpoint_index.get("schema_version") != _PROGRAM_CHECKPOINT_INDEX_SCHEMA
        or checkpoint_index.get("study_manifest_id") != frozen.study_manifest_id
        or checkpoint_index.get("raw_execution_semantics_revision")
        != RAW_EXECUTION_SEMANTICS_REVISION
        or checkpoint_index.get("program_runtime_implementation_version")
        != PROGRAM_RUNTIME_IMPLEMENTATION_VERSION
        or checkpoint_index.get("program_count") != frozen.program_count
        or checkpoint_index.get("frozen_program_ids") != sorted(frozen.program_ids)
        or checkpoint_index.get("authority_granted") is not False
        or checkpoint_index.get("proved_commute") is not False
        or not isinstance(entries, list)
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        raise ValueError("program checkpoint index metadata mismatch")
    completed_count = checkpoint_index.get("completed_program_count")
    if (
        isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or completed_count != len(entries)
        or not _sha256_shaped(checkpoint_index.get("progress_id", ""))
    ):
        raise ValueError("program checkpoint index progress metadata mismatch")
    progress_body = {
        str(key): value
        for key, value in checkpoint_index.items()
        if key != "progress_id"
    }
    if checkpoint_index.get("progress_id") != _sha256_bytes(
        json.dumps(
            progress_body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    ):
        raise ValueError("program checkpoint index self-hash mismatch")
    by_program: dict[str, Mapping[str, object]] = {}
    for raw_entry in entries:
        entry = raw_entry
        program_id = str(entry.get("program_id", ""))
        if not program_id or program_id in by_program:
            raise ValueError("program checkpoint index program coverage is ambiguous")
        by_program[program_id] = entry
    if not set(by_program).issubset(frozen.program_ids):
        raise ValueError("program checkpoint index escapes the frozen program set")
    if require_complete and (
        set(by_program) != set(frozen.program_ids)
        or len(entries) != frozen.program_count
    ):
        raise ValueError("program checkpoint index does not cover the frozen program set")

    payloads: dict[str, dict[str, object]] = {}
    for program_id in sorted(by_program):
        entry = by_program[program_id]
        input_sha256 = str(entry.get("checkpoint_input_sha256", ""))
        root_sha256 = str(entry.get("root_ir_sha256", ""))
        if (
            entry.get("authority_granted") is not False
            or entry.get("proved_commute") is not False
            or entry.get("raw_execution_semantics_revision")
            != RAW_EXECUTION_SEMANTICS_REVISION
            or entry.get("program_runtime_implementation_version")
            != PROGRAM_RUNTIME_IMPLEMENTATION_VERSION
            or not _sha256_shaped(input_sha256)
            or not _sha256_shaped(root_sha256)
        ):
            raise ValueError(f"program checkpoint index entry mismatch: {program_id}")
        expected_input = program_checkpoint_input_sha256(
            study_manifest_id=frozen.study_manifest_id,
            program_id=program_id,
            root_ir_sha256=root_sha256,
            group_action_ids=frozen.groups,
            runner_semantics_id=RAW_EXECUTION_SEMANTICS_REVISION,
        )
        checkpoint_dir = program_checkpoint_path(frozen.out_dir, expected_input)
        try:
            indexed_dir = Path(str(entry.get("checkpoint_dir", ""))).resolve(strict=False)
            indexed_relative = checkpoint_dir.resolve(strict=False).relative_to(
                frozen.out_dir.resolve(strict=False)
            ).as_posix()
        except ValueError as error:
            raise ValueError(
                f"program checkpoint index path escapes the output: {program_id}"
            ) from error
        if (
            input_sha256 != expected_input
            or indexed_dir != checkpoint_dir.resolve(strict=False)
            or entry.get("checkpoint_relative_path") != indexed_relative
        ):
            raise ValueError(f"program checkpoint index input/path mismatch: {program_id}")
        loaded = load_program_checkpoint(
            checkpoint_dir,
            expected_input_sha256=expected_input,
            isolation_root=frozen.out_dir,
        )
        if loaded is None or loaded[1] != "complete":
            raise ValueError(f"program checkpoint selected complete evidence is invalid: {program_id}")
        payload = loaded[0]
        pointer = _json_mapping(
            checkpoint_dir / "active.json", label="indexed program checkpoint pointer"
        )
        ledger = payload.get("cleanup_ledger")
        staging_ledger = payload.get("unpublished_staging_cleanup")
        cleanup_entries = ledger.get("entries") if isinstance(ledger, Mapping) else None
        staging_entries = (
            staging_ledger.get("entries")
            if isinstance(staging_ledger, Mapping)
            else None
        )
        if (
            not isinstance(ledger, Mapping)
            or not isinstance(cleanup_entries, list)
            or not isinstance(staging_ledger, Mapping)
            or not isinstance(staging_entries, list)
        ):
            raise ValueError(f"program checkpoint cleanup ledger is invalid: {program_id}")
        cleanup_ids = sorted(
            str(cleanup_entry.get("cleanup_id", ""))
            for cleanup_entry in cleanup_entries
            if isinstance(cleanup_entry, Mapping)
        )
        ledger_sha256 = _sha256_bytes(
            json.dumps(ledger, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
        staging_cleanup_ids = sorted(
            str(cleanup_entry.get("cleanup_id", ""))
            for cleanup_entry in staging_entries
            if isinstance(cleanup_entry, Mapping)
        )
        staging_ledger_sha256 = _sha256_bytes(
            json.dumps(
                staging_ledger,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if (
            pointer.get("version_id") != entry.get("checkpoint_version_id")
            or pointer.get("result_sha256") != entry.get("checkpoint_result_sha256")
            or entry.get("cleanup_ledger_sha256") != ledger_sha256
            or entry.get("cleanup_ids") != cleanup_ids
            or entry.get("unpublished_staging_cleanup_sha256")
            != staging_ledger_sha256
            or entry.get("unpublished_staging_cleanup_ids")
            != staging_cleanup_ids
            or payload.get("study_manifest_id") != frozen.study_manifest_id
            or payload.get("program_id") != program_id
            or payload.get("program_status") != entry.get("program_status")
            or payload.get("limitation_kind") != entry.get("limitation_kind")
        ):
            raise ValueError(f"program checkpoint index content binding mismatch: {program_id}")
        payloads[program_id] = payload
    return payloads


def _cleanup_ledger_from_checkpoint_index(
    frozen: FrozenPhase, *, checkpoint_index: Mapping[str, object]
) -> dict[str, object]:
    """Combine already-complete per-program ledgers without deleting again."""

    from .cleanup import CLEANUP_LEDGER_SCHEMA_VERSION

    payloads = _validate_program_checkpoint_index(frozen, checkpoint_index)
    entries: list[dict[str, object]] = []
    protected_pairs: set[str] = set()
    protected_directionals: set[tuple[str, str, str]] = set()
    seen_cleanup_ids: set[str] = set()
    for program_id in sorted(payloads):
        ledger = payloads[program_id]["cleanup_ledger"]
        assert isinstance(ledger, Mapping)
        raw_entries = ledger.get("entries")
        raw_pairs = ledger.get("protected_pair_row_ids")
        raw_directionals = ledger.get("protected_directionals")
        if (
            ledger.get("cleanup_state") != "complete"
            or not isinstance(raw_entries, list)
            or not isinstance(raw_pairs, list)
            or not isinstance(raw_directionals, list)
        ):
            raise ValueError(f"program checkpoint cleanup ledger is incomplete: {program_id}")
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"program checkpoint cleanup entry is malformed: {program_id}")
            cleanup_id = str(raw_entry.get("cleanup_id", ""))
            if not _sha256_shaped(cleanup_id) or cleanup_id in seen_cleanup_ids:
                raise ValueError("combined program cleanup IDs are missing or ambiguous")
            seen_cleanup_ids.add(cleanup_id)
            entries.append(dict(raw_entry))
        protected_pairs.update(str(value) for value in raw_pairs)
        for value in raw_directionals:
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError("program checkpoint protected directional is malformed")
            protected_directionals.add(tuple(str(part) for part in value))
    ordered = sorted(entries, key=lambda entry: str(entry["cleanup_id"]))
    summary_fields = (
        "reclaimed_file_count",
        "reclaimed_bytes",
        "retained_file_count",
        "retained_bytes",
        "planned_file_count",
        "planned_bytes",
    )
    summary = {
        field: sum(
            int(entry.get(field, 0))
            for entry in ordered
            if isinstance(entry.get(field, 0), int)
            and not isinstance(entry.get(field, 0), bool)
        )
        for field in summary_fields
    }
    return {
        "schema_version": CLEANUP_LEDGER_SCHEMA_VERSION,
        "cleanup_state": "complete",
        "study_manifest_id": frozen.study_manifest_id,
        "authority_granted": False,
        "proved_commute": False,
        "protected_pair_row_ids": sorted(protected_pairs),
        "protected_directionals": [list(value) for value in sorted(protected_directionals)],
        "entries": ordered,
        "summary": summary,
    }


def _publish_cleanup_handoff(
    out_dir: Path,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    manifest_id: str,
    cleanup_ledger: Mapping[str, object],
    *,
    checkpoint_index: Mapping[str, object] | None = None,
) -> None:
    """Stage and validate all files, then atomically switch one active pointer."""

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    staging = raw_dir / f".cleanup-handoff-staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        rows_path = staging / "study_rows.json"
        ledger_path = staging / "cleanup_ledger.json"
        index_path = staging / "program_checkpoint_index.json"
        _write_handoff_file(rows_path, _handoff_payload(rows, manifest_id))
        _write_handoff_file(
            ledger_path,
            json.dumps(dict(cleanup_ledger), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        )
        if checkpoint_index is not None:
            if (
                checkpoint_index.get("schema_version") != _PROGRAM_CHECKPOINT_INDEX_SCHEMA
                or checkpoint_index.get("study_manifest_id") != manifest_id
                or checkpoint_index.get("raw_execution_semantics_revision")
                != RAW_EXECUTION_SEMANTICS_REVISION
                or checkpoint_index.get("authority_granted") is not False
                or checkpoint_index.get("proved_commute") is not False
            ):
                raise ValueError("program checkpoint index publication metadata mismatch")
            _write_handoff_file(
                index_path,
                json.dumps(
                    dict(checkpoint_index),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )
        staged_payload = _json_mapping(rows_path, label="staged raw study rows")
        staged_tables = staged_payload.get("tables")
        if not isinstance(staged_tables, Mapping) or not all(
            isinstance(name, str) and isinstance(values, list)
            and all(isinstance(row, Mapping) for row in values)
            for name, values in staged_tables.items()
        ):
            raise ValueError("staged cleanup raw tables are malformed")
        # Entry-to-row binding is checked before the active pointer is
        # exposed.  Full frozen coverage is checked again after publication
        # and before the cleanup callback may delete any IR.
        _validate_cleanup_ledger(
            _json_mapping(ledger_path, label="staged cleanup ledger"),
            {str(name): list(values) for name, values in staged_tables.items()},
            manifest_id,
        )
        files_sha256 = {
            "study_rows.json": _sha256_file(rows_path),
            "cleanup_ledger.json": _sha256_file(ledger_path),
        }
        if checkpoint_index is not None:
            files_sha256["program_checkpoint_index.json"] = _sha256_file(index_path)
        completion = {
            "schema_version": _RAW_COMPLETION_SCHEMA,
            "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
            "study_manifest_id": manifest_id,
            "authority_granted": False,
            "proved_commute": False,
            "files_sha256": files_sha256,
        }
        completion_path = staging / "complete.json"
        _write_handoff_file(
            completion_path,
            json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\n",
        )
        # Validate the staged bytes before exposing them through the pointer.
        if (
            _sha256_file(rows_path) != completion["files_sha256"]["study_rows.json"]
            or _sha256_file(ledger_path) != completion["files_sha256"]["cleanup_ledger.json"]
            or (
                checkpoint_index is not None
                and _sha256_file(index_path)
                != completion["files_sha256"]["program_checkpoint_index.json"]
            )
            or not _json_mapping(ledger_path, label="staged cleanup ledger")
        ):
            raise ValueError("staged cleanup handoff is not self-validating")
        handoff_id = hashlib.sha256(
            (
                manifest_id
                + "".join(files_sha256[name] for name in sorted(files_sha256))
            ).encode("utf-8")
        ).hexdigest()[:24]
        handoffs = raw_dir / "handoffs"
        handoffs.mkdir(exist_ok=True)
        published = handoffs / handoff_id
        if published.exists():
            # Identical bytes already have a valid immutable hand-off.  Leave
            # it intact and discard only this never-active staging directory.
            shutil.rmtree(staging)
        else:
            staging.replace(published)
        pointer = {
            "schema_version": _ACTIVE_RAW_HANDOFF_SCHEMA,
            "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
            "study_manifest_id": manifest_id,
            "handoff_dir": f"handoffs/{handoff_id}",
            "authority_granted": False,
            "proved_commute": False,
        }
        if checkpoint_index is not None:
            pointer["program_checkpoint_index_sha256"] = files_sha256[
                "program_checkpoint_index.json"
            ]
        pointer_staging = raw_dir / f".{_ACTIVE_RAW_HANDOFF_FILE}.{uuid.uuid4().hex}.staging"
        _write_handoff_file(
            pointer_staging,
            json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
        )
        pointer_staging.replace(raw_dir / _ACTIVE_RAW_HANDOFF_FILE)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _execute_after_planned_cleanup_handoff(
    *,
    out_dir: Path,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    manifest_id: str,
    planned_ledger: Mapping[str, object],
    execute_cleanup: Callable[[], _CleanupResultT],
    validate_published_handoff: Callable[[], None] | None = None,
) -> _CleanupResultT:
    """Publish an auditable recovery point before executing any deletion."""

    _write_raw_completion(out_dir, rows, manifest_id, cleanup_ledger=planned_ledger)
    if validate_published_handoff is not None:
        validate_published_handoff()
    return execute_cleanup()


def _raw_handoff_dir(raw_root: Path, frozen: FrozenPhase) -> Path:
    """Resolve the immutable active cleanup hand-off, or the legacy root."""

    pointer_path = raw_root / _ACTIVE_RAW_HANDOFF_FILE
    if not pointer_path.is_file():
        if (raw_root / "handoffs").exists():
            raise ValueError("versioned raw handoffs require an active pointer")
        return raw_root
    pointer = _json_mapping(pointer_path, label="active raw handoff")
    relative = pointer.get("handoff_dir")
    if (
        pointer.get("schema_version") != _ACTIVE_RAW_HANDOFF_SCHEMA
        or pointer.get("raw_execution_semantics_revision")
        != RAW_EXECUTION_SEMANTICS_REVISION
        or pointer.get("study_manifest_id") != frozen.study_manifest_id
        or pointer.get("authority_granted") is not False
        or pointer.get("proved_commute") is not False
        or not isinstance(relative, str)
        or not relative
    ):
        raise ValueError("active raw handoff pointer mismatch")
    candidate = (raw_root / relative).resolve(strict=False)
    try:
        candidate.relative_to((raw_root / "handoffs").resolve(strict=False))
    except ValueError as error:
        raise ValueError("active raw handoff escapes raw handoff directory") from error
    if not candidate.is_dir():
        raise ValueError("active raw handoff directory is missing")
    if not (candidate / "cleanup_ledger.json").is_file():
        raise ValueError("active versioned raw handoff is missing cleanup ledger")
    return candidate


def _raw_rows_from_complete(
    frozen: FrozenPhase,
    *,
    expected_checkpoint_index: Mapping[str, object] | None = None,
    expected_rows: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    materialized_artifact_bindings_out: list[dict[str, object]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    raw_dir = _raw_handoff_dir(frozen.out_dir / "raw", frozen)
    completion = _json_mapping(raw_dir / "complete.json", label="raw complete")
    if (
        completion.get("schema_version") != _RAW_COMPLETION_SCHEMA
        or completion.get("raw_execution_semantics_revision")
        != RAW_EXECUTION_SEMANTICS_REVISION
        or completion.get("study_manifest_id") != frozen.study_manifest_id
        or completion.get("authority_granted") is not False
        or completion.get("proved_commute") is not False
    ):
        raise ValueError("raw evidence completion mismatch")
    expected = completion.get("files_sha256")
    rows_path = raw_dir / "study_rows.json"
    if not isinstance(expected, Mapping) or expected.get("study_rows.json") != _sha256_file(rows_path):
        raise ValueError("raw evidence is not hash-validated")
    is_legacy_handoff = "cleanup_ledger.json" not in expected
    cleanup: Mapping[str, object] | None = None
    checkpoint_index: Mapping[str, object] | None = None
    checkpoint_payloads: Mapping[str, Mapping[str, object]] | None = None
    if materialized_artifact_bindings_out is not None and materialized_artifact_bindings_out:
        raise ValueError("materialized artifact binding sink must initially be empty")
    if not is_legacy_handoff:
        cleanup_path = raw_dir / "cleanup_ledger.json"
        has_checkpoint_index = "program_checkpoint_index.json" in expected
        expected_files = {"study_rows.json", "cleanup_ledger.json"}
        if has_checkpoint_index:
            expected_files.add("program_checkpoint_index.json")
        if (
            set(expected) != expected_files
            or not cleanup_path.is_file()
            or expected.get("cleanup_ledger.json") != _sha256_file(cleanup_path)
        ):
            raise ValueError("new cleanup raw evidence is not hash-validated")
        cleanup = _json_mapping(cleanup_path, label="cleanup ledger")
        summary = cleanup.get("summary")
        from .cleanup import CLEANUP_LEDGER_SCHEMA_VERSION
        if (
            cleanup.get("schema_version") != CLEANUP_LEDGER_SCHEMA_VERSION
            or cleanup.get("study_manifest_id") != frozen.study_manifest_id
            or cleanup.get("authority_granted") is not False
            or cleanup.get("proved_commute") is not False
            or cleanup.get("cleanup_state") not in {"planned", "complete"}
            or not isinstance(cleanup.get("entries"), list)
            or not isinstance(summary, Mapping)
            or any(
                not isinstance(summary.get(field), int)
                or isinstance(summary.get(field), bool)
                or int(summary[field]) < 0
                for field in (
                    "reclaimed_file_count",
                    "reclaimed_bytes",
                    "retained_file_count",
                    "retained_bytes",
                    "planned_file_count",
                    "planned_bytes",
                )
            )
        ):
            raise ValueError("new cleanup raw evidence has an invalid cleanup ledger")
        if has_checkpoint_index:
            index_path = raw_dir / "program_checkpoint_index.json"
            if (
                not index_path.is_file()
                or expected.get("program_checkpoint_index.json")
                != _sha256_file(index_path)
            ):
                raise ValueError("program checkpoint index is not hash-validated")
            checkpoint_index = _json_mapping(
                index_path, label="program checkpoint index"
            )
            pointer_path = frozen.out_dir / "raw" / _ACTIVE_RAW_HANDOFF_FILE
            pointer = _json_mapping(pointer_path, label="active raw handoff")
            if pointer.get("program_checkpoint_index_sha256") != expected.get(
                "program_checkpoint_index.json"
            ):
                raise ValueError("active raw handoff does not bind the checkpoint index")
            checkpoint_payloads = _validate_program_checkpoint_index(
                frozen, checkpoint_index
            )
        elif expected_checkpoint_index is not None:
            raise ValueError("active raw handoff is missing its program checkpoint index")
    elif expected_checkpoint_index is not None:
        raise ValueError("legacy raw handoff cannot satisfy a program checkpoint index")
    payload = _json_mapping(rows_path, label="raw study rows")
    if (
        payload.get("study_manifest_id") != frozen.study_manifest_id
        or payload.get("raw_execution_semantics_revision")
        != RAW_EXECUTION_SEMANTICS_REVISION
        or payload.get("authority_granted") is not False
        or payload.get("proved_commute") is not False
    ):
        raise ValueError("raw evidence manifest mismatch")
    tables = payload.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("raw evidence tables are missing")
    required = {
        "single_pass_observations.csv",
        "pair_observations.csv",
        "advisor_2n_group_results.csv",
        "advisor_2n_directional_results.csv",
        "advisor_2n_pair_validation.csv",
    }
    if set(tables) != required:
        raise ValueError("raw evidence is missing required tables or contains an unexpected table")
    output: dict[str, list[dict[str, object]]] = {}
    cleanup_fields: dict[str, frozenset[str]] = {
        "pair_observations.csv": frozenset({
            "ab_output_path", "ab_output_sha256", "ba_output_path",
            "ba_output_sha256", "ab_verifier_status", "ba_verifier_status",
            "cleanup_status",
        }),
        "advisor_2n_directional_results.csv": frozenset({
            "merged_input_path", "second_output_path", "second_output_sha256",
            "second_output_materialized", "cleanup_status",
        }),
    }
    legacy_cleanup_defaults: dict[str, dict[str, object]] = {
        "pair_observations.csv": {
            "ab_output_path": "",
            "ab_output_sha256": "",
            "ba_output_path": "",
            "ba_output_sha256": "",
            "ab_verifier_status": "unknown",
            "ba_verifier_status": "unknown",
            "cleanup_status": "legacy_uncompacted",
        },
        "advisor_2n_directional_results.csv": {
            "merged_input_path": "",
            "second_output_path": "",
            "second_output_sha256": "",
            "second_output_materialized": "unknown",
            "cleanup_status": "legacy_uncompacted",
        },
    }
    for name, values in tables.items():
        if not isinstance(name, str) or name not in TABLE_FIELDS or not isinstance(values, list):
            raise ValueError("raw evidence table shape is invalid")
        if not all(isinstance(row, Mapping) for row in values):
            raise ValueError("raw evidence rows must be mappings")
        rows = [dict(row) for row in values]
        for row in rows:
            if is_legacy_handoff:
                for field, default in legacy_cleanup_defaults.get(name, {}).items():
                    row.setdefault(field, default)
            else:
                missing_cleanup = cleanup_fields.get(name, frozenset()).difference(row)
                if missing_cleanup:
                    raise ValueError(
                        "new cleanup raw evidence missing persisted cleanup fields: "
                        + ",".join(sorted(missing_cleanup))
                    )
        identifiers = [str(row.get("row_id", "")) for row in rows]
        if any(not item for item in identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("raw evidence requires unique non-empty row_id values")
        for row in rows:
            if (
                str(row.get("study_manifest_id", "")) != frozen.study_manifest_id
                or str(row.get("authority_granted", "")).lower() != "false"
                or str(row.get("proved_commute", "")).lower() != "false"
            ):
                raise ValueError("raw evidence row manifest/authority binding mismatch")
        output[name] = rows
    if cleanup is not None:
        _validate_cleanup_ledger(cleanup, output, frozen.study_manifest_id)
    _validate_raw_coverage(output, frozen)
    if expected_checkpoint_index is not None:
        if checkpoint_index is None or json.dumps(
            checkpoint_index, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ) != json.dumps(
            dict(expected_checkpoint_index),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ):
            raise ValueError("active raw handoff checkpoint index differs from this run")
    if expected_rows is not None and json.dumps(
        output, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ) != json.dumps(
        {name: list(rows) for name, rows in expected_rows.items()},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ):
        raise ValueError("active raw handoff rows differ from complete program checkpoints")
    if materialized_artifact_bindings_out is not None:
        if checkpoint_payloads is None:
            raise ValueError(
                "materialized artifact bindings require a validated program checkpoint index"
            )
        for program_id in sorted(checkpoint_payloads):
            bindings = checkpoint_payloads[program_id].get(
                "materialized_artifact_bindings"
            )
            if not isinstance(bindings, list) or any(
                not isinstance(binding, Mapping) for binding in bindings
            ):
                raise ValueError(
                    f"program checkpoint artifact bindings are malformed: {program_id}"
                )
            materialized_artifact_bindings_out.extend(
                dict(binding) for binding in bindings
            )
    return output


def _validate_cleanup_ledger(
    ledger: Mapping[str, object], tables: Mapping[str, Sequence[Mapping[str, object]]],
    study_manifest_id: str,
) -> None:
    """Bind every cleanup entry exactly to one persisted raw evidence row."""

    entries = ledger.get("entries")
    summary = ledger.get("summary")
    if not isinstance(entries, list) or not isinstance(summary, Mapping):
        raise ValueError("cleanup ledger shape is invalid")
    expected_rows: dict[tuple[str, str], Mapping[str, object]] = {}
    for kind, table_name in (
        ("pair_ab_ba", "pair_observations.csv"),
        ("two_n_second_round", "advisor_2n_directional_results.csv"),
    ):
        for row in tables[table_name]:
            key = (kind, str(row.get("row_id", "")))
            if not key[1] or key in expected_rows:
                raise ValueError("cleanup ledger source rows are ambiguous")
            expected_rows[key] = row
    if len(entries) != len(expected_rows):
        raise ValueError("cleanup ledger coverage does not equal affected raw rows")
    seen_ids: set[str] = set()
    seen_rows: set[tuple[str, str]] = set()
    totals = {field: 0 for field in (
        "reclaimed_file_count", "reclaimed_bytes", "retained_file_count",
        "retained_bytes", "planned_file_count", "planned_bytes",
    )}
    cleanup_state = str(ledger.get("cleanup_state", ""))
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("cleanup ledger entry must be a mapping")
        cleanup_id = entry.get("cleanup_id")
        kind = str(entry.get("artifact_kind", ""))
        source_row_id = str(entry.get("source_row_id", ""))
        key = (kind, source_row_id)
        if not isinstance(cleanup_id, str) or len(cleanup_id) != 64 or cleanup_id in seen_ids:
            raise ValueError("cleanup ledger requires unique hash-shaped cleanup_id")
        if key not in expected_rows or key in seen_rows:
            raise ValueError("cleanup ledger entry does not bind exactly one affected raw row")
        row = expected_rows[key]
        identity = {
            "study_manifest_id": study_manifest_id,
            "artifact_kind": kind,
            "source_row_id": source_row_id,
            "group_id": str(row.get("group_id", "")),
            "program_id": str(row.get("program_id", "")),
            "action_id": str(row.get("action_id", "")),
        }
        expected_id = hashlib.sha256(
            json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if cleanup_id != expected_id or any(str(entry.get(field, "")) != str(value) for field, value in identity.items()):
            raise ValueError("cleanup ledger identity/hash binding mismatch")
        _validate_cleanup_entry(entry, row, kind, cleanup_state)
        for field in totals:
            value = entry.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("cleanup ledger entry summary field is invalid")
            totals[field] += value
        seen_ids.add(cleanup_id)
        seen_rows.add(key)
    if seen_rows != set(expected_rows):
        raise ValueError("cleanup ledger omits an affected raw row")
    if any(summary.get(field) != value for field, value in totals.items()):
        raise ValueError("cleanup ledger summary does not equal entry totals")


def _validate_cleanup_entry(
    entry: Mapping[str, object], row: Mapping[str, object], kind: str, cleanup_state: str,
) -> None:
    """Check status, path/hash and materialization consistency for one row."""

    artifacts = entry.get("artifacts")
    expected_names = ("AB", "BA") if kind == "pair_ab_ba" else ("second_round",)
    materialized_field = "artifact_materialized" if kind == "pair_ab_ba" else "second_output_materialized"
    path_fields = (("AB", "ab_output_path", "ab_output_sha256"), ("BA", "ba_output_path", "ba_output_sha256")) if kind == "pair_ab_ba" else (("second_round", "second_output_path", "second_output_sha256"),)
    if (
        str(entry.get("cleanup_status", "")) not in {"planned", "reclaimed", "retained"}
        or str(entry.get("row_cleanup_status", "")) != str(row.get("cleanup_status", ""))
        or str(entry.get("artifact_materialized", "")) != str(row.get(materialized_field, ""))
        or not isinstance(artifacts, list)
        or len(artifacts) != len(expected_names)
        or {str(item.get("name", "")) for item in artifacts if isinstance(item, Mapping)} != set(expected_names)
    ):
        raise ValueError("cleanup ledger entry status/materialization shape mismatch")
    by_name = {str(item["name"]): item for item in artifacts if isinstance(item, Mapping)}
    reclaimed_count = retained_count = reclaimed_bytes = retained_bytes = 0
    for name, path_field, sha_field in path_fields:
        item = by_name[name]
        required = {"original_path", "actual_path", "quarantine_path", "sha256", "size_bytes", "reclaimed"}
        if not required.issubset(item):
            raise ValueError("cleanup ledger artifact is incomplete")
        if (
            not isinstance(item["size_bytes"], int) or isinstance(item["size_bytes"], bool)
            or item["size_bytes"] < 0 or str(item["sha256"]) != str(row.get(sha_field, ""))
            or (str(entry.get("cleanup_status", "")) in {"planned", "reclaimed"} and not str(item["original_path"]))
        ):
            raise ValueError("cleanup ledger artifact path/hash binding mismatch")
        reclaimed = item["reclaimed"]
        if not isinstance(reclaimed, bool):
            raise ValueError("cleanup ledger artifact reclaimed flag is invalid")
        actual = str(item["actual_path"])
        if reclaimed:
            if actual:
                raise ValueError("reclaimed cleanup artifact still has an actual path")
            reclaimed_count += 1
            reclaimed_bytes += item["size_bytes"]
        else:
            retained_count += 1
            retained_bytes += item["size_bytes"]
            if actual != str(row.get(path_field, "")):
                raise ValueError("retained cleanup artifact path does not match raw row")
    if (
        entry.get("file_count") != len(expected_names)
        or entry.get("size_bytes") != reclaimed_bytes + retained_bytes
        or entry.get("reclaimed_file_count") != reclaimed_count
        or entry.get("reclaimed_bytes") != reclaimed_bytes
        or entry.get("retained_file_count") != retained_count
        or entry.get("retained_bytes") != retained_bytes
        or entry.get("planned_file_count") != (len(expected_names) if str(entry.get("cleanup_status")) == "planned" else 0)
        or entry.get("planned_bytes") != (reclaimed_bytes + retained_bytes if str(entry.get("cleanup_status")) == "planned" else 0)
    ):
        raise ValueError("cleanup ledger entry counters are inconsistent")
    status = str(entry.get("cleanup_status"))
    if kind == "pair_ab_ba" and status in {"planned", "reclaimed"} and (
        str(row.get("ab_status", "")) != "success"
        or str(row.get("ba_status", "")) != "success"
        or str(row.get("ab_verifier_status", "")) != "success"
        or str(row.get("ba_verifier_status", "")) != "success"
    ):
        raise ValueError("cleanup ledger pair verifier predicate is not persisted")
    if status == "planned":
        if cleanup_state != "planned" or str(row.get(materialized_field, "")) != "true" or any(bool(item["reclaimed"]) for item in by_name.values()):
            raise ValueError("planned cleanup entry is not recoverably materialized")
    elif status == "reclaimed":
        if cleanup_state != "complete" or str(row.get(materialized_field, "")) != "false" or reclaimed_count != len(expected_names):
            raise ValueError("reclaimed cleanup entry is inconsistent")
    elif not str(row.get("cleanup_status", "")).startswith("retained_"):
        raise ValueError("retained cleanup entry is inconsistent")


def _validate_raw_coverage(tables: Mapping[str, Sequence[Mapping[str, object]]], frozen: FrozenPhase) -> None:
    """Reject a partial raw hand-off before aggregate derivation starts."""

    expected_programs = frozen.program_count
    programs = set(frozen.program_ids)
    group_rows = tables["advisor_2n_group_results.csv"]
    observed_group_keys = {
        (str(row.get("program_id", "")), str(row.get("group_id", "")))
        for row in group_rows
    }
    if len(group_rows) != expected_programs * len(_GROUP_IDS) or any(
        not program or group not in _GROUP_IDS for program, group in observed_group_keys
    ) or len(observed_group_keys) != len(group_rows):
        raise ValueError("raw 2N group evidence does not cover every frozen program/group")
    if {program for program, _group in observed_group_keys} != programs:
        raise ValueError("raw 2N group evidence program coverage mismatch")
    if observed_group_keys != {(program, group) for program in programs for group in _GROUP_IDS}:
        raise ValueError("raw 2N group evidence exact set mismatch")
    for table_name, action_field, expected_count in (
        ("single_pass_observations.csv", "action_id", expected_programs * len(frozen.groups["Uall"])),
        ("advisor_2n_directional_results.csv", "action_id", expected_programs * sum(len(frozen.groups[group]) for group in _GROUP_IDS)),
    ):
        rows = tables[table_name]
        if len(rows) != expected_count or any(not str(row.get(action_field, "")) for row in rows):
            raise ValueError(f"raw {table_name} action coverage mismatch")
    single_keys = {(str(r.get("program_id")), str(r.get("action_id"))) for r in tables["single_pass_observations.csv"]}
    if single_keys != {(program, action) for program in programs for action in frozen.groups["Uall"]}:
        raise ValueError("raw single-pass evidence exact set mismatch")
    if any(str(row.get("group_id", "")) != "Uall" for row in tables["single_pass_observations.csv"]):
        raise ValueError("raw single-pass evidence group binding mismatch")
    directional_keys = {(str(r.get("program_id")), str(r.get("group_id")), str(r.get("action_id"))) for r in tables["advisor_2n_directional_results.csv"]}
    if directional_keys != {(program, group, action) for program in programs for group in _GROUP_IDS for action in frozen.groups[group]}:
        raise ValueError("raw directional evidence exact set mismatch")
    for table_name in ("pair_observations.csv", "advisor_2n_pair_validation.csv"):
        rows = tables[table_name]
        if any(
            not str(row.get("program_id", ""))
            or not str(row.get("action_a_id", ""))
            or not str(row.get("action_b_id", ""))
            for row in rows
        ):
            raise ValueError(f"raw {table_name} pair binding mismatch")
    for row in tables["pair_observations.csv"]:
        if str(row["program_id"]) not in programs or str(row["group_id"]) != "Uall" or {str(row["action_a_id"]), str(row["action_b_id"])} - set(frozen.groups["Uall"]):
            raise ValueError("raw Uall pair evidence exact binding mismatch")
    for row in tables["advisor_2n_pair_validation.csv"]:
        group = str(row.get("group_id", ""))
        if str(row["program_id"]) not in programs or group not in _GROUP_IDS or {str(row["action_a_id"]), str(row["action_b_id"])} - set(frozen.groups[group]):
            raise ValueError("raw 2N pair evidence exact binding mismatch")

    # Preserve the full frozen Uall matrix, including terminal first-round
    # actions.  AB/BA applies only when both endpoints succeeded, but a
    # terminal row is still evidence that the pair was considered and that its
    # second stage was correctly not run.
    single_statuses = {
        (str(row["program_id"]), str(row["action_id"])): str(row.get("execution_status", ""))
        for row in tables["single_pass_observations.csv"]
    }
    expected_uall_pairs = {
        (program, left, right)
        for program in programs
        for left, right in _unordered_pair_keys(frozen.groups["Uall"])
    }
    observed_uall_pairs = {
        (str(row["program_id"]), str(row["action_a_id"]), str(row["action_b_id"]))
        for row in tables["pair_observations.csv"]
    }
    if (
        len(tables["pair_observations.csv"]) != len(expected_uall_pairs)
        or observed_uall_pairs != expected_uall_pairs
    ):
        raise ValueError("raw Uall pair evidence exact set mismatch")
    for row in tables["pair_observations.csv"]:
        program_id = str(row["program_id"])
        action_a_id = str(row["action_a_id"])
        action_b_id = str(row["action_b_id"])
        a_status = str(row.get("a_status", ""))
        b_status = str(row.get("b_status", ""))
        expected_a_status = single_statuses[(program_id, action_a_id)]
        expected_b_status = single_statuses[(program_id, action_b_id)]
        if a_status != expected_a_status or b_status != expected_b_status:
            raise ValueError("raw Uall pair endpoint status mismatch")
        if expected_a_status == "success" and expected_b_status == "success":
            ab_status = str(row.get("ab_status", ""))
            ba_status = str(row.get("ba_status", ""))
            if ab_status not in {"success", "error", "timeout", "invalid"} or ba_status not in {
                "success",
                "error",
                "timeout",
                "invalid",
            }:
                raise ValueError("raw Uall successful AB/BA status mismatch")
            dynamic_result = str(row.get("dynamic_result", ""))
            if "timeout" in {ab_status, ba_status}:
                allowed_dynamic_results = {"timeout"}
            elif {"error", "invalid"} & {ab_status, ba_status}:
                allowed_dynamic_results = {"failed"}
            else:
                allowed_dynamic_results = {"commute", "order_sensitive", "unknown"}
            if dynamic_result not in allowed_dynamic_results:
                raise ValueError("raw Uall successful pair dynamic result mismatch")
            continue
        if str(row.get("ab_status", "")) != "not_run" or str(row.get("ba_status", "")) != "not_run":
            raise ValueError("raw Uall terminal AB/BA status mismatch")
        expected_terminal_result = (
            "timeout" if "timeout" in {expected_a_status, expected_b_status} else "failed"
        )
        if str(row.get("dynamic_result", "")) != expected_terminal_result:
            raise ValueError("raw Uall terminal dynamic result mismatch")

    # 2N must retain failed/no-op/timeout first-round actions and emits one
    # typed validation row for every configured unordered pair.  Its coverage
    # therefore uses each frozen group, not the successful-only AB/BA subset.
    expected_two_n_pairs = {
        (program, group, left, right)
        for program in programs
        for group in _GROUP_IDS
        for left, right in _unordered_pair_keys(frozen.groups[group])
    }
    observed_two_n_pairs = {
        (
            str(row["program_id"]),
            str(row["group_id"]),
            str(row["action_a_id"]),
            str(row["action_b_id"]),
        )
        for row in tables["advisor_2n_pair_validation.csv"]
    }
    if (
        len(tables["advisor_2n_pair_validation.csv"]) != len(expected_two_n_pairs)
        or observed_two_n_pairs != expected_two_n_pairs
    ):
        raise ValueError("raw 2N pair evidence exact set mismatch")


def _unordered_pair_keys(action_ids: Sequence[str] | set[str]) -> set[tuple[str, str]]:
    """Canonical unordered action keys; self and reversed rows never appear."""

    ordered = sorted({str(action_id) for action_id in action_ids})
    return {
        (ordered[left_index], ordered[right_index])
        for left_index in range(len(ordered))
        for right_index in range(left_index + 1, len(ordered))
    }


@contextmanager
def _study_run_writer_lock(out_dir: Path) -> Iterator[None]:
    """Hold one crash-released OS lock for the complete run/handoff lifecycle."""

    root = Path(out_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".study-run-writer.lock"
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
                f"study run writer lock is already held: {lock_path}"
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


def _csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _monitor_category_counts(payload: Mapping[str, object]) -> dict[str, dict[str, int]]:
    """Return small typed counters for the external wall-time monitor."""

    tables = payload.get("tables")
    if not isinstance(tables, Mapping):
        return {}
    categories = (
        ("single_pass_execution_status", "single_pass_observations.csv", "execution_status"),
        ("pair_dynamic_result", "pair_observations.csv", "dynamic_result"),
        ("two_n_group_authorization", "advisor_2n_group_results.csv", "group_authorization_status"),
        ("two_n_directional_status", "advisor_2n_directional_results.csv", "directional_status"),
        ("two_n_validation_status", "advisor_2n_pair_validation.csv", "validation_status"),
    )
    output: dict[str, dict[str, int]] = {}
    for category, table_name, field in categories:
        values = tables.get(table_name)
        if not isinstance(values, list):
            continue
        counts: dict[str, int] = {}
        for row in values:
            if not isinstance(row, Mapping):
                continue
            value = str(row.get(field, "")).strip() or "empty"
            counts[value] = counts.get(value, 0) + 1
        output[category] = dict(sorted(counts.items()))
    return output


def _publish_program_monitor_event(
    *,
    frozen: FrozenPhase,
    program_id: str,
    status: str,
    program_wall_time_budget_s: int,
    checkpoint_input_sha256: str,
    payload: Mapping[str, object] | None = None,
) -> None:
    if status not in {"start", "complete", "reused", "coverage_limitation"}:
        raise ValueError(f"unknown program monitor status: {status}")
    event = {
        "schema_version": "advisor-pair-scale-2n/current-program-v1",
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "study_manifest_id": frozen.study_manifest_id,
        "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
        "program_id": str(program_id),
        "status": status,
        "program_wall_time_budget_s": int(program_wall_time_budget_s),
        "checkpoint_input_sha256": str(checkpoint_input_sha256),
        "category_counts": _monitor_category_counts(payload or {}),
        "authority_granted": False,
        "proved_commute": False,
    }
    logs_dir = frozen.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    current = logs_dir / "current_program.json"
    staging = logs_dir / f".{current.name}.{uuid.uuid4().hex}.staging"
    _write_handoff_file(
        staging,
        json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
    )
    staging.replace(current)
    print(json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")), flush=True)


def _checkpoint_index_entry(
    *,
    frozen: FrozenPhase,
    program_id: str,
    root_ir_sha256: str,
    checkpoint_input_sha256: str,
    checkpoint_dir: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    from .program_runtime import PROGRAM_RUNTIME_IMPLEMENTATION_VERSION

    pointer = _json_mapping(checkpoint_dir / "active.json", label="program checkpoint pointer")
    ledger = payload.get("cleanup_ledger")
    staging_ledger = payload.get("unpublished_staging_cleanup")
    if (
        pointer.get("input_sha256") != checkpoint_input_sha256
        or pointer.get("checkpoint_state") != "complete"
        or not isinstance(ledger, Mapping)
        or ledger.get("cleanup_state") != "complete"
        or not isinstance(staging_ledger, Mapping)
        or staging_ledger.get("cleanup_state") != "complete"
    ):
        raise ValueError(f"program checkpoint selection is not complete: {program_id}")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError(f"program checkpoint cleanup ledger is malformed: {program_id}")
    cleanup_ids = sorted(str(entry.get("cleanup_id", "")) for entry in entries)
    if any(len(cleanup_id) != 64 for cleanup_id in cleanup_ids):
        raise ValueError(f"program checkpoint cleanup IDs are malformed: {program_id}")
    staging_entries = staging_ledger.get("entries")
    if not isinstance(staging_entries, list) or any(
        not isinstance(entry, Mapping) for entry in staging_entries
    ):
        raise ValueError(
            f"program checkpoint unpublished staging ledger is malformed: {program_id}"
        )
    staging_cleanup_ids = sorted(
        str(entry.get("cleanup_id", "")) for entry in staging_entries
    )
    if any(len(cleanup_id) != 64 for cleanup_id in staging_cleanup_ids):
        raise ValueError(
            f"program checkpoint unpublished staging cleanup IDs are malformed: {program_id}"
        )
    relative = checkpoint_dir.resolve(strict=False).relative_to(
        frozen.out_dir.resolve(strict=False)
    )
    result_sha256 = str(pointer.get("result_sha256", ""))
    version_id = str(pointer.get("version_id", ""))
    if len(result_sha256) != 64 or len(version_id) != 64:
        raise ValueError(f"program checkpoint content identity is malformed: {program_id}")
    return {
        "program_id": str(program_id),
        "root_ir_sha256": str(root_ir_sha256),
        "checkpoint_input_sha256": str(checkpoint_input_sha256),
        "checkpoint_dir": str(checkpoint_dir.resolve(strict=False)),
        "checkpoint_relative_path": relative.as_posix(),
        "checkpoint_version_id": version_id,
        "checkpoint_result_sha256": result_sha256,
        "cleanup_ledger_sha256": _sha256_bytes(
            json.dumps(ledger, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        ),
        "cleanup_ids": cleanup_ids,
        "unpublished_staging_cleanup_sha256": _sha256_bytes(
            json.dumps(
                staging_ledger,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "unpublished_staging_cleanup_ids": staging_cleanup_ids,
        "program_status": str(payload.get("program_status", "")),
        "limitation_kind": str(payload.get("limitation_kind", "")),
        "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
        "program_runtime_implementation_version": (
            PROGRAM_RUNTIME_IMPLEMENTATION_VERSION
        ),
        "authority_granted": False,
        "proved_commute": False,
    }


def _checkpoint_progress_payload(
    frozen: FrozenPhase, entries: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    from .program_runtime import PROGRAM_RUNTIME_IMPLEMENTATION_VERSION

    ordered = sorted((dict(entry) for entry in entries), key=lambda row: str(row["program_id"]))
    payload: dict[str, object] = {
        "schema_version": _PROGRAM_CHECKPOINT_INDEX_SCHEMA,
        "study_manifest_id": frozen.study_manifest_id,
        "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
        "program_runtime_implementation_version": (
            PROGRAM_RUNTIME_IMPLEMENTATION_VERSION
        ),
        "program_count": frozen.program_count,
        "completed_program_count": len(ordered),
        "frozen_program_ids": sorted(frozen.program_ids),
        "entries": ordered,
        "authority_granted": False,
        "proved_commute": False,
    }
    payload["progress_id"] = _sha256_bytes(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )
    return payload


def _load_program_checkpoint_progress(frozen: FrozenPhase) -> dict[str, object] | None:
    from .program_runtime import PROGRAM_RUNTIME_IMPLEMENTATION_VERSION

    path = frozen.out_dir / "raw" / _PROGRAM_CHECKPOINT_PROGRESS_FILE
    if not path.is_file():
        return None
    raw = _json_mapping(path, label="program checkpoint progress")
    entries = raw.get("entries")
    progress_body = {
        str(key): value for key, value in raw.items() if key != "progress_id"
    }
    if (
        raw.get("schema_version") != _PROGRAM_CHECKPOINT_INDEX_SCHEMA
        or raw.get("study_manifest_id") != frozen.study_manifest_id
        or raw.get("program_count") != frozen.program_count
        or raw.get("frozen_program_ids") != sorted(frozen.program_ids)
        or raw.get("authority_granted") is not False
        or raw.get("proved_commute") is not False
        or not isinstance(entries, list)
        or raw.get("completed_program_count") != len(entries)
        or raw.get("progress_id")
        != _sha256_bytes(
            json.dumps(
                progress_body,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    ):
        raise ValueError("program checkpoint progress envelope/self-hash mismatch")
    if (
        raw.get("program_runtime_implementation_version")
        != PROGRAM_RUNTIME_IMPLEMENTATION_VERSION
    ):
        # A runtime implementation revision changes checkpoint input identities.
        # Invalidate the old monotonic chain before reopening its now-obsolete
        # entries, so the first current-version result can start a clean chain.
        return None
    if (
        raw.get("raw_execution_semantics_revision") != RAW_EXECUTION_SEMANTICS_REVISION
    ):
        # A semantics revision deliberately invalidates all older checkpoint
        # identities.  Its new monotonic chain begins with the first new entry.
        return None
    _validate_program_checkpoint_index(frozen, raw, require_complete=False)
    return dict(raw)


def _publish_program_checkpoint_progress(
    frozen: FrozenPhase, entry: Mapping[str, object]
) -> dict[str, object]:
    existing = _load_program_checkpoint_progress(frozen)
    entries = [] if existing is None else list(existing["entries"])
    assert all(isinstance(item, Mapping) for item in entries)
    program_id = str(entry.get("program_id", ""))
    prior = next(
        (item for item in entries if str(item.get("program_id", "")) == program_id),
        None,
    )
    if prior is not None:
        if json.dumps(
            prior, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ) != json.dumps(
            dict(entry), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ):
            raise ValueError(
                f"program checkpoint progress cannot rewrite a completed entry: {program_id}"
            )
        assert existing is not None
        return existing
    entries.append(dict(entry))
    payload = _checkpoint_progress_payload(frozen, entries)
    _validate_program_checkpoint_index(frozen, payload, require_complete=False)
    raw_dir = frozen.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / _PROGRAM_CHECKPOINT_PROGRESS_FILE
    staging = raw_dir / f".{path.name}.{uuid.uuid4().hex}.staging"
    data = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    try:
        with staging.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    return payload


def _run_program_checkpoints(
    *,
    frozen: FrozenPhase,
    programs: Mapping[str, Path],
    program_families: Mapping[str, str],
    groups: Mapping[str, Sequence[object]],
    run_program: Callable[[str, Path], object],
    recover_program: Callable[[str, Path, object], object] | None = None,
) -> tuple[object, dict[str, object]]:
    """Run/resume one frozen program at a time and compact it before advancing."""

    from .program_runtime import (
        combine_program_results,
        finalize_program_evidence,
        load_program_checkpoint,
        plan_unpublished_staging_cleanup,
        program_checkpoint_input_sha256,
        program_checkpoint_path,
        program_result_payload,
        result_from_program_payload,
        runtime_budget_limited_result,
    )
    from .run_control import ensure_run_control, load_run_control

    expected_programs = set(frozen.program_ids)
    if set(programs) != expected_programs or set(program_families) != expected_programs:
        raise ValueError("program checkpoint loop requires the exact frozen program set")
    if set(groups) != set(_GROUP_IDS):
        raise ValueError("program checkpoint loop requires exactly U14, U30, and Uall")
    for group_id in _GROUP_IDS:
        action_ids = [
            str(
                action.get("action_id", "")
                if isinstance(action, Mapping)
                else getattr(action, "action_id", "")
            )
            for action in groups[group_id]
        ]
        if sorted(action_ids) != sorted(frozen.groups[group_id]):
            raise ValueError(f"program checkpoint group differs from frozen {group_id}")
    ensure_run_control(
        frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        program_ids=frozen.program_ids,
    )
    results = []
    index_entries: list[dict[str, object]] = []
    progress: dict[str, object] | None = None
    for program_id in sorted(frozen.program_ids):
        root = Path(programs[program_id]).resolve(strict=False)
        if not root.is_file():
            raise ValueError(f"frozen program root is missing: {program_id}")
        control = load_run_control(
            frozen.out_dir / "run_control.json",
            study_manifest_id=frozen.study_manifest_id,
            program_ids=frozen.program_ids,
        )
        decision = control.decision_for(program_id)
        root_sha256 = _sha256_file(root)
        input_sha256 = program_checkpoint_input_sha256(
            study_manifest_id=frozen.study_manifest_id,
            program_id=program_id,
            root_ir_sha256=root_sha256,
            group_action_ids=frozen.groups,
            runner_semantics_id=RAW_EXECUTION_SEMANTICS_REVISION,
        )
        checkpoint_dir = program_checkpoint_path(frozen.out_dir, input_sha256)
        loaded = load_program_checkpoint(
            checkpoint_dir,
            expected_input_sha256=input_sha256,
            isolation_root=frozen.out_dir,
        )
        if loaded is not None and loaded[1] == "complete":
            payload = loaded[0]
            status = "reused"
        else:
            _publish_program_monitor_event(
                frozen=frozen,
                program_id=program_id,
                status="start",
                program_wall_time_budget_s=control.program_wall_time_budget_s,
                checkpoint_input_sha256=input_sha256,
            )
            if loaded is not None:
                initial_payload = loaded[0]
            elif decision.decision == "skip":
                limited = (
                    recover_program(program_id, root, decision)
                    if recover_program is not None
                    else runtime_budget_limited_result(
                        out_dir=frozen.out_dir,
                        study_manifest_id=frozen.study_manifest_id,
                        program_id=program_id,
                        program_family=program_families[program_id],
                        groups=groups,
                        provenance=decision.provenance(),
                    )
                )
                staging_plan = plan_unpublished_staging_cleanup(
                    isolation_root=frozen.out_dir,
                    stage_paths=limited.stage_paths,  # type: ignore[attr-defined]
                )
                initial_payload = program_result_payload(
                    limited,
                    program_id=program_id,
                    program_status="coverage_limitation",
                    limitation_kind=decision.limitation_kind,
                    run_control_provenance={
                        **decision.provenance(),
                        "control_payload": dict(control.raw_payload),
                    },
                    unpublished_staging_cleanup=staging_plan,
                )
            else:
                executed = run_program(program_id, root)
                staging_plan = plan_unpublished_staging_cleanup(
                    isolation_root=frozen.out_dir,
                    stage_paths=executed.stage_paths,  # type: ignore[attr-defined]
                )
                initial_payload = program_result_payload(
                    executed,  # type: ignore[arg-type]
                    program_id=program_id,
                    program_status="complete",
                    limitation_kind="",
                    run_control_provenance={
                        **decision.provenance(),
                        "control_payload": dict(control.raw_payload),
                    },
                    unpublished_staging_cleanup=staging_plan,
                )
            finalize_program_evidence(
                checkpoint_dir,
                initial_payload,
                expected_input_sha256=input_sha256,
                isolation_root=frozen.out_dir,
            )
            reloaded = load_program_checkpoint(
                checkpoint_dir,
                expected_input_sha256=input_sha256,
                isolation_root=frozen.out_dir,
            )
            if reloaded is None or reloaded[1] != "complete":
                raise ValueError(f"program checkpoint did not finalize: {program_id}")
            payload = reloaded[0]
            status = (
                "coverage_limitation"
                if payload.get("program_status") == "coverage_limitation"
                else "complete"
            )
        result = result_from_program_payload(payload, out_dir=frozen.out_dir)
        results.append(result)
        entry = _checkpoint_index_entry(
            frozen=frozen,
            program_id=program_id,
            root_ir_sha256=root_sha256,
            checkpoint_input_sha256=input_sha256,
            checkpoint_dir=checkpoint_dir,
            payload=payload,
        )
        index_entries.append(entry)
        progress = _publish_program_checkpoint_progress(frozen, entry)
        _publish_program_monitor_event(
            frozen=frozen,
            program_id=program_id,
            status=status,
            program_wall_time_budget_s=control.program_wall_time_budget_s,
            checkpoint_input_sha256=input_sha256,
            payload=payload,
        )
    combined = combine_program_results(results)
    if progress is None:
        raise ValueError("program checkpoint progress was not published")
    checkpoint_index = progress
    expected_progress = _checkpoint_progress_payload(frozen, index_entries)
    if json.dumps(
        checkpoint_index, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ) != json.dumps(
        expected_progress, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ):
        raise ValueError("program checkpoint progress contains non-current entries")
    _validate_program_checkpoint_index(frozen, checkpoint_index)
    return combined, checkpoint_index


def _run_frozen(
    frozen: FrozenPhase,
    *,
    replay_dependency_factory: ReplayDependencyFactory | None = None,
) -> None:
    """Hold the run-wide writer lock across checkpoints and final handoff."""

    with _study_run_writer_lock(frozen.out_dir):
        _run_frozen_locked(
            frozen, replay_dependency_factory=replay_dependency_factory
        )


def _run_frozen_locked(
    frozen: FrozenPhase,
    *,
    replay_dependency_factory: ReplayDependencyFactory | None = None,
) -> None:
    """Run the complete Uall pair oracle plus each exact group 2N check."""

    from .direct_merge import DirectMergeClient, evaluate_group_2n
    from .orchestration import OrchestrationDependencies, run_study_orchestration
    from .pair_matrix import (
        profile_single_passes,
        run_complete_pair_matrix,
    )
    from .program_runtime import recover_runtime_budget_limited_program

    manifest = _manifest_for_frozen(frozen)
    actions = _load_actions(frozen.out_dir)
    if any(action_id not in actions for ids in frozen.groups.values() for action_id in ids):
        raise ValueError("frozen pass group action is absent from pass_inventory")
    tools = manifest["tools"]
    assert isinstance(tools, Mapping)
    opt = Path(str(dict(tools["opt"])["path"]))
    helper = Path(str(dict(tools["merge_helper"])["path"]))
    worker_path = Path(str(dict(tools["worker"])["path"]))
    programs_raw = manifest["program_manifest"]
    assert isinstance(programs_raw, list)
    programs = {
        str(row["program_id"]): Path(str(row["root_ir_path"]))
        for row in programs_raw
        if isinstance(row, Mapping)
    }
    root_hard_state_ids = {
        str(row["program_id"]): str(row.get("root_hard_state_id", ""))
        for row in programs_raw
        if isinstance(row, Mapping)
    }
    families = {
        str(row["program_id"]): str(row.get("program_family", ""))
        for row in programs_raw
        if isinstance(row, Mapping)
    }
    if len(programs) != frozen.program_count or any(not root.is_file() for root in programs.values()):
        raise ValueError("frozen program roots are missing or malformed")
    for program_id, root in programs.items():
        expected = root_hard_state_ids.get(program_id, "")
        if len(expected) != 64 or _phasebatch_hard_state_id(root) != expected:
            raise ValueError(f"frozen root hard-state identity mismatch: {program_id}")
    worker: _WorkerRunner | object | None = None
    pair_worker_pool: _WorkerRunnerPool | None = None
    merge_client: DirectMergeClient | None = None
    try:
        with ExitStack() as resource_stack:
            def ensure_runtime_resources() -> None:
                """Start real tools only when a checkpoint actually needs execution."""

                nonlocal worker, pair_worker_pool, merge_client
                if worker is not None or merge_client is not None:
                    if worker is None or merge_client is None:
                        raise RuntimeError("partial study runtime resource initialization")
                    return
                worker, pair_worker_pool = _create_worker_runners(
                    worker_path,
                    timeout_s=frozen.timeout_s,
                    jobs=frozen.jobs,
                )
                merge_client = resource_stack.enter_context(
                    DirectMergeClient((str(helper),), timeout_s=frozen.timeout_s)
                )
                # A successful helper ping is an explicit tool/protocol gate
                # immediately before the first real pair execution.
                merge_client.ping()

            def profile(root: Path, configured: tuple[object, ...], directory: Path) -> list[dict[str, object]]:
                typed = tuple(action for action in configured if isinstance(action, ActionRecord))
                if len(typed) != len(configured):
                    raise ValueError("Uall profile requires ActionRecord values")
                program_id = next(key for key, value in programs.items() if value.resolve() == root.resolve())
                rows = profile_single_passes(
                    root_ir=root,
                    actions=typed,
                    out_dir=directory,
                    run_single=worker.apply,
                    verify_ir=lambda path: _verify_with_opt(opt, frozen.timeout_s, path),
                    root_hard_state_id=root_hard_state_ids[program_id],
                    extract_observed_effect=lambda base, output, _action: {
                        "changed_functions": merge_client.inspect_patch(base, output).changed_functions,
                        "changed_blocks": (),
                        "changed_module_regions": (),
                    },
                    study_manifest_id=frozen.study_manifest_id,
                )
                return [
                    {**row, "program_id": program_id, "group_id": "Uall", "program_family": families[program_id], "wall_time_ms": 0}
                    for row in rows
                ]

            def pairs(root: Path, profiles: list[dict[str, object]], configured: dict[str, object], directory: Path) -> list[dict[str, object]]:
                typed = {key: value for key, value in configured.items() if isinstance(value, ActionRecord)}
                if len(typed) != len(configured):
                    raise ValueError("Uall pair oracle requires ActionRecord values")
                program_id = next(key for key, value in programs.items() if value.resolve() == root.resolve())
                pair_apply = (
                    worker.apply
                    if pair_worker_pool is None
                    else pair_worker_pool.apply
                )
                try:
                    rows = run_complete_pair_matrix(
                        root_ir=root,
                        profiles=profiles,
                        actions=typed,
                        out_dir=directory,
                        profile_artifact_root=Path(str(profiles[0]["output_path"])).resolve().parent.parent,
                        run_second=pair_apply,
                        verify_ir=lambda path: _verify_with_opt(opt, frozen.timeout_s, path),
                        compare=lambda ab, ba: _compare_phasebatch_hard_states(
                            ab,
                            ba,
                            opt=opt,
                            timeout_s=frozen.timeout_s,
                        ),
                        jobs=frozen.jobs,
                        timeout=frozen.timeout_s,
                        study_manifest_id=frozen.study_manifest_id,
                        program_id=program_id,
                        group_id="Uall",
                    )
                finally:
                    if pair_worker_pool is not None:
                        pair_worker_pool.release_thread_bindings()
                return [{**row, "program_family": families[program_id], "cache_reused": "false", "wall_time_ms": 0} for row in rows]

            def two_n(root: Path, group_id: str, configured: dict[str, object], profiles: list[dict[str, object]], directory: Path, pair_view: list[dict[str, object]]) -> object:
                program_id = next(key for key, value in programs.items() if value.resolve() == root.resolve())
                return evaluate_group_2n(
                    root_ir=root,
                    group_id=group_id,
                    program_id=program_id,
                    study_manifest_id=frozen.study_manifest_id,
                    actions=configured,
                    profiles=profiles,
                    merge_client=merge_client,
                    out_dir=directory,
                    run_second=worker.apply,
                    verify_ir=lambda path: _verify_with_opt(opt, frozen.timeout_s, path),
                    pair_observations=pair_view,
                )

            def external_apply(parent: Path, action: ActionRecord, output: Path) -> dict[str, object]:
                return _external_replay_apply(
                    opt, frozen.timeout_s, parent, action, output
                )

            def replay_common(
                case: dict[str, object],
                repetition: int,
                directory: Path,
                *,
                use_external: bool,
                family: str = "",
            ) -> Mapping[str, object]:
                """Replay one AB/BA witness with a real Worker or external opt."""

                pair = case["advisor_pair_row"]
                assert isinstance(pair, Mapping)
                left_id, right_id = str(pair["action_a_id"]), str(pair["action_b_id"])
                left, right = actions[left_id], actions[right_id]
                root = programs[str(case["program_id"])]
                runner = external_apply if use_external else worker.apply
                authorized_id = str(case.get("authorized_action_id", ""))
                expected_field = (
                    "action_a_directional_status"
                    if authorized_id == left_id
                    else "action_b_directional_status"
                    if authorized_id == right_id
                    else ""
                )
                if (
                    not expected_field
                    or str(pair.get(expected_field, ""))
                    != "authorized_all_others"
                ):
                    raise ValueError(
                        "replay case is not bound to one authorized Pi direction"
                    )
                return _build_pair_only_replay_record(
                    root=root,
                    left=left,
                    right=right,
                    directory=directory,
                    runner=runner,
                    family=(
                        family
                        or ("external_opt" if use_external else "worker")
                    ),
                    repetition=repetition,
                )

            def replay_two_n(case: dict[str, object], repetition: int, directory: Path) -> Mapping[str, object]:
                """Run the exact group 2N evaluator again for a false-auth case."""

                base = dict(
                    replay_common(
                        case,
                        repetition,
                        directory,
                        use_external=False,
                        family="two_n",
                    )
                )
                pair = case["advisor_pair_row"]
                assert isinstance(pair, Mapping)
                group_id = str(case["group_id"])
                root = programs[str(case["program_id"])]
                group_actions = {action_id: actions[action_id] for action_id in frozen.groups[group_id]}
                profiles = profile(root, tuple(group_actions.values()), directory / "two_n_profiles")
                evaluation = evaluate_group_2n(
                    root_ir=root,
                    group_id=group_id,
                    program_id=str(case["program_id"]),
                    study_manifest_id=frozen.study_manifest_id,
                    actions=group_actions,
                    profiles=profiles,
                    merge_client=merge_client,
                    out_dir=directory / "two_n",
                    run_second=worker.apply,
                    verify_ir=lambda path: _verify_with_opt(opt, frozen.timeout_s, path),
                    pair_observations=[dict(case["pair_observation"])],
                )
                matching = [
                    row for row in evaluation.pair_rows
                    if str(row["action_a_id"]) == str(pair["action_a_id"])
                    and str(row["action_b_id"]) == str(pair["action_b_id"])
                ]
                if len(matching) != 1:
                    base["status"] = "error"
                    base["stderr"] = str(base["stderr"]) + "\n2N replay pair binding unavailable"
                    return base
                two_n_result = {
                    field: str(matching[0][field])
                    for field in ("two_n_pair_status", "action_a_directional_status", "action_b_directional_status")
                }
                authorized_id = str(case.get("authorized_action_id", ""))
                expected_field = (
                    "action_a_directional_status"
                    if authorized_id == str(pair["action_a_id"])
                    else "action_b_directional_status"
                    if authorized_id == str(pair["action_b_id"])
                    else ""
                )
                if not expected_field or str(matching[0].get(expected_field, "")) != "authorized_all_others":
                    base["status"] = "error"
                    base["stderr"] = str(base["stderr"]) + "\n2N replay case lost its bound authorized direction"
                    return base
                matching_directionals = [
                    row
                    for row in evaluation.directional_rows
                    if str(row.get("action_id", "")) == authorized_id
                ]
                if len(matching_directionals) != 1:
                    base["status"] = "error"
                    base["stderr"] = str(base["stderr"]) + "\n2N replay directional binding unavailable"
                    return base
                directional = matching_directionals[0]
                two_n_result.update(
                    {
                        field: str(directional.get(field, ""))
                        for field in (
                            "directional_status",
                            "first_round_effect_sha256",
                            "second_round_effect_sha256",
                            "second_output_sha256",
                        )
                    }
                )
                artifact_identities = _bind_two_n_replay_artifacts(
                    base, directional, directory
                )
                if artifact_identities is None:
                    return base
                two_n_result.update(artifact_identities)
                base["two_n_result"] = two_n_result
                return base

            groups = {group: tuple(actions[action_id] for action_id in ids) for group, ids in frozen.groups.items()}
            replay_callbacks = {
                "worker": lambda case, repetition, directory: replay_common(case, repetition, directory, use_external=False),
                "external_opt": lambda case, repetition, directory: replay_common(case, repetition, directory, use_external=True),
                "two_n": replay_two_n,
            }
            replay_callbacks = _build_replay_dependencies(replay_callbacks, replay_dependency_factory)
            dependencies = OrchestrationDependencies(
                profile_uall=profile,
                run_uall_pairs=pairs,
                run_group_two_n=two_n,
                replay_worker=replay_callbacks["worker"],
                replay_external_opt=replay_callbacks["external_opt"],
                replay_two_n=replay_callbacks["two_n"],
            )

            def run_program(program_id: str, root: Path) -> object:
                ensure_runtime_resources()
                return run_study_orchestration(
                    out_dir=frozen.out_dir,
                    isolation_root=EXPERIMENT_ROOT,
                    study_manifest_id=frozen.study_manifest_id,
                    programs={program_id: root},
                    groups=groups,
                    dependencies=dependencies,
                )

            def lazy_replay(callback: ReplayCallback) -> ReplayCallback:
                def invoke(
                    case: dict[str, object], repetition: int, directory: Path
                ) -> Mapping[str, object]:
                    ensure_runtime_resources()
                    return callback(case, repetition, directory)

                return invoke

            recovery_replays = {
                name: lazy_replay(callback)
                for name, callback in replay_callbacks.items()
            }

            def recover_program(
                program_id: str, root: Path, decision: object
            ) -> object:
                provenance = getattr(decision, "provenance")
                return recover_runtime_budget_limited_program(
                    out_dir=frozen.out_dir,
                    isolation_root=EXPERIMENT_ROOT,
                    study_manifest_id=frozen.study_manifest_id,
                    program_id=program_id,
                    root_ir=root,
                    program_family=families[program_id],
                    groups=groups,
                    provenance=provenance(),
                    replay_worker=recovery_replays["worker"],
                    replay_external_opt=recovery_replays["external_opt"],
                    replay_two_n=recovery_replays["two_n"],
                )

            result, checkpoint_index = _run_program_checkpoints(
                frozen=frozen,
                programs=programs,
                program_families=families,
                groups=groups,
                run_program=run_program,
                recover_program=recover_program,
            )
    finally:
        if pair_worker_pool is not None:
            pair_worker_pool.close()
        elif worker is not None:
            worker.close()
    rows = _raw_rows_from_compacted_result(result)
    cleanup_ledger = _cleanup_ledger_from_checkpoint_index(
        frozen, checkpoint_index=checkpoint_index
    )
    active_pointer = frozen.out_dir / "raw" / _ACTIVE_RAW_HANDOFF_FILE
    if not active_pointer.is_file():
        _publish_cleanup_handoff(
            frozen.out_dir,
            rows,
            frozen.study_manifest_id,
            cleanup_ledger,
            checkpoint_index=checkpoint_index,
        )
    # An existing active handoff is never an unconditional fast path.  Re-open
    # every selected checkpoint and compare the merged rows byte-for-byte at
    # the canonical JSON level before accepting it.
    _raw_rows_from_complete(
        frozen,
        expected_checkpoint_index=checkpoint_index,
        expected_rows=rows,
    )


def _raw_rows_from_compacted_result(
    result: object,
) -> dict[str, list[dict[str, object]]]:
    """Project already-compacted checkpoint results; this function never deletes."""

    profile_rows = getattr(result, "profile_rows")
    pair_views = getattr(result, "pair_views")
    two_n_results = getattr(result, "two_n_results")
    return {
        "single_pass_observations.csv": [
            _project_row("single_pass_observations.csv", row)
            for program_id in sorted(profile_rows)
            for row in profile_rows[program_id]
        ],
        "pair_observations.csv": [
            _project_row("pair_observations.csv", row)
            for row in pair_views["Uall"]
        ],
        "advisor_2n_group_results.csv": [
            _project_row("advisor_2n_group_results.csv", row)
            for group in _GROUP_IDS
            for row in two_n_results[group]["group_rows"]
        ],
        "advisor_2n_directional_results.csv": [
            _project_row("advisor_2n_directional_results.csv", row)
            for group in _GROUP_IDS
            for row in two_n_results[group]["directional_rows"]
        ],
        "advisor_2n_pair_validation.csv": [
            _project_row("advisor_2n_pair_validation.csv", row)
            for group in _GROUP_IDS
            for row in two_n_results[group]["pair_rows"]
        ],
    }


def _raw_rows_for_cleanup(result: object, cleanup: object) -> dict[str, Sequence[Mapping[str, object]]]:
    """Project an orchestration result with one planned or completed cleanup."""

    pair_rows = getattr(cleanup, "pair_rows")
    directional_rows_by_group = getattr(cleanup, "directional_rows_by_group")
    profile_rows = getattr(result, "profile_rows")
    two_n_results = getattr(result, "two_n_results")
    return {
        "single_pass_observations.csv": [
            _project_row("single_pass_observations.csv", row)
            for rows in profile_rows.values() for row in rows
        ],
        "pair_observations.csv": [
            _project_row("pair_observations.csv", row)
            for row in pair_rows
        ],
        "advisor_2n_group_results.csv": [
            _project_row("advisor_2n_group_results.csv", row)
            for group in _GROUP_IDS for row in two_n_results[group]["group_rows"]
        ],
        "advisor_2n_directional_results.csv": [
            _project_row("advisor_2n_directional_results.csv", row)
            for group in _GROUP_IDS for row in directional_rows_by_group[group]
        ],
        "advisor_2n_pair_validation.csv": [
            _project_row("advisor_2n_pair_validation.csv", row)
            for group in _GROUP_IDS for row in two_n_results[group]["pair_rows"]
        ],
    }


def _resume_planned_cleanup(
    frozen: FrozenPhase, tables: Mapping[str, Sequence[Mapping[str, object]]],
    ledger: Mapping[str, object],
) -> None:
    """Complete a pre-published cleanup after a process/publish interruption."""

    from .cleanup import cleanup_journals_resolved, compact_intermediate_artifacts

    protected_pairs_raw = ledger.get("protected_pair_row_ids", ())
    protected_directionals_raw = ledger.get("protected_directionals", ())
    if (
        not isinstance(protected_pairs_raw, list)
        or not isinstance(protected_directionals_raw, list)
        or any(not isinstance(value, str) for value in protected_pairs_raw)
        or any(not isinstance(value, list) or len(value) != 3 or any(not isinstance(part, str) for part in value) for value in protected_directionals_raw)
    ):
        raise ValueError("planned cleanup protection binding is malformed")
    cleanup = compact_intermediate_artifacts(
        isolation_root=frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        pair_rows=tables["pair_observations.csv"],
        directional_rows_by_group={
            group: tuple(
                row for row in tables["advisor_2n_directional_results.csv"]
                if str(row.get("group_id", "")) == group
            )
            for group in _GROUP_IDS
        },
        false_authorizations=(),
        protected_pair_ids=protected_pairs_raw,
        protected_directionals=[tuple(value) for value in protected_directionals_raw],
        planned_ledger=ledger,
    )
    if not cleanup_journals_resolved(isolation_root=frozen.out_dir, ledger=cleanup.ledger):
        raise RuntimeError("cleanup journal remains prepared; planned handoff is recoverable")
    final_rows = {name: list(rows) for name, rows in tables.items()}
    final_rows["pair_observations.csv"] = [
        _project_row("pair_observations.csv", row) for row in cleanup.pair_rows
    ]
    final_rows["advisor_2n_directional_results.csv"] = [
        _project_row("advisor_2n_directional_results.csv", row)
        for group in _GROUP_IDS for row in cleanup.directional_rows_by_group[group]
    ]
    # Resolve journal/tombstones first, then publish a final versioned pointer;
    # do not fall back to the legacy root-level completion writer.
    _publish_cleanup_handoff(
        frozen.out_dir,
        final_rows,
        frozen.study_manifest_id,
        cleanup.ledger,
    )


def _summarize_frozen(frozen: FrozenPhase) -> None:
    """Derive deterministic Chinese aggregate/report artifacts from raw rows."""

    from .aggregate import materialize_aggregate

    # The manifest and raw evidence are independently self-hashed.  No pair
    # runner, Worker, compiler, or merge helper is constructed in this phase.
    _manifest_for_frozen(frozen)
    materialized_artifact_bindings: list[dict[str, object]] = []
    tables: dict[str, Sequence[Mapping[str, object]]] = _raw_rows_from_complete(
        frozen,
        materialized_artifact_bindings_out=materialized_artifact_bindings,
    )
    for table_name in ("program_manifest.csv", "pass_inventory.csv", "pass_preflight.csv", "pass_groups.csv"):
        tables[table_name] = _csv_rows(frozen.out_dir / table_name)
    materialize_aggregate(
        out_dir=frozen.out_dir,
        study_manifest_id=frozen.study_manifest_id,
        group_actions=frozen.groups,
        tables=tables,
        materialized_artifact_bindings=materialized_artifact_bindings,
        isolation_root=EXPERIMENT_ROOT,
        program_count=frozen.program_count,
    )


def _run_process(command: Sequence[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            tuple(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("subprocess timed out") from error


def _source_entries(source_manifest: Path, single_source_root: Path) -> list[tuple[str, Path, str]]:
    """Read the frozen existing-50 selection without observing experiment data."""

    try:
        raw = yaml.safe_load(source_manifest.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"source manifest is unreadable: {source_manifest}") from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get("benchmarks"), list):
        raise ValueError("source manifest must contain a benchmarks list")
    source_root = single_source_root.resolve()
    if source_root.name.casefold() != "singlesource":
        candidate = source_root / "SingleSource"
        if not candidate.is_dir():
            raise ValueError("--single-source-root must be SingleSource or contain SingleSource")
        source_root = candidate
    entries: list[tuple[str, Path, str]] = []
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for item in raw["benchmarks"]:
        if not isinstance(item, Mapping):
            raise ValueError("source manifest benchmarks must be mappings")
        name, relative = item.get("name"), item.get("path")
        if not isinstance(name, str) or not name.strip() or not isinstance(relative, str):
            raise ValueError("source manifest benchmark requires name and path")
        normalized = relative.replace("\\", "/")
        if not normalized.startswith("SingleSource/") or not normalized.endswith(".c"):
            raise ValueError("source manifest benchmark path must be a SingleSource C path")
        source = (source_root.parent / normalized).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as error:
            raise ValueError("source manifest benchmark escapes SingleSource") from error
        if not source.is_file() or source.suffix.casefold() != ".c":
            raise ValueError(f"source manifest benchmark is missing: {source}")
        if name in seen_names or source in seen_paths:
            raise ValueError("source manifest has duplicate benchmark identity")
        seen_names.add(name)
        seen_paths.add(source)
        entries.append((name, source, normalized))
    return entries


@dataclass(frozen=True)
class _SourceInventoryEntry:
    """One deterministic row from the approved SingleSource ``**/*.c`` scan."""

    program_id: str
    source_path: Path
    relative_path: str


@dataclass(frozen=True)
class _CandidatePreflightInventory:
    records: tuple[ProgramRecord, ...]
    exclusions: tuple[dict[str, object], ...]
    inventory_count: int


def _single_source_root(path: Path) -> Path:
    root = Path(path).resolve()
    if root.name.casefold() == "singlesource":
        return root
    nested = root / "SingleSource"
    if nested.is_dir():
        return nested.resolve()
    raise ValueError("--single-source-root must be SingleSource or contain SingleSource")


def _safe_inventory_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or "benchmark"


def _require_resolved_single_source_path(
    single_source_root: Path, candidate: Path
) -> Path:
    """Resolve one source-tree entry and reject reparse/symlink escapes."""

    root = Path(single_source_root).resolve(strict=True)
    resolved = Path(candidate).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"path escapes resolved SingleSource root: {resolved}"
        ) from error
    return resolved


def _walk_single_source_paths(single_source_root: Path) -> tuple[Path, ...]:
    """Walk resolved in-tree C files without following an escaping entry."""

    root = Path(single_source_root).resolve(strict=True)
    pending = [root]
    visited_directories: set[Path] = set()
    seen_sources: set[Path] = set()
    sources: list[Path] = []
    while pending:
        directory = pending.pop()
        if directory in visited_directories:
            continue
        visited_directories.add(directory)
        with os.scandir(directory) as stream:
            entries = sorted(stream, key=lambda entry: entry.name.casefold())
        for entry in entries:
            resolved = _require_resolved_single_source_path(root, Path(entry.path))
            # Containment is established before either file/dir metadata is
            # followed.  This is the critical junction/reparse fail-closed
            # boundary.
            if resolved.is_dir():
                pending.append(resolved)
            elif (
                resolved.suffix.lower() == ".c"
                and resolved.is_file()
                and resolved not in seen_sources
            ):
                seen_sources.add(resolved)
                sources.append(resolved)
    return tuple(sources)


def _scan_single_source_inventory(single_source_root: Path) -> tuple[_SourceInventoryEntry, ...]:
    """Mirror ``advisor_benchmarks`` scan/naming without running a compiler."""

    root = _single_source_root(single_source_root)
    # Recheck walker output here as a defence-in-depth seam: even a replaced
    # walker implementation cannot hand inventory an already resolved escape.
    sources = sorted(
        (
            _require_resolved_single_source_path(root, path)
            for path in _walk_single_source_paths(root)
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    name_counts: dict[str, int] = {}
    rows: list[_SourceInventoryEntry] = []
    for source in sources:
        base = _safe_inventory_name(source.stem)
        key = base.casefold()
        name_counts[key] = name_counts.get(key, 0) + 1
        program_id = base if name_counts[key] == 1 else f"{base}_{name_counts[key]}"
        rows.append(
            _SourceInventoryEntry(
                program_id=program_id,
                source_path=source,
                relative_path=(Path("SingleSource") / source.relative_to(root)).as_posix(),
            )
        )
    return tuple(rows)


def _llvm_identity_from_ir(path: Path) -> tuple[str, str]:
    target = ""
    data_layout = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("target triple ="):
            target = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("target datalayout ="):
            data_layout = stripped.split("=", 1)[1].strip().strip('"')
        if target and data_layout:
            break
    return target, data_layout


def _bound_root_target_identity(
    experiment_manifest: Mapping[str, object], root_ir: Path
) -> tuple[str, str]:
    """Cross-bind the old experiment target record to its retained root IR."""

    target_record = experiment_manifest.get("target")
    if not isinstance(target_record, Mapping):
        raise ValueError("existing root-only target identity is missing")
    manifest_target = str(target_record.get("triple", "")).strip()
    manifest_layout = str(target_record.get("data_layout", "")).strip()
    ir_target, ir_layout = _llvm_identity_from_ir(Path(root_ir))
    if not manifest_target or not manifest_layout or not ir_target or not ir_layout:
        raise ValueError("existing root-only target identity is incomplete")
    if (manifest_target, manifest_layout) != (ir_target, ir_layout):
        raise ValueError("existing root-only target identity mismatch")
    return manifest_target, manifest_layout


def _preflight_candidate_records(
    entries: Sequence[_SourceInventoryEntry],
    *,
    fixed_programs: Sequence[ProgramRecord],
    single_source_root: Path,
    preflight_root: Path,
    compile_source: Callable[[Path, Path], RunResult],
    verify_ir: Callable[[Path], bool],
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> _CandidatePreflightInventory:
    """Compile-preflight every scanned candidate before any pass observation.

    This deliberately follows the existing ``advisor_benchmarks`` policy:
    every C candidate at or below the frozen size limit is compiled, failures
    remain explicit rows, and selection happens only after the complete scan.
    Exact copies of the fixed 50 reuse their already bound root records.
    """

    if max_source_bytes < 1:
        raise ValueError("max_source_bytes must be positive")
    root = _single_source_root(single_source_root)
    fixed_by_source: dict[Path, ProgramRecord] = {}
    for program in fixed_programs:
        source = _require_resolved_single_source_path(
            root, Path(program.source_path)
        )
        if source.suffix.lower() != ".c" or not source.is_file():
            raise ValueError(f"fixed source is not an existing C file: {source}")
        expected_relative = (
            Path("SingleSource") / source.relative_to(root)
        ).as_posix()
        if program.relative_path != expected_relative:
            raise ValueError(f"fixed source path drift: {program.relative_path}")
        if source in fixed_by_source:
            raise ValueError(f"fixed source path is ambiguous: {source}")
        fixed_by_source[source] = program

    validated_entries: list[_SourceInventoryEntry] = []
    seen_sources: set[Path] = set()
    for entry in entries:
        source = _require_resolved_single_source_path(root, entry.source_path)
        if source.suffix.lower() != ".c" or not source.is_file():
            raise ValueError(f"candidate source is not an existing C file: {source}")
        expected_relative = (
            Path("SingleSource") / source.relative_to(root)
        ).as_posix()
        try:
            program_id = normalize_program_id(entry.program_id)
            relative = normalize_program_relative_path(entry.relative_path)
        except ValueError as error:
            raise ValueError(f"candidate inventory identity is invalid: {error}") from error
        if relative != entry.relative_path or relative != expected_relative:
            raise ValueError(f"candidate source path drift: {entry.relative_path}")
        if source in seen_sources:
            raise ValueError(f"candidate source path is ambiguous: {source}")
        seen_sources.add(source)
        validated_entries.append(
            _SourceInventoryEntry(
                program_id=program_id,
                source_path=source,
                relative_path=relative,
            )
        )

    # Every candidate/fixed resolved path has passed the containment boundary
    # before the first source byte is read or the first compiler can run.
    entries = tuple(validated_entries)
    source_bytes_by_path = {
        entry.source_path: entry.source_path.read_bytes()
        for entry in entries
    }
    digest_by_path = {
        path: _sha256_bytes(payload) for path, payload in source_bytes_by_path.items()
    }
    entries_by_digest: dict[str, list[_SourceInventoryEntry]] = {}
    for entry in entries:
        source = entry.source_path.resolve()
        entries_by_digest.setdefault(digest_by_path[source], []).append(entry)

    canonical_by_digest: dict[str, _SourceInventoryEntry] = {}
    for digest, duplicates in entries_by_digest.items():
        fixed_matches = [
            entry
            for entry in duplicates
            if entry.source_path.resolve() in fixed_by_source
        ]
        if len(fixed_matches) > 1:
            raise ValueError(f"fixed source SHA is ambiguous: {digest}")
        canonical_by_digest[digest] = (
            fixed_matches[0]
            if fixed_matches
            else min(
                duplicates,
                key=lambda entry: (
                    stable_rank(DEFAULT_SELECTION_SEED, entry.relative_path),
                    entry.relative_path.casefold(),
                ),
            )
        )

    exclusions: list[dict[str, object]] = []
    for digest, duplicates in entries_by_digest.items():
        canonical = canonical_by_digest[digest]
        for entry in duplicates:
            if entry == canonical:
                continue
            source = entry.source_path.resolve()
            exclusions.append(
                {
                    "program_id": entry.program_id,
                    "relative_path": entry.relative_path,
                    "source_path": str(source),
                    "source_sha256": digest,
                    "source_size_bytes": len(source_bytes_by_path[source]),
                    "stable_rank": stable_rank(
                        DEFAULT_SELECTION_SEED, entry.relative_path
                    ),
                    "canonical_program_id": canonical.program_id,
                    "canonical_relative_path": canonical.relative_path,
                    "canonical_source_path": str(canonical.source_path.resolve()),
                    "canonical_source_sha256": digest,
                    "canonical_stable_rank": stable_rank(
                        DEFAULT_SELECTION_SEED, canonical.relative_path
                    ),
                    "reason": "duplicate_source_sha256",
                }
            )
    records: list[ProgramRecord] = []
    for index, entry in enumerate(entries):
        source = entry.source_path.resolve()
        source_bytes = source_bytes_by_path[source]
        source_digest = digest_by_path[source]
        if canonical_by_digest[source_digest] != entry:
            continue
        fixed = fixed_by_source.get(source)
        if fixed is not None:
            if (
                fixed.relative_path != entry.relative_path
                or fixed.source_sha256 != _sha256_file(source)
            ):
                raise ValueError(f"fixed source identity drift: {entry.relative_path}")
            records.append(
                replace(
                    fixed,
                    selection_class="candidate",
                    selection_order=None,
                    reserve_rank=None,
                )
            )
            continue

        output = (
            Path(preflight_root)
            / f"{index:05d}_{_safe_inventory_name(entry.program_id)}"
            / "input.ll"
        ).resolve()
        compile_status = "skipped"
        preflight_status = "source_too_large"
        command: tuple[str, ...] = ("not-run", "source_too_large", str(source))
        stderr = ""
        root_sha256 = ""
        hard_state_id = ""
        target = "x86_64-w64-windows-gnu"
        data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

        if len(source_bytes) <= max_source_bytes:
            try:
                result = compile_source(source, output)
            except Exception as error:  # Candidate rows survive tool failures.
                compile_status = "failed"
                preflight_status = "compile_exception"
                command = ("compile-exception", str(source))
                stderr = " ".join(str(error).split())
            else:
                command = tuple(str(part) for part in result.command) or (
                    "compile",
                    str(source),
                )
                stderr = str(result.stderr)
                if result.timed_out:
                    compile_status = "timeout"
                    preflight_status = "compile_timeout"
                elif not result.success or not output.is_file():
                    compile_status = "failed"
                    preflight_status = "compile_failed"
                elif not verify_ir(output):
                    compile_status = "failed"
                    preflight_status = "root_ir_invalid"
                else:
                    observed_target, observed_layout = _llvm_identity_from_ir(output)
                    if not observed_target or not observed_layout:
                        compile_status = "failed"
                        preflight_status = "root_ir_identity_missing"
                    elif observed_target != target:
                        compile_status = "failed"
                        preflight_status = "target_mismatch"
                        target = observed_target
                        data_layout = observed_layout
                    else:
                        compile_status = "success"
                        preflight_status = "success"
                        data_layout = observed_layout
                        root_sha256 = _sha256_file(output)
                        hard_state_id = (
                            str(result.hard_state_id).strip()
                            or _phasebatch_hard_state_id(output)
                        )

        records.append(
            ProgramRecord(
                program_id=entry.program_id,
                source_path=str(source),
                relative_path=entry.relative_path,
                program_family=str(Path(entry.relative_path).parent).replace("\\", "/"),
                source_sha256=source_digest,
                source_size_bytes=len(source_bytes),
                compile_command=command,
                compile_status=compile_status,
                compile_stderr_sha256=_sha256_bytes(stderr) if stderr else "",
                root_ir_path=str(output),
                root_ir_sha256=root_sha256,
                root_hard_state_id=hard_state_id,
                target=target,
                data_layout=data_layout,
                preflight_status=preflight_status,
                selection_class="candidate",
            )
        )
    return _CandidatePreflightInventory(
        records=tuple(records),
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda row: (str(row["stable_rank"]), str(row["relative_path"])),
            )
        ),
        inventory_count=len(entries),
    )


def _existing_root_only_records(
    source_entries: Sequence[tuple[str, Path, str]],
    *,
    root_only_root: Path,
    expected_actions: Sequence[ActionRecord],
) -> tuple[tuple["ProgramRecord", ...], dict[str, str]]:
    """Bind every fixed program to the prior root-only source/root/action data.

    This is intentionally a read-only bridge.  The old experiment's per
    program manifest identifies the exact U14 action family, while its root
    state CSV identifies the retained ``S`` artifact and Worker hard hash.
    The isolated study copies those bytes to its own output; it never rewrites
    the old result directory or substitutes a fresh compiler root.
    """

    expected_ids = tuple(action.action_id for action in expected_actions)
    records: list["ProgramRecord"] = []
    hashes: dict[str, str] = {}
    for index, (name, source, relative) in enumerate(source_entries):
        directory = root_only_root / "programs" / name / "root_only"
        manifest_path = directory / "experiment_manifest.json"
        states_path = directory / "states.csv"
        metadata_path = directory / "metadata.json"
        if not manifest_path.is_file() or not states_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"existing root-only evidence is missing for {name}")
        manifest = _json_mapping(manifest_path, label="existing root-only experiment_manifest")
        pass_config = manifest.get("pass_config")
        actions = pass_config.get("actions") if isinstance(pass_config, Mapping) else None
        if not isinstance(actions, list):
            raise ValueError(f"existing root-only action inventory is missing for {name}")
        try:
            observed_ids = tuple(ActionRecord.from_manifest_record(action).action_id for action in actions if isinstance(action, Mapping))
        except ValueError as error:
            raise ValueError(f"existing root-only action inventory is invalid for {name}") from error
        if observed_ids != expected_ids or len(observed_ids) != len(actions):
            raise ValueError(f"existing root-only U14 action identity mismatch for {name}")
        metadata = _json_mapping(metadata_path, label="existing root-only metadata")
        if Path(str(metadata.get("input", ""))).resolve() != source.resolve():
            raise ValueError(f"existing root-only source identity mismatch for {name}")
        with states_path.open("r", encoding="utf-8", newline="") as stream:
            state_rows = list(csv.DictReader(stream))
        roots = [row for row in state_rows if row.get("depth") == "0"]
        if len(roots) != 1:
            raise ValueError(f"existing root-only state identity is ambiguous for {name}")
        root_row = roots[0]
        root_ir = Path(str(root_row.get("ir_path", ""))).resolve()
        hard_id = str(root_row.get("state_hash", ""))
        if not root_ir.is_file() or len(hard_id) != 64:
            raise ValueError(f"existing root-only root artifact is missing for {name}")
        try:
            target, data_layout = _bound_root_target_identity(manifest, root_ir)
        except ValueError as error:
            raise ValueError(
                f"existing root-only target identity mismatch for {name}: {error}"
            ) from error
        records.append(_program_from_source(
            program_id=name,
            source=source,
            relative=relative,
            selection_class="fixed",
            selection_order=index + 1,
            root_ir_path=root_ir,
            root_ir_sha256=_sha256_file(root_ir),
            root_hard_state_id=hard_id,
            compile_command=("root-only-artifact", str(root_ir)),
            target=target,
            data_layout=data_layout,
        ))
        hashes[f"{name}/experiment_manifest.json"] = _sha256_file(manifest_path)
        hashes[f"{name}/states.csv"] = _sha256_file(states_path)
        hashes[f"{name}/metadata.json"] = _sha256_file(metadata_path)
        hashes[f"{name}/S.ll"] = _sha256_file(root_ir)
    return tuple(records), dict(sorted(hashes.items()))


def _program_from_source(
    *,
    program_id: str,
    source: Path,
    relative: str,
    selection_class: str,
    selection_order: int | None,
    root_ir_path: Path,
    root_ir_sha256: str,
    root_hard_state_id: str,
    compile_command: Sequence[str],
    target: str,
    data_layout: str,
) -> "ProgramRecord":
    from .manifest import ProgramRecord

    data = source.read_bytes()
    return ProgramRecord(
        program_id=program_id,
        source_path=str(source.resolve()),
        relative_path=relative,
        program_family=str(Path(relative).parent).replace("\\", "/"),
        source_sha256=_sha256_bytes(data),
        source_size_bytes=len(data),
        compile_command=tuple(compile_command),
        compile_status="success",
        compile_stderr_sha256="",
        root_ir_path=str(root_ir_path.resolve(strict=False)),
        root_ir_sha256=root_ir_sha256,
        root_hard_state_id=root_hard_state_id,
        target=target,
        data_layout=data_layout,
        preflight_status="success",
        selection_class=selection_class,
        selection_order=selection_order,
    )


def _tool_record(path: Path, name: str) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} tool is missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _prepare_frozen(args: argparse.Namespace) -> None:
    """Compile/freeze only source, tool, pass, and preflight identities.

    No pair observation is read or written here.  The temporary reference
    records live under the system temporary directory and are removed before
    this command returns; all durable artifacts are published only through
    :func:`prepare_study` under the isolated output root.
    """

    out_dir = Path(args.output).resolve()
    kind = _require_output_kind(out_dir)
    smoke_count = args.smoke_programs
    if kind == "formal":
        if smoke_count is not None:
            raise ValueError("formal output rejects --smoke-programs")
        program_target = FORMAL_PROGRAM_TARGET
    else:
        if smoke_count != 3:
            raise ValueError("smoke output requires --smoke-programs 3")
        program_target = 3
    if args.jobs < 1 or args.timeout <= 0:
        raise ValueError("--jobs and --timeout must be positive")
    policy = load_frozen_policy(Path(args.pass_policy))
    source_entries = _source_entries(Path(args.source_manifest), Path(args.single_source_root))
    if len(source_entries) != 50:
        raise ValueError("source manifest must freeze exactly the existing 50 programs")
    core_actions = load_u14_actions(Path(args.core_passes))
    root_only_root = Path(args.source_manifest).resolve().parent
    all_fixed, root_only_hashes = _existing_root_only_records(
        source_entries,
        root_only_root=root_only_root,
        expected_actions=core_actions,
    )
    selected_fixed = all_fixed if kind == "formal" else all_fixed[:program_target]
    tools = {
        "opt": _tool_record(Path(args.opt), "opt"),
        "clang": _tool_record(Path(args.clang), "clang"),
        "worker": _tool_record(Path(args.worker), "worker"),
        "merge_helper": _tool_record(Path(args.merge_helper), "merge_helper"),
    }
    opt_version = _run_process((str(args.opt), "--version"), timeout_s=args.timeout)
    clang_version = _run_process((str(args.clang), "--version"), timeout_s=args.timeout)
    expected_commit = str(policy["llvm_commit"])
    if opt_version.returncode != 0 or clang_version.returncode != 0 or expected_commit not in (opt_version.stdout + opt_version.stderr + clang_version.stdout + clang_version.stderr):
        raise ValueError("LLVM tool version does not bind the frozen policy commit")

    compile_command_prefix = (
        str(Path(args.clang).resolve()),
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-S",
        "-emit-llvm",
    )

    def compile_one(source: Path, output: Path) -> RunResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        command = (*compile_command_prefix, str(source), "-o", str(output))
        try:
            result = _run_process(command, timeout_s=args.timeout)
        except TimeoutError:
            return RunResult(False, output, timed_out=True, stderr="clang timeout", command=tuple(command))
        if result.returncode != 0:
            return RunResult(False, output, stderr=result.stderr, command=tuple(command))
        digest = _sha256_file(output) if output.is_file() else ""
        hard_state_id = _phasebatch_hard_state_id(output) if digest else ""
        return RunResult(
            bool(digest),
            output,
            hard_state_id=hard_state_id,
            stderr=result.stderr,
            command=tuple(command),
        )

    with tempfile.TemporaryDirectory(prefix="advisor-pair-scale-prepare-") as temporary:
        temp_root = Path(temporary)
        candidate_records: Sequence[ProgramRecord] = ()
        candidate_identity_exclusions: Sequence[Mapping[str, object]] = ()
        candidate_inventory_count = 0
        # Formal Task 15 is frozen to the existing root-only fixed 50.  Do not
        # scan, compile, rank, or select any extension candidate.

        reference = temp_root / "fixed_reference.json"
        reference.write_text(
            json.dumps(
                {
                    "pass_config": {"actions": [action.as_manifest_record() for action in core_actions]},
                    # The smoke path intentionally retains the existing-50
                    # action identity but only needs its selected three roots.
                    "program_manifest": [program.as_manifest_record() for program in all_fixed],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        def materialize_frozen_root(record: object, output: Path) -> RunResult:
            source_root = Path(str(getattr(record, "root_ir_path")))
            if not source_root.is_file():
                return RunResult(False, output, stderr="compile-preflight root artifact is missing")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root, output)
            return RunResult(
                success=True,
                output_path=output,
                hard_state_id=str(getattr(record, "root_hard_state_id")),
                command=("copy-frozen-preflight-root", str(source_root), str(output)),
            )

        from phasebatch.ir_equivalence import DEFAULT_HARD_STATE_POLICY

        llvm_diff = Path(args.opt).resolve().parent / "llvm-diff.exe"
        comparator_tool = _tool_record(llvm_diff, "llvm-diff")
        prepare_worker = _WorkerRunner(Path(args.worker).resolve(), timeout_s=args.timeout)
        dependencies = PrepareDependencies(
            compile_source=materialize_frozen_root,
            print_passes=lambda: _run_process((str(args.opt), "--print-passes"), timeout_s=args.timeout).stdout,
            run_single=prepare_worker.apply,
            verify_ir=lambda path: _verify_with_opt(Path(args.opt), args.timeout, path),
            tool_records=tools,
            fixed_programs=tuple(selected_fixed),
            candidate_programs=tuple(candidate_records),
            single_source_root=Path(args.single_source_root),
            llvm_commit=expected_commit,
            target="x86_64-w64-windows-gnu",
            hard_state_policy={
                "policy_id": DEFAULT_HARD_STATE_POLICY.policy_id,
                "implementation": "phasebatch.ir_equivalence.hard_state_hash",
                "worker_execution_path": str(Path(args.worker).resolve()),
                "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
            },
            comparator={
                "comparator_id": (
                    f"{DEFAULT_HARD_STATE_POLICY.policy_id}@{_HARD_STATE_COMPARATOR_VERSION}"
                ),
                "implementation": "phasebatch.ir_equivalence.compare_hard_states",
                "comparator_version": _HARD_STATE_COMPARATOR_VERSION,
                "llvm_diff": comparator_tool,
                "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
            },
            artifact_policy={
                "retain_roots": True,
                "source_manifest": str(Path(args.source_manifest).resolve()),
                "source_manifest_sha256": _sha256_file(Path(args.source_manifest)),
                "root_only_manifest_root": str(root_only_root),
                "root_only_identity_sha256": canonical_sha256(root_only_hashes),
                "smoke_programs": program_target if kind == "smoke" else 0,
                "formal_program_count": FORMAL_PROGRAM_TARGET if kind == "formal" else 0,
                "fixed_program_count": (
                    FORMAL_PROGRAM_TARGET if kind == "formal" else len(selected_fixed)
                ),
                "formal_source_inventory_count": (
                    FORMAL_SOURCE_INVENTORY_COUNT if kind == "formal" else 0
                ),
                "formal_selection_rule_id": (
                    FORMAL_SELECTION_RULE_ID if kind == "formal" else ""
                ),
                "formal_source_positions": (
                    list(FORMAL_SOURCE_POSITIONS) if kind == "formal" else []
                ),
                "candidate_reserve_count": len(candidate_records),
                "candidate_inventory_count": candidate_inventory_count,
                "candidate_identity_exclusion_count": (
                    candidate_inventory_count - len(candidate_records)
                ),
                "selection_seed": DEFAULT_SELECTION_SEED,
                "raw_execution_semantics_revision": RAW_EXECUTION_SEMANTICS_REVISION,
            },
            candidate_identity_exclusions=candidate_identity_exclusions,
        )
        try:
            result = prepare_study(
                repo_root=EXPERIMENT_ROOT.parents[1],
                out_dir=out_dir,
                existing_50_manifest=reference,
                pass_policy=Path(args.pass_policy),
                dependencies=dependencies,
                core_passes=Path(args.core_passes),
                program_target=program_target,
                jobs=args.jobs,
                timeout_s=args.timeout,
            )
        finally:
            prepare_worker.close()
    print(json.dumps({
        "study_manifest": str(result.study_manifest_path),
        "study_manifest_id": result.study_manifest_id,
        "program_count": result.program_count,
        "group_sizes": result.group_sizes,
        "scale_gate": result.scale_gate,
    }, ensure_ascii=False, sort_keys=True))


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--single-source-root", required=True, type=Path)
    parser.add_argument("--clang", required=True, type=Path)
    parser.add_argument("--opt", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--merge-helper", required=True, type=Path)
    parser.add_argument("--pass-policy", required=True, type=Path)
    parser.add_argument("--core-passes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--smoke-programs", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m advisor_study.cli")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="freeze isolated study inputs")
    _add_prepare_arguments(prepare)
    run = subcommands.add_parser("run", help="write raw evidence from a frozen manifest")
    run.add_argument("--manifest", required=True, type=Path)
    summarize = subcommands.add_parser("summarize", help="derive aggregate outputs from complete evidence")
    summarize.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        validate_prepare_output(args.output, require_experiment_root=True)
        _prepare_frozen(args)
    elif args.command == "run":
        _run_frozen(load_frozen_phase(args.manifest, phase="run"))
    elif args.command == "summarize":
        _summarize_frozen(load_frozen_phase(args.manifest, phase="summarize"))
    else:  # pragma: no cover - argparse enforces the known subcommands.
        raise AssertionError(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through Python -m.
    raise SystemExit(main())
