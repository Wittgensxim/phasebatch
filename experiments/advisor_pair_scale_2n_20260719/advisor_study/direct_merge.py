"""Fail-closed JSON-lines client for the isolated direct-merge helper.

This module is an experiment-only adapter.  It never merges text, never runs
passes, and never turns a helper failure into evidence for a 2N claim.  The
LLVM helper remains the sole implementation of structured whole-function
merge; this client only validates its narrow protocol and records provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any

from phasebatch.opt_worker import (
    WorkerError,
    WorkerProcess,
    WorkerProtocolError,
    WorkerTimeoutError,
)


_PROTOCOL_VERSION = 1
_OPERATIONS = ("ping", "inspect_patch", "merge", "compare_effect")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class DirectMergeProtocolError(RuntimeError):
    """A malformed, unavailable, or otherwise unusable helper interaction.

    ``unavailable`` deliberately has no success interpretation.  Callers must
    record a named unavailable/failure row instead of inferring mergeability or
    commutativity from this exception.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        operation_record_id: str,
        unavailable: bool = True,
        error_kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.operation_record_id = operation_record_id
        self.unavailable = unavailable
        self.error_kind = error_kind


class DirectMergeUnavailable(DirectMergeProtocolError):
    """A typed ``status=error`` result emitted by the helper itself."""


@dataclass(frozen=True)
class PatchRecord:
    """A validated whole-function patch relative to one exact base artifact."""

    operation_record_id: str
    base_path: Path
    output_path: Path
    base_artifact_sha256: str
    output_artifact_sha256: str
    base_module_hash: str
    output_module_hash: str
    base_skeleton_hash: str
    output_skeleton_hash: str
    base_symbol_inventory_hash: str
    output_symbol_inventory_hash: str
    changed_functions: tuple[str, ...]
    changed_function_hashes: tuple[tuple[str, str, str], ...]
    patch_hash: str


@dataclass(frozen=True)
class MergeRecord:
    """A validated direct-merge output and its exact first-round inputs."""

    operation_record_id: str
    base_path: Path
    output_paths: tuple[Path, ...]
    merged_path: Path
    base_artifact_sha256: str
    merged_artifact_sha256: str
    base_module_hash: str
    base_skeleton_hash: str
    output_module_hash: str
    output_skeleton_hash: str
    merged_functions: tuple[str, ...]
    contributed_functions: tuple[str, ...]
    input_patch_hashes: tuple[str, ...]
    input_output_module_hashes: tuple[str, ...]
    merge_input_count: int
    merge_wall_time_ns: int


@dataclass(frozen=True)
class EffectRecord:
    """A validated first/second-round effect comparison result."""

    operation_record_id: str
    first_base_path: Path
    first_output_path: Path
    second_base_path: Path
    second_output_path: Path
    first_base_artifact_sha256: str
    first_output_artifact_sha256: str
    second_base_artifact_sha256: str
    second_output_artifact_sha256: str
    same_effect: bool
    first_changed_functions: tuple[str, ...]
    second_changed_functions: tuple[str, ...]
    first_patch_hash: str
    second_patch_hash: str
    protected_functions: tuple[str, ...]
    expected_protected_functions: tuple[str, ...]
    protected_functions_preserved: bool
    skeletons_unchanged: bool
    symbol_inventories_unchanged: bool


@dataclass(frozen=True)
class Advisor2NGroupResult:
    """Evidence rows from one exact, report-only group-level 2N evaluation.

    The result deliberately contains rows rather than an authority decision.
    Both authority fields are fixed to ``false`` in every emitted row.  A
    caller must retain the independent AB/BA evidence for any claim expanded
    from a directional authorization.
    """

    group_row: Mapping[str, object]
    directional_rows: tuple[Mapping[str, object], ...]
    pair_rows: tuple[Mapping[str, object], ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _record_id(operation: str, inputs: Mapping[str, object]) -> str:
    canonical = _canonical_json({"operation": operation, "inputs": inputs})
    return f"{operation}-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _normal_path(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


def _path_text(path: Path) -> str:
    return path.as_posix()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class DirectMergeClient:
    """Persistent, fail-closed client for ``phasebatch-2n-merge`` JSONL.

    ``WorkerProcess`` owns monotonic wire request IDs.  Persisted operation IDs
    deliberately derive only from canonical operation inputs, so they remain
    stable across process restarts and do not encode wall-clock or wire state.
    """

    def __init__(self, command: Sequence[str | Path], timeout_s: float) -> None:
        if not command:
            raise ValueError("direct-merge helper command must not be empty")
        if timeout_s <= 0:
            raise ValueError("direct-merge helper timeout_s must be positive")
        self._worker = WorkerProcess(tuple(command), worker_id=0)
        self._timeout_s = float(timeout_s)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def stderr_text(self) -> str:
        return self._worker.stderr_text

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._worker.close()

    def __enter__(self) -> "DirectMergeClient":
        if self._closed:
            raise RuntimeError("direct-merge client is closed")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def ping(self) -> Mapping[str, object]:
        inputs: dict[str, object] = {"protocol_version": _PROTOCOL_VERSION}
        record_id = _record_id("ping", inputs)
        reply = self._request("ping", record_id=record_id)
        protocol_version = reply.get("protocol_version")
        llvm_version = reply.get("llvm_version")
        operations = reply.get("operations")
        if protocol_version != _PROTOCOL_VERSION:
            self._fatal(
                "ping protocol_version mismatch",
                operation="ping",
                record_id=record_id,
            )
        if not isinstance(llvm_version, str) or not llvm_version.strip():
            self._fatal("ping missing llvm_version", operation="ping", record_id=record_id)
        if not isinstance(operations, list) or tuple(operations) != _OPERATIONS:
            self._fatal("ping operations mismatch", operation="ping", record_id=record_id)
        return MappingProxyType(
            {
                "protocol_version": protocol_version,
                "llvm_version": llvm_version,
                "operations": tuple(operations),
                "operation_record_id": record_id,
            }
        )

    def inspect_patch(self, base: str | Path, output: str | Path) -> PatchRecord:
        base_path = self._input_file(base, "base_path", "inspect_patch")
        output_path = self._input_file(output, "output_path", "inspect_patch")
        before_base = self._current_input_digest(base_path, "base_path", "inspect_patch")
        before_output = self._current_input_digest(
            output_path, "output_path", "inspect_patch"
        )
        inputs: dict[str, object] = {
            "base_path": _path_text(base_path),
            "output_path": _path_text(output_path),
            "base_artifact_sha256": before_base,
            "output_artifact_sha256": before_output,
        }
        record_id = _record_id("inspect_patch", inputs)
        reply = self._request(
            "inspect_patch",
            record_id=record_id,
            base_path=str(base_path),
            output_path=str(output_path),
        )
        record = self._parse_patch(
            reply,
            record_id=record_id,
            base_path=base_path,
            output_path=output_path,
            base_artifact_sha256=before_base,
            output_artifact_sha256=before_output,
        )
        after_base = self._artifact_digest(base_path, "base_path", "inspect_patch", record_id)
        after_output = self._artifact_digest(
            output_path, "output_path", "inspect_patch", record_id
        )
        if (after_base, after_output) != (before_base, before_output):
            self._fatal(
                "inspect_patch input artifact changed during helper request",
                operation="inspect_patch",
                record_id=record_id,
            )
        return record

    def merge(
        self,
        base: str | Path,
        patches: Sequence[PatchRecord],
        merged_path: str | Path,
    ) -> MergeRecord:
        base_path = self._input_file(base, "base_path", "merge")
        target = _normal_path(merged_path)
        if target == base_path:
            self._local_error("merged_path must not overwrite base_path", "merge")
        validated = self._validate_patches(base_path, patches)
        if any(target == patch.output_path for patch in validated):
            self._local_error("merged_path must not overwrite an output patch", "merge")
        canonical_patches = tuple(sorted(validated, key=lambda patch: _path_text(patch.output_path)))
        before_base = self._current_input_digest(base_path, "base_path", "merge")
        current_patch_artifacts = tuple(
            self._current_input_digest(patch.output_path, "patch output_path", "merge")
            for patch in canonical_patches
        )
        inputs: dict[str, object] = {
            "base_path": _path_text(base_path),
            "base_artifact_sha256": before_base,
            "patches": [
                {
                    "output_path": _path_text(patch.output_path),
                    "output_artifact_sha256": artifact_sha256,
                    "output_module_hash": patch.output_module_hash,
                    "patch_hash": patch.patch_hash,
                }
                for patch, artifact_sha256 in zip(canonical_patches, current_patch_artifacts)
            ],
            "merged_path": _path_text(target),
        }
        record_id = _record_id("merge", inputs)
        # A PatchRecord is evidence about exact input bytes, not only a path and
        # a helper-reported canonical IR hash.  Rehash the complete family
        # immediately before issuing ``merge`` so a stale or replaced first-
        # round artifact cannot be sent to the helper.
        for patch, current_output in zip(canonical_patches, current_patch_artifacts):
            if patch.base_artifact_sha256 != before_base:
                self._fatal(
                    "base artifact hash mismatch",
                    operation="merge",
                    record_id=record_id,
                    error_kind="artifact_mismatch",
                )
            if current_output != patch.output_artifact_sha256:
                self._fatal(
                    "patch output artifact hash mismatch",
                    operation="merge",
                    record_id=record_id,
                    error_kind="artifact_mismatch",
                )
        reply = self._request(
            "merge",
            record_id=record_id,
            base_path=str(base_path),
            output_paths=[str(patch.output_path) for patch in canonical_patches],
            merged_path=str(target),
        )
        if not target.is_file():
            self._fatal("missing merged artifact", operation="merge", record_id=record_id)
        merged_digest = self._artifact_digest(target, "merged_path", "merge", record_id)
        after_base = self._artifact_digest(base_path, "base_path", "merge", record_id)
        if after_base != before_base:
            self._fatal(
                "merge base artifact changed during helper request",
                operation="merge",
                record_id=record_id,
            )
        for patch in canonical_patches:
            if (
                self._artifact_digest(
                    patch.output_path,
                    "patch output_path",
                    "merge",
                    record_id,
                )
                != patch.output_artifact_sha256
            ):
                self._fatal(
                    "merge patch output artifact changed during helper request",
                    operation="merge",
                    record_id=record_id,
                )
        return self._parse_merge(
            reply,
            record_id=record_id,
            base_path=base_path,
            patches=canonical_patches,
            merged_path=target,
            base_artifact_sha256=before_base,
            merged_artifact_sha256=merged_digest,
        )

    def compare_effect(
        self,
        *,
        first_base: str | Path,
        first_output: str | Path,
        second_base: str | Path,
        second_output: str | Path,
        protected_functions: Sequence[str],
        expected_first_patch: PatchRecord | None = None,
        expected_second_patch: PatchRecord | None = None,
    ) -> EffectRecord:
        first_base_path = self._input_file(first_base, "first_base_path", "compare_effect")
        first_output_path = self._input_file(
            first_output, "first_output_path", "compare_effect"
        )
        second_base_path = self._input_file(second_base, "second_base_path", "compare_effect")
        second_output_path = self._input_file(
            second_output, "second_output_path", "compare_effect"
        )
        protected = self._function_names(protected_functions, "protected_functions", "compare_effect")
        artifacts = tuple(
            self._current_input_digest(path, name, "compare_effect")
            for path, name in (
                (first_base_path, "first_base_path"),
                (first_output_path, "first_output_path"),
                (second_base_path, "second_base_path"),
                (second_output_path, "second_output_path"),
            )
        )
        inputs: dict[str, object] = {
            "first_base_path": _path_text(first_base_path),
            "first_output_path": _path_text(first_output_path),
            "second_base_path": _path_text(second_base_path),
            "second_output_path": _path_text(second_output_path),
            "first_base_artifact_sha256": artifacts[0],
            "first_output_artifact_sha256": artifacts[1],
            "second_base_artifact_sha256": artifacts[2],
            "second_output_artifact_sha256": artifacts[3],
            "protected_functions": list(protected),
            "expected_first_patch_hash": (
                expected_first_patch.patch_hash if expected_first_patch is not None else ""
            ),
            "expected_second_patch_hash": (
                expected_second_patch.patch_hash if expected_second_patch is not None else ""
            ),
        }
        record_id = _record_id("compare_effect", inputs)
        self._bind_expected_patch_artifacts(
            expected_first_patch,
            base_path=first_base_path,
            output_path=first_output_path,
            base_artifact_sha256=artifacts[0],
            output_artifact_sha256=artifacts[1],
            field="first",
            record_id=record_id,
        )
        self._bind_expected_patch_artifacts(
            expected_second_patch,
            base_path=second_base_path,
            output_path=second_output_path,
            base_artifact_sha256=artifacts[2],
            output_artifact_sha256=artifacts[3],
            field="second",
            record_id=record_id,
        )
        reply = self._request(
            "compare_effect",
            record_id=record_id,
            first_base_path=str(first_base_path),
            first_output_path=str(first_output_path),
            second_base_path=str(second_base_path),
            second_output_path=str(second_output_path),
            protected_functions=list(protected),
        )
        record = self._parse_effect(
            reply,
            record_id=record_id,
            first_base_path=first_base_path,
            first_output_path=first_output_path,
            second_base_path=second_base_path,
            second_output_path=second_output_path,
            artifacts=artifacts,
            protected=protected,
            expected_first_patch=expected_first_patch,
            expected_second_patch=expected_second_patch,
        )
        after = tuple(
            self._artifact_digest(path, name, "compare_effect", record_id)
            for path, name in (
                (first_base_path, "first_base_path"),
                (first_output_path, "first_output_path"),
                (second_base_path, "second_base_path"),
                (second_output_path, "second_output_path"),
            )
        )
        if after != artifacts:
            self._fatal(
                "compare_effect input artifact changed during helper request",
                operation="compare_effect",
                record_id=record_id,
            )
        return record

    def _request(self, operation: str, *, record_id: str, **payload: object) -> Mapping[str, Any]:
        if self._closed:
            raise DirectMergeProtocolError(
                "direct-merge client is closed",
                operation=operation,
                operation_record_id=record_id,
                unavailable=True,
                error_kind="client_closed",
            )
        try:
            reply = self._worker.request(operation, timeout=self._timeout_s, **payload).payload
        except WorkerTimeoutError as error:
            self._fatal(
                str(error),
                operation=operation,
                record_id=record_id,
                cause=error,
                error_kind="timeout",
            )
        except WorkerProtocolError as error:
            detail = str(error).replace("response ID", "response request_id")
            self._fatal(
                detail,
                operation=operation,
                record_id=record_id,
                cause=error,
                error_kind=(
                    "transport_error"
                    if detail.startswith("worker exited") or detail.startswith("worker stdin")
                    else "protocol_error"
                ),
            )
        except WorkerError as error:
            self._fatal(
                str(error),
                operation=operation,
                record_id=record_id,
                cause=error,
                error_kind="transport_error",
            )
        if not isinstance(reply, Mapping):
            self._fatal("helper reply must be an object", operation=operation, record_id=record_id)
        status = reply.get("status")
        if status == "error":
            kind = reply.get("error_kind")
            message = reply.get("error_message")
            if not isinstance(kind, str) or not kind or not isinstance(message, str) or not message:
                self._fatal(
                    "helper error reply is malformed",
                    operation=operation,
                    record_id=record_id,
                )
            raise DirectMergeUnavailable(
                f"{operation} unavailable: {kind}: {message}",
                operation=operation,
                operation_record_id=record_id,
                unavailable=True,
                error_kind=kind,
            )
        if status != "ok":
            self._fatal(
                f"helper returned unknown status: {status!r}",
                operation=operation,
                record_id=record_id,
            )
        return reply

    def _parse_patch(
        self,
        reply: Mapping[str, Any],
        *,
        record_id: str,
        base_path: Path,
        output_path: Path,
        base_artifact_sha256: str,
        output_artifact_sha256: str,
    ) -> PatchRecord:
        operation = "inspect_patch"
        base_module_hash = self._digest_field(reply, "base_module_hash", operation, record_id)
        output_module_hash = self._digest_field(reply, "output_module_hash", operation, record_id)
        base_skeleton_hash = self._digest_field(reply, "base_skeleton_hash", operation, record_id)
        output_skeleton_hash = self._digest_field(reply, "output_skeleton_hash", operation, record_id)
        base_inventory = self._digest_field(
            reply, "base_symbol_inventory_hash", operation, record_id
        )
        output_inventory = self._digest_field(
            reply, "output_symbol_inventory_hash", operation, record_id
        )
        if base_skeleton_hash != output_skeleton_hash or base_inventory != output_inventory:
            self._fatal(
                "inspect_patch returned a non-body structural change as success",
                operation=operation,
                record_id=record_id,
            )
        changed = self._function_names(
            reply.get("changed_functions"), "changed_functions", operation, record_id
        )
        raw_hashes = reply.get("changed_function_hashes")
        raw_record = reply.get("patch_record")
        if not isinstance(raw_hashes, list) or not isinstance(raw_record, Mapping):
            self._fatal("inspect_patch missing changed patch records", operation=operation, record_id=record_id)
        if raw_record.get("schema_version") != 1 or raw_record.get("changed_functions") != raw_hashes:
            self._fatal("inspect_patch patch_record mismatch", operation=operation, record_id=record_id)
        changed_hashes: list[tuple[str, str, str]] = []
        for item in raw_hashes:
            if not isinstance(item, Mapping):
                self._fatal("changed_function_hashes entry must be an object", operation=operation, record_id=record_id)
            name = item.get("name")
            base_hash = item.get("base_isolated_hash")
            output_hash = item.get("output_isolated_hash")
            if not isinstance(name, str) or not name or not _is_digest(base_hash) or not _is_digest(output_hash):
                self._fatal("invalid changed_function_hashes entry", operation=operation, record_id=record_id)
            changed_hashes.append((name, base_hash, output_hash))
        if tuple(name for name, _, _ in changed_hashes) != changed:
            self._fatal("changed_functions mismatch", operation=operation, record_id=record_id)
        if tuple(sorted(changed_hashes)) != tuple(changed_hashes):
            self._fatal("changed_function_hashes must be canonical", operation=operation, record_id=record_id)
        patch_hash = self._digest_field(reply, "patch_hash", operation, record_id)
        canonical = "advisor_2n_patch_record_v1\n" + "".join(
            f"{name}\n{base_hash}\n{output_hash}\n"
            for name, base_hash, output_hash in changed_hashes
        )
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if patch_hash != expected_hash:
            self._fatal("patch_hash mismatch", operation=operation, record_id=record_id)
        return PatchRecord(
            operation_record_id=record_id,
            base_path=base_path,
            output_path=output_path,
            base_artifact_sha256=base_artifact_sha256,
            output_artifact_sha256=output_artifact_sha256,
            base_module_hash=base_module_hash,
            output_module_hash=output_module_hash,
            base_skeleton_hash=base_skeleton_hash,
            output_skeleton_hash=output_skeleton_hash,
            base_symbol_inventory_hash=base_inventory,
            output_symbol_inventory_hash=output_inventory,
            changed_functions=changed,
            changed_function_hashes=tuple(changed_hashes),
            patch_hash=patch_hash,
        )

    def _parse_merge(
        self,
        reply: Mapping[str, Any],
        *,
        record_id: str,
        base_path: Path,
        patches: tuple[PatchRecord, ...],
        merged_path: Path,
        base_artifact_sha256: str,
        merged_artifact_sha256: str,
    ) -> MergeRecord:
        operation = "merge"
        base_hash = self._digest_field(reply, "base_module_hash", operation, record_id)
        base_skeleton = self._digest_field(reply, "base_skeleton_hash", operation, record_id)
        output_hash = self._digest_field(reply, "output_module_hash", operation, record_id)
        output_skeleton = self._digest_field(reply, "output_skeleton_hash", operation, record_id)
        if any(patch.base_module_hash != base_hash for patch in patches):
            self._fatal("base_module_hash mismatch", operation=operation, record_id=record_id)
        if any(patch.base_skeleton_hash != base_skeleton for patch in patches):
            self._fatal("base_skeleton_hash mismatch", operation=operation, record_id=record_id)
        if output_skeleton != base_skeleton:
            self._fatal("merge output_skeleton_hash mismatch", operation=operation, record_id=record_id)
        merged_functions = self._function_names(
            reply.get("merged_functions"), "merged_functions", operation, record_id
        )
        contributed = self._function_names(
            reply.get("contributed_functions"), "contributed_functions", operation, record_id
        )
        expected_functions = tuple(sorted({name for patch in patches for name in patch.changed_functions}))
        if merged_functions != expected_functions or contributed != expected_functions:
            self._fatal("merged_functions mismatch", operation=operation, record_id=record_id)
        patch_hashes = self._digest_list(reply, "input_patch_hashes", operation, record_id)
        output_hashes = self._digest_list(
            reply, "input_output_module_hashes", operation, record_id
        )
        expected_patch_hashes = tuple(sorted(patch.patch_hash for patch in patches))
        expected_output_hashes = tuple(sorted(patch.output_module_hash for patch in patches))
        if patch_hashes != expected_patch_hashes:
            self._fatal("input_patch_hashes mismatch", operation=operation, record_id=record_id)
        if output_hashes != expected_output_hashes:
            self._fatal(
                "input_output_module_hashes mismatch", operation=operation, record_id=record_id
            )
        count = reply.get("merge_input_count")
        elapsed = reply.get("merge_wall_time_ns")
        if not _is_int(count) or count != len(patches) or not _is_int(elapsed) or elapsed < 0:
            self._fatal("invalid merge counts", operation=operation, record_id=record_id)
        return MergeRecord(
            operation_record_id=record_id,
            base_path=base_path,
            output_paths=tuple(patch.output_path for patch in patches),
            merged_path=merged_path,
            base_artifact_sha256=base_artifact_sha256,
            merged_artifact_sha256=merged_artifact_sha256,
            base_module_hash=base_hash,
            base_skeleton_hash=base_skeleton,
            output_module_hash=output_hash,
            output_skeleton_hash=output_skeleton,
            merged_functions=merged_functions,
            contributed_functions=contributed,
            input_patch_hashes=patch_hashes,
            input_output_module_hashes=output_hashes,
            merge_input_count=count,
            merge_wall_time_ns=elapsed,
        )

    def _parse_effect(
        self,
        reply: Mapping[str, Any],
        *,
        record_id: str,
        first_base_path: Path,
        first_output_path: Path,
        second_base_path: Path,
        second_output_path: Path,
        artifacts: tuple[str, str, str, str],
        protected: tuple[str, ...],
        expected_first_patch: PatchRecord | None,
        expected_second_patch: PatchRecord | None,
    ) -> EffectRecord:
        operation = "compare_effect"
        bool_fields = (
            "same_effect",
            "protected_functions_preserved",
            "skeletons_unchanged",
            "symbol_inventories_unchanged",
        )
        booleans: dict[str, bool] = {}
        for field in bool_fields:
            value = reply.get(field)
            if not isinstance(value, bool):
                self._fatal(f"invalid {field}", operation=operation, record_id=record_id)
            booleans[field] = value
        first_changed = self._function_names(
            reply.get("first_changed_functions"), "first_changed_functions", operation, record_id
        )
        second_changed = self._function_names(
            reply.get("second_changed_functions"), "second_changed_functions", operation, record_id
        )
        first_hash = self._digest_field(reply, "first_patch_hash", operation, record_id)
        second_hash = self._digest_field(reply, "second_patch_hash", operation, record_id)
        returned_protected = self._function_names(
            reply.get("protected_functions"), "protected_functions", operation, record_id
        )
        expected_protected = self._function_names(
            reply.get("expected_protected_functions"),
            "expected_protected_functions",
            operation,
            record_id,
        )
        if returned_protected != protected or expected_protected != protected:
            self._fatal("protected_functions mismatch", operation=operation, record_id=record_id)
        self._validate_expected_patch(
            expected_first_patch,
            base_path=first_base_path,
            output_path=first_output_path,
            changed=first_changed,
            patch_hash=first_hash,
            field="first_patch_hash",
            record_id=record_id,
        )
        self._validate_expected_patch(
            expected_second_patch,
            base_path=second_base_path,
            output_path=second_output_path,
            changed=second_changed,
            patch_hash=second_hash,
            field="second_patch_hash",
            record_id=record_id,
        )
        if booleans["same_effect"] and (
            first_changed != second_changed
            or first_hash != second_hash
            or not booleans["protected_functions_preserved"]
            or not booleans["skeletons_unchanged"]
            or not booleans["symbol_inventories_unchanged"]
        ):
            self._fatal("same_effect contradicts comparison evidence", operation=operation, record_id=record_id)
        return EffectRecord(
            operation_record_id=record_id,
            first_base_path=first_base_path,
            first_output_path=first_output_path,
            second_base_path=second_base_path,
            second_output_path=second_output_path,
            first_base_artifact_sha256=artifacts[0],
            first_output_artifact_sha256=artifacts[1],
            second_base_artifact_sha256=artifacts[2],
            second_output_artifact_sha256=artifacts[3],
            same_effect=booleans["same_effect"],
            first_changed_functions=first_changed,
            second_changed_functions=second_changed,
            first_patch_hash=first_hash,
            second_patch_hash=second_hash,
            protected_functions=returned_protected,
            expected_protected_functions=expected_protected,
            protected_functions_preserved=booleans["protected_functions_preserved"],
            skeletons_unchanged=booleans["skeletons_unchanged"],
            symbol_inventories_unchanged=booleans["symbol_inventories_unchanged"],
        )

    def _validate_patches(
        self, base_path: Path, patches: Sequence[PatchRecord]
    ) -> tuple[PatchRecord, ...]:
        validated: list[PatchRecord] = []
        seen_paths: set[Path] = set()
        for patch in patches:
            if not isinstance(patch, PatchRecord):
                self._local_error("merge patches must be PatchRecord instances", "merge")
            if patch.base_path != base_path:
                self._local_error("patch base_path mismatch", "merge")
            if patch.output_path in seen_paths:
                self._local_error("duplicate patch output_path", "merge")
            if not patch.output_path.is_file():
                self._local_error("missing input artifact: patch output_path", "merge")
            seen_paths.add(patch.output_path)
            validated.append(patch)
        return tuple(validated)

    def _validate_expected_patch(
        self,
        expected: PatchRecord | None,
        *,
        base_path: Path,
        output_path: Path,
        changed: tuple[str, ...],
        patch_hash: str,
        field: str,
        record_id: str,
    ) -> None:
        if expected is None:
            return
        if expected.base_path != base_path or expected.output_path != output_path:
            self._fatal(f"{field} expected patch path mismatch", operation="compare_effect", record_id=record_id)
        if expected.changed_functions != changed or expected.patch_hash != patch_hash:
            self._fatal(f"{field} mismatch", operation="compare_effect", record_id=record_id)

    def _bind_expected_patch_artifacts(
        self,
        expected: PatchRecord | None,
        *,
        base_path: Path,
        output_path: Path,
        base_artifact_sha256: str,
        output_artifact_sha256: str,
        field: str,
        record_id: str,
    ) -> None:
        """Bind a comparison input to the exact bytes its PatchRecord described.

        This happens before the JSONL request.  A replacement at an identical
        path therefore cannot be accepted merely because the path still exists.
        """

        if expected is None:
            return
        if expected.base_path != base_path or expected.output_path != output_path:
            self._fatal(
                f"{field} expected patch path mismatch",
                operation="compare_effect",
                record_id=record_id,
                error_kind="artifact_mismatch",
            )
        if (
            expected.base_artifact_sha256 != base_artifact_sha256
            or expected.output_artifact_sha256 != output_artifact_sha256
        ):
            self._fatal(
                f"{field} expected patch artifact hash mismatch",
                operation="compare_effect",
                record_id=record_id,
                error_kind="artifact_mismatch",
            )

    def _input_file(self, path: str | Path, field: str, operation: str) -> Path:
        resolved = _normal_path(path)
        if not resolved.is_file():
            self._local_error(f"missing input artifact: {field}", operation)
        return resolved

    def _artifact_digest(self, path: Path, field: str, operation: str, record_id: str) -> str:
        try:
            if not path.is_file():
                self._fatal(
                    f"missing artifact: {field}", operation=operation, record_id=record_id
                )
            return _file_digest(path)
        except OSError as error:
            self._fatal(
                f"cannot hash artifact {field}: {error}",
                operation=operation,
                record_id=record_id,
                cause=error,
            )

    def _current_input_digest(self, path: Path, field: str, operation: str) -> str:
        """Hash an already-validated input before deriving its operation ID."""

        try:
            if not path.is_file():
                self._local_error(f"missing input artifact: {field}", operation)
            return _file_digest(path)
        except OSError as error:
            self._local_error(f"cannot hash input artifact {field}: {error}", operation)

    def _function_names(
        self,
        value: object,
        field: str,
        operation: str,
        record_id: str | None = None,
    ) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(name, str) or not name for name in value
        ):
            if record_id is None:
                self._local_error(f"invalid {field}", operation)
            self._fatal(f"invalid {field}", operation=operation, record_id=record_id)
        names = tuple(value)
        if tuple(sorted(names)) != names or len(set(names)) != len(names):
            if record_id is None:
                self._local_error(f"{field} must be sorted and unique", operation)
            self._fatal(
                f"{field} must be sorted and unique", operation=operation, record_id=record_id
            )
        return names

    def _digest_field(
        self, reply: Mapping[str, Any], field: str, operation: str, record_id: str
    ) -> str:
        value = reply.get(field)
        if not _is_digest(value):
            self._fatal(f"missing or invalid {field}", operation=operation, record_id=record_id)
        return value

    def _digest_list(
        self, reply: Mapping[str, Any], field: str, operation: str, record_id: str
    ) -> tuple[str, ...]:
        value = reply.get(field)
        if not isinstance(value, list) or any(not _is_digest(item) for item in value):
            self._fatal(f"missing or invalid {field}", operation=operation, record_id=record_id)
        result = tuple(value)
        if tuple(sorted(result)) != result:
            self._fatal(f"{field} must be canonical", operation=operation, record_id=record_id)
        return result

    def _local_error(self, message: str, operation: str) -> None:
        record_id = _record_id(operation, {"local_error": message})
        raise DirectMergeProtocolError(
            message,
            operation=operation,
            operation_record_id=record_id,
            unavailable=True,
            error_kind="invalid_input",
        )

    def _fatal(
        self,
        message: str,
        *,
        operation: str,
        record_id: str,
        cause: Exception | None = None,
        error_kind: str = "protocol_error",
    ) -> None:
        self.close()
        error = DirectMergeProtocolError(
            message,
            operation=operation,
            operation_record_id=record_id,
            unavailable=True,
            error_kind=error_kind,
        )
        if cause is None:
            raise error
        raise error from cause


# The evaluator below is intentionally kept in this experiment-only module so
# its only merge implementation is the strict LLVM helper above.  It neither
# imports production relation/batch code nor substitutes a pass sequence for a
# direct merge.
_TWO_N_SUCCESS = "success"
_TWO_N_TIMEOUT = "timeout"
_TWO_N_STAGE_STATUSES = frozenset(
    ("success", "invalid", "error", "timeout", "unknown", "not_run")
)


def _two_n_clock_value(clock: Callable[[], int]) -> int:
    value = clock()
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("2N clock must return a non-negative integer nanosecond value")
    return value


def _two_n_elapsed_ms(clock: Callable[[], int], start_ns: int) -> int:
    end_ns = _two_n_clock_value(clock)
    if end_ns < start_ns:
        raise ValueError("2N clock moved backwards")
    # Ceiling avoids silently recording zero for a completed sub-millisecond
    # evaluator run while retaining a non-negative measured integer.
    return (end_ns - start_ns + 999_999) // 1_000_000


def evaluate_group_2n(
    *,
    root_ir: Path,
    group_id: str,
    program_id: str,
    study_manifest_id: str,
    actions: Mapping[str, object] | Sequence[object],
    profiles: Sequence[Mapping[str, object]],
    merge_client: object,
    out_dir: Path,
    run_second: Callable[[Path, object, Path], object],
    verify_ir: Callable[[Path], bool] | None = None,
    pair_observations: Sequence[Mapping[str, object]] = (),
    clock_ns: Callable[[], int] | None = None,
) -> Advisor2NGroupResult:
    """Evaluate the advisor's exact group-level direct-merge 2N rule.

    One retained profile is required for every configured action.  Only after
    all are successful, verifier-valid, and the *complete* patch family is
    mutually disjoint does the evaluator run exactly one direct merge of all
    other first-round outputs and exactly one second-round action per endpoint.
    A failed gate is reported as applicability/coverage evidence, never as a
    soundness conclusion.  This function creates no certificate and keeps
    ``authority_granted=false`` and ``proved_commute=false`` throughout.
    """

    clock = clock_ns or time.perf_counter_ns
    start_ns = _two_n_clock_value(clock)
    root = Path(root_ir).resolve(strict=False)
    if not root.is_file():
        raise ValueError(f"root IR is missing: {root}")
    if not str(group_id).strip() or not str(program_id).strip() or not str(study_manifest_id).strip():
        raise ValueError("group_id, program_id, and study_manifest_id must be non-empty")
    action_items = _two_n_action_items(actions)
    if len(action_items) < 2:
        raise ValueError("advisor 2N evaluation requires at least two configured actions")
    target_root = Path(out_dir).resolve(strict=False)
    target_root.mkdir(parents=True, exist_ok=True)
    profiles_by_action = _two_n_profiles_by_action(profiles, action_items)

    counts = _two_n_profile_counts(profiles_by_action, action_items)
    base_rows = {
        "group_id": str(group_id),
        "program_id": str(program_id),
        "study_manifest_id": str(study_manifest_id),
        "authority_granted": "false",
        "proved_commute": "false",
    }
    first_round_issue = _two_n_first_round_issue(profiles_by_action, action_items)
    if first_round_issue is not None:
        status, reason = first_round_issue
        directional_rows = tuple(
            _two_n_unavailable_directional_row(
                base_rows,
                action_id,
                profiles_by_action[action_id],
                directional_status=status,
                reason=reason,
            )
            for action_id, _ in action_items
        )
        return _two_n_finish_precondition(
            base_rows=base_rows,
            action_items=action_items,
            counts=counts,
            directional_rows=directional_rows,
            profiles_by_action=profiles_by_action,
            round1_status="timeout" if status == "timeout" else "round1_precondition_failed",
            disjoint_status="unknown",
            merge_status="unknown",
            second_status="unknown",
            reason=reason,
            pair_observations=pair_observations,
            wall_time_ms=_two_n_elapsed_ms(clock, start_ns),
        )

    patches: dict[str, PatchRecord] = {}
    patch_error: tuple[str, str] | None = None
    for action_id, _ in action_items:
        profile = profiles_by_action[action_id]
        output = _two_n_profile_output(profile)
        try:
            patch = _two_n_inspect_patch(merge_client, root, output)
        except DirectMergeProtocolError as error:
            patch_error = (_two_n_merge_error_status(error), _two_n_error_reason(error))
            break
        if not isinstance(patch, PatchRecord):
            patch_error = ("unknown", "inspect_patch returned a non-PatchRecord")
            break
        patches[action_id] = patch

    if patch_error is not None or len(patches) != len(action_items):
        status, reason = patch_error or ("unknown", "incomplete first-round patch family")
        directional_status = "timeout" if status == "timeout" else "direct_merge_not_defined"
        directional_rows = tuple(
            _two_n_unavailable_directional_row(
                base_rows,
                action_id,
                profiles_by_action[action_id],
                directional_status=directional_status,
                reason=reason,
                first_patch=patches.get(action_id),
            )
            for action_id, _ in action_items
        )
        return _two_n_finish_precondition(
            base_rows=base_rows,
            action_items=action_items,
            counts=counts,
            directional_rows=directional_rows,
            profiles_by_action=profiles_by_action,
            round1_status="complete",
            disjoint_status="unknown",
            merge_status=status if status in {"timeout", "unknown"} else "direct_merge_not_defined",
            second_status="unknown",
            reason=reason,
            pair_observations=pair_observations,
            wall_time_ms=_two_n_elapsed_ms(clock, start_ns),
        )

    overlapping = _two_n_overlapping_actions(action_items, patches)
    if overlapping:
        reason = "first-round patches overlap: " + ",".join(overlapping)
        directional_rows = tuple(
            _two_n_unavailable_directional_row(
                base_rows,
                action_id,
                profiles_by_action[action_id],
                directional_status="direct_merge_not_defined",
                reason=reason,
                first_patch=patches[action_id],
            )
            for action_id, _ in action_items
        )
        return _two_n_finish_precondition(
            base_rows=base_rows,
            action_items=action_items,
            counts=counts,
            directional_rows=directional_rows,
            profiles_by_action=profiles_by_action,
            round1_status="complete",
            disjoint_status="overlap",
            merge_status="direct_merge_not_defined",
            second_status="unknown",
            reason=reason,
            pair_observations=pair_observations,
            wall_time_ms=_two_n_elapsed_ms(clock, start_ns),
        )

    directional: list[dict[str, object]] = []
    for action_id, action in action_items:
        profile = profiles_by_action[action_id]
        others = [patches[other_id] for other_id, _ in action_items if other_id != action_id]
        merge_path = target_root / action_id / "merged_input.ll"
        try:
            merged = _two_n_merge(merge_client, root, others, merge_path)
        except DirectMergeProtocolError as error:
            merge_status = _two_n_merge_error_status(error)
            directional.append(
                _two_n_unavailable_directional_row(
                    base_rows,
                    action_id,
                    profile,
                    directional_status=("timeout" if merge_status == "timeout" else "direct_merge_not_defined" if merge_status == "direct_merge_not_defined" else "merge_invalid" if merge_status == "merge_invalid" else "unknown"),
                    reason=_two_n_error_reason(error),
                    first_patch=patches[action_id],
                    merged_input_status=merge_status,
                )
            )
            continue
        if not isinstance(merged, MergeRecord):
            directional.append(
                _two_n_unavailable_directional_row(
                    base_rows,
                    action_id,
                    profile,
                    directional_status="unknown",
                    reason="merge returned a non-MergeRecord",
                    first_patch=patches[action_id],
                    merged_input_status="unknown",
                )
            )
            continue
        if merged.merge_input_count != len(others) or tuple(sorted(merged.input_patch_hashes)) != tuple(
            sorted(patch.patch_hash for patch in others)
        ):
            directional.append(
                _two_n_unavailable_directional_row(
                    base_rows,
                    action_id,
                    profile,
                    directional_status="unknown",
                    reason="merge did not bind exactly the N-1 first-round patches",
                    first_patch=patches[action_id],
                    merged_input_status="unknown",
                )
            )
            continue

        second_path = target_root / action_id / "second_round.ll"
        second = _two_n_run_second(run_second, merged.merged_path, action, second_path, verify_ir)
        if second["status"] != _TWO_N_SUCCESS:
            directional.append(
                _two_n_second_failure_row(
                    base_rows,
                    action_id,
                    profile,
                    patches[action_id],
                    merged,
                    second,
                )
            )
            continue
        try:
            effect = _two_n_compare_effect(
                merge_client,
                first_base=root,
                first_output=_two_n_profile_output(profile),
                second_base=merged.merged_path,
                second_output=second["output_path"],
                protected_functions=merged.contributed_functions,
                expected_first_patch=patches[action_id],
            )
        except DirectMergeProtocolError as error:
            directional.append(
                _two_n_unavailable_directional_row(
                    base_rows,
                    action_id,
                    profile,
                    directional_status="timeout" if _two_n_merge_error_status(error) == "timeout" else "unknown",
                    reason=_two_n_error_reason(error),
                    first_patch=patches[action_id],
                    merged_input_status="complete",
                    merged=merged,
                    second=second,
                )
            )
            continue
        if not isinstance(effect, EffectRecord):
            directional.append(
                _two_n_unavailable_directional_row(
                    base_rows,
                    action_id,
                    profile,
                    directional_status="unknown",
                    reason="compare_effect returned a non-EffectRecord",
                    first_patch=patches[action_id],
                    merged_input_status="complete",
                    merged=merged,
                    second=second,
                )
            )
            continue
        same_effect = (
            effect.same_effect
            and effect.first_changed_functions == effect.second_changed_functions
            and effect.first_patch_hash == effect.second_patch_hash
            and effect.protected_functions_preserved
            and effect.skeletons_unchanged
            and effect.symbol_inventories_unchanged
        )
        directional.append(
            _two_n_directional_row(
                base_rows,
                action_id,
                profile,
                patches[action_id],
                merged=merged,
                second=second,
                effect=effect,
                directional_status=("authorized_all_others" if same_effect else "rejected_effect_changed"),
                reason="" if same_effect else "second-round effect differs from first-round effect",
            )
        )

    directional_rows = tuple(directional)
    merge_status = _two_n_all_merge_status(directional_rows)
    second_status = _two_n_all_second_status(directional_rows)
    group_status = _two_n_group_authorization(directional_rows, merge_status, second_status)
    group_reason = _two_n_group_reason(directional_rows, group_status)
    return _two_n_finish(
        base_rows=base_rows,
        action_items=action_items,
        counts=counts,
        directional_rows=directional_rows,
        profiles_by_action=profiles_by_action,
        round1_status="complete",
        disjoint_status="disjoint",
        merge_status=merge_status,
        second_status=second_status,
        group_status=group_status,
        reason=group_reason,
        pair_observations=pair_observations,
        wall_time_ms=_two_n_elapsed_ms(clock, start_ns),
    )


def _two_n_action_items(actions: Mapping[str, object] | Sequence[object]) -> tuple[tuple[str, object], ...]:
    if isinstance(actions, Mapping):
        # Mapping insertion order is caller-controlled, so normalize only its
        # keys while retaining the mapped action object passed to the runner.
        # This keeps row IDs, pair endpoint orientation, merge inputs, and
        # action invocation order deterministic without replacing a rich
        # ActionRecord (or other runner payload) with its string key.
        items = tuple(sorted(((str(action_id), action) for action_id, action in actions.items()), key=lambda item: item[0]))
    else:
        items = tuple((_two_n_text(getattr(action, "action_id", "")), action) for action in actions)
    if not items or any(not action_id for action_id, _ in items):
        raise ValueError("every configured action requires a non-empty action_id")
    action_ids = tuple(action_id for action_id, _ in items)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("configured actions must have unique action_id values")
    return items


def _two_n_profiles_by_action(
    profiles: Sequence[Mapping[str, object]], action_items: Sequence[tuple[str, object]]
) -> dict[str, Mapping[str, object]]:
    selected = {action_id for action_id, _ in action_items}
    rows: dict[str, Mapping[str, object]] = {}
    for profile in profiles:
        action_id = _two_n_text(profile.get("action_id", ""))
        if action_id not in selected:
            continue
        if action_id in rows:
            raise ValueError(f"duplicate first-round profile for action {action_id}")
        rows[action_id] = profile
    missing = sorted(selected - set(rows))
    if missing:
        raise ValueError(f"missing first-round profile(s): {', '.join(missing)}")
    return rows


def _two_n_profile_counts(
    profiles: Mapping[str, Mapping[str, object]], action_items: Sequence[tuple[str, object]]
) -> dict[str, int]:
    values = [profiles[action_id] for action_id, _ in action_items]
    execution = [_two_n_text(row.get("execution_status", "unknown")) for row in values]
    return {
        "configured_n": len(values),
        "successful_n": sum(status == _TWO_N_SUCCESS for status in execution),
        "active_n": sum(
            status == _TWO_N_SUCCESS and _two_n_text(row.get("activity_status")) == "active"
            for row, status in zip(values, execution)
        ),
        "no_op_n": sum(
            status == _TWO_N_SUCCESS and _two_n_text(row.get("activity_status")) == "no_op"
            for row, status in zip(values, execution)
        ),
        "failed_n": sum(status in {"invalid", "error", "unknown", "not_run"} for status in execution),
        "timeout_n": sum(status == _TWO_N_TIMEOUT for status in execution),
    }


def _two_n_first_round_issue(
    profiles: Mapping[str, Mapping[str, object]], action_items: Sequence[tuple[str, object]]
) -> tuple[str, str] | None:
    reasons: list[str] = []
    timed_out = False
    for action_id, _ in action_items:
        profile = profiles[action_id]
        status = _two_n_text(profile.get("execution_status", "unknown"))
        verifier = _two_n_text(profile.get("verifier_status", "unknown"))
        output = _two_n_profile_output(profile, required=False)
        if status == _TWO_N_TIMEOUT:
            timed_out = True
            reasons.append(f"{action_id}: first-round timeout")
        elif status != _TWO_N_SUCCESS:
            reasons.append(f"{action_id}: first-round status={status or 'unknown'}")
        elif verifier != _TWO_N_SUCCESS:
            reasons.append(f"{action_id}: first-round verifier={verifier or 'unknown'}")
        elif output is None or not output.is_file():
            reasons.append(f"{action_id}: missing first-round output")
    if not reasons:
        return None
    return ("timeout" if timed_out else "round1_precondition_failed", "; ".join(reasons))


def _two_n_profile_output(profile: Mapping[str, object], *, required: bool = True) -> Path | None:
    value = profile.get("output_path", "")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        if required:
            raise ValueError("successful first-round profile requires output_path")
        return None
    return Path(value).resolve(strict=False)


def _two_n_overlapping_actions(
    action_items: Sequence[tuple[str, object]], patches: Mapping[str, PatchRecord]
) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    overlaps: set[str] = set()
    for action_id, _ in action_items:
        for function in patches[action_id].changed_functions:
            earlier = seen.get(function)
            if earlier is not None:
                overlaps.update((earlier, action_id))
            else:
                seen[function] = action_id
    return tuple(sorted(overlaps))


def _two_n_inspect_patch(client: object, base: Path, output: Path) -> object:
    method = getattr(client, "inspect_patch", None)
    if not callable(method):
        raise TypeError("merge_client must provide inspect_patch")
    return method(base, output)


def _two_n_merge(client: object, base: Path, patches: Sequence[PatchRecord], target: Path) -> object:
    method = getattr(client, "merge", None)
    if not callable(method):
        raise TypeError("merge_client must provide merge")
    return method(base, list(patches), target)


def _two_n_compare_effect(client: object, **kwargs: object) -> object:
    method = getattr(client, "compare_effect", None)
    if not callable(method):
        raise TypeError("merge_client must provide compare_effect")
    return method(**kwargs)


def _two_n_run_second(
    runner: Callable[[Path, object, Path], object],
    base: Path,
    action: object,
    output: Path,
    verify_ir: Callable[[Path], bool] | None,
) -> dict[str, object]:
    try:
        raw = runner(base, action, output)
    except TimeoutError as error:
        return _two_n_stage("timeout", "not_run", output, reason=f"second-round timeout: {error}")
    except Exception as error:
        return _two_n_stage("error", "not_run", output, reason=f"second-round runner error: {type(error).__name__}")
    timed_out = bool(_two_n_value(raw, "timed_out", False))
    explicit = _two_n_text(_two_n_value(raw, "execution_status", _two_n_value(raw, "status", "")))
    if timed_out:
        status = "timeout"
    elif explicit in _TWO_N_STAGE_STATUSES:
        status = explicit
    elif bool(_two_n_value(raw, "success", False)):
        status = "success"
    else:
        status = "error"
    actual_output = _two_n_value(raw, "output_path", output)
    path = Path(actual_output).resolve(strict=False) if isinstance(actual_output, (str, Path)) else output
    verifier = _two_n_text(_two_n_value(raw, "verifier_status", ""))
    if status == "success" and (not path.is_file()):
        status, verifier = "error", "not_run"
        reason = "second-round runner reported success without output artifact"
    elif status == "success" and verify_ir is not None:
        try:
            verifier_ok = bool(verify_ir(path))
        except Exception as error:
            verifier_ok = False
            reason = f"second-round verifier error: {type(error).__name__}"
        else:
            reason = "" if verifier_ok else "second-round verifier rejected output"
        if not verifier_ok:
            status, verifier = "invalid", "invalid"
        else:
            verifier = "success"
    else:
        reason = ""
        if status == "success":
            # A second-round success is not usable for a 2N authorization
            # unless verifier success is explicit.  The runner may otherwise
            # have omitted verification entirely, and treating that omission
            # as success would turn unavailable evidence into an authorization.
            if verifier != "success":
                status = "invalid"
                reason = "second-round verifier status is missing or not success"
        else:
            verifier = verifier or "not_run"
    return _two_n_stage(
        status,
        verifier,
        path,
        hard_state_id=_two_n_text(_two_n_value(raw, "hard_state_id", _two_n_value(raw, "output_hard_state_id", ""))),
        stderr=_two_n_text(_two_n_value(raw, "stderr", "")),
        command=_two_n_command(_two_n_value(raw, "command", ())),
        physical=_two_n_nonnegative_int(_two_n_value(raw, "physical_pass_invocations", 1)),
        reason=reason,
    )


def _two_n_stage(
    status: str,
    verifier_status: str,
    output_path: Path,
    *,
    hard_state_id: str = "",
    stderr: str = "",
    command: tuple[str, ...] = (),
    physical: int = 1,
    reason: str = "",
) -> dict[str, object]:
    return {
        "status": status,
        "verifier_status": verifier_status,
        "output_path": output_path,
        "hard_state_id": hard_state_id,
        "stderr": stderr,
        "command": command,
        "physical_pass_invocations": physical,
        "reason": reason,
    }


def _two_n_directional_row(
    base: Mapping[str, object],
    action_id: str,
    profile: Mapping[str, object],
    first_patch: PatchRecord,
    *,
    merged: MergeRecord,
    second: Mapping[str, object],
    effect: EffectRecord,
    directional_status: str,
    reason: str,
) -> dict[str, object]:
    return _two_n_row(
        base,
        action_id,
        profile,
        directional_status=directional_status,
        first_patch=first_patch,
        merged_input_status="complete",
        merged=merged,
        second=second,
        effect=effect,
        reason=reason,
    )


def _two_n_second_failure_row(
    base: Mapping[str, object],
    action_id: str,
    profile: Mapping[str, object],
    first_patch: PatchRecord,
    merged: MergeRecord,
    second: Mapping[str, object],
) -> dict[str, object]:
    status = _two_n_text(second["status"])
    return _two_n_row(
        base,
        action_id,
        profile,
        directional_status="timeout" if status == "timeout" else "second_round_failed",
        first_patch=first_patch,
        merged_input_status="complete",
        merged=merged,
        second=second,
        reason=_two_n_text(second.get("reason", "second-round execution failed")),
    )


def _two_n_unavailable_directional_row(
    base: Mapping[str, object],
    action_id: str,
    profile: Mapping[str, object],
    *,
    directional_status: str,
    reason: str,
    first_patch: PatchRecord | None = None,
    merged_input_status: str = "unknown",
    merged: MergeRecord | None = None,
    second: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return _two_n_row(
        base,
        action_id,
        profile,
        directional_status=directional_status,
        first_patch=first_patch,
        merged_input_status=merged_input_status,
        merged=merged,
        second=second,
        reason=reason,
    )


def _two_n_row(
    base: Mapping[str, object],
    action_id: str,
    profile: Mapping[str, object],
    *,
    directional_status: str,
    first_patch: PatchRecord | None = None,
    merged_input_status: str,
    merged: MergeRecord | None = None,
    second: Mapping[str, object] | None = None,
    effect: EffectRecord | None = None,
    reason: str = "",
) -> dict[str, object]:
    first_status = _two_n_text(profile.get("execution_status", "unknown")) or "unknown"
    second_status = _two_n_text(second.get("status", "not_run")) if second is not None else "not_run"
    if second_status not in _TWO_N_STAGE_STATUSES:
        second_status = "unknown"
    row_id = _record_id(
        "advisor-2n-directional",
        {
            "study_manifest_id": base["study_manifest_id"],
            "group_id": base["group_id"],
            "program_id": base["program_id"],
            "action_id": action_id,
        },
    )
    merged_sha = merged.merged_artifact_sha256 if merged is not None else ""
    second_path = Path(str(second["output_path"])).resolve(strict=False) if second is not None else None
    second_sha = (
        hashlib.sha256(second_path.read_bytes()).hexdigest()
        if second_path is not None and second_path.is_file()
        else ""
    )
    return {
        **base,
        "row_id": row_id,
        "action_id": action_id,
        "directional_status": directional_status,
        "first_round_status": first_status,
        "first_round_effect_sha256": first_patch.patch_hash if first_patch else "",
        "first_output_path": str(_two_n_profile_output(profile, required=False) or ""),
        "merged_input_status": merged_input_status,
        "merged_input_hard_state_id": (
            _two_n_materialized_hard_state_id(merged.merged_path)
            if merged is not None
            else ""
        ),
        "merged_input_sha256": merged_sha,
        "merged_input_path": str(merged.merged_path) if merged is not None else "",
        "second_round_status": second_status,
        "second_round_effect_sha256": effect.second_patch_hash if effect is not None else "",
        "second_output_path": str(second_path) if second_path is not None else "",
        "second_output_sha256": second_sha,
        "second_output_materialized": "true" if second_sha else "false",
        "other_contributions_preserved": (
            "true" if effect is not None and effect.protected_functions_preserved else "false"
        ),
        "verifier_status": _two_n_text(second.get("verifier_status", "not_run")) if second else "not_run",
        "logical_pass_applications": 2 if second is not None else 1,
        "physical_pass_invocations": _two_n_nonnegative_int(profile.get("physical_pass_invocations", 1)) + (
            _two_n_nonnegative_int(second.get("physical_pass_invocations", 1)) if second else 0
        ),
        "merge_helper_calls": int(first_patch is not None) + int(merged is not None) + int(effect is not None),
        "artifact_id": merged.operation_record_id if merged is not None else "",
        "cleanup_status": "not_eligible",
        "fail_closed_reason": reason,
        "command_sha256": hashlib.sha256("\0".join(second.get("command", ())).encode("utf-8")).hexdigest() if second else "",
        "stderr_sha256": hashlib.sha256(_two_n_text(second.get("stderr", "")).encode("utf-8")).hexdigest() if second else "",
        "wall_time_ms": 0,
    }


def _two_n_materialized_hard_state_id(path: Path) -> str:
    """Use the frozen Phasebatch hard-state policy for retained 2N IR."""

    from phasebatch.ir_equivalence import DEFAULT_HARD_STATE_POLICY, hard_state_hash

    digest = hard_state_hash(Path(path), DEFAULT_HARD_STATE_POLICY)
    if len(digest) != 64:
        raise ValueError("2N merged input hard-state hash is malformed")
    return digest


def _two_n_finish_precondition(
    *,
    base_rows: Mapping[str, object],
    action_items: Sequence[tuple[str, object]],
    counts: Mapping[str, int],
    directional_rows: tuple[Mapping[str, object], ...],
    profiles_by_action: Mapping[str, Mapping[str, object]],
    round1_status: str,
    disjoint_status: str,
    merge_status: str,
    second_status: str,
    reason: str,
    pair_observations: Sequence[Mapping[str, object]],
    wall_time_ms: int,
) -> Advisor2NGroupResult:
    return _two_n_finish(
        base_rows=base_rows,
        action_items=action_items,
        counts=counts,
        directional_rows=directional_rows,
        profiles_by_action=profiles_by_action,
        round1_status=round1_status,
        disjoint_status=disjoint_status,
        merge_status=merge_status,
        second_status=second_status,
        group_status="group_precondition_unavailable",
        reason=reason,
        pair_observations=pair_observations,
        wall_time_ms=wall_time_ms,
    )


def _two_n_finish(
    *,
    base_rows: Mapping[str, object],
    action_items: Sequence[tuple[str, object]],
    counts: Mapping[str, int],
    directional_rows: tuple[Mapping[str, object], ...],
    profiles_by_action: Mapping[str, Mapping[str, object]],
    round1_status: str,
    disjoint_status: str,
    merge_status: str,
    second_status: str,
    group_status: str,
    reason: str,
    pair_observations: Sequence[Mapping[str, object]],
    wall_time_ms: int,
) -> Advisor2NGroupResult:
    group_id = _two_n_text(base_rows["group_id"])
    program_id = _two_n_text(base_rows["program_id"])
    manifest_id = _two_n_text(base_rows["study_manifest_id"])
    pair_rows = _two_n_pair_rows(
        base_rows,
        action_items,
        directional_rows,
        pair_observations,
    )
    first_physical = sum(
        _two_n_nonnegative_int(profiles_by_action[action_id].get("physical_pass_invocations", 1))
        for action_id, _ in action_items
    )
    second_physical = sum(
        max(0, _two_n_nonnegative_int(row.get("physical_pass_invocations", 0)) - _two_n_nonnegative_int(profiles_by_action[_two_n_text(row["action_id"])].get("physical_pass_invocations", 1)))
        for row in directional_rows
    )
    helper_calls = len(action_items) if round1_status == "complete" else 0
    helper_calls += sum(_two_n_nonnegative_int(row.get("merge_helper_calls", 0)) - int(bool(row.get("first_round_effect_sha256"))) for row in directional_rows)
    group_row: dict[str, object] = {
        **base_rows,
        "row_id": _record_id("advisor-2n-group", {"study_manifest_id": manifest_id, "group_id": group_id, "program_id": program_id}),
        **counts,
        "round1_status": round1_status,
        "first_round_disjoint_status": disjoint_status,
        "all_n_merge_status": merge_status,
        "all_n_second_round_status": second_status,
        "group_authorization_status": group_status,
        "directional_authorized_count": sum(row["directional_status"] == "authorized_all_others" for row in directional_rows),
        "directional_unavailable_count": sum(row["directional_status"] != "authorized_all_others" for row in directional_rows),
        "logical_pass_applications": counts["configured_n"] + sum(row["second_round_status"] != "not_run" for row in directional_rows),
        "physical_pass_invocations": first_physical + second_physical,
        "merge_helper_calls": helper_calls,
        # Detailed timing is filled by the frozen runner when it materializes
        # evidence.  Keep explicit zero fields here so the aggregate contract
        # never silently substitutes a missing component.
        "merge_construction_time_ms": 0,
        "parse_time_ms": 0,
        "verifier_time_ms": 0,
        "worker_time_ms": 0,
        "replay_time_ms": 0,
        "fail_closed_reason": reason,
        "source_row_ids": _canonical_json([row["row_id"] for row in directional_rows]),
        "wall_time_ms": wall_time_ms,
    }
    return Advisor2NGroupResult(
        group_row=MappingProxyType(group_row),
        directional_rows=tuple(MappingProxyType(dict(row)) for row in directional_rows),
        pair_rows=tuple(MappingProxyType(row) for row in pair_rows),
    )


def _two_n_pair_rows(
    base: Mapping[str, object],
    action_items: Sequence[tuple[str, object]],
    directional_rows: Sequence[Mapping[str, object]],
    observations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    by_action = {_two_n_text(row["action_id"]): row for row in directional_rows}
    lookup = _two_n_pair_observation_lookup(base, observations)
    rows: list[dict[str, object]] = []
    for index, (left, _) in enumerate(action_items):
        for right, _ in action_items[index + 1 :]:
            left_row, right_row = by_action[left], by_action[right]
            left_status = _two_n_text(left_row["directional_status"])
            right_status = _two_n_text(right_row["directional_status"])
            observed = lookup.get(tuple(sorted((left, right))))
            derived = derive_two_n_pair_fields(
                left_status,
                right_status,
                observed.get("dynamic_result", "unknown") if observed else "unknown",
                observation_available=observed is not None,
            )
            row_id = _record_id(
                "advisor-2n-pair",
                {
                    "study_manifest_id": base["study_manifest_id"],
                    "group_id": base["group_id"],
                    "program_id": base["program_id"],
                    "action_a_id": left,
                    "action_b_id": right,
                },
            )
            rows.append(
                {
                    **base,
                    "row_id": row_id,
                    "action_a_id": left,
                    "action_b_id": right,
                    **derived,
                    "pair_observation_row_id": _two_n_text(observed.get("row_id", "")) if observed else "",
                    # Task11 upgrades this only after all six mandatory replay
                    # observations agree; its vocabulary is strictly boolean.
                    "stable_false_authorization": "false",
                    "worker_replay_status": (
                        "unavailable" if derived["false_authorization"] == "true" else "not_required"
                    ),
                    "external_opt_replay_status": (
                        "unavailable" if derived["false_authorization"] == "true" else "not_required"
                    ),
                    "two_n_replay_status": (
                        "unavailable" if derived["false_authorization"] == "true" else "not_required"
                    ),
                    "replay_artifact_id": "",
                    "replay_time_ms": 0,
                    "fail_closed_reason": "" if derived["validation_status"] in {"agree", "false_authorization"} else "AB/BA ground truth unavailable for an authorization" if (left_status == "authorized_all_others" or right_status == "authorized_all_others") else "no 2N directional authorization",
                    "source_row_ids": _canonical_json([left_row["row_id"], right_row["row_id"]]),
                }
            )
    return tuple(rows)


def derive_two_n_pair_fields(
    action_a_directional_status: object,
    action_b_directional_status: object,
    dynamic_result: object,
    *,
    observation_available: bool,
) -> dict[str, str]:
    """Derive semantic 2N-pair fields from directional and AB/BA evidence."""

    left_status = _two_n_text(action_a_directional_status)
    right_status = _two_n_text(action_b_directional_status)
    dynamic = _two_n_text(dynamic_result) if observation_available else "unknown"
    left_authorized = left_status == "authorized_all_others"
    right_authorized = right_status == "authorized_all_others"
    if left_authorized and right_authorized:
        pair_status = "both_directions_authorized"
    elif left_authorized or right_authorized:
        pair_status = "one_direction_only"
    elif left_status == right_status == "rejected_effect_changed":
        pair_status = "both_rejected"
    else:
        pair_status = "group_precondition_unavailable"

    if left_authorized or right_authorized:
        if dynamic in {"order_sensitive", "failed"}:
            validation, false_authorization = "false_authorization", "true"
        elif dynamic == "commute":
            validation, false_authorization = "agree", "false"
        elif dynamic == "timeout":
            validation, false_authorization = "ground_truth_timeout", "false"
        elif dynamic == "unknown" or not observation_available:
            validation = (
                "ground_truth_unknown" if observation_available else "unavailable"
            )
            false_authorization = "false"
        else:
            validation, false_authorization = "unknown", "false"
    else:
        validation, false_authorization = "unavailable", "false"
    return {
        "action_a_directional_status": left_status,
        "action_b_directional_status": right_status,
        "two_n_pair_status": pair_status,
        "dynamic_result": dynamic,
        "validation_status": validation,
        "false_authorization": false_authorization,
    }


def _two_n_pair_observation_lookup(
    base: Mapping[str, object], observations: Sequence[Mapping[str, object]]
) -> dict[tuple[str, str], Mapping[str, object]]:
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in observations:
        if _two_n_text(row.get("study_manifest_id", "")) != _two_n_text(
            base["study_manifest_id"]
        ):
            continue
        if _two_n_text(row.get("program_id", "")) != _two_n_text(base["program_id"]):
            continue
        observed_group = _two_n_text(row.get("group_id", ""))
        if observed_group and observed_group != _two_n_text(base["group_id"]):
            continue
        left, right = _two_n_text(row.get("action_a_id", "")), _two_n_text(row.get("action_b_id", ""))
        if not left or not right or left == right:
            continue
        key = tuple(sorted((left, right)))
        if key in result:
            raise ValueError(f"duplicate AB/BA pair observation for {key[0]}/{key[1]}")
        result[key] = row
    return result


def _two_n_all_merge_status(rows: Sequence[Mapping[str, object]]) -> str:
    statuses = {_two_n_text(row.get("merged_input_status", "unknown")) for row in rows}
    if statuses == {"complete"}:
        return "complete"
    if "timeout" in statuses:
        return "timeout"
    if "merge_invalid" in statuses:
        return "merge_invalid"
    if "direct_merge_not_defined" in statuses:
        return "direct_merge_not_defined"
    return "unknown"


def _two_n_all_second_status(rows: Sequence[Mapping[str, object]]) -> str:
    statuses = {_two_n_text(row.get("second_round_status", "unknown")) for row in rows}
    if all(status == "success" for status in statuses):
        return "complete"
    if "timeout" in statuses:
        return "timeout"
    if statuses & {"error", "invalid"}:
        return "second_round_failed"
    return "unknown"


def _two_n_group_authorization(
    rows: Sequence[Mapping[str, object]], merge_status: str, second_status: str
) -> str:
    statuses = {_two_n_text(row["directional_status"]) for row in rows}
    if statuses == {"authorized_all_others"}:
        return "authorized"
    if merge_status != "complete" or second_status != "complete":
        return "group_precondition_unavailable"
    if statuses <= {"authorized_all_others", "rejected_effect_changed"}:
        return "rejected"
    return "unknown"


def _two_n_group_reason(rows: Sequence[Mapping[str, object]], group_status: str) -> str:
    if group_status == "authorized":
        return ""
    return "; ".join(
        _two_n_text(row.get("fail_closed_reason", ""))
        for row in rows
        if _two_n_text(row.get("fail_closed_reason", ""))
    )


def _two_n_merge_error_status(error: DirectMergeProtocolError) -> str:
    kind = _two_n_text(error.error_kind).lower()
    if kind == "timeout":
        return "timeout"
    if kind == "merge_invalid":
        return "merge_invalid"
    if isinstance(error, DirectMergeUnavailable) or kind in {
        "patch_not_mergeable",
        "module_skeleton_changed",
        "non_function_change",
        "function_conflict",
    }:
        return "direct_merge_not_defined"
    return "unknown"


def _two_n_error_reason(error: DirectMergeProtocolError) -> str:
    kind = _two_n_text(error.error_kind) or "unknown"
    return f"{error.operation}:{kind}:{error}"


def _two_n_value(value: object, field: str, default: object) -> object:
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _two_n_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _two_n_nonnegative_int(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _two_n_command(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) for item in value):
        return ()
    return tuple(value)
