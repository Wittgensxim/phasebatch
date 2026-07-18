"""Frozen, deterministic program and study manifests for the isolated study.

The module deliberately has no dependency on Phasebatch authority code.  It
only freezes input identities and chooses compile/root-IR-valid programs before
any pair or advisor-2N observation exists.  That separation is important: a
program may never be selected, replaced, or removed because of an experimental
outcome.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any


STUDY_SCHEMA_VERSION = "advisor-pair-scale-2n/v1"
DEFAULT_SELECTION_SEED = 0
DEFAULT_MAX_SOURCE_BYTES = 200_000
DEFAULT_CATEGORY_CAP = 5
FORMAL_SOURCE_INVENTORY_COUNT = 50
FORMAL_PROGRAM_TARGET = 10
FORMAL_SELECTION_RULE_ID = "systematic_midpoint_fixed50_n10_v1"
FORMAL_SOURCE_POSITIONS = (3, 8, 13, 18, 23, 28, 33, 38, 43, 48)
FORMAL_SAMPLING_FRAME_SCHEMA_VERSION = (
    "advisor-pair-scale-2n/formal-sampling-frame-v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_TOOL_NAMES = frozenset({"opt", "clang", "worker", "merge_helper"})
_TOOL_RECORD_OPTIONAL_FIELDS = frozenset({"size_bytes", "version"})


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_nonempty(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _require_sha256(value: object, field: str, *, allow_empty: bool = False) -> str:
    text = str(value).strip()
    if allow_empty and not text:
        return ""
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def normalize_program_id(value: object, *, field: str = "program_id") -> str:
    """Return one canonical, output-safe program identifier.

    Program identities are manifest authority.  Coercing another JSON type,
    trimming whitespace, or accepting a path-shaped value would make a signed
    identity compare differently at different call sites.
    """

    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    if (
        not value
        or value in {".", ".."}
        or any(mark in value for mark in ("/", "\\"))
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def normalize_program_relative_path(value: object) -> str:
    """Apply the canonical ``ProgramRecord`` SingleSource path policy."""

    text = _require_nonempty(value, "relative_path").replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path must be a non-escaping relative path")
    if not path.parts or path.parts[0] != "SingleSource":
        raise ValueError("relative_path must start with SingleSource")
    if path.suffix.lower() != ".c":
        raise ValueError("relative_path must identify a C source")
    return path.as_posix()


def _normalise_source_path(value: object) -> str:
    text = _require_nonempty(value, "source_path")
    # ``Path.is_absolute`` is host-dependent; recognising Windows spelling
    # explicitly keeps manifests portable when inspected on another host.
    if not Path(text).is_absolute() and not PureWindowsPath(text).is_absolute():
        raise ValueError("source_path must be absolute")
    return text


def _canonical_value(value: object) -> object:
    """Return a JSON-compatible value with deterministic mapping ordering."""

    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(
            (_canonical_value(item) for item in value),
            key=lambda item: _canonical_json(item),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical manifest value: {type(value)!r}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON so logical input order cannot affect identity."""

    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def stable_rank(seed: int, value: str) -> str:
    """Match the frozen ``advisor_benchmarks`` stable-rank policy exactly."""

    return _sha256_bytes(f"{seed}\0{value}".encode("utf-8"))


@dataclass(frozen=True)
class ProgramRecord:
    """One source/root-IR preflight record, independent of pair outcomes."""

    program_id: str
    source_path: str
    relative_path: str
    program_family: str
    source_sha256: str
    source_size_bytes: int
    compile_command: tuple[str, ...]
    compile_status: str
    compile_stderr_sha256: str
    root_ir_path: str
    root_ir_sha256: str
    root_hard_state_id: str
    target: str
    data_layout: str
    preflight_status: str
    selection_class: str = "candidate"
    selection_order: int | None = None
    reserve_rank: int | None = None
    replacement_for_program_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", normalize_program_id(self.program_id))
        object.__setattr__(self, "source_path", _normalise_source_path(self.source_path))
        object.__setattr__(
            self, "relative_path", normalize_program_relative_path(self.relative_path)
        )
        family = _require_nonempty(self.program_family, "program_family").replace(
            "\\", "/"
        )
        family_path = PurePosixPath(family)
        if family_path.is_absolute() or any(
            part in {"", ".", ".."} for part in family_path.parts
        ):
            raise ValueError("program_family must be a non-escaping relative path")
        expected_family = PurePosixPath(self.relative_path).parent.as_posix()
        if family_path.as_posix() != expected_family:
            raise ValueError("program_family must equal source parent")
        object.__setattr__(
            self, "program_family", expected_family
        )
        object.__setattr__(
            self, "source_sha256", _require_sha256(self.source_sha256, "source_sha256")
        )
        if (
            not isinstance(self.source_size_bytes, int)
            or isinstance(self.source_size_bytes, bool)
            or self.source_size_bytes < 0
        ):
            raise ValueError("source_size_bytes must be a non-negative integer")
        command = tuple(str(part) for part in self.compile_command)
        if not command or not all(part.strip() for part in command):
            raise ValueError("compile_command must contain non-empty command parts")
        object.__setattr__(self, "compile_command", command)
        object.__setattr__(
            self, "compile_status", _require_nonempty(self.compile_status, "compile_status")
        )
        object.__setattr__(
            self,
            "compile_stderr_sha256",
            _require_sha256(
                self.compile_stderr_sha256,
                "compile_stderr_sha256",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self, "root_ir_path", _require_nonempty(self.root_ir_path, "root_ir_path")
        )
        object.__setattr__(
            self,
            "root_ir_sha256",
            _require_sha256(self.root_ir_sha256, "root_ir_sha256", allow_empty=True),
        )
        object.__setattr__(
            self,
            "root_hard_state_id",
            str(self.root_hard_state_id).strip(),
        )
        object.__setattr__(self, "target", _require_nonempty(self.target, "target"))
        object.__setattr__(
            self, "data_layout", _require_nonempty(self.data_layout, "data_layout")
        )
        object.__setattr__(
            self,
            "preflight_status",
            _require_nonempty(self.preflight_status, "preflight_status"),
        )
        object.__setattr__(
            self,
            "selection_class",
            _require_nonempty(self.selection_class, "selection_class"),
        )
        if self.selection_order is not None and (
            not isinstance(self.selection_order, int)
            or isinstance(self.selection_order, bool)
            or self.selection_order < 1
        ):
            raise ValueError("selection_order must be positive when present")
        if self.reserve_rank is not None and (
            not isinstance(self.reserve_rank, int)
            or isinstance(self.reserve_rank, bool)
            or self.reserve_rank < 1
        ):
            raise ValueError("reserve_rank must be positive when present")
        object.__setattr__(
            self,
            "replacement_for_program_id",
            str(self.replacement_for_program_id).strip(),
        )

    @property
    def compile_command_sha256(self) -> str:
        return canonical_sha256(list(self.compile_command))

    @property
    def selection_eligible(self) -> bool:
        return (
            self.source_size_bytes <= DEFAULT_MAX_SOURCE_BYTES
            and self.compile_status == "success"
            and self.preflight_status == "success"
            and bool(self.root_ir_sha256)
        )

    def as_manifest_record(self) -> dict[str, object]:
        """Return the complete machine-readable program-manifest row."""

        return {
            "program_id": self.program_id,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "program_family": self.program_family,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "compile_command": list(self.compile_command),
            "compile_command_sha256": self.compile_command_sha256,
            "compile_status": self.compile_status,
            "compile_stderr_sha256": self.compile_stderr_sha256,
            "root_ir_path": self.root_ir_path,
            "root_ir_sha256": self.root_ir_sha256,
            "root_hard_state_id": self.root_hard_state_id,
            "target": self.target,
            "data_layout": self.data_layout,
            "preflight_status": self.preflight_status,
            "selection_class": self.selection_class,
            "selection_order": self.selection_order,
            "reserve_rank": self.reserve_rank,
            "replacement_for_program_id": self.replacement_for_program_id,
        }


@dataclass(frozen=True)
class PreflightLedgerEntry:
    """An auditable non-selection reason that remains outside pair results."""

    program_id: str
    relative_path: str
    source_sha256: str
    reserve_rank: int
    compile_status: str
    preflight_status: str
    reason: str


@dataclass(frozen=True)
class FrozenProgramManifest:
    """Final programs plus the fixed reserve ordering and retained failures."""

    programs: tuple[ProgramRecord, ...]
    reserve_order: tuple[ProgramRecord, ...]
    preflight_ledger: tuple[PreflightLedgerEntry, ...]
    target: int
    selection_seed: int
    per_category_cap: int
    max_source_bytes: int

    @property
    def program_manifest_sha256(self) -> str:
        return canonical_sha256(
            [record.as_manifest_record() for record in self.programs]
        )


def _record_path_key(record: ProgramRecord) -> str:
    return record.relative_path.casefold()


def _is_exact_fixed_inventory_copy(
    candidate: ProgramRecord, fixed: ProgramRecord
) -> bool:
    return (
        candidate.source_sha256 == fixed.source_sha256
        and candidate.relative_path.casefold() == fixed.relative_path.casefold()
        and _canonical_source_path(candidate.source_path)
        == _canonical_source_path(fixed.source_path)
    )


def _canonical_source_path(value: str) -> str:
    """Canonicalize absolute source spelling independent of the host OS."""

    text = _normalise_source_path(value).replace("\\", "/")
    windows_path = PureWindowsPath(text)
    if windows_path.is_absolute():
        return windows_path.as_posix().casefold()
    return Path(text).resolve().as_posix().casefold()


def _validate_program_identities(
    fixed: Sequence[ProgramRecord], candidates: Sequence[ProgramRecord]
) -> set[int]:
    """Reject ambiguous source identities while allowing inventory copies of fixed 50."""

    seen_ids: set[str] = set()
    fixed_by_hash: dict[str, ProgramRecord] = {}
    fixed_by_path: dict[str, ProgramRecord] = {}
    for row in fixed:
        if row.program_id in seen_ids:
            raise ValueError(f"duplicate program_id: {row.program_id}")
        seen_ids.add(row.program_id)
        if row.source_sha256 in fixed_by_hash:
            raise ValueError(f"duplicate source_sha256: {row.source_sha256}")
        path_key = _record_path_key(row)
        if path_key in fixed_by_path:
            raise ValueError(f"duplicate relative_path: {row.relative_path}")
        fixed_by_hash[row.source_sha256] = row
        fixed_by_path[path_key] = row

    inventory_copies: set[int] = set()
    candidate_hashes: set[str] = set()
    candidate_paths: set[str] = set()
    for index, row in enumerate(candidates):
        if row.program_id in seen_ids and row.source_sha256 not in fixed_by_hash:
            raise ValueError(f"duplicate program_id: {row.program_id}")
        fixed_hash = fixed_by_hash.get(row.source_sha256)
        fixed_path = fixed_by_path.get(_record_path_key(row))
        if fixed_path is not None and fixed_path.source_sha256 != row.source_sha256:
            raise ValueError(f"source hash drift: {row.relative_path}")
        if fixed_hash is not None:
            if not _is_exact_fixed_inventory_copy(row, fixed_hash):
                raise ValueError(f"duplicate source_sha256: {row.source_sha256}")
            inventory_copies.add(index)
            continue
        if row.source_sha256 in candidate_hashes:
            raise ValueError(f"duplicate source_sha256: {row.source_sha256}")
        path_key = _record_path_key(row)
        if path_key in candidate_paths:
            raise ValueError(f"source hash drift: {row.relative_path}")
        candidate_hashes.add(row.source_sha256)
        candidate_paths.add(path_key)
        seen_ids.add(row.program_id)
    return inventory_copies


def _selection_reason(record: ProgramRecord, max_source_bytes: int) -> str | None:
    if record.source_size_bytes > max_source_bytes:
        return "source_too_large"
    if record.compile_status != "success":
        return record.preflight_status if record.preflight_status != "success" else (
            f"compile_{record.compile_status}"
        )
    if record.preflight_status != "success":
        return record.preflight_status
    if not record.root_ir_sha256:
        return "missing_root_ir_sha256"
    return None


def _ranked_reserves(
    candidates: Sequence[ProgramRecord], *, seed: int
) -> tuple[ProgramRecord, ...]:
    ranked = sorted(
        candidates,
        key=lambda row: (stable_rank(seed, row.relative_path), row.relative_path.casefold()),
    )
    return tuple(
        replace(row, selection_class="reserve", reserve_rank=index)
        for index, row in enumerate(ranked, start=1)
    )


def select_extension_candidates(
    fixed: Sequence[ProgramRecord],
    candidates: Sequence[ProgramRecord],
    *,
    needed: int,
    seed: int,
    per_category_cap: int,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> tuple[ProgramRecord, ...]:
    """Apply the advisor_benchmarks category-priority selection policy.

    Existing fixed programs are preselected.  Task 3 then continues one
    global stable rank of source paths while carrying forward the fixed
    category counts and enforcing the frozen cap.  There is no fresh
    category-priority bootstrap after the fixed 50.
    """

    if needed < 0:
        raise ValueError("needed must be non-negative")
    if per_category_cap < 1:
        raise ValueError("per_category_cap must be positive")
    fixed_hashes = {row.source_sha256 for row in fixed}
    ranked = sorted(
        (
            row
            for row in candidates
            if _selection_reason(row, max_source_bytes) is None
            and row.source_sha256 not in fixed_hashes
        ),
        key=lambda row: (stable_rank(seed, row.relative_path), row.relative_path.casefold()),
    )
    counts = Counter(row.program_family for row in fixed)
    selected: list[ProgramRecord] = []
    for row in ranked:
        if len(selected) >= needed:
            break
        if counts[row.program_family] >= per_category_cap:
            continue
        selected.append(row)
        counts[row.program_family] += 1
    return tuple(selected)


def require_formal_program_boundary(
    fixed: Sequence[ProgramRecord], *, target: int
) -> None:
    """Enforce the frozen 50-row source inventory before midpoint sampling."""

    if len(fixed) != FORMAL_SOURCE_INVENTORY_COUNT:
        raise ValueError("formal source inventory requires exactly 50 fixed programs")
    if target != FORMAL_PROGRAM_TARGET:
        raise ValueError("formal study requires target=10")
    if any(row.selection_class != "fixed" for row in fixed):
        raise ValueError("formal study requires only existing fixed programs")
    orders = [row.selection_order for row in fixed]
    if (
        any(not isinstance(order, int) or isinstance(order, bool) for order in orders)
        or sorted(orders) != list(range(1, FORMAL_SOURCE_INVENTORY_COUNT + 1))
    ):
        raise ValueError("formal study requires the complete fixed program order")


def freeze_formal_program_manifest(
    fixed: Sequence[ProgramRecord],
    candidates: Sequence[ProgramRecord],
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    per_category_cap: int = DEFAULT_CATEGORY_CAP,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> FrozenProgramManifest:
    """Select the frozen midpoint sample from the existing fixed-50 inventory."""

    require_formal_program_boundary(fixed, target=FORMAL_PROGRAM_TARGET)
    if candidates:
        raise ValueError("formal midpoint scope forbids candidate programs")
    programs_by_source_order = {
        row.selection_order: row
        for row in fixed
    }
    selected = tuple(
        replace(
            programs_by_source_order[source_position],
            selection_order=selection_order,
        )
        for selection_order, source_position in enumerate(
            FORMAL_SOURCE_POSITIONS,
            start=1,
        )
    )
    if len({row.program_family for row in selected}) != FORMAL_PROGRAM_TARGET:
        raise ValueError("formal midpoint selection requires 10 distinct program_family values")
    return freeze_program_manifest(
        selected,
        (),
        target=FORMAL_PROGRAM_TARGET,
        seed=seed,
        per_category_cap=per_category_cap,
        max_source_bytes=max_source_bytes,
    )


def freeze_program_manifest(
    fixed: Sequence[ProgramRecord],
    candidates: Sequence[ProgramRecord],
    *,
    target: int,
    seed: int = DEFAULT_SELECTION_SEED,
    per_category_cap: int = DEFAULT_CATEGORY_CAP,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> FrozenProgramManifest:
    """Freeze a generic deterministic corpus without observing pair or 2N outcomes.

    The existing fixed manifest is never re-ranked or altered.  A scan may
    contain an exact copy of one of its sources; those copies remain visible in
    the reserve inventory but are not eligible to duplicate the fixed source.
    Compile/root-IR failures become ledger rows and cannot be bypassed except
    by the next member of this already frozen reserve ordering.
    """

    if target < 1:
        raise ValueError("target must be positive")
    if len(fixed) > target:
        raise ValueError("fixed program count exceeds target")
    if per_category_cap < 1:
        raise ValueError("per_category_cap must be positive")
    if max_source_bytes < 1:
        raise ValueError("max_source_bytes must be positive")

    fixed_rows = tuple(fixed)
    candidate_rows = tuple(candidates)
    inventory_copies = _validate_program_identities(fixed_rows, candidate_rows)
    reserve_order = _ranked_reserves(candidate_rows, seed=seed)

    selected = list(fixed_rows)
    ledger: list[PreflightLedgerEntry] = []
    candidate_indexes = {
        (row.program_id, row.source_sha256, row.relative_path): index
        for index, row in enumerate(candidate_rows)
    }

    usable_reserves: list[ProgramRecord] = []
    for reserve in reserve_order:
        original_index = candidate_indexes[
            (reserve.program_id, reserve.source_sha256, reserve.relative_path)
        ]
        if original_index in inventory_copies:
            continue
        reason = _selection_reason(reserve, max_source_bytes)
        if reason is not None:
            ledger.append(
                PreflightLedgerEntry(
                    program_id=reserve.program_id,
                    relative_path=reserve.relative_path,
                    source_sha256=reserve.source_sha256,
                    reserve_rank=reserve.reserve_rank or 0,
                    compile_status=reserve.compile_status,
                    preflight_status=reserve.preflight_status,
                    reason=reason,
                )
            )
            continue
        usable_reserves.append(reserve)

    selected_reserves = select_extension_candidates(
        fixed_rows,
        usable_reserves,
        needed=target - len(selected),
        seed=seed,
        per_category_cap=per_category_cap,
        max_source_bytes=max_source_bytes,
    )
    selected_keys = {
        (row.program_id, row.source_sha256, row.relative_path)
        for row in selected_reserves
    }
    final_category_counts = Counter(row.program_family for row in fixed_rows)
    final_category_counts.update(row.program_family for row in selected_reserves)
    for reserve in usable_reserves:
        key = (reserve.program_id, reserve.source_sha256, reserve.relative_path)
        if key not in selected_keys and final_category_counts[reserve.program_family] >= per_category_cap:
            ledger.append(
                PreflightLedgerEntry(
                    program_id=reserve.program_id,
                    relative_path=reserve.relative_path,
                    source_sha256=reserve.source_sha256,
                    reserve_rank=reserve.reserve_rank or 0,
                    compile_status=reserve.compile_status,
                    preflight_status=reserve.preflight_status,
                    reason="category_cap",
                )
            )
    for reserve in selected_reserves:
        selected.append(
            replace(
                reserve,
                selection_class="extension",
                selection_order=len(selected) + 1,
            )
        )

    if len(selected) != target:
        raise ValueError(f"needed {target} programs, selected {len(selected)}")

    canonical_programs = tuple(sorted(selected, key=lambda row: row.relative_path.casefold()))
    canonical_ledger = tuple(
        sorted(
            ledger,
            key=lambda row: (row.reserve_rank, row.relative_path.casefold(), row.reason),
        )
    )
    return FrozenProgramManifest(
        programs=canonical_programs,
        reserve_order=reserve_order,
        preflight_ledger=canonical_ledger,
        target=target,
        selection_seed=seed,
        per_category_cap=per_category_cap,
        max_source_bytes=max_source_bytes,
    )


def extend_program_manifest(
    fixed: Sequence[ProgramRecord],
    candidates: Sequence[ProgramRecord],
    *,
    target: int,
    seed: int,
    per_category_cap: int = DEFAULT_CATEGORY_CAP,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
) -> tuple[ProgramRecord, ...]:
    """Return only the final deterministic manifest for compact callers."""

    return freeze_program_manifest(
        fixed,
        candidates,
        target=target,
        seed=seed,
        per_category_cap=per_category_cap,
        max_source_bytes=max_source_bytes,
    ).programs


def _single_source_directory(root: Path) -> Path:
    resolved = root.resolve()
    if resolved.name.casefold() == "singlesource":
        return resolved
    candidate = resolved / "SingleSource"
    if candidate.is_dir():
        return candidate
    raise ValueError("SingleSource root is required for source-path validation")


def validate_program_source_paths(
    programs: Sequence[ProgramRecord], single_source_root: Path
) -> None:
    """Fail closed on source escape, path drift, source-size drift, or hash drift."""

    root = _single_source_directory(Path(single_source_root))
    expected_prefix = root.parent
    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    for record in programs:
        source = Path(record.source_path).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"source path escapes SingleSource root: {record.source_path}"
            ) from error
        if source.suffix.lower() != ".c" or not source.is_file():
            raise ValueError(f"source path is not an existing C source: {record.source_path}")
        expected_relative = source.relative_to(expected_prefix).as_posix()
        if expected_relative != record.relative_path:
            raise ValueError(f"source path drift: {record.relative_path}")
        if record.source_sha256 in seen_hashes:
            raise ValueError(f"duplicate source_sha256: {record.source_sha256}")
        if record.relative_path.casefold() in seen_paths:
            raise ValueError(f"source path drift: {record.relative_path}")
        seen_hashes.add(record.source_sha256)
        seen_paths.add(record.relative_path.casefold())
        actual_size = source.stat().st_size
        if actual_size != record.source_size_bytes:
            raise ValueError(f"source size drift: {record.relative_path}")
        actual_hash = _sha256_bytes(source.read_bytes())
        if actual_hash != record.source_sha256:
            raise ValueError(f"source hash drift: {record.relative_path}")


def _component_identity(value: object, name: str) -> dict[str, object]:
    """Freeze an exact supplied file or a canonical in-memory configuration."""

    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} input file not found: {path}")
        data = path.read_bytes()
        return {
            "kind": "file",
            "path": path.as_posix(),
            "sha256": _sha256_bytes(data),
            "size_bytes": len(data),
        }
    canonical = _canonical_value(value)
    return {
        "kind": "canonical_object",
        "sha256": canonical_sha256(canonical),
        "value": canonical,
    }


def _canonical_program_records(
    programs: Sequence[ProgramRecord],
) -> list[dict[str, object]]:
    rows = list(programs)
    _validate_program_identities(rows, ())
    return [
        record.as_manifest_record()
        for record in sorted(rows, key=lambda record: record.relative_path.casefold())
    ]


def _normalise_tools(tools: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    if not isinstance(tools, Mapping):
        raise ValueError("tools must be a mapping")
    missing = _REQUIRED_TOOL_NAMES.difference(tools)
    if missing:
        raise ValueError(f"missing required tool records: {', '.join(sorted(missing))}")
    unexpected = set(tools).difference(_REQUIRED_TOOL_NAMES)
    if unexpected:
        raise ValueError(f"unexpected tool records: {', '.join(sorted(unexpected))}")
    normalized: dict[str, dict[str, object]] = {}
    for name, record in sorted(tools.items()):
        if not isinstance(record, Mapping):
            raise ValueError(f"tools.{name} must be a mapping")
        unexpected_fields = set(record).difference(
            {"path", "sha256", *_TOOL_RECORD_OPTIONAL_FIELDS}
        )
        if unexpected_fields:
            raise ValueError(
                f"tools.{name} contains unexpected fields: {', '.join(sorted(map(str, unexpected_fields)))}"
            )
        path = _require_nonempty(record.get("path", ""), f"tools.{name}.path")
        digest = _require_sha256(record.get("sha256", ""), f"tools.{name}.sha256")
        if "size_bytes" in record and (
            not isinstance(record["size_bytes"], int)
            or isinstance(record["size_bytes"], bool)
            or record["size_bytes"] < 0
        ):
            raise ValueError(f"tools.{name}.size_bytes must be a non-negative integer")
        if "version" in record and record["version"] is not None and not isinstance(record["version"], str):
            raise ValueError(f"tools.{name}.version must be a string or null")
        normalized[name] = {
            str(key): _canonical_value(value) for key, value in sorted(record.items())
        }
        normalized[name]["path"] = path
        normalized[name]["sha256"] = digest
    return normalized


def build_study_manifest(
    *,
    programs: Sequence[ProgramRecord],
    pass_policy: object,
    pass_inventory: object,
    pass_preflight: object,
    pass_groups: object,
    llvm_commit: str,
    target: str,
    tools: Mapping[str, Mapping[str, object]],
    hard_state_policy: object,
    comparator: object,
    jobs: int,
    timeout_s: int | float,
    artifact_policy: object,
) -> dict[str, object]:
    """Create a self-validating study identity from every frozen input.

    The returned mapping is deterministic and deliberately has no timestamp.
    ``study_manifest_id`` and ``study_manifest_sha256`` are aliases of the
    digest of all non-derived fields, making both reuse and mismatch checks
    fail closed.
    """

    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    llvm_commit = _require_nonempty(llvm_commit, "llvm_commit")
    target = _require_nonempty(target, "target")
    program_rows = _canonical_program_records(programs)
    program_sha256 = canonical_sha256(program_rows)
    identity: dict[str, object] = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "program_manifest": program_rows,
        "program_manifest_sha256": program_sha256,
        "frozen_inputs": {
            "pass_policy": _component_identity(pass_policy, "pass_policy"),
            "pass_inventory": _component_identity(pass_inventory, "pass_inventory"),
            "pass_preflight": _component_identity(pass_preflight, "pass_preflight"),
            "pass_groups": _component_identity(pass_groups, "pass_groups"),
        },
        "llvm": {"commit": llvm_commit, "target": target},
        "tools": _normalise_tools(tools),
        "hard_state_policy": _component_identity(
            hard_state_policy, "hard_state_policy"
        ),
        "comparator": _component_identity(comparator, "comparator"),
        "execution": {"jobs": jobs, "timeout_s": timeout_s},
        "artifact_policy": _component_identity(artifact_policy, "artifact_policy"),
        "authority_granted": False,
        "proved_commute": False,
    }
    digest = canonical_sha256(identity)
    manifest = dict(identity)
    manifest["study_manifest_id"] = digest
    manifest["study_manifest_sha256"] = digest
    return manifest


def _load_manifest(value: Mapping[str, object] | str | Path) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    path = Path(value)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("study manifest JSON must be a mapping")
    return {str(key): _canonical_value(item) for key, item in loaded.items()}


_MANIFEST_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "program_manifest",
        "program_manifest_sha256",
        "frozen_inputs",
        "llvm",
        "tools",
        "hard_state_policy",
        "comparator",
        "execution",
        "artifact_policy",
        "authority_granted",
        "proved_commute",
    }
)
_MANIFEST_DERIVED_FIELDS = frozenset({"study_manifest_id", "study_manifest_sha256"})
_PROGRAM_MANIFEST_FIELDS = frozenset(
    {
        "program_id",
        "source_path",
        "relative_path",
        "program_family",
        "source_sha256",
        "source_size_bytes",
        "compile_command",
        "compile_command_sha256",
        "compile_status",
        "compile_stderr_sha256",
        "root_ir_path",
        "root_ir_sha256",
        "root_hard_state_id",
        "target",
        "data_layout",
        "preflight_status",
        "selection_class",
        "selection_order",
        "reserve_rank",
        "replacement_for_program_id",
    }
)


def _require_exact_keys(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    keys = {str(key) for key in value}
    missing = expected.difference(keys)
    unexpected = keys.difference(expected)
    if missing:
        raise ValueError(f"{name} missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(f"{name} unexpected fields: {', '.join(sorted(unexpected))}")
    return value


def _validate_component_identity(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    kind = value.get("kind")
    if kind == "file":
        _require_exact_keys(value, frozenset({"kind", "path", "sha256", "size_bytes"}), name)
        _require_nonempty(value.get("path", ""), f"{name}.path")
        _require_sha256(value.get("sha256", ""), f"{name}.sha256")
        size = value.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"{name}.size_bytes must be a non-negative integer")
        return
    if kind == "canonical_object":
        _require_exact_keys(value, frozenset({"kind", "sha256", "value"}), name)
        digest = _require_sha256(value.get("sha256", ""), f"{name}.sha256")
        if canonical_sha256(value.get("value")) != digest:
            raise ValueError(f"{name}.sha256 does not match value")
        return
    raise ValueError(f"{name}.kind is unsupported")


def _validate_program_manifest_rows(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("program_manifest must be a list")
    rows: list[ProgramRecord] = []
    for index, raw in enumerate(value):
        row = _require_exact_keys(raw, _PROGRAM_MANIFEST_FIELDS, f"program_manifest[{index}]")
        record = ProgramRecord(
            program_id=row["program_id"],
            source_path=row["source_path"],
            relative_path=row["relative_path"],
            program_family=row["program_family"],
            source_sha256=row["source_sha256"],
            source_size_bytes=row["source_size_bytes"],
            compile_command=tuple(row["compile_command"]) if isinstance(row["compile_command"], list) else (),
            compile_status=row["compile_status"],
            compile_stderr_sha256=row["compile_stderr_sha256"],
            root_ir_path=row["root_ir_path"],
            root_ir_sha256=row["root_ir_sha256"],
            root_hard_state_id=row["root_hard_state_id"],
            target=row["target"],
            data_layout=row["data_layout"],
            preflight_status=row["preflight_status"],
            selection_class=row["selection_class"],
            selection_order=row["selection_order"],
            reserve_rank=row["reserve_rank"],
            replacement_for_program_id=row["replacement_for_program_id"],
        )
        if record.as_manifest_record() != dict(row):
            raise ValueError(f"program_manifest[{index}] has non-canonical fields")
        rows.append(record)
    _validate_program_identities(rows, ())


def _validated_manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    _require_exact_keys(
        manifest,
        _MANIFEST_IDENTITY_FIELDS | _MANIFEST_DERIVED_FIELDS,
        "study manifest",
    )
    if manifest["schema_version"] != STUDY_SCHEMA_VERSION:
        raise ValueError("study manifest schema_version mismatch")
    if manifest["authority_granted"] is not False or manifest["proved_commute"] is not False:
        raise ValueError("study manifest identity mismatch: authority must remain false")
    _validate_program_manifest_rows(manifest["program_manifest"])
    program_digest = _require_sha256(
        manifest["program_manifest_sha256"], "program_manifest_sha256"
    )
    if program_digest != canonical_sha256(manifest["program_manifest"]):
        raise ValueError("program_manifest_sha256 does not match program_manifest")
    frozen_inputs = _require_exact_keys(
        manifest["frozen_inputs"],
        frozenset({"pass_policy", "pass_inventory", "pass_preflight", "pass_groups"}),
        "frozen_inputs",
    )
    for name, value in frozen_inputs.items():
        _validate_component_identity(value, f"frozen_inputs.{name}")
    llvm = _require_exact_keys(manifest["llvm"], frozenset({"commit", "target"}), "llvm")
    _require_nonempty(llvm["commit"], "llvm.commit")
    _require_nonempty(llvm["target"], "llvm.target")
    normalized_tools = _normalise_tools(manifest["tools"])
    if _canonical_json(normalized_tools) != _canonical_json(manifest["tools"]):
        raise ValueError("tools contain non-canonical records")
    _validate_component_identity(manifest["hard_state_policy"], "hard_state_policy")
    _validate_component_identity(manifest["comparator"], "comparator")
    _validate_component_identity(manifest["artifact_policy"], "artifact_policy")
    execution = _require_exact_keys(
        manifest["execution"], frozenset({"jobs", "timeout_s"}), "execution"
    )
    jobs = execution["jobs"]
    timeout_s = execution["timeout_s"]
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError("execution.jobs must be a positive integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("execution.timeout_s must be positive")
    manifest_id = _require_sha256(manifest["study_manifest_id"], "study_manifest_id")
    manifest_sha = _require_sha256(manifest["study_manifest_sha256"], "study_manifest_sha256")
    identity = {
        str(key): _canonical_value(value)
        for key, value in manifest.items()
        if key not in {"study_manifest_id", "study_manifest_sha256"}
    }
    digest = canonical_sha256(identity)
    if (
        manifest_id != digest
        or manifest_sha != digest
    ):
        raise ValueError("study manifest identity mismatch: digest does not match content")
    return identity


def require_study_manifest(
    expected: Mapping[str, object] | str | Path,
    actual: Mapping[str, object] | str | Path,
) -> None:
    """Require byte-logical identity before artifact reuse or result emission."""

    expected_identity = _validated_manifest_identity(_load_manifest(expected))
    actual_identity = _validated_manifest_identity(_load_manifest(actual))
    if _canonical_json(expected_identity) != _canonical_json(actual_identity):
        raise ValueError("study manifest identity mismatch")
