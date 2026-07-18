from __future__ import annotations

import math
from pathlib import Path
import csv
from typing import Iterable

from .deterministic_io import write_csv
from .extractors import run_extractor
from .models import (
    ExtractionLevel,
    FrozenDataset,
    PairedIncrementalRow,
    RuntimeRecord,
    TimingTask,
)


LEVEL_ORDER = (
    ExtractionLevel.FUNC_ONLY,
    ExtractionLevel.BLOCK_ONLY,
    ExtractionLevel.EFFECT_ONLY,
    ExtractionLevel.INSTRUCTION_ONLY,
)

RUNTIME_FIELDS = (
    "phase",
    "cycle",
    "measured_repetition",
    "source_repetition",
    "sequence_index",
    "level",
    "artifact_read_ms",
    "parse_ms",
    "feature_build_ms",
    "pair_selection_ms",
    "total_extraction_ms",
    "programs",
    "transitions",
    "pairs",
    "selected_count",
    "unknown_count",
    "function_builds",
    "block_builds",
    "effect_builds",
    "instruction_builds",
)


def build_timing_schedule(*, warmups: int = 5, measured: int = 30) -> tuple[TimingTask, ...]:
    if warmups < 0 or measured <= 0:
        raise ValueError("warmups must be non-negative and measured must be positive")
    result: list[TimingTask] = []
    sequence = 0
    total = warmups + measured
    for cycle in range(1, total + 1):
        phase = "warmup" if cycle <= warmups else "measured"
        measured_repetition = 0 if phase == "warmup" else cycle - warmups
        source_repetition = ((cycle - 1) % 3) + 1
        for level in LEVEL_ORDER:
            sequence += 1
            result.append(
                TimingTask(
                    phase=phase,
                    cycle=cycle,
                    measured_repetition=measured_repetition,
                    source_repetition=source_repetition,
                    level=level,
                    sequence_index=sequence,
                )
            )
    return tuple(result)


def nearest_rank_percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty values")
    if not 0 < quantile <= 1:
        raise ValueError(f"invalid quantile: {quantile}")
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def paired_incremental_rows(
    records: Iterable[RuntimeRecord],
) -> tuple[PairedIncrementalRow, ...]:
    measured = [record for record in records if record.phase == "measured"]
    by_key = {
        (record.measured_repetition, record.level): record for record in measured
    }
    expected = len({record.measured_repetition for record in measured}) * len(LEVEL_ORDER)
    if len(by_key) != expected:
        raise ValueError("duplicate or missing measured runtime rows")
    comparisons = (
        ("BLOCK-FUNC", ExtractionLevel.FUNC_ONLY, ExtractionLevel.BLOCK_ONLY),
        ("EFFECT-BLOCK", ExtractionLevel.BLOCK_ONLY, ExtractionLevel.EFFECT_ONLY),
        (
            "INSTRUCTION-EFFECT",
            ExtractionLevel.EFFECT_ONLY,
            ExtractionLevel.INSTRUCTION_ONLY,
        ),
    )
    result: list[PairedIncrementalRow] = []
    repetitions = sorted({record.measured_repetition for record in measured})
    for name, lower_level, upper_level in comparisons:
        for repetition in repetitions:
            lower = by_key[(repetition, lower_level)]
            upper = by_key[(repetition, upper_level)]
            if lower.source_repetition != upper.source_repetition:
                raise ValueError("paired rows use different source repetitions")
            result.append(
                PairedIncrementalRow(
                    comparison=name,
                    measured_repetition=repetition,
                    source_repetition=lower.source_repetition,
                    lower_level=lower_level.value,
                    upper_level=upper_level.value,
                    lower_total_ms=lower.total_extraction_ms,
                    upper_total_ms=upper.total_extraction_ms,
                    delta_ms=upper.total_extraction_ms - lower.total_extraction_ms,
                )
            )
    return tuple(result)


def run_timing_experiment(
    dataset: FrozenDataset,
    raw_csv: Path,
    *,
    warmups: int = 5,
    measured: int = 30,
    progress=None,  # noqa: ANN001
) -> tuple[RuntimeRecord, ...]:
    """Run the fixed formal protocol once and checkpoint after every level."""

    raw_csv = Path(raw_csv)
    tasks = build_timing_schedule(warmups=warmups, measured=measured)
    records: list[RuntimeRecord] = []
    if raw_csv.exists():
        existing = load_runtime_records(raw_csv)
        expected_rows = (warmups + measured) * len(LEVEL_ORDER)
        if len(existing) == expected_rows:
            _validate_runtime_records(existing, warmups=warmups, measured=measured)
            return existing
        if len(existing) > expected_rows:
            raise ValueError(
                f"formal timing has too many rows: {raw_csv}:rows={len(existing)}"
            )
        _validate_runtime_prefix(existing, tasks)
        records.extend(existing)

    expected_selected = {
        ExtractionLevel.FUNC_ONLY: 30,
        ExtractionLevel.BLOCK_ONLY: 46,
        ExtractionLevel.EFFECT_ONLY: 47,
    }
    for task in tasks[len(records) :]:
        run = run_extractor(dataset, task.source_repetition, task.level)
        selected_count = sum(
            decision.status == "selected" for decision in run.decisions.values()
        )
        unknown_count = sum(
            decision.status == "unknown" for decision in run.decisions.values()
        )
        expected = expected_selected.get(task.level)
        if expected is not None and selected_count != expected:
            raise ValueError(
                f"legacy selection gate failed during timing: {task.level.value}:"
                f"expected={expected}:actual={selected_count}:"
                f"source={task.source_repetition}"
            )
        record = RuntimeRecord(
            phase=task.phase,
            cycle=task.cycle,
            measured_repetition=task.measured_repetition,
            source_repetition=task.source_repetition,
            level=task.level,
            artifact_read_ms=run.artifact_read_ms,
            parse_ms=run.parse_ms,
            feature_build_ms=run.feature_build_ms,
            pair_selection_ms=run.pair_selection_ms,
            total_extraction_ms=run.total_extraction_ms,
            programs=49,
            transitions=len(run.features),
            pairs=len(run.decisions),
            sequence_index=task.sequence_index,
            selected_count=selected_count,
            unknown_count=unknown_count,
            function_builds=run.trace.function_builds,
            block_builds=run.trace.block_builds,
            effect_builds=run.trace.effect_builds,
            instruction_builds=run.trace.instruction_builds,
        )
        records.append(record)
        write_runtime_records(raw_csv, records)
        if progress is not None:
            progress(record, len(records), (warmups + measured) * len(LEVEL_ORDER))
    result = tuple(records)
    _validate_runtime_records(result, warmups=warmups, measured=measured)
    return result


def write_runtime_records(path: Path, records: Iterable[RuntimeRecord]) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "phase": record.phase,
                "cycle": record.cycle,
                "measured_repetition": record.measured_repetition,
                "source_repetition": record.source_repetition,
                "sequence_index": record.sequence_index,
                "level": record.level.value,
                "artifact_read_ms": _format_ms(record.artifact_read_ms),
                "parse_ms": _format_ms(record.parse_ms),
                "feature_build_ms": _format_ms(record.feature_build_ms),
                "pair_selection_ms": _format_ms(record.pair_selection_ms),
                "total_extraction_ms": _format_ms(record.total_extraction_ms),
                "programs": record.programs,
                "transitions": record.transitions,
                "pairs": record.pairs,
                "selected_count": record.selected_count,
                "unknown_count": record.unknown_count,
                "function_builds": record.function_builds,
                "block_builds": record.block_builds,
                "effect_builds": record.effect_builds,
                "instruction_builds": record.instruction_builds,
            }
        )
    write_csv(Path(path), RUNTIME_FIELDS, rows)


def load_runtime_records(path: Path) -> tuple[RuntimeRecord, ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return tuple(
        RuntimeRecord(
            phase=row["phase"],
            cycle=int(row["cycle"]),
            measured_repetition=int(row["measured_repetition"]),
            source_repetition=int(row["source_repetition"]),
            level=ExtractionLevel(row["level"]),
            artifact_read_ms=float(row["artifact_read_ms"]),
            parse_ms=float(row["parse_ms"]),
            feature_build_ms=float(row["feature_build_ms"]),
            pair_selection_ms=float(row["pair_selection_ms"]),
            total_extraction_ms=float(row["total_extraction_ms"]),
            programs=int(row["programs"]),
            transitions=int(row["transitions"]),
            pairs=int(row["pairs"]),
            sequence_index=int(row["sequence_index"]),
            selected_count=int(row["selected_count"]),
            unknown_count=int(row["unknown_count"]),
            function_builds=int(row["function_builds"]),
            block_builds=int(row["block_builds"]),
            effect_builds=int(row["effect_builds"]),
            instruction_builds=int(row["instruction_builds"]),
        )
        for row in rows
    )


def _validate_runtime_records(
    records: Iterable[RuntimeRecord], *, warmups: int, measured: int
) -> None:
    values = tuple(records)
    expected_tasks = build_timing_schedule(warmups=warmups, measured=measured)
    if len(values) != len(expected_tasks):
        raise ValueError(
            f"runtime row count mismatch: expected={len(expected_tasks)}:actual={len(values)}"
        )
    for record, task in zip(values, expected_tasks, strict=True):
        if (
            record.phase,
            record.cycle,
            record.measured_repetition,
            record.source_repetition,
            record.level,
            record.sequence_index,
        ) != (
            task.phase,
            task.cycle,
            task.measured_repetition,
            task.source_repetition,
            task.level,
            task.sequence_index,
        ):
            raise ValueError(f"runtime schedule mismatch at sequence {task.sequence_index}")
        if (record.programs, record.transitions, record.pairs) != (49, 686, 1411):
            raise ValueError(f"runtime hard counts mismatch: {record}")
        if min(
            record.artifact_read_ms,
            record.parse_ms,
            record.feature_build_ms,
            record.pair_selection_ms,
            record.total_extraction_ms,
        ) < 0:
            raise ValueError(f"negative runtime component: {record}")
    measured_rows = [record for record in values if record.phase == "measured"]
    if len(measured_rows) != measured * len(LEVEL_ORDER):
        raise ValueError("measured runtime count mismatch")
    for level in LEVEL_ORDER:
        if sum(record.level == level for record in measured_rows) != measured:
            raise ValueError(f"measured level count mismatch: {level.value}")


def _validate_runtime_prefix(
    records: Iterable[RuntimeRecord], tasks: tuple[TimingTask, ...]
) -> None:
    values = tuple(records)
    for record, task in zip(values, tasks, strict=False):
        if (
            record.phase,
            record.cycle,
            record.measured_repetition,
            record.source_repetition,
            record.level,
            record.sequence_index,
            record.programs,
            record.transitions,
            record.pairs,
        ) != (
            task.phase,
            task.cycle,
            task.measured_repetition,
            task.source_repetition,
            task.level,
            task.sequence_index,
            49,
            686,
            1411,
        ):
            raise ValueError(
                f"partial formal timing prefix mismatch at sequence {task.sequence_index}"
            )


def _format_ms(value: float) -> str:
    return f"{value:.6f}"
