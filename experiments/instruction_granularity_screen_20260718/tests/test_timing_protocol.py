from __future__ import annotations

from instruction_granularity.models import ExtractionLevel, RuntimeRecord
from instruction_granularity.timing import (
    LEVEL_ORDER,
    build_timing_schedule,
    nearest_rank_percentile,
    paired_incremental_rows,
)


def test_schedule_is_five_warmups_plus_thirty_measured_in_fixed_order() -> None:
    tasks = build_timing_schedule(warmups=5, measured=30)

    assert len(tasks) == 35 * 4
    for cycle in range(1, 36):
        rows = [task for task in tasks if task.cycle == cycle]
        assert tuple(row.level for row in rows) == LEVEL_ORDER
        assert {row.source_repetition for row in rows} == {((cycle - 1) % 3) + 1}
        assert {row.phase for row in rows} == {"warmup" if cycle <= 5 else "measured"}
    measured = [task for task in tasks if task.phase == "measured"]
    assert len(measured) == 120
    for source_repetition in (1, 2, 3):
        assert len(
            {
                task.measured_repetition
                for task in measured
                if task.source_repetition == source_repetition
            }
        ) == 10


def _record(repetition: int, level: ExtractionLevel, total: float) -> RuntimeRecord:
    return RuntimeRecord(
        phase="measured",
        cycle=repetition + 5,
        measured_repetition=repetition,
        source_repetition=((repetition - 1) % 3) + 1,
        level=level,
        artifact_read_ms=1.0,
        parse_ms=1.0,
        feature_build_ms=1.0,
        pair_selection_ms=1.0,
        total_extraction_ms=total,
        programs=49,
        transitions=686,
        pairs=1411,
    )


def test_paired_differences_happen_before_summary() -> None:
    records = []
    for repetition in range(1, 31):
        records.extend(
            (
                _record(repetition, ExtractionLevel.FUNC_ONLY, 10 + repetition),
                _record(repetition, ExtractionLevel.BLOCK_ONLY, 20 + 2 * repetition),
                _record(repetition, ExtractionLevel.EFFECT_ONLY, 40 + 4 * repetition),
                _record(repetition, ExtractionLevel.INSTRUCTION_ONLY, 80 + 8 * repetition),
            )
        )
    rows = paired_incremental_rows(records)

    block_func = [row for row in rows if row.comparison == "BLOCK-FUNC"]
    effect_block = [row for row in rows if row.comparison == "EFFECT-BLOCK"]
    instruction_effect = [row for row in rows if row.comparison == "INSTRUCTION-EFFECT"]
    assert [row.delta_ms for row in block_func] == [10 + i for i in range(1, 31)]
    assert [row.delta_ms for row in effect_block] == [20 + 2 * i for i in range(1, 31)]
    assert [row.delta_ms for row in instruction_effect] == [40 + 4 * i for i in range(1, 31)]
    assert nearest_rank_percentile([row.delta_ms for row in block_func], 0.9) == 37

