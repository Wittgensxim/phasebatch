from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from instruction_granularity.aggregate import generate_aggregate_outputs
from instruction_granularity.dataset import load_frozen_dataset
from instruction_granularity.timing import load_runtime_records


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "aggregate"


def _rows(name: str) -> list[dict[str, str]]:
    with (AGGREGATE / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_screen_retains_all_rows_and_exact_incremental_cumulative_counts() -> None:
    pairs = _rows("instruction_screen_pairs.csv")
    incremental = _rows("instruction_incremental_summary.csv")[0]
    cumulative = _rows("instruction_cumulative_summary.csv")[0]

    assert len(pairs) == 1411
    assert len({row["observation_id"] for row in pairs}) == 1411
    assert sum(row["h_effect_selected"] == "true" for row in pairs) == 47
    assert incremental["incremental_selected_count"] == "1"
    assert incremental["incremental_commute"] == "1"
    assert incremental["incremental_order_sensitive"] == "0"
    assert incremental["incremental_failed"] == "0"
    assert incremental["incremental_unknown"] == "1342"
    assert cumulative["cumulative_selected"] == "48"
    assert cumulative["cumulative_commute"] == "46"
    assert cumulative["cumulative_unsafe"] == "2"
    assert cumulative["cumulative_order_sensitive"] == "2"
    assert cumulative["cumulative_failed"] == "0"


def test_four_level_legacy_and_instruction_counts() -> None:
    rows = {row["heuristic"]: row for row in _rows("granularity_coverage_summary.csv")}

    assert (rows["H_func"]["selected_count"], rows["H_func"]["selected_commute"], rows["H_func"]["selected_order_sensitive"]) == ("30", "28", "2")
    assert (rows["H_block"]["selected_count"], rows["H_block"]["selected_commute"], rows["H_block"]["selected_order_sensitive"]) == ("46", "44", "2")
    assert (rows["H_effect"]["selected_count"], rows["H_effect"]["selected_commute"], rows["H_effect"]["selected_order_sensitive"]) == ("47", "45", "2")
    assert (rows["H_inst"]["selected_count"], rows["H_inst"]["selected_commute"], rows["H_inst"]["selected_order_sensitive"]) == ("48", "46", "2")


def test_runtime_has_five_warmups_thirty_measured_and_layered_payment() -> None:
    rows = _rows("extraction_runtime_raw.csv")
    assert len(rows) == 140
    for level in ("FUNC_ONLY", "BLOCK_ONLY", "EFFECT_ONLY", "INSTRUCTION_ONLY"):
        level_rows = [row for row in rows if row["level"] == level]
        assert sum(row["phase"] == "warmup" for row in level_rows) == 5
        assert sum(row["phase"] == "measured" for row in level_rows) == 30
        assert all((row["programs"], row["transitions"], row["pairs"]) == ("49", "686", "1411") for row in level_rows)
    expected_zero = {
        "FUNC_ONLY": ("0", "0", "0"),
        "BLOCK_ONLY": (None, "0", "0"),
        "EFFECT_ONLY": (None, None, "0"),
        "INSTRUCTION_ONLY": (None, None, None),
    }
    for row in rows:
        block, effect, instruction = expected_zero[row["level"]]
        if block is not None:
            assert row["block_builds"] == block
        if effect is not None:
            assert row["effect_builds"] == effect
        if instruction is not None:
            assert row["instruction_builds"] == instruction


def test_paired_incremental_cost_has_thirty_same_repetition_differences() -> None:
    rows = _rows("extraction_incremental_cost.csv")
    assert [row["comparison"] for row in rows] == [
        "BLOCK-FUNC",
        "EFFECT-BLOCK",
        "INSTRUCTION-EFFECT",
    ]
    assert all(row["paired_repetitions"] == "30" for row in rows)
    assert all(len(json.loads(row["paired_values_json"])) == 30 for row in rows)


def test_aggregate_regeneration_is_byte_deterministic(tmp_path: Path) -> None:
    dataset = load_frozen_dataset(ROOT)
    snapshot = json.loads((ROOT / "raw" / "instruction_screen_snapshot.json").read_text(encoding="utf-8"))
    runtime = load_runtime_records(AGGREGATE / "extraction_runtime_raw.csv")
    one = tmp_path / "one" / "aggregate"
    two = tmp_path / "two" / "aggregate"
    generate_aggregate_outputs(dataset, snapshot, runtime, one)
    generate_aggregate_outputs(dataset, snapshot, runtime, two)

    def manifest(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert manifest(one) == manifest(two)

