from __future__ import annotations

from pathlib import Path

import pytest

from instruction_granularity.deterministic_io import (
    canonical_json_bytes,
    write_csv,
    write_json,
)
from instruction_granularity.isolation import (
    assert_safe_source_tree,
    assert_within_root,
    build_inventory,
    compare_inventories,
    inventory_record_sha256,
)


def test_canonical_json_and_csv_are_byte_deterministic(tmp_path: Path) -> None:
    left_json = tmp_path / "left.json"
    right_json = tmp_path / "right.json"
    payload = {"z": [3, 2, 1], "a": {"β": True, "x": 1}}
    write_json(left_json, payload)
    write_json(right_json, payload)
    assert left_json.read_bytes() == right_json.read_bytes() == canonical_json_bytes(payload)

    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    rows = [{"b": "二", "a": 1}, {"b": "x", "a": 2}]
    write_csv(left_csv, ("a", "b"), rows)
    write_csv(right_csv, ("a", "b"), rows)
    assert left_csv.read_bytes() == right_csv.read_bytes()
    assert b"\r\n" not in left_csv.read_bytes()


def test_write_boundary_rejects_sibling_and_parent(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    assert assert_within_root(root / "aggregate" / "x.csv", root) == root / "aggregate" / "x.csv"
    with pytest.raises(ValueError):
        assert_within_root(tmp_path / "outside.csv", root)
    with pytest.raises(ValueError):
        assert_within_root(root.parent, root)


def test_inventory_compare_detects_every_difference_kind() -> None:
    baseline = {
        "files": [
            {"path": "a", "size": 1, "sha256": "a"},
            {"path": "b", "size": 2, "sha256": "b"},
            {"path": "c", "size": 3, "sha256": "c"},
        ]
    }
    final = {
        "files": [
            {"path": "a", "size": 1, "sha256": "changed"},
            {"path": "b", "size": 9, "sha256": "b"},
            {"path": "d", "size": 4, "sha256": "d"},
        ]
    }
    diff = compare_inventories(baseline, final)
    assert diff.added == ("d",)
    assert diff.deleted == ("c",)
    assert diff.size_changed == ("b",)
    assert diff.content_changed == ("a",)
    assert not diff.is_clean


def test_source_tree_contains_no_forbidden_process_launches() -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    violations = assert_safe_source_tree(experiment_root / "src")
    assert violations == ()


def test_inventory_hashes_all_nested_files_and_is_order_stable(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "b" / "deep").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"a")
    (root / "b" / "deep" / "z.bin").write_bytes(b"z" * 3)
    first = build_inventory(root, schema_version="test-v1")
    second = build_inventory(root, schema_version="test-v1")

    assert first["directory_count"] == 3
    assert [row["path"] for row in first["files"]] == [
        "a.txt",
        "b/deep/z.bin",
    ]
    assert inventory_record_sha256(first) == inventory_record_sha256(second)
    assert compare_inventories(first, second).is_clean
