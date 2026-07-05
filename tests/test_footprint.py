import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.footprint import build_footprint_overlap, parse_set_field


class FootprintOverlapTests(unittest.TestCase):
    def test_parse_set_field_handles_common_serializations(self) -> None:
        self.assertEqual(parse_set_field(""), set())
        self.assertEqual(parse_set_field("foo;bar"), {"foo", "bar"})
        self.assertEqual(parse_set_field("foo,bar"), {"foo", "bar"})
        self.assertEqual(parse_set_field("['foo','bar']"), {"foo", "bar"})
        self.assertEqual(parse_set_field("[]"), set())

    def test_disjoint_write_when_exact_changed_sets_do_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(
                state_dir,
                [
                    _profile("A", changed_functions="f", changed_blocks="f::entry"),
                    _profile("B", changed_functions="g", changed_blocks="g::entry"),
                ],
            )
            _write_pairs(state_dir, [_pair("A", "B", "dynamic_commute", "final_commute")])

            rows = build_footprint_overlap(state_dir)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["overlap_kind"], "disjoint_write")
        self.assertEqual(rows[0]["write_func_overlap"], "0")
        self.assertEqual(rows[0]["write_block_overlap"], "0")
        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_commute")
        self.assertEqual(rows[0]["final_relation"], "final_commute")

    def test_same_function_overlap_when_functions_overlap_but_blocks_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(
                state_dir,
                [
                    _profile("A", changed_functions="f", changed_blocks="f::entry"),
                    _profile("B", changed_functions="f", changed_blocks="f::exit"),
                ],
            )

            rows = build_footprint_overlap(state_dir)

        self.assertEqual(rows[0]["overlap_kind"], "same_function_overlap")
        self.assertEqual(rows[0]["same_function"], "true")
        self.assertEqual(rows[0]["same_block"], "false")
        self.assertEqual(rows[0]["final_relation"], "unknown")

    def test_same_block_overlap_when_blocks_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(
                state_dir,
                [
                    _profile("A", changed_functions="f", changed_blocks="f::entry"),
                    _profile("B", changed_functions="f", changed_blocks="f::entry"),
                ],
            )

            rows = build_footprint_overlap(state_dir)

        self.assertEqual(rows[0]["overlap_kind"], "same_block_overlap")
        self.assertEqual(rows[0]["write_block_overlap"], "1")
        self.assertEqual(rows[0]["same_block"], "true")

    def test_missing_changed_sets_are_unknown_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            fieldnames = [
                "program",
                "state_id",
                "state_hash",
                "pass",
                "success",
                "active",
                "funcs_changed",
                "blocks_changed",
            ]
            _write_csv(
                state_dir / "pass_profile.csv",
                fieldnames,
                [
                    _profile("A", funcs_changed="1", blocks_changed="1"),
                    _profile("B", funcs_changed="1", blocks_changed="1"),
                ],
            )

            rows = build_footprint_overlap(state_dir)

        self.assertEqual(rows[0]["overlap_kind"], "unknown_overlap")
        self.assertEqual(rows[0]["pass_a_changed_functions"], "")
        self.assertEqual(rows[0]["pass_b_changed_blocks"], "")

    def test_pair_relation_join_uses_unordered_pair_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_profiles(
                state_dir,
                [
                    _profile("A", changed_functions="f", changed_blocks="f::entry"),
                    _profile("B", changed_functions="f", changed_blocks="f::entry"),
                ],
            )
            _write_pairs(state_dir, [_pair("B", "A", "dynamic_order_sensitive", "final_order_sensitive")])

            rows = build_footprint_overlap(state_dir)

        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_order_sensitive")
        self.assertEqual(rows[0]["final_relation"], "final_order_sensitive")


def _profile(pass_name: str, **overrides: str) -> dict[str, str]:
    row = {
        "program": "testprog",
        "state_id": "S0000",
        "state_hash": "hash0",
        "pass": pass_name,
        "success": "true",
        "active": "true",
        "funcs_changed": "1",
        "blocks_changed": "1",
        "changed_functions": "",
        "changed_blocks": "",
    }
    row.update(overrides)
    return row


def _pair(pass_a: str, pass_b: str, dynamic_relation: str, final_relation: str) -> dict[str, str]:
    return {
        "program": "testprog",
        "state_id": "S0000",
        "state_hash": "hash0",
        "pass_a": pass_a,
        "pass_b": pass_b,
        "dynamic_relation": dynamic_relation,
        "final_relation": final_relation,
    }


def _write_profiles(state_dir: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(
        state_dir / "pass_profile.csv",
        [
            "program",
            "state_id",
            "state_hash",
            "pass",
            "success",
            "active",
            "funcs_changed",
            "blocks_changed",
            "changed_functions",
            "changed_blocks",
        ],
        rows,
    )


def _write_pairs(state_dir: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(
        state_dir / "pair_relation.csv",
        ["program", "state_id", "state_hash", "pass_a", "pass_b", "dynamic_relation", "final_relation"],
        rows,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
