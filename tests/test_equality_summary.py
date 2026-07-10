import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.equality_summary import (
    equality_tier_markdown,
    equality_tier_summary_for_run,
    equality_tier_summary_from_rows,
    write_equality_tier_summary,
)


class EqualitySummaryTests(unittest.TestCase):
    def test_counts_tiers_and_hard_fold_rows(self) -> None:
        rows = [
            {"equality_tier": "canonical_hash", "can_hard_fold": "true"},
            {"equality_tier": "canonical_hash", "can_hard_fold": "true"},
            {"equality_tier": "structural_diff", "can_hard_fold": "true"},
            {"equality_tier": "different", "can_hard_fold": "false"},
            {"equality_tier": "failed", "can_hard_fold": "false"},
        ]

        summary = {row["tier"]: row for row in equality_tier_summary_from_rows(rows)}

        self.assertEqual(summary["canonical_hash"]["count"], 2)
        self.assertEqual(summary["canonical_hash"]["hard_fold"], 2)
        self.assertEqual(summary["structural_diff"]["count"], 1)
        self.assertEqual(summary["structural_diff"]["hard_fold"], 1)
        self.assertEqual(summary["different"]["count"], 1)
        self.assertEqual(summary["different"]["hard_fold"], 0)
        self.assertEqual(summary["failed"]["count"], 1)
        self.assertEqual(summary["failed"]["hard_fold"], 0)

    def test_markdown_uses_stable_table_shape(self) -> None:
        lines = equality_tier_markdown([
            {"tier": "canonical_hash", "count": 2, "hard_fold": 2},
            {"tier": "structural_diff", "count": 1, "hard_fold": 1},
        ])

        text = "\n".join(lines)

        self.assertIn("Equality Tier Summary", text)
        self.assertIn("| tier | count | hard_fold |", text)
        self.assertIn("| canonical_hash | 2 | 2 |", text)

    def test_run_summary_reads_unique_state_pair_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            s0 = run_dir / "states" / "S0000"
            s1 = run_dir / "states" / "S0001"
            s0.mkdir(parents=True)
            s1.mkdir(parents=True)
            _write_csv(
                run_dir / "states.csv",
                ["state_id", "state_dir", "is_duplicate"],
                [
                    {"state_id": "S0000", "state_dir": str(s0), "is_duplicate": "false"},
                    {"state_id": "S0001", "state_dir": str(s1), "is_duplicate": "false"},
                    {"state_id": "S0002", "state_dir": str(s1), "is_duplicate": "true"},
                ],
            )
            _write_csv(
                s0 / "pair_relation.csv",
                ["equality_tier", "can_hard_fold"],
                [{"equality_tier": "canonical_hash", "can_hard_fold": "true"}],
            )
            _write_csv(
                s1 / "pair_relation.csv",
                ["equality_tier", "can_hard_fold"],
                [{"equality_tier": "structural_diff", "can_hard_fold": "true"}],
            )

            summary = {row["tier"]: row for row in equality_tier_summary_for_run(run_dir)}

        self.assertEqual(summary["canonical_hash"]["count"], 1)
        self.assertEqual(summary["structural_diff"]["count"], 1)

    def test_write_equality_tier_summary_aggregates_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            s0 = run_dir / "states" / "S0000"
            s0.mkdir(parents=True)
            _write_csv(
                run_dir / "states.csv",
                ["state_id", "state_dir", "is_duplicate"],
                [{"state_id": "S0000", "state_dir": str(s0), "is_duplicate": "false"}],
            )
            _write_csv(
                s0 / "pair_relation.csv",
                ["equality_tier", "equality_reason", "can_hard_fold"],
                [
                    {
                        "equality_tier": "structural_diff",
                        "equality_reason": "llvm_diff_equal_and_module_fingerprint_equal",
                        "can_hard_fold": "true",
                    },
                    {
                        "equality_tier": "different",
                        "equality_reason": "module_fingerprint_difference",
                        "can_hard_fold": "false",
                    },
                ],
            )
            _write_csv(
                s0 / "batch_validation.csv",
                ["validation_equality_tier", "validation_equality_reason", "validation_status"],
                [
                    {
                        "validation_equality_tier": "structural_diff",
                        "validation_equality_reason": "llvm_diff_equal_and_module_fingerprint_equal",
                        "validation_status": "all_permutations_same",
                    },
                    {
                        "validation_equality_tier": "failed",
                        "validation_equality_reason": "tool_failed",
                        "validation_status": "failed",
                    },
                ],
            )
            _write_csv(
                run_dir / "pipeline_replay.csv",
                ["equality_tier", "equality_reason", "can_hard_fold"],
                [
                    {
                        "equality_tier": "canonical_hash",
                        "equality_reason": "hash_equal",
                        "can_hard_fold": "true",
                    }
                ],
            )

            result = write_equality_tier_summary(run_dir)
            rows = _read_csv(run_dir / "equality_tier_summary.csv")
            markdown = (run_dir / "equality_tier_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["equality_tier_summary_csv"], str(run_dir / "equality_tier_summary.csv"))
        self.assertEqual(result["equality_tier_summary_md"], str(run_dir / "equality_tier_summary.md"))
        indexed = {(row["source"], row["equality_tier"], row["equality_reason"]): row for row in rows}
        self.assertEqual(indexed[("pair_relation", "structural_diff", "llvm_diff_equal_and_module_fingerprint_equal")]["hard_fold_count"], "1")
        self.assertEqual(indexed[("batch_validation", "structural_diff", "llvm_diff_equal_and_module_fingerprint_equal")]["hard_fold_count"], "1")
        self.assertEqual(indexed[("batch_validation", "failed", "tool_failed")]["hard_fold_count"], "0")
        self.assertEqual(indexed[("pipeline_replay", "canonical_hash", "hash_equal")]["hard_fold_count"], "1")
        self.assertIn("# Equality Tier Summary", markdown)
        self.assertIn("## Pair Relations", markdown)
        self.assertIn("## Batch Validation", markdown)
        self.assertIn("## Replay", markdown)
        self.assertIn(
            "Structural fallback is used only to avoid false conflicts from local naming or harmless structural noise. Module safety fingerprint guards against false commutativity from optimization-relevant attributes, metadata, globals, target information, or datalayout differences.",
            markdown,
        )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
