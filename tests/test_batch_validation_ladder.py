import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.batch_validation_ladder import write_batch_validation_ladder_summary


class BatchValidationLadderSummaryTests(unittest.TestCase):
    def test_write_batch_validation_ladder_summary_aggregates_validation_and_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state_dir = run_dir / "states" / "S0000"
            state_dir.mkdir(parents=True)
            _write_csv(
                state_dir / "batch_validation.csv",
                [
                    "program",
                    "state_id",
                    "batch_id",
                    "validation_status",
                    "validation_mode",
                    "validation_tier",
                    "validation_sequences_tested",
                    "validation_hard_certificate",
                    "validation_opt_invocations",
                    "validation_pass_invocations_baseline",
                    "validation_pass_invocations_actual",
                    "validation_pass_invocations_saved",
                    "validation_profile_reuse_hits",
                    "validation_state_transition_cache_hits",
                    "validation_state_equivalence_cache_hits",
                    "time_ms",
                ],
                [
                    {
                        "program": "branch",
                        "state_id": "S0000",
                        "batch_id": "B0000",
                        "validation_status": "all_permutations_same",
                        "validation_mode": "auto",
                        "validation_tier": "exhaustive_all_permutations",
                        "validation_sequences_tested": "6",
                        "validation_hard_certificate": "true",
                        "validation_opt_invocations": "6",
                        "validation_pass_invocations_baseline": "8",
                        "validation_pass_invocations_actual": "6",
                        "validation_pass_invocations_saved": "2",
                        "validation_profile_reuse_hits": "1",
                        "validation_state_transition_cache_hits": "1",
                        "validation_state_equivalence_cache_hits": "0",
                        "time_ms": "10.5",
                    },
                    {
                        "program": "branch",
                        "state_id": "S0000",
                        "batch_id": "B0001",
                        "validation_status": "bounded_same",
                        "validation_mode": "bounded",
                        "validation_tier": "bounded_insertion",
                        "validation_sequences_tested": "5",
                        "validation_hard_certificate": "false",
                        "validation_opt_invocations": "4",
                        "validation_pass_invocations_baseline": "5",
                        "validation_pass_invocations_actual": "4",
                        "validation_pass_invocations_saved": "1",
                        "validation_profile_reuse_hits": "0",
                        "validation_state_transition_cache_hits": "0",
                        "validation_state_equivalence_cache_hits": "2",
                        "time_ms": "2.5",
                    },
                ],
            )
            _write_csv(
                state_dir / "batch_correctness.csv",
                ["batch_id", "correctness_class", "can_execute"],
                [
                    {"batch_id": "B0000", "correctness_class": "certified_batch", "can_execute": "true"},
                    {"batch_id": "B0001", "correctness_class": "bounded_batch", "can_execute": "false"},
                ],
            )

            result = write_batch_validation_ladder_summary(run_dir)
            rows = _read_csv(run_dir / "batch_validation_ladder_summary.csv")
            markdown = (run_dir / "batch_validation_ladder_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["batch_validation_ladder_summary_csv"], str(run_dir / "batch_validation_ladder_summary.csv"))
        self.assertEqual(rows[0]["program"], "branch")
        self.assertEqual(rows[0]["batch_candidates"], "2")
        self.assertEqual(rows[0]["exhaustive_batches"], "1")
        self.assertEqual(rows[0]["bounded_batches"], "1")
        self.assertEqual(rows[0]["hard_certified_batches"], "1")
        self.assertEqual(rows[0]["executable_batches"], "1")
        self.assertEqual(rows[0]["validation_sequences_tested"], "11")
        self.assertEqual(rows[0]["validation_opt_invocations"], "10")
        self.assertEqual(rows[0]["validation_pass_invocations_baseline"], "13")
        self.assertEqual(rows[0]["validation_pass_invocations_actual"], "10")
        self.assertEqual(rows[0]["validation_pass_invocations_saved"], "3")
        self.assertEqual(rows[0]["validation_profile_reuse_hits"], "1")
        self.assertEqual(rows[0]["validation_state_transition_cache_hits"], "1")
        self.assertEqual(rows[0]["validation_state_equivalence_cache_hits"], "2")
        self.assertEqual(rows[0]["validation_time_ms"], "13.000")
        self.assertIn("# Batch Validation Ladder Summary", markdown)
        self.assertIn("- validation opt invocations: 10", markdown)
        self.assertIn("- validation pass invocations saved: 3", markdown)
        self.assertIn("Bounded and sampled validation reduce validation cost but do not become hard commutation proof by default. Only complete all-permutations validation or explicitly complete certificates are hard-foldable.", markdown)

    def test_write_batch_validation_ladder_summary_includes_dag_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state_dir = run_dir / "states" / "S0000"
            state_dir.mkdir(parents=True)
            _write_csv(
                state_dir / "batch_validation.csv",
                [
                    "program",
                    "state_id",
                    "state_hash",
                    "batch_id",
                    "batch_size",
                    "validation_status",
                    "validation_mode",
                    "validation_tier",
                    "validation_sequences_tested",
                    "validation_hard_certificate",
                    "validation_dag_nodes",
                    "validation_dag_edges",
                    "validation_dag_final_classes",
                    "validation_dag_hash_merges",
                    "validation_dag_structural_merges",
                    "validation_dag_transition_cache_hits",
                    "validation_dag_equivalence_cache_hits",
                    "factorial_permutations_log10",
                    "compression_vs_permutation",
                    "time_ms",
                ],
                [
                    {
                        "program": "branch",
                        "state_id": "S0000",
                        "state_hash": "hash0",
                        "batch_id": "B0000",
                        "batch_size": "4",
                        "validation_status": "all_permutations_same",
                        "validation_mode": "dag",
                        "validation_tier": "permutation_dag_exact",
                        "validation_sequences_tested": "20",
                        "validation_hard_certificate": "true",
                        "validation_dag_nodes": "16",
                        "validation_dag_edges": "32",
                        "validation_dag_final_classes": "1",
                        "validation_dag_hash_merges": "3",
                        "validation_dag_structural_merges": "1",
                        "validation_dag_transition_cache_hits": "2",
                        "validation_dag_equivalence_cache_hits": "1",
                        "factorial_permutations_log10": "1.380211",
                        "compression_vs_permutation": "0.75",
                        "time_ms": "10",
                    }
                ],
            )
            _write_csv(
                state_dir / "batch_correctness.csv",
                ["batch_id", "correctness_class", "can_execute"],
                [{"batch_id": "B0000", "correctness_class": "certified_batch", "can_execute": "true"}],
            )

            write_batch_validation_ladder_summary(run_dir)
            markdown = (run_dir / "batch_validation_ladder_summary.md").read_text(encoding="utf-8")

        self.assertIn("## Permutation DAG Validation", markdown)
        self.assertIn("Permutation DAG validation is a hard certificate only when exploration is complete and all full-subset paths merge into one final IR equivalence class. If the DAG budget is exceeded, the result is incomplete and cannot be used for hard folding.", markdown)
        self.assertIn("| branch | S0000 | B0000 | 4 | 1.380211 | 16 | 32 | 1 | permutation_dag_exact | 0.75 |", markdown)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
