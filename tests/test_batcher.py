import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.batcher import build_batch_family, validate_batch_candidates
from phasebatch.ir_equivalence import EqualityResult
from phasebatch.pass_config import PassRegistry, PassSpec
from phasebatch.schema import RunResult
from phasebatch.validation_runtime import ValidationRuntime


class BatcherTests(unittest.TestCase):
    def test_worker_exhaustive_validation_keep_ir_forces_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}],
            )
            _write_summary(state_dir)
            materialize_flags = []

            def fake_run_opt(_opt, _src, _passes, output, _timeout, *, materialize=True):
                materialize_flags.append(materialize)
                Path(output).write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult(
                    ["worker"],
                    0,
                    "",
                    "",
                    1.0,
                    output_path=Path(output),
                    backend="worker",
                    materialized=True,
                )

            with mock.patch("phasebatch.batcher.worker_handles_enabled", return_value=True), \
                mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="exhaustive",
                    keep_ir_artifacts=True,
                )

            validation_dir = state_dir / "artifacts" / "batch_validation" / "B0000"
            self.assertEqual(materialize_flags, [True, True])
            self.assertTrue((validation_dir / "canonical.ll").exists())
            self.assertEqual(len(list(validation_dir.glob("order_*.ll"))), 1)

    def test_worker_exhaustive_validation_uses_hash_fast_path_without_ir_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}],
            )
            _write_summary(state_dir)

            def fake_run_opt(_opt, _src, passes, output, _timeout, *, materialize=True):
                self.assertFalse(materialize)
                return RunResult(
                    ["worker"],
                    0,
                    "",
                    "",
                    1.0,
                    output_path=Path(output),
                    backend="worker",
                    worker_id=0,
                    worker_generation=1,
                    module_handle=";".join(passes),
                    canonical_hash="same-full-ir-hash",
                    materialized=False,
                )

            with mock.patch("phasebatch.batcher.worker_handles_enabled", return_value=True), \
                mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt) as run, \
                mock.patch("phasebatch.batcher.materialize_run_result") as materialize, \
                mock.patch("phasebatch.batcher.compare_ir_equivalence") as compare:
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="exhaustive",
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(run.call_count, 6)
        self.assertEqual(rows[0]["validation_status"], "all_permutations_same")
        self.assertEqual(rows[0]["validation_materializations"], "0")
        self.assertEqual(rows[0]["validation_materializations_avoided"], "6")
        materialize.assert_not_called()
        compare.assert_not_called()

    def test_selected_candidate_validation_emits_explicit_unvalidated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [
                    {"batch_id": "B0000", "batch_passes": "A", "batch_size": "1", "canonical_order": "A"},
                    {"batch_id": "B0001", "batch_passes": "B", "batch_size": "1", "canonical_order": "B"},
                    {"batch_id": "B0002", "batch_passes": "C", "batch_size": "1", "canonical_order": "C"},
                    {"batch_id": "B0003", "batch_passes": "D", "batch_size": "1", "canonical_order": "D"},
                ],
            )
            _write_summary(state_dir)
            calls = []

            def fake_run_opt(opt, src, passes, out, timeout):
                calls.append(passes[0])
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                runtime = ValidationRuntime(state_dir, max_workers=1)
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    candidate_ids=["B0001", "B0002"],
                    runtime=runtime,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(calls, ["B", "C"])
        self.assertEqual([row["batch_id"] for row in rows], ["B0000", "B0001", "B0002", "B0003"])
        self.assertEqual([row["validation_status"] for row in rows], ["not_validated", "all_permutations_same", "all_permutations_same", "not_validated"])
        self.assertEqual(rows[0]["validation_tier"], "unvalidated")
        self.assertEqual(rows[0]["validation_complete"], "false")
        self.assertEqual(rows[0]["validation_hard_certificate"], "false")
        self.assertEqual(rows[0]["validation_incomplete_reason"], "budgeted_on_demand_not_selected")

    def test_candidate_validation_is_parallel_but_output_order_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [
                    {"batch_id": "B0000", "batch_passes": "A", "batch_size": "1", "canonical_order": "A"},
                    {"batch_id": "B0001", "batch_passes": "B", "batch_size": "1", "canonical_order": "B"},
                    {"batch_id": "B0002", "batch_passes": "C", "batch_size": "1", "canonical_order": "C"},
                ],
            )
            _write_summary(state_dir)
            active = 0
            peak = 0
            lock = threading.Lock()

            def fake_run_opt(opt, src, passes, out, timeout):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep({"A": 0.06, "B": 0.03, "C": 0.01}[passes[0]])
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                with lock:
                    active -= 1
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(state_dir, {"opt": "opt"}, timeout=1, jobs=2, max_permutation_factorial=1)
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(peak, 2)
        self.assertEqual([row["batch_id"] for row in rows], ["B0000", "B0001", "B0002"])
    def test_fully_commuting_four_passes_produces_one_full_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_state(state_dir, ["A", "B", "C", "D"], _all_pairs(["A", "B", "C", "D"], "final_commute"))

            result = build_batch_family(state_dir)
            candidates = _read_csv(state_dir / "batch_candidates.csv")
            summary = _read_csv(state_dir / "batch_summary.csv")[0]

        self.assertEqual(result["batch_candidates"], 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["batch_passes"], "A;B;C;D")
        self.assertEqual(candidates[0]["is_exact"], "true")
        self.assertEqual(summary["active_pairs"], "6")
        self.assertEqual(summary["commute_pairs"], "6")
        self.assertEqual(summary["conflict_pairs"], "0")
        self.assertEqual(summary["naive_orderings_estimate"], "24")
        self.assertEqual(summary["truncated"], "false")
        self.assertEqual(summary["max_batch_candidates"], "200")

    def test_fully_conflicting_three_passes_with_two_independent_passes_produces_three_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            passes = ["A", "B", "C", "D", "E"]
            relations = _all_pairs(passes, "final_commute")
            relations.update({("A", "B"): "final_order_sensitive", ("A", "C"): "final_order_sensitive", ("B", "C"): "final_order_sensitive"})
            _write_state(state_dir, passes, relations)

            result = build_batch_family(state_dir)
            batches = {row["batch_passes"] for row in _read_csv(state_dir / "batch_candidates.csv")}
            summary = _read_csv(state_dir / "batch_summary.csv")[0]

        self.assertEqual(result["batch_candidates"], 3)
        self.assertEqual(batches, {"A;D;E", "B;D;E", "C;D;E"})
        self.assertEqual(summary["conflict_pairs"], "3")
        self.assertEqual(summary["batch_reduction_estimate"], "40.00")

    def test_path_conflict_component_uses_maximal_independent_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            relations = {("A", "B"): "final_order_sensitive", ("A", "C"): "final_commute", ("B", "C"): "final_order_sensitive"}
            _write_state(state_dir, ["A", "B", "C"], relations)

            build_result = build_batch_family(state_dir)
            batches = {row["batch_passes"] for row in _read_csv(state_dir / "batch_candidates.csv")}

        self.assertEqual(build_result["batch_candidates"], 2)
        self.assertEqual(batches, {"A;C", "B"})

    def test_unknown_relation_is_conservative_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_state(state_dir, ["A", "B"], {("A", "B"): "final_unknown"})

            result = build_batch_family(state_dir)
            batches = {row["batch_passes"] for row in _read_csv(state_dir / "batch_candidates.csv")}
            components = _read_csv(state_dir / "batch_components.csv")
            summary_text = (state_dir / "batch_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["batch_candidates"], 2)
        self.assertEqual(batches, {"A", "B"})
        self.assertEqual(components[0]["conflict_edges"], "A--B")
        self.assertIn("Batch Summary", summary_text)

    def test_lazy_skipped_pair_is_conservative_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_csv(
                state_dir / "pass_profile.csv",
                ["program", "state_id", "state_hash", "pass", "success", "active"],
                [
                    {"program": "testprog", "state_id": "S0000", "state_hash": "hash0", "pass": "A", "success": "true", "active": "true"},
                    {"program": "testprog", "state_id": "S0000", "state_hash": "hash0", "pass": "B", "success": "true", "active": "true"},
                ],
            )
            _write_csv(
                state_dir / "pair_relation.csv",
                ["program", "state_id", "state_hash", "pass_a", "pass_b", "dynamic_relation", "final_relation", "failure_kind", "skipped_by_budget"],
                [
                    {
                        "program": "testprog",
                        "state_id": "S0000",
                        "state_hash": "hash0",
                        "pass_a": "A",
                        "pass_b": "B",
                        "dynamic_relation": "not_tested",
                        "final_relation": "final_unknown",
                        "failure_kind": "lazy_budget",
                        "skipped_by_budget": "true",
                    }
                ],
            )

            result = build_batch_family(state_dir)
            batches = {row["batch_passes"] for row in _read_csv(state_dir / "batch_candidates.csv")}
            components = _read_csv(state_dir / "batch_components.csv")

        self.assertEqual(result["batch_candidates"], 2)
        self.assertEqual(batches, {"A", "B"})
        self.assertEqual(components[0]["conflict_edges"], "A--B")

    def test_no_active_passes_produces_no_empty_batch_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_state(state_dir, [], {})

            result = build_batch_family(state_dir)
            candidates = _read_csv(state_dir / "batch_candidates.csv")
            summary = _read_csv(state_dir / "batch_summary.csv")[0]

        self.assertEqual(result["active_passes"], 0)
        self.assertEqual(result["batch_candidates"], 0)
        self.assertEqual(candidates, [])
        self.assertEqual(summary["batch_candidates"], "0")

    def test_pair_relation_keys_are_unordered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_state(state_dir, ["B", "A"], {("A", "B"): "final_commute"})

            result = build_batch_family(state_dir)
            candidates = _read_csv(state_dir / "batch_candidates.csv")

        self.assertEqual(result["batch_candidates"], 1)
        self.assertEqual(candidates[0]["batch_passes"], "B;A")

    def test_validate_batch_candidates_all_permutations_same_is_strong_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}])
            _write_summary(state_dir)

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                result = validate_batch_candidates(state_dir, {"opt": "opt"}, timeout=1, jobs=2, max_permutation_factorial=6)
            rows = _read_csv(state_dir / "batch_validation.csv")
            summary_text = (state_dir / "batch_summary.md").read_text(encoding="utf-8")

        self.assertEqual(result["validated_batches"], 1)
        self.assertEqual(rows[0]["tested_orders"], "6")
        self.assertEqual(rows[0]["same_hash_count"], "6")
        self.assertEqual(rows[0]["different_hash_count"], "0")
        self.assertEqual(rows[0]["validation_status"], "all_permutations_same")
        self.assertEqual(rows[0]["validation_mode"], "auto")
        self.assertEqual(rows[0]["validation_tier"], "exhaustive_all_permutations")
        self.assertEqual(rows[0]["validation_complete"], "true")
        self.assertEqual(rows[0]["validation_hard_certificate"], "true")
        self.assertIn("all_permutations_same", summary_text)

    def test_validation_reuses_profile_output_for_noncanonical_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [
                    {
                        "batch_id": "B0000",
                        "batch_passes": "A;B",
                        "batch_size": "2",
                        "canonical_order": "A;B",
                    }
                ],
            )
            _write_summary(state_dir)
            profile_dir = state_dir / "artifacts" / "single_pass"
            profile_dir.mkdir(parents=True)
            profile_a = profile_dir / "A.ll"
            profile_b = profile_dir / "B.ll"
            profile_a.write_text("profile A\n", encoding="utf-8")
            profile_b.write_text("profile B\n", encoding="utf-8")
            _write_csv(
                state_dir / "pass_profile.csv",
                ["state_hash", "pass", "success", "active", "output_path"],
                [
                    {
                        "state_hash": "hash0",
                        "pass": "A",
                        "success": "true",
                        "active": "true",
                        "output_path": str(profile_a),
                    },
                    {
                        "state_hash": "hash0",
                        "pass": "B",
                        "success": "true",
                        "active": "true",
                        "output_path": str(profile_b),
                    },
                ],
            )
            seen_calls: list[dict] = []

            def fake_run_opt(opt, src, passes, out, timeout):
                seen_calls.append({"src": Path(src), "passes": list(passes)})
                out.write_text(
                    "define i32 @f() {\n  ret i32 0\n}\n",
                    encoding="utf-8",
                )
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    max_permutation_factorial=2,
                )
            row = _read_csv(state_dir / "batch_validation.csv")[0]

        self.assertEqual(seen_calls[0]["passes"], ["A", "B"])
        self.assertEqual(seen_calls[1]["src"], profile_b)
        self.assertEqual(seen_calls[1]["passes"], ["A"])
        self.assertEqual(row["validation_opt_invocations"], "2")
        self.assertEqual(row["validation_pass_invocations_baseline"], "4")
        self.assertEqual(row["validation_pass_invocations_actual"], "3")
        self.assertEqual(row["validation_pass_invocations_saved"], "1")
        self.assertEqual(row["validation_profile_reuse_hits"], "1")

    def test_validation_ignores_profile_output_from_another_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}],
            )
            _write_summary(state_dir)
            stale_output = state_dir / "stale-B.ll"
            stale_output.write_text("stale\n", encoding="utf-8")
            _write_csv(
                state_dir / "pass_profile.csv",
                ["state_hash", "pass", "success", "active", "output_path"],
                [{"state_hash": "other-hash", "pass": "B", "success": "true", "active": "true", "output_path": str(stale_output)}],
            )
            seen_calls: list[tuple[Path, list[str]]] = []

            def fake_run_opt(opt, src, passes, out, timeout):
                seen_calls.append((Path(src), list(passes)))
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(state_dir, {"opt": "opt"}, timeout=1, jobs=1, max_permutation_factorial=2)
            row = _read_csv(state_dir / "batch_validation.csv")[0]

        self.assertEqual(seen_calls[1], (state_dir / "input.ll", ["B", "A"]))
        self.assertEqual(row["validation_profile_reuse_hits"], "0")
        self.assertEqual(row["validation_pass_invocations_saved"], "0")

    def test_singleton_validation_keeps_full_canonical_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A", "batch_size": "1", "canonical_order": "A"}],
            )
            _write_summary(state_dir)
            profile_output = state_dir / "profile-A.ll"
            profile_output.write_text("profile\n", encoding="utf-8")
            _write_csv(
                state_dir / "pass_profile.csv",
                ["state_hash", "pass", "success", "active", "output_path"],
                [{"state_hash": "hash0", "pass": "A", "success": "true", "active": "true", "output_path": str(profile_output)}],
            )
            seen_calls: list[tuple[Path, list[str]]] = []

            def fake_run_opt(opt, src, passes, out, timeout):
                seen_calls.append((Path(src), list(passes)))
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(state_dir, {"opt": "opt"}, timeout=1, jobs=1, max_permutation_factorial=1)
            row = _read_csv(state_dir / "batch_validation.csv")[0]

        self.assertEqual(seen_calls, [(state_dir / "input.ll", ["A"])])
        self.assertEqual(row["validation_opt_invocations"], "1")
        self.assertEqual(row["validation_pass_invocations_baseline"], "1")
        self.assertEqual(row["validation_pass_invocations_actual"], "1")
        self.assertEqual(row["validation_profile_reuse_hits"], "0")

    def test_validate_batch_candidates_auto_uses_dag_for_large_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C;D", "batch_size": "4", "canonical_order": "A;B;C;D"}])
            _write_summary(state_dir)

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    max_permutation_factorial=2,
                    max_validation_sequences=5,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "all_permutations_same")
        self.assertEqual(rows[0]["validation_tier"], "permutation_dag_exact")
        self.assertEqual(rows[0]["validation_complete"], "true")
        self.assertEqual(rows[0]["validation_hard_certificate"], "true")
        self.assertGreater(int(rows[0]["validation_dag_edges"]), 0)
        self.assertEqual(rows[0]["validation_sequences_total_estimate"], "24")

    def test_validate_batch_candidates_detects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}])
            _write_summary(state_dir)

            def fake_run_opt(opt, src, passes, out, timeout):
                value = 0 if passes == ["A", "B"] else 1
                out.write_text(f"define i32 @f() {{\n  ret i32 {value}\n}}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(state_dir, {"opt": "opt"}, timeout=1, jobs=1, max_permutation_factorial=2)
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "mismatch")
        self.assertEqual(rows[0]["tested_orders"], "2")
        self.assertEqual(rows[0]["same_hash_count"], "1")
        self.assertEqual(rows[0]["different_hash_count"], "1")
        self.assertEqual(rows[0]["first_mismatch_order"], "B;A")
        self.assertTrue(rows[0]["first_mismatch_hash"])

    def test_bounded_validation_detects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}])
            _write_summary(state_dir)

            def fake_run_opt(opt, src, passes, out, timeout):
                value = 0 if passes == ["A", "B", "C"] else 1
                out.write_text(f"define i32 @f() {{\n  ret i32 {value}\n}}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="bounded",
                    max_validation_sequences=4,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "mismatch")
        self.assertEqual(rows[0]["validation_tier"], "bounded_insertion")
        self.assertEqual(rows[0]["validation_hard_certificate"], "false")

    def test_validate_batch_candidates_accepts_structural_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}])
            _write_summary(state_dir)

            def fake_run_opt(opt, src, passes, out, timeout):
                if passes == ["A", "B"]:
                    out.write_text("define i32 @f(i32 %x) {\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n", encoding="utf-8")
                else:
                    out.write_text("define i32 @f(i32 %x) {\n  %b = add i32 %x, 0\n  ret i32 %b\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            structural = EqualityResult(
                equal=True,
                tier="structural_diff",
                can_hard_fold=True,
                reason="llvm_diff_equal_and_module_fingerprint_equal",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=True,
                left_hash="canonical",
                right_hash="permutation",
            )

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.batcher.compare_ir_equivalence", return_value=structural):
                validate_batch_candidates(state_dir, {"opt": "opt", "llvm-diff": "llvm-diff"}, timeout=1, jobs=1, max_permutation_factorial=2)
            rows = _read_csv(state_dir / "batch_validation.csv")
            summary_text = (state_dir / "batch_summary.md").read_text(encoding="utf-8")

        self.assertEqual(rows[0]["validation_status"], "all_permutations_same")
        self.assertEqual(rows[0]["tested_orders"], "2")
        self.assertEqual(rows[0]["same_hash_count"], "1")
        self.assertEqual(rows[0]["different_hash_count"], "1")
        self.assertEqual(rows[0]["hash_equal_count"], "1")
        self.assertEqual(rows[0]["structural_equal_count"], "1")
        self.assertEqual(rows[0]["different_count"], "0")
        self.assertEqual(rows[0]["canonical_hash_equal_count"], "1")
        self.assertEqual(rows[0]["structural_diff_equal_count"], "1")
        self.assertEqual(rows[0]["equality_failed_count"], "0")
        self.assertEqual(rows[0]["validation_equality_tier"], "structural_diff")
        self.assertEqual(rows[0]["validation_equality_reason"], "llvm_diff_equal_and_module_fingerprint_equal")
        self.assertIn("Equality Tier Summary", summary_text)
        self.assertIn("| tier | count | hard_fold |", summary_text)
        self.assertIn("| structural_diff | 1 | 1 |", summary_text)
        self.assertIn("structural_diff", summary_text)

    def test_validate_batch_candidates_samples_large_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C;D;E;F", "batch_size": "6", "canonical_order": "A;B;C;D;E;F"}])
            _write_summary(state_dir)

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=2,
                    max_permutation_factorial=2,
                    batch_validation_mode="sampled",
                    samples=3,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "sampled_same")
        self.assertEqual(rows[0]["tested_orders"], "4")
        self.assertEqual(rows[0]["same_hash_count"], "4")

    def test_validate_batch_candidates_executes_pipeline_text_but_records_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "licm;dce", "batch_size": "2", "canonical_order": "licm;dce"}])
            _write_summary(state_dir)
            registry = PassRegistry.from_specs(
                [
                    PassSpec("licm", "function(loop(licm))", ["function(loop(licm))"], "loop", "v3", True),
                    PassSpec("dce", "dce", ["dce"], "cleanup", "v1", True),
                ]
            )
            seen_orders: list[list[str]] = []

            def fake_run_opt(opt, src, passes, out, timeout):
                seen_orders.append(passes)
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    max_permutation_factorial=2,
                    pass_registry=registry,
                )

            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(seen_orders[0], ["function(loop(licm))", "dce"])
        self.assertEqual(rows[0]["canonical_order"], "licm;dce")

    def test_validate_batch_candidates_marks_comparator_failure_as_failed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}])
            _write_summary(state_dir)

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            failed = EqualityResult(
                equal=False,
                tier="failed",
                can_hard_fold=False,
                reason="tool_failed",
                text_hash_equal=False,
                left_hash="canonical",
                right_hash="permutation",
                error_message="llvm-diff not found",
            )

            with mock.patch("phasebatch.batcher.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.batcher.compare_ir_equivalence", return_value=failed):
                validate_batch_candidates(state_dir, {"opt": "opt"}, timeout=1, jobs=1, max_permutation_factorial=2)

            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "failed")
        self.assertEqual(rows[0]["validation_equality_tier"], "failed")
        self.assertEqual(rows[0]["validation_equality_reason"], "tool_failed")
        self.assertEqual(rows[0]["different_count"], "0")
        self.assertEqual(rows[0]["equality_failed_count"], "1")
        self.assertEqual(rows[0]["first_mismatch_order"], "")


def _write_state(state_dir: Path, passes: list[str], relations: dict[tuple[str, str], str]) -> None:
    _write_csv(
        state_dir / "pass_profile.csv",
        ["program", "state_id", "state_hash", "pass", "success", "active"],
        [
            {"program": "testprog", "state_id": "S0000", "state_hash": "hash0", "pass": pass_name, "success": "true", "active": "true"}
            for pass_name in passes
        ],
    )
    _write_csv(
        state_dir / "pair_relation.csv",
        ["program", "state_id", "state_hash", "pass_a", "pass_b", "final_relation"],
        [
            {
                "program": "testprog",
                "state_id": "S0000",
                "state_hash": "hash0",
                "pass_a": pass_a,
                "pass_b": pass_b,
                "final_relation": relation,
            }
            for (pass_a, pass_b), relation in relations.items()
        ],
    )


def _write_input(state_dir: Path) -> None:
    (state_dir / "input.ll").write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")


def _write_candidates(state_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "program",
        "state_id",
        "state_hash",
        "batch_id",
        "batch_passes",
        "batch_size",
        "component_choices",
        "is_exact",
        "num_conflict_components",
        "unresolved_components",
        "canonical_order",
    ]
    for row in rows:
        row.setdefault("program", "testprog")
        row.setdefault("state_id", "S0000")
        row.setdefault("state_hash", "hash0")
        row.setdefault("component_choices", "")
        row.setdefault("is_exact", "true")
        row.setdefault("num_conflict_components", "1")
        row.setdefault("unresolved_components", "0")
    _write_csv(state_dir / "batch_candidates.csv", fieldnames, rows)


def _write_summary(state_dir: Path) -> None:
    (state_dir / "batch_summary.md").write_text("# Batch Summary\n", encoding="utf-8")


def _all_pairs(passes: list[str], relation: str) -> dict[tuple[str, str], str]:
    return {(passes[i], passes[j]): relation for i in range(len(passes)) for j in range(i + 1, len(passes))}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
