import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.batch_correctness import classify_batch_correctness
from phasebatch.batch_validation_dag import validate_batch_with_permutation_dag
from phasebatch.batcher import validate_batch_candidates
from phasebatch.ir_equivalence import EqualityResult
from phasebatch.schema import RunResult
from phasebatch.validation_runtime import ValidationRuntime


class BatchValidationDagTests(unittest.TestCase):
    def test_candidate_validation_closes_owned_runtime_when_validation_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}],
            )
            _write_summary(state_dir)
            with mock.patch(
                "phasebatch.batcher._validate_one_batch",
                side_effect=RuntimeError("validation failed"),
            ), mock.patch.object(
                ValidationRuntime,
                "close",
                autospec=True,
                return_value=0,
            ) as close:
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    validate_batch_candidates(
                        state_dir,
                        {"opt": "opt"},
                        timeout=1,
                        jobs=1,
                    )

        close.assert_called_once()

    def test_worker_dag_keep_ir_forces_file_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            materialize_flags = []

            def fake_run_opt(_opt, _src, _passes, output, _timeout, *, materialize=True):
                materialize_flags.append(materialize)
                Path(output).write_text("define void @f() { ret void }\n", encoding="utf-8")
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

            out_dir = root / "dag"
            with mock.patch("phasebatch.batch_validation_dag.worker_handles_enabled", return_value=True), \
                mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.batch_validation_dag.run_opt_from_result") as child_run:
                row = validate_batch_with_permutation_dag(
                    input_ll,
                    ["A", "B"],
                    ["A", "B"],
                    None,
                    {"opt": "opt"},
                    out_dir,
                    max_nodes=100,
                    max_edges=100,
                    timeout=1,
                    keep_ir_artifacts=True,
                )

            self.assertEqual(row["validation_status"], "all_permutations_same")
            self.assertTrue(materialize_flags)
            self.assertTrue(all(materialize_flags))
            self.assertTrue(list(out_dir.rglob("*.ll")))
            self.assertTrue((out_dir / ".keep_ir_artifacts").exists())
            child_run.assert_not_called()

    def test_candidate_validation_closes_owned_state_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}],
            )
            _write_summary(state_dir)
            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_commuting_run_opt), \
                mock.patch.object(ValidationRuntime, "close", autospec=True, return_value=0) as close:
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                )

        close.assert_called_once()

    def test_worker_handle_dag_avoids_materializing_hash_equal_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")

            def result_for(tokens, output):
                state = ";".join(sorted(set(tokens)))
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
                    module_handle=state,
                    canonical_hash=state,
                    feature_counts={"instructions": len(tokens)},
                    materialized=False,
                )

            def fake_root(_opt, _src, passes, output, _timeout, *, materialize=True):
                self.assertFalse(materialize)
                return result_for(passes, output)

            def fake_child(parent, passes, output, _timeout, *, materialize=True):
                self.assertFalse(materialize)
                prior = [token for token in parent.module_handle.split(";") if token]
                return result_for([*prior, *passes], output)

            with mock.patch("phasebatch.batch_validation_dag.worker_handles_enabled", return_value=True), \
                mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=fake_root) as root_run, \
                mock.patch("phasebatch.batch_validation_dag.run_opt_from_result", side_effect=fake_child) as child_run, \
                mock.patch("phasebatch.batch_validation_dag.materialize_run_result") as materialize, \
                mock.patch("phasebatch.batch_validation_dag.compare_ir_equivalence") as compare, \
                mock.patch("phasebatch.validation_runtime.release_run_result", return_value=True):
                row = validate_batch_with_permutation_dag(
                    input_ll,
                    ["A", "B", "C"],
                    ["A", "B", "C"],
                    None,
                    {"opt": "opt"},
                    root / "dag",
                    max_nodes=100,
                    max_edges=100,
                    timeout=1,
                )

        self.assertEqual(row["validation_status"], "all_permutations_same")
        self.assertEqual(row["validation_materializations"], "0")
        self.assertEqual(row["validation_materializations_avoided"], "12")
        self.assertEqual(root_run.call_count, 3)
        self.assertEqual(child_run.call_count, 9)
        materialize.assert_not_called()
        compare.assert_not_called()

    def test_worker_handle_dag_materializes_only_hash_different_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")

            def result_for(tokens, output):
                state = ";".join(tokens)
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
                    module_handle=state,
                    canonical_hash=state,
                    materialized=False,
                )

            def fake_root(_opt, _src, passes, output, _timeout, *, materialize=True):
                self.assertFalse(materialize)
                return result_for(passes, output)

            def fake_child(parent, passes, output, _timeout, *, materialize=True):
                self.assertFalse(materialize)
                prior = [token for token in parent.module_handle.split(";") if token]
                return result_for([*prior, *passes], output)

            def materialize(result, path, *, timeout):
                del timeout
                Path(path).write_text(f"define void @f() {{ ret void }} ; {result.module_handle}\n", encoding="utf-8")
                result.materialized = True
                return Path(path)

            structural = EqualityResult(
                equal=True,
                tier="structural_diff",
                can_hard_fold=True,
                reason="llvm_diff_equal_and_module_fingerprint_equal",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=True,
            )
            with mock.patch("phasebatch.batch_validation_dag.worker_handles_enabled", return_value=True), \
                mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=fake_root), \
                mock.patch("phasebatch.batch_validation_dag.run_opt_from_result", side_effect=fake_child), \
                mock.patch("phasebatch.batch_validation_dag.materialize_run_result", side_effect=materialize) as materialize_call, \
                mock.patch("phasebatch.batch_validation_dag.compare_ir_equivalence", return_value=structural) as compare, \
                mock.patch("phasebatch.validation_runtime.release_run_result", return_value=True):
                row = validate_batch_with_permutation_dag(
                    input_ll,
                    ["A", "B"],
                    ["A", "B"],
                    None,
                    {"opt": "opt", "llvm-diff": "llvm-diff"},
                    root / "dag",
                    max_nodes=100,
                    max_edges=100,
                    timeout=1,
                )

        self.assertEqual(row["validation_status"], "all_permutations_same")
        self.assertEqual(row["validation_dag_structural_merges"], "1")
        self.assertEqual(row["validation_materializations"], "2")
        self.assertEqual(row["validation_materializations_avoided"], "2")
        self.assertEqual(materialize_call.call_count, 2)
        compare.assert_called_once()

    def test_parallel_dag_matches_serial_node_and_edge_order(self) -> None:
        snapshots = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for jobs in (1, 4):
                state_dir = root / f"jobs_{jobs}"
                state_dir.mkdir()
                _write_input(state_dir)
                _write_candidates(
                    state_dir,
                    [{"batch_id": "B0000", "batch_passes": "A;B;C;D", "batch_size": "4", "canonical_order": "A;B;C;D"}],
                )
                _write_summary(state_dir)

                def delayed_run_opt(opt, src, passes, out, timeout):
                    time.sleep({"A": 0.012, "B": 0.009, "C": 0.006, "D": 0.003}[passes[0]])
                    return _commuting_run_opt(opt, src, passes, out, timeout)

                with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=delayed_run_opt):
                    validate_batch_candidates(
                        state_dir,
                        {"opt": "opt"},
                        timeout=1,
                        jobs=jobs,
                        batch_validation_mode="dag",
                        max_validation_dag_nodes=5000,
                        max_validation_dag_edges=20000,
                        dump_validation_dag=True,
                    )
                row = _read_csv(state_dir / "batch_validation.csv")[0]
                nodes = _read_csv(
                    state_dir / "artifacts" / "batch_validation" / "B0000" / "validation_dag_B0000_nodes.csv"
                )
                edges = _read_csv(
                    state_dir / "artifacts" / "batch_validation" / "B0000" / "validation_dag_B0000_edges.csv"
                )
                snapshots.append(
                    (
                        {
                            key: row[key]
                            for key in (
                                "validation_status",
                                "validation_tier",
                                "validation_complete",
                                "validation_hard_certificate",
                                "validation_dag_nodes",
                                "validation_dag_edges",
                                "validation_dag_final_classes",
                                "validation_dag_transition_cache_hits",
                                "validation_dag_transition_cache_misses",
                            )
                        },
                        nodes,
                        edges,
                    )
                )

        self.assertEqual(snapshots[0], snapshots[1])

    def test_parallel_failures_report_earliest_deterministic_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}],
            )
            _write_summary(state_dir)

            def failing_run_opt(opt, src, passes, out, timeout):
                pass_name = passes[0]
                time.sleep({"A": 0.03, "B": 0.005, "C": 0.01}[pass_name])
                if pass_name in {"A", "B"}:
                    return RunResult([opt], 1, "", f"failed {pass_name}", 1.0)
                return _commuting_run_opt(opt, src, passes, out, timeout)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=failing_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=3,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            row = _read_csv(state_dir / "batch_validation.csv")[0]

        self.assertEqual(row["validation_status"], "failed")
        self.assertEqual(row["validation_equality_reason"], "validation_dag_opt_failed:A")

    def test_same_depth_transitions_use_requested_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B;C;D", "batch_size": "4", "canonical_order": "A;B;C;D"}],
            )
            _write_summary(state_dir)
            active = 0
            peak = 0
            lock = threading.Lock()

            def delayed_run_opt(opt, src, passes, out, timeout):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.02)
                    return _commuting_run_opt(opt, src, passes, out, timeout)
                finally:
                    with lock:
                        active -= 1

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=delayed_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=3,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )

        self.assertEqual(peak, 3)

    def test_dag_root_transitions_reuse_profile_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}],
            )
            _write_summary(state_dir)
            profile_dir = state_dir / "artifacts" / "single_pass"
            profile_dir.mkdir(parents=True)
            profile_rows = []
            for pass_name in ("A", "B", "C"):
                output = profile_dir / f"{pass_name}.ll"
                _write_ir(output, pass_name)
                profile_rows.append(
                    {
                        "state_hash": "hash0",
                        "pass": pass_name,
                        "success": "true",
                        "active": "true",
                        "output_path": str(output),
                    }
                )
            _write_csv(
                state_dir / "pass_profile.csv",
                ["state_hash", "pass", "success", "active", "output_path"],
                profile_rows,
            )
            calls: list[tuple[Path, list[str]]] = []

            def recording_run_opt(opt, src, passes, out, timeout):
                calls.append((Path(src), list(passes)))
                return _commuting_run_opt(opt, src, passes, out, timeout)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=recording_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            row = _read_csv(state_dir / "batch_validation.csv")[0]

        self.assertEqual(len(calls), 9)
        self.assertFalse(any(src == state_dir / "input.ll" for src, _passes in calls))
        self.assertEqual(row["validation_profile_reuse_hits"], "3")
        self.assertGreaterEqual(int(row["validation_state_transition_cache_hits"]), 3)
        self.assertEqual(row["validation_opt_invocations"], "9")
        self.assertEqual(row["validation_pass_invocations_baseline"], "12")
        self.assertEqual(row["validation_pass_invocations_actual"], "9")
        self.assertEqual(row["validation_pass_invocations_saved"], "3")

    def test_overlapping_dag_candidates_share_state_transition_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(
                state_dir,
                [
                    {"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"},
                    {"batch_id": "B0001", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"},
                ],
            )
            _write_summary(state_dir)
            calls = 0

            def counting_run_opt(opt, src, passes, out, timeout):
                nonlocal calls
                calls += 1
                return _commuting_run_opt(opt, src, passes, out, timeout)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=counting_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertLess(calls, 16)
        self.assertGreater(int(rows[1]["validation_state_transition_cache_hits"]), 0)
        self.assertLess(
            int(rows[1]["validation_opt_invocations"]),
            int(rows[1]["validation_dag_edges"]),
        )

    def test_simple_commuting_batch_is_certified_and_dumped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}])
            _write_summary(state_dir)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_commuting_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                    dump_validation_dag=True,
                )
            correctness = classify_batch_correctness(state_dir)
            rows = _read_csv(state_dir / "batch_validation.csv")
            dag_summary = _read_csv(state_dir / "batch_validation_dag_summary.csv")
            nodes_dump_exists = (state_dir / "artifacts" / "batch_validation" / "B0000" / "validation_dag_B0000_nodes.csv").exists()
            edges_dump_exists = (state_dir / "artifacts" / "batch_validation" / "B0000" / "validation_dag_B0000_edges.csv").exists()
            dot_dump_exists = (state_dir / "artifacts" / "batch_validation" / "B0000" / "validation_dag_B0000.dot").exists()

        self.assertEqual(rows[0]["validation_status"], "all_permutations_same")
        self.assertEqual(rows[0]["validation_tier"], "permutation_dag_exact")
        self.assertEqual(rows[0]["validation_complete"], "true")
        self.assertEqual(rows[0]["validation_hard_certificate"], "true")
        self.assertEqual(rows[0]["validation_dag_final_classes"], "1")
        self.assertEqual(rows[0]["factorial_permutations"], "6")
        self.assertEqual(rows[0]["validation_dag_budget_exceeded"], "false")
        self.assertEqual(correctness[0]["correctness_class"], "certified_batch")
        self.assertEqual(correctness[0]["can_hard_fold"], "true")
        self.assertEqual(correctness[0]["can_execute"], "true")
        self.assertEqual(dag_summary[0]["validation_tier"], "permutation_dag_exact")
        self.assertTrue(nodes_dump_exists)
        self.assertTrue(edges_dump_exists)
        self.assertTrue(dot_dump_exists)

    def test_mismatch_batch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}])
            _write_summary(state_dir)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_order_sensitive_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            correctness = classify_batch_correctness(state_dir)
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "mismatch")
        self.assertEqual(rows[0]["validation_tier"], "permutation_dag_mismatch")
        self.assertEqual(rows[0]["validation_complete"], "true")
        self.assertGreater(int(rows[0]["validation_dag_final_classes"]), 1)
        self.assertEqual(correctness[0]["correctness_class"], "rejected_batch")
        self.assertEqual(correctness[0]["can_execute"], "false")

    def test_intermediate_split_can_finally_merge_and_certify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}])
            _write_summary(state_dir)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_split_then_final_merge_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "all_permutations_same")
        self.assertEqual(rows[0]["validation_tier"], "permutation_dag_exact")
        self.assertEqual(rows[0]["validation_dag_final_classes"], "1")
        self.assertGreater(int(rows[0]["validation_dag_nodes"]), 8)

    def test_budget_exceeded_marks_incomplete_and_unvalidated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}])
            _write_summary(state_dir)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_commuting_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=2,
                    max_validation_dag_edges=20000,
                )
            correctness = classify_batch_correctness(state_dir)
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "incomplete")
        self.assertEqual(rows[0]["validation_tier"], "permutation_dag_incomplete")
        self.assertEqual(rows[0]["validation_dag_budget_exceeded"], "true")
        self.assertEqual(correctness[0]["correctness_class"], "unvalidated_batch")
        self.assertEqual(correctness[0]["can_hard_fold"], "false")
        self.assertEqual(correctness[0]["can_execute"], "false")

    def test_transition_cache_records_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}])
            _write_summary(state_dir)

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_noop_a_b_run_opt):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertGreater(int(rows[0]["validation_dag_transition_cache_hits"]), 0)
        self.assertGreater(int(rows[0]["validation_dag_transition_cache_misses"]), 0)

    def test_equivalence_cache_records_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B;C", "batch_size": "3", "canonical_order": "A;B;C"}])
            _write_summary(state_dir)
            different = EqualityResult(
                equal=False,
                tier="different",
                can_hard_fold=False,
                reason="llvm_diff_difference",
                text_hash_equal=False,
            )

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_repeated_hash_pair_run_opt), \
                mock.patch("phasebatch.batch_validation_dag.compare_ir_equivalence", return_value=different):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt", "llvm-diff": "llvm-diff"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertGreater(int(rows[0]["validation_dag_equivalence_cache_hits"]), 0)
        self.assertGreater(int(rows[0]["validation_dag_equivalence_cache_misses"]), 0)

    def test_structural_merge_certifies_hash_different_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            _write_input(state_dir)
            _write_candidates(state_dir, [{"batch_id": "B0000", "batch_passes": "A;B", "batch_size": "2", "canonical_order": "A;B"}])
            _write_summary(state_dir)
            structural = EqualityResult(
                equal=True,
                tier="structural_diff",
                can_hard_fold=True,
                reason="llvm_diff_equal_and_module_fingerprint_equal",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=True,
            )

            with mock.patch("phasebatch.batch_validation_dag.run_opt", side_effect=_order_sensitive_run_opt), \
                mock.patch("phasebatch.batch_validation_dag.compare_ir_equivalence", return_value=structural):
                validate_batch_candidates(
                    state_dir,
                    {"opt": "opt", "llvm-diff": "llvm-diff"},
                    timeout=1,
                    jobs=1,
                    batch_validation_mode="dag",
                    max_validation_dag_nodes=5000,
                    max_validation_dag_edges=20000,
                )
            rows = _read_csv(state_dir / "batch_validation.csv")

        self.assertEqual(rows[0]["validation_status"], "all_permutations_same")
        self.assertEqual(rows[0]["validation_tier"], "permutation_dag_exact")
        self.assertGreater(int(rows[0]["validation_dag_structural_merges"]), 0)
        self.assertEqual(rows[0]["validation_equality_tier"], "structural_diff")


def _write_input(state_dir: Path) -> None:
    (state_dir / "input.ll").write_text("root\n", encoding="utf-8")


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


def _commuting_run_opt(opt, src, passes, out, timeout):
    tokens = sorted(set(_tokens(Path(src).read_text(encoding="utf-8")) + [passes[0]]))
    _write_ir(out, "|".join(tokens))
    return RunResult([opt], 0, "", "", 1.0)


def _order_sensitive_run_opt(opt, src, passes, out, timeout):
    tokens = _tokens(Path(src).read_text(encoding="utf-8")) + [passes[0]]
    _write_ir(out, "|".join(tokens))
    return RunResult([opt], 0, "", "", 1.0)


def _split_then_final_merge_run_opt(opt, src, passes, out, timeout):
    tokens = _tokens(Path(src).read_text(encoding="utf-8")) + [passes[0]]
    if "C" in tokens:
        _write_ir(out, "FINAL:" + "|".join(sorted(set(tokens))))
    else:
        _write_ir(out, "SEQ:" + "|".join(tokens))
    return RunResult([opt], 0, "", "", 1.0)


def _noop_a_b_run_opt(opt, src, passes, out, timeout):
    pass_name = passes[0]
    source_text = Path(src).read_text(encoding="utf-8")
    if pass_name in {"A", "B"}:
        _write_ir(out, source_text.strip())
    else:
        tokens = sorted(set(_tokens(source_text) + [pass_name]))
        _write_ir(out, "|".join(tokens))
    return RunResult([opt], 0, "", "", 1.0)


def _repeated_hash_pair_run_opt(opt, src, passes, out, timeout):
    tokens = _tokens(Path(src).read_text(encoding="utf-8")) + [passes[0]]
    if len(tokens) == 2:
        _write_ir(out, "PAIR_LEFT" if tokens == sorted(tokens) else "PAIR_RIGHT")
    else:
        _write_ir(out, "|".join(tokens))
    return RunResult([opt], 0, "", "", 1.0)


def _tokens(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped or stripped == "root":
        return []
    if stripped.startswith("FINAL:") or stripped.startswith("SEQ:"):
        stripped = stripped.split(":", 1)[1]
    return [part for part in stripped.split("|") if part]


def _write_ir(path: Path, text: str) -> None:
    Path(path).write_text((text or "root") + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
