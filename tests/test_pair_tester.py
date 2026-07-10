import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.ir_equivalence import EqualityResult
from phasebatch.pair_cache import PairRelationCache
from phasebatch.pass_config import PassRegistry, PassSpec
from phasebatch.pair_tester import run_pair_tests
from phasebatch.schema import RunResult


class PairTesterTests(unittest.TestCase):
    def test_worker_equal_hash_pair_avoids_both_materializations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "state_hash": "s", "pass": name, "active": "true"}
                for name in ("a", "b")
            ]
            results = [
                RunResult(
                    ["worker"],
                    0,
                    "",
                    "",
                    1.0,
                    backend="worker",
                    worker_id=0,
                    worker_generation=1,
                    module_handle=f"h{index}",
                    canonical_hash="same-full-ir-hash",
                    feature_counts={"instructions": 7},
                    materialized=False,
                )
                for index in range(2)
            ]

            with mock.patch("phasebatch.pair_tester.worker_handles_enabled", return_value=True), \
                mock.patch("phasebatch.pair_tester.run_opt", side_effect=results) as fake_run, \
                mock.patch("phasebatch.pair_tester.materialize_run_result") as fake_materialize, \
                mock.patch("phasebatch.pair_tester.compare_ir_equivalence") as fake_compare:
                rows = run_pair_tests(
                    input_ll,
                    profiles,
                    {"opt": "opt"},
                    root / "pairs",
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                )

        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_commute")
        self.assertEqual(rows[0]["equality_tier"], "canonical_hash")
        self.assertEqual(rows[0]["ab_inst"], 7)
        self.assertEqual(rows[0]["pair_materializations"], "0")
        self.assertEqual(rows[0]["pair_materializations_avoided"], "2")
        self.assertEqual(rows[0]["worker_hash_fast_path"], "true")
        self.assertTrue(all(call.kwargs["materialize"] is False for call in fake_run.call_args_list))
        fake_materialize.assert_not_called()
        fake_compare.assert_not_called()

    def test_worker_hash_difference_materializes_before_structural_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "state_hash": "s", "pass": name, "active": "true"}
                for name in ("a", "b")
            ]
            results = [
                RunResult(
                    ["worker"],
                    0,
                    "",
                    "",
                    1.0,
                    backend="worker",
                    worker_id=0,
                    worker_generation=1,
                    module_handle=f"h{index}",
                    canonical_hash=f"hash-{index}",
                    feature_counts={"instructions": 3},
                    materialized=False,
                )
                for index in range(2)
            ]
            structural = EqualityResult(
                equal=True,
                tier="structural_diff",
                can_hard_fold=True,
                reason="llvm_diff_equal_and_module_fingerprint_equal",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=True,
                left_hash="safe-a",
                right_hash="safe-b",
            )

            def materialize(result, path, *, timeout):
                del timeout
                Path(path).write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
                result.materialized = True
                return Path(path)

            with mock.patch("phasebatch.pair_tester.worker_handles_enabled", return_value=True), \
                mock.patch("phasebatch.pair_tester.run_opt", side_effect=results), \
                mock.patch("phasebatch.pair_tester.materialize_run_result", side_effect=materialize) as fake_materialize, \
                mock.patch("phasebatch.pair_tester.compare_ir_equivalence", return_value=structural):
                rows = run_pair_tests(
                    input_ll,
                    profiles,
                    {"opt": "opt", "llvm-diff": "llvm-diff"},
                    root / "pairs",
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                )

        self.assertEqual(rows[0]["equality_tier"], "structural_diff")
        self.assertEqual(rows[0]["pair_materializations"], "2")
        self.assertEqual(rows[0]["pair_materializations_avoided"], "0")
        self.assertEqual(rows[0]["worker_hash_fast_path"], "false")
        self.assertEqual(fake_materialize.call_count, 2)

    def test_lazy_pair_testing_budget_marks_remaining_pairs_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "inst_delta": "-10", "blocks_changed": "3", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "inst_delta": "-7", "blocks_changed": "2", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "c", "active": "true", "inst_delta": "1", "blocks_changed": "1", "changed_functions": "g", "changed_blocks": "g::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "d", "active": "true", "inst_delta": "0", "blocks_changed": "0", "changed_functions": "h", "changed_blocks": "h::entry"},
            ]
            call_count = 0

            def fake_run_opt(opt, src, passes, out, timeout):
                nonlocal call_count
                call_count += 1
                out.write_text(input_ll.read_text(encoding="utf-8"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                rows = run_pair_tests(
                    input_ll,
                    profiles,
                    {"opt": "opt"},
                    root,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    pair_testing_mode="lazy",
                    pair_test_budget_per_state=2,
                    pair_priority_policy="effect-size",
                )

        tested = [row for row in rows if row["dynamic_relation"] != "not_tested"]
        skipped = [row for row in rows if row["dynamic_relation"] == "not_tested"]
        self.assertEqual(call_count, 4)
        self.assertEqual(len(tested), 2)
        self.assertEqual(len(skipped), 4)
        self.assertEqual((rows[0]["pass_a"], rows[0]["pass_b"]), ("a", "b"))
        self.assertTrue(all(row["pair_testing_mode"] == "lazy" for row in rows))
        self.assertTrue(all(row["skipped_by_budget"] == "true" for row in skipped))
        self.assertTrue(all(row["final_relation"] == "final_unknown" for row in skipped))
        self.assertTrue(all(row["failure_kind"] == "lazy_budget" for row in skipped))
        self.assertTrue(all(row["can_hard_fold"] == "false" for row in skipped))
        self.assertTrue(all(row["pair_priority_score"] for row in rows))
        self.assertIn("effect", rows[0]["pair_priority_reason"])

    def test_lazy_pair_priority_sorting_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "c", "active": "true", "inst_delta": "-2", "blocks_changed": "1", "changed_functions": "c", "changed_blocks": "c::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "inst_delta": "-9", "blocks_changed": "1", "changed_functions": "a", "changed_blocks": "a::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "inst_delta": "-9", "blocks_changed": "1", "changed_functions": "b", "changed_blocks": "b::entry"},
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text(input_ll.read_text(encoding="utf-8"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                first = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root / "first", jobs=1, timeout=1, max_pairs=None, pair_testing_mode="lazy", pair_test_budget_per_state=1, pair_priority_policy="mixed")
                second = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root / "second", jobs=1, timeout=1, max_pairs=None, pair_testing_mode="lazy", pair_test_budget_per_state=1, pair_priority_policy="mixed")

        self.assertEqual([(row["pass_a"], row["pass_b"]) for row in first], [(row["pass_a"], row["pass_b"]) for row in second])
        self.assertEqual((first[0]["pass_a"], first[0]["pass_b"]), ("a", "b"))

    def test_run_pair_tests_reuses_cached_relation_for_same_state_and_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "same-state", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "same-state", "pass": "b", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
            ]
            cache = PairRelationCache(llvm_version="LLVM test", target_triple="x86_64-test")
            call_count = 0

            def fake_run_opt(opt, src, passes, out, timeout):
                nonlocal call_count
                call_count += 1
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 3.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                first = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root / "first", jobs=1, timeout=1, max_pairs=None, pair_cache=cache)
                second = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root / "second", jobs=1, timeout=1, max_pairs=None, pair_cache=cache)

        self.assertEqual(call_count, 2)
        self.assertEqual(first[0]["cache_hit"], "false")
        self.assertEqual(first[0]["pair_test_opt_runs"], "2")
        self.assertEqual(first[0]["pair_test_pass_invocations_baseline"], "4")
        self.assertEqual(first[0]["pair_test_pass_invocations_actual"], "4")
        self.assertEqual(first[0]["pair_test_pass_invocations_saved"], "0")
        self.assertEqual(second[0]["cache_hit"], "true")
        self.assertEqual(second[0]["pair_test_opt_runs"], "0")
        self.assertEqual(second[0]["avoided_opt_runs"], "2")
        self.assertEqual(second[0]["pair_test_pass_invocations_baseline"], "4")
        self.assertEqual(second[0]["pair_test_pass_invocations_actual"], "0")
        self.assertEqual(second[0]["pair_test_pass_invocations_saved"], "4")
        self.assertEqual(second[0]["equality_tier"], first[0]["equality_tier"])
        self.assertEqual(second[0]["can_hard_fold"], first[0]["can_hard_fold"])

    def test_run_pair_tests_does_not_reuse_cache_across_state_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles_a = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "state-a", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "state-a", "pass": "b", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
            ]
            profiles_b = [dict(row, state_hash="state-b", state_id="S0001") for row in profiles_a]
            cache = PairRelationCache(llvm_version="LLVM test", target_triple="x86_64-test")
            call_count = 0

            def fake_run_opt(opt, src, passes, out, timeout):
                nonlocal call_count
                call_count += 1
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                first = run_pair_tests(input_ll, profiles_a, {"opt": "opt"}, root / "first", jobs=1, timeout=1, max_pairs=None, pair_cache=cache)
                second = run_pair_tests(input_ll, profiles_b, {"opt": "opt"}, root / "second", jobs=1, timeout=1, max_pairs=None, pair_cache=cache)

        self.assertEqual(call_count, 4)
        self.assertEqual(first[0]["cache_hit"], "false")
        self.assertEqual(second[0]["cache_hit"], "false")
        self.assertNotEqual(first[0]["cache_key"], second[0]["cache_key"])

    def test_run_pair_tests_does_not_reuse_cache_when_pipeline_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "same-state", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "same-state", "pass": "b", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
            ]
            first_registry = PassRegistry.from_specs(
                [
                    PassSpec("a", "a-v1", ["a-v1"], "scalar", "v1", True),
                    PassSpec("b", "b", ["b"], "scalar", "v1", True),
                ]
            )
            second_registry = PassRegistry.from_specs(
                [
                    PassSpec("a", "a-v2", ["a-v2"], "scalar", "v1", True),
                    PassSpec("b", "b", ["b"], "scalar", "v1", True),
                ]
            )
            cache = PairRelationCache(llvm_version="LLVM test", target_triple="x86_64-test")
            call_count = 0

            def fake_run_opt(opt, src, passes, out, timeout):
                nonlocal call_count
                call_count += 1
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                first = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root / "first", jobs=1, timeout=1, max_pairs=None, pass_registry=first_registry, pair_cache=cache)
                second = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root / "second", jobs=1, timeout=1, max_pairs=None, pass_registry=second_registry, pair_cache=cache)

        self.assertEqual(call_count, 4)
        self.assertEqual(first[0]["cache_hit"], "false")
        self.assertEqual(second[0]["cache_hit"], "false")
        self.assertNotEqual(first[0]["cache_key"], second[0]["cache_key"])

    def test_run_pair_tests_reuses_single_pass_outputs_for_second_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            a_output = root / "a.ll"
            b_output = root / "b.ll"
            a_output.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            b_output.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry", "output_path": str(a_output)},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry", "output_path": str(b_output)},
            ]
            seen: list[tuple[Path, list[str]]] = []

            def fake_run_opt(opt, src, passes, out, timeout):
                seen.append((Path(src), list(passes)))
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root / "pairs", jobs=1, timeout=1, max_pairs=None)

        self.assertEqual(seen, [(a_output, ["b"]), (b_output, ["a"])])
        self.assertEqual(rows[0]["reused_single_pass_outputs"], "true")
        self.assertEqual(rows[0]["full_pipeline_runs_avoided"], "2")
        self.assertEqual(rows[0]["second_stage_runs"], "2")
        self.assertEqual(rows[0]["pair_test_pass_invocations_baseline"], "4")
        self.assertEqual(rows[0]["pair_test_pass_invocations_actual"], "2")
        self.assertEqual(rows[0]["pair_test_pass_invocations_saved"], "2")

    def test_run_pair_tests_classifies_equal_hashes_as_commute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {
                    "program": "x",
                    "state_id": "S0001",
                    "depth": 1,
                    "parent_state_id": "S0000",
                    "transition_pass": "mem2reg",
                    "state_hash": "s",
                    "pass": "a",
                    "active": "true",
                    "changed_functions": "f",
                    "changed_blocks": "f::entry",
                },
                {
                    "program": "x",
                    "state_id": "S0001",
                    "depth": 1,
                    "parent_state_id": "S0000",
                    "transition_pass": "mem2reg",
                    "state_hash": "s",
                    "pass": "b",
                    "active": "true",
                    "changed_functions": "f",
                    "changed_blocks": "f::entry",
                },
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 3.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root, jobs=1, timeout=1, max_pairs=None)

        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_commute")
        self.assertEqual(rows[0]["final_relation"], "final_commute")
        self.assertEqual(rows[0]["same_hash"], "true")
        self.assertEqual(rows[0]["state_id"], "S0001")
        self.assertEqual(rows[0]["depth"], 1)
        self.assertEqual(rows[0]["parent_state_id"], "S0000")
        self.assertEqual(rows[0]["transition_pass"], "mem2reg")

    def test_run_pair_tests_uses_structural_fallback_when_hashes_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f(i32 %x) {\n  ret i32 %x\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                text = "define i32 @f(i32 %x) {\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n"
                if passes == ["b", "a"]:
                    text = "define i32 @f(i32 %x) {\n  %b = add i32 %x, 0\n  ret i32 %b\n}\n"
                out.write_text(text, encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            structural = EqualityResult(
                equal=True,
                tier="structural_diff",
                can_hard_fold=True,
                reason="llvm_diff_equal_and_module_fingerprint_equal",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=True,
                left_hash="hash-ab",
                right_hash="hash-ba",
            )

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.pair_tester.compare_ir_equivalence", return_value=structural):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt", "llvm-diff": "llvm-diff"}, root, jobs=1, timeout=1, max_pairs=None)

        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_commute")
        self.assertEqual(rows[0]["final_relation"], "final_commute")
        self.assertEqual(rows[0]["same_hash"], "false")
        self.assertEqual(rows[0]["text_hash_equal"], "false")
        self.assertEqual(rows[0]["llvm_diff_equal"], "true")
        self.assertEqual(rows[0]["module_fingerprint_equal"], "true")
        self.assertEqual(rows[0]["equality_tier"], "structural_diff")
        self.assertEqual(rows[0]["equality_reason"], "llvm_diff_equal_and_module_fingerprint_equal")
        self.assertEqual(rows[0]["can_hard_fold"], "true")

    def test_run_pair_tests_marks_comparator_failure_as_dynamic_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            failed = EqualityResult(
                equal=False,
                tier="failed",
                can_hard_fold=False,
                reason="tool_failed",
                text_hash_equal=False,
                left_hash="hash-ab",
                right_hash="hash-ba",
                error_message="llvm-diff not found",
            )

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.pair_tester.compare_ir_equivalence", return_value=failed):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root, jobs=1, timeout=1, max_pairs=None)

        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_failed")
        self.assertEqual(rows[0]["final_relation"], "final_unknown")
        self.assertEqual(rows[0]["same_hash"], "false")
        self.assertEqual(rows[0]["equality_tier"], "failed")
        self.assertEqual(rows[0]["equality_reason"], "tool_failed")
        self.assertEqual(rows[0]["can_hard_fold"], "false")
        self.assertEqual(rows[0]["failure_kind"], "tool_failed")

    def test_run_pair_tests_marks_successful_difference_as_order_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            different = EqualityResult(
                equal=False,
                tier="different",
                can_hard_fold=False,
                reason="module_fingerprint_difference",
                text_hash_equal=False,
                llvm_diff_equal=True,
                module_fingerprint_equal=False,
                left_hash="hash-ab",
                right_hash="hash-ba",
            )

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt), \
                mock.patch("phasebatch.pair_tester.compare_ir_equivalence", return_value=different):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt", "llvm-diff": "llvm-diff"}, root, jobs=1, timeout=1, max_pairs=None)

        self.assertEqual(rows[0]["dynamic_relation"], "dynamic_order_sensitive")
        self.assertEqual(rows[0]["final_relation"], "final_order_sensitive")
        self.assertEqual(rows[0]["equality_tier"], "different")
        self.assertEqual(rows[0]["equality_reason"], "module_fingerprint_difference")
        self.assertEqual(rows[0]["can_hard_fold"], "false")
        self.assertEqual(rows[0]["failure_kind"], "")

    def test_run_pair_tests_records_not_tested_when_max_pairs_caps_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "a", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "b", "active": "true", "changed_functions": "g", "changed_blocks": "g::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "c", "active": "true", "changed_functions": "h", "changed_blocks": "h::entry"},
            ]

            def fake_run_opt(opt, src, passes, out, timeout):
                out.write_text(input_ll.read_text(encoding="utf-8"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                rows = run_pair_tests(input_ll, profiles, {"opt": "opt"}, root, jobs=1, timeout=1, max_pairs=1)

        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(1 for row in rows if row["dynamic_relation"] == "not_tested"), 2)
        self.assertTrue(all(row["state_id"] == "S0000" for row in rows))

    def test_run_pair_tests_executes_pipeline_text_but_records_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            profiles = [
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "licm", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
                {"program": "x", "state_id": "S0000", "depth": 0, "parent_state_id": "", "transition_pass": "", "state_hash": "s", "pass": "dce", "active": "true", "changed_functions": "f", "changed_blocks": "f::entry"},
            ]
            registry = PassRegistry.from_specs(
                [
                    PassSpec("licm", "function(loop(licm))", ["function(loop(licm))"], "loop", "v3", True),
                    PassSpec("dce", "dce", ["dce"], "cleanup", "v1", True),
                ]
            )
            seen_orders: list[list[str]] = []

            def fake_run_opt(opt, src, passes, out, timeout):
                seen_orders.append(passes)
                out.write_text(input_ll.read_text(encoding="utf-8"), encoding="utf-8")
                return RunResult([opt], 0, "", "", 1.0)

            with mock.patch("phasebatch.pair_tester.run_opt", side_effect=fake_run_opt):
                rows = run_pair_tests(
                    input_ll,
                    profiles,
                    {"opt": "opt"},
                    root,
                    jobs=1,
                    timeout=1,
                    max_pairs=None,
                    pass_registry=registry,
                )

        self.assertEqual(seen_orders, [["dce", "function(loop(licm))"], ["function(loop(licm))", "dce"]])
        self.assertEqual(rows[0]["pass_a"], "dce")
        self.assertEqual(rows[0]["pass_b"], "licm")
