import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.ir_equivalence import EqualityResult
from phasebatch.schema import RunResult
from phasebatch.worker_differential import DifferentialCase, build_differential_cases, verify_opt_worker


class WorkerDifferentialTests(unittest.TestCase):
    def test_case_builder_includes_pipeline_candidates_and_nested_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "passes.yaml"
            config.write_text(
                "passes:\n"
                "  - name: rotate\n"
                "    pipeline_candidates:\n"
                "      - loop-rotate\n"
                "      - function(loop(loop-rotate))\n"
                "  - name: cleanup\n"
                "    pipeline: instcombine\n",
                encoding="utf-8",
            )
            cases = build_differential_cases(config)

        candidate_pipelines = [case.pipeline for case in cases if case.case_kind == "pipeline_candidate"]
        nested_pipelines = [case.pipeline for case in cases if case.case_kind == "nested_loop"]
        self.assertIn("function(loop(loop-rotate))", candidate_pipelines)
        self.assertTrue(nested_pipelines)
        self.assertIn("loop(", nested_pipelines[0])

    def test_equal_and_invalid_pipeline_parity_pass_the_gate(self) -> None:
        cases = [
            DifferentialCase("C0000", "active", "function(instcombine)"),
            DifferentialCase("C0001", "invalid", "not-a-real-pass"),
        ]

        def execute(backend, _input, pipeline, output, _timeout):
            self.assertTrue(output.parent.is_dir())
            if pipeline == "not-a-real-pass":
                return RunResult(
                    command=[backend, pipeline],
                    returncode=1,
                    stdout="",
                    stderr="unknown pass name 'not-a-real-pass'",
                    time_ms=2.0,
                    failure_kind="invalid_pipeline" if backend == "worker" else "nonzero_exit",
                    output_path=output,
                    backend=backend,
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
            return _success(backend, output)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
            result = verify_opt_worker(
                [input_ll],
                root / "out",
                cases=cases,
                execute=execute,
            )
            rows = _read_csv(Path(result["worker_differential_csv"]))
            artifacts_exists = (root / "out" / "artifacts").exists()

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["worker_default_recommended"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["external_success"], "true")
        self.assertEqual(rows[0]["worker_success"], "true")
        self.assertEqual(rows[0]["semantic_equal"], "true")
        self.assertEqual(rows[1]["failure_parity"], "true")
        self.assertEqual(rows[1]["semantic_equal"], "true")
        self.assertFalse(artifacts_exists)

    def test_canonical_hash_mismatch_fails_summary_and_recommendation(self) -> None:
        def execute(backend, _input, _pipeline, output, _timeout):
            value = "0" if backend == "external" else "1"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"define i32 @f() {{ ret i32 {value} }}\n", encoding="utf-8")
            return _success(backend, output)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
            result = verify_opt_worker(
                [input_ll],
                root / "out",
                cases=[DifferentialCase("C0000", "active", "function(instcombine)")],
                execute=execute,
            )
            rows = _read_csv(Path(result["worker_differential_csv"]))
            summary_rows = _read_csv(Path(result["worker_differential_summary_csv"]))

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["worker_default_recommended"])
        self.assertEqual(rows[0]["canonical_hash_equal"], "false")
        self.assertIn("canonical_hash_mismatch", rows[0]["mismatch_reason"])
        self.assertEqual(summary_rows[0]["failed_cases"], "1")

    def test_fingerprint_mismatch_fails_even_when_hashes_match(self) -> None:
        def execute(backend, _input, _pipeline, output, _timeout):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
            return _success(backend, output)

        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch("phasebatch.worker_differential.safe_canonical_hash", return_value="same"), \
            mock.patch(
                "phasebatch.worker_differential.module_safety_fingerprint",
                side_effect=["external-fingerprint", "worker-fingerprint"],
            ):
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
            result = verify_opt_worker(
                [input_ll],
                root / "out",
                cases=[DifferentialCase("C0000", "active", "function(instcombine)")],
                execute=execute,
            )
            rows = _read_csv(Path(result["worker_differential_csv"]))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(rows[0]["canonical_hash_equal"], "true")
        self.assertEqual(rows[0]["module_fingerprint_equal"], "false")
        self.assertIn("module_fingerprint_mismatch", rows[0]["mismatch_reason"])

    def test_structural_diff_accepts_local_name_only_hash_difference(self) -> None:
        def execute(backend, _input, _pipeline, output, _timeout):
            output.parent.mkdir(parents=True, exist_ok=True)
            local = "%left" if backend == "external" else "%right"
            output.write_text(
                f"define i32 @f(i32 %x) {{\n  {local} = add i32 %x, 1\n  ret i32 {local}\n}}\n",
                encoding="utf-8",
            )
            return _success(backend, output)

        structural = EqualityResult(
            equal=True,
            tier="structural_diff",
            can_hard_fold=True,
            reason="llvm_diff_equal_and_module_fingerprint_equal",
            text_hash_equal=False,
            llvm_diff_equal=True,
            module_fingerprint_equal=True,
        )
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch("phasebatch.worker_differential.compare_ir_equivalence", return_value=structural):
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define i32 @f(i32 %x) { ret i32 %x }\n", encoding="utf-8")
            result = verify_opt_worker(
                [input_ll],
                root / "out",
                cases=[DifferentialCase("C0000", "pair_ab", "licm,indvars")],
                execute=execute,
            )
            rows = _read_csv(Path(result["worker_differential_csv"]))

        self.assertEqual(result["status"], "passed")
        self.assertEqual(rows[0]["canonical_hash_equal"], "false")
        self.assertEqual(rows[0]["structural_equal"], "true")
        self.assertEqual(rows[0]["equality_tier"], "structural_diff")
        self.assertEqual(rows[0]["semantic_equal"], "true")

    def test_matching_llvm_fatal_failures_have_parity(self) -> None:
        def execute(backend, _input, _pipeline, output, _timeout):
            diagnostic = (
                "LLVM ERROR: LICM requires MemorySSA (loop-mssa)"
                if backend == "external"
                else "worker exited before sending a response: LLVM ERROR: LICM requires MemorySSA (loop-mssa)"
            )
            return RunResult(
                command=[backend],
                returncode=1,
                stdout="",
                stderr=diagnostic,
                time_ms=1.0,
                failure_kind="nonzero_exit" if backend == "external" else "backend_exception",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            result = verify_opt_worker(
                [input_ll],
                root / "out",
                cases=[DifferentialCase("C0000", "pipeline_candidate", "loop(licm)")],
                execute=execute,
            )
            rows = _read_csv(Path(result["worker_differential_csv"]))

        self.assertEqual(result["status"], "passed")
        self.assertEqual(rows[0]["external_failure_kind"], "llvm_fatal")
        self.assertEqual(rows[0]["worker_failure_kind"], "llvm_fatal")
        self.assertEqual(rows[0]["failure_parity"], "true")

    def test_status_mismatch_is_reported_without_stopping_later_cases(self) -> None:
        def execute(backend, _input, pipeline, output, _timeout):
            if pipeline == "bad" and backend == "worker":
                return RunResult([backend], 1, "", "worker failure", 1.0, failure_kind="worker_error")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("define void @f() { ret void }\n", encoding="utf-8")
            return _success(backend, output)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            result = verify_opt_worker(
                [input_ll],
                root / "out",
                cases=[
                    DifferentialCase("C0000", "status", "bad"),
                    DifferentialCase("C0001", "active", "good"),
                ],
                execute=execute,
            )
            rows = _read_csv(Path(result["worker_differential_csv"]))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status_equal"], "false")
        self.assertEqual(rows[1]["semantic_equal"], "true")


def _success(backend: str, output: Path) -> RunResult:
    return RunResult(
        command=[backend],
        returncode=0,
        stdout="",
        stderr="",
        time_ms=1.0,
        output_path=output,
        backend=backend,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
