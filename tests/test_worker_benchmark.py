import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.schema import RunResult
from phasebatch.worker_benchmark import benchmark_opt_worker


class WorkerBenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_percentiles_speedup_cache_and_acceptance(self) -> None:
        def execute(backend, _input, _pipeline, output, _timeout):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("define void @f() { ret void }\n", encoding="utf-8")
            return RunResult(
                [backend],
                0,
                "",
                "",
                10.0 if backend == "external" else 2.0,
                output_path=output,
                backend=backend,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            result = benchmark_opt_worker(
                input_ll,
                root / "out",
                iterations=4,
                workloads={
                    "no_op": "verify",
                    "single": "function(instcombine)",
                    "pair": "mem2reg,sroa",
                    "validation_shaped": "mem2reg,sroa,instcombine",
                },
                execute=execute,
                worker_stats={
                    "module_loads": 1,
                    "module_load_cache_hits": 15,
                    "starts": 1,
                    "restarts": 0,
                },
            )
            summaries = _read_csv(Path(result["worker_benchmark_summary_csv"]))
            samples = _read_csv(Path(result["worker_benchmark_samples_csv"]))
            artifacts_exists = (root / "out" / "artifacts").exists()

        total = next(row for row in summaries if row["workload"] == "all_file_compatible")
        self.assertEqual(len(samples), 32)
        self.assertEqual(total["external_median_ms"], "10.000")
        self.assertEqual(total["worker_median_ms"], "2.000")
        self.assertEqual(total["speedup"], "5.000")
        self.assertEqual(total["cache_hit_rate"], "0.937500")
        self.assertEqual(total["acceptance_status"], "passed")
        self.assertEqual(result["acceptance_status"], "passed")
        self.assertFalse(artifacts_exists)

    def test_benchmark_fails_acceptance_below_three_x_or_on_errors(self) -> None:
        def execute(backend, _input, pipeline, output, _timeout):
            success = not (backend == "worker" and pipeline == "bad")
            if success:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("define void @f() { ret void }\n", encoding="utf-8")
            return RunResult(
                [backend],
                0 if success else 1,
                "",
                "failure" if not success else "",
                4.0 if backend == "external" else 2.0,
                output_path=output,
                backend=backend,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            result = benchmark_opt_worker(
                input_ll,
                root / "out",
                iterations=2,
                workloads={"single": "bad"},
                execute=execute,
            )

        self.assertEqual(result["acceptance_status"], "failed")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
