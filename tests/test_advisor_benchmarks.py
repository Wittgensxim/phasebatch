import csv
import tempfile
import unittest
from pathlib import Path

from phasebatch.advisor_benchmarks import discover_advisor_benchmarks
from phasebatch.schema import RunResult


class AdvisorBenchmarkTests(unittest.TestCase):
    def test_discovery_records_every_candidate_and_selects_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "SingleSource"
            for relative in (
                "A/one.c",
                "A/two.c",
                "B/three.c",
                "C/four.c",
                "C/fail.c",
            ):
                path = single / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            (single / "A" / "ignored.cpp").write_text("int main() {}\n", encoding="utf-8")
            (single / "B" / "large.c").write_text("x" * 200, encoding="utf-8")

            def fake_compile(_clang, source, output, _timeout):
                success = source.name != "fail.c"
                if success:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text("define i32 @main() { ret i32 0 }\n", encoding="utf-8")
                return RunResult(
                    command=["clang", str(source)],
                    returncode=0 if success else 1,
                    stdout="",
                    stderr="" if success else "synthetic failure",
                    time_ms=12.5,
                    output_path=output,
                )

            first = discover_advisor_benchmarks(
                test_suite_root=root,
                out_dir=root / "out1",
                clang="clang",
                num_programs=4,
                max_source_bytes=100,
                selection_seed=7,
                timeout=3,
                compile_runner=fake_compile,
            )
            second = discover_advisor_benchmarks(
                test_suite_root=root,
                out_dir=root / "out2",
                clang="clang",
                num_programs=4,
                max_source_bytes=100,
                selection_seed=7,
                timeout=3,
                compile_runner=fake_compile,
            )

            candidates = _read_csv(root / "out1" / "benchmark_candidates.csv")
            selected = _read_csv(root / "out1" / "benchmark_selection.csv")

        self.assertEqual(len(candidates), 6)
        self.assertNotIn("ignored.cpp", {Path(row["relative_path"]).name for row in candidates})
        large = next(row for row in candidates if row["relative_path"].endswith("large.c"))
        failed = next(row for row in candidates if row["relative_path"].endswith("fail.c"))
        self.assertEqual(large["compile_status"], "skipped")
        self.assertEqual(large["skip_reason"], "source_too_large")
        self.assertEqual(failed["compile_status"], "failed")
        self.assertIn("synthetic failure", failed["error_message"])
        self.assertEqual(len(selected), 4)
        self.assertGreaterEqual(len({row["category"] for row in selected}), 3)
        self.assertEqual(first["selected"], second["selected"])
        self.assertTrue(first["selected_benchmarks_yaml"].endswith("selected_benchmarks.yaml"))

    def test_manifest_preserves_names_and_rejects_non_c_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "SingleSource" / "Bench" / "ok.c"
            source.parent.mkdir(parents=True)
            source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "benchmarks:\n"
                "  - name: chosen-name\n"
                "    path: SingleSource/Bench/ok.c\n",
                encoding="utf-8",
            )

            def fake_compile(_clang, _source, output, _timeout):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("define i32 @main() { ret i32 0 }\n", encoding="utf-8")
                return RunResult(["clang"], 0, "", "", 1.0, output_path=output)

            result = discover_advisor_benchmarks(
                test_suite_root=root,
                out_dir=root / "out",
                clang="clang",
                benchmark_manifest=manifest,
                num_programs=15,
                max_source_bytes=1000,
                selection_seed=0,
                timeout=3,
                compile_runner=fake_compile,
            )

        self.assertEqual(result["selected"], [{"name": "chosen-name", "path": str(source.resolve())}])

    def test_program_ids_are_unique_on_case_insensitive_filesystems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upper = root / "SingleSource" / "A" / "Queens.c"
            lower = root / "SingleSource" / "B" / "queens.c"
            for source in (upper, lower):
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("int main(void) { return 0; }\n", encoding="utf-8")

            def fake_compile(_clang, _source, output, _timeout):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("define i32 @main() { ret i32 0 }\n", encoding="utf-8")
                return RunResult(["clang"], 0, "", "", 1.0, output_path=output)

            result = discover_advisor_benchmarks(
                test_suite_root=root,
                out_dir=root / "out",
                clang="clang",
                num_programs=2,
                max_source_bytes=1000,
                selection_seed=0,
                timeout=3,
                compile_runner=fake_compile,
            )

        names = [row["name"] for row in result["selected"]]
        self.assertEqual(len({name.casefold() for name in names}), 2)
        self.assertEqual(names, ["Queens", "queens_2"])


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
