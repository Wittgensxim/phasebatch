import subprocess
import sys
import unittest
from pathlib import Path


RETAINED_COMMANDS = [
    "analyze",
    "batch",
    "explore",
    "explore-batches",
    "optimize-batches",
    "optimize-staged",
    "audit-passes",
    "batchify",
    "eval-batches",
    "compare-baselines",
    "summarize-final",
    "summarize-reduction",
    "summarize-components",
    "export-evidence-pack",
    "diagnose-paths",
    "visualize-dag",
    "replay-final-pipeline",
    "verify-opt-worker",
    "benchmark-opt-worker",
    "run-advisor-report-zh",
    "summarize-advisor-report-zh",
]

REMOVED_COMMANDS = [
    "compare-cegar-pairwise",
    "run-mainline",
    "run-method-comparison",
    "run-round-sensitivity",
    "run-reduction-study",
    "run-budgeted-sensitivity",
    "summarize-exact-reduction-study",
    "summarize-core-v1-case-study",
    "run-core-v1-budgeted-study",
    "select-and-run-exact-reference",
    "run-passset-smoke",
    "run-v2-extension-study",
    "run-v3-loop-smoke",
    "summarize-passsets",
    "summarize-mainline",
    "export-case-studies",
]


class CliBootstrapTests(unittest.TestCase):
    def test_module_help_lists_only_retained_mainline_commands(self) -> None:
        result = _run_help()

        self.assertEqual(result.returncode, 0, result.stderr)
        for command in RETAINED_COMMANDS:
            with self.subTest(command=command):
                self.assertIn(command, result.stdout)
        for command in REMOVED_COMMANDS:
            with self.subTest(command=command):
                self.assertNotIn(command, result.stdout)

    def test_retained_command_help_runs(self) -> None:
        for command in RETAINED_COMMANDS:
            with self.subTest(command=command):
                result = _run_help(command)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_advisor_report_help_exposes_reproducibility_controls(self) -> None:
        result = _run_help("run-advisor-report-zh")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--test-suite-root", result.stdout)
        self.assertIn("--benchmark-manifest", result.stdout)
        self.assertIn("--num-programs", result.stdout)
        self.assertIn("--resume", result.stdout)


def _run_help(command: str | None = None) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, "-m", "phasebatch"]
    if command:
        args.append(command)
    args.append("--help")
    return subprocess.run(
        args,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


if __name__ == "__main__":
    unittest.main()
