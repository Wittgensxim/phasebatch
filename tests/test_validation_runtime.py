import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.schema import RunResult
from phasebatch.validation_runtime import ValidationRuntime, ValidationTransition, ValidationTransitionKey


class ValidationRuntimeTests(unittest.TestCase):
    def test_close_continues_after_release_error_and_retries_owned_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=1)
            for index in range(2):
                result = RunResult(
                    ["worker"],
                    0,
                    "",
                    "",
                    1.0,
                    backend="worker",
                    worker_id=0,
                    worker_generation=1,
                    module_handle=f"h{index}",
                    materialized=False,
                )
                runtime.seed_transition(
                    ValidationTransitionKey("root", f"P{index}", f"P{index}"),
                    ValidationTransition(Path(tmp) / f"{index}.ll", f"hash{index}", "computed", result),
                )

            with mock.patch(
                "phasebatch.validation_runtime.release_run_result",
                side_effect=[RuntimeError("release failed"), True, True],
            ) as release:
                with self.assertRaisesRegex(RuntimeError, "release failed"):
                    runtime.close(timeout=3)
                retried = runtime.close(timeout=3)
                final = runtime.close(timeout=3)

        self.assertEqual(release.call_count, 3)
        self.assertEqual(retried, 1)
        self.assertEqual(final, 0)
        self.assertEqual(runtime.snapshot().released_handles, 2)

    def test_close_releases_each_cached_worker_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=1)
            for index in range(2):
                result = RunResult(
                    ["worker"],
                    0,
                    "",
                    "",
                    1.0,
                    backend="worker",
                    worker_id=0,
                    worker_generation=1,
                    module_handle=f"h{index}",
                    materialized=False,
                )
                runtime.seed_transition(
                    ValidationTransitionKey("root", f"P{index}", f"P{index}"),
                    ValidationTransition(Path(tmp) / f"{index}.ll", f"hash{index}", "computed", result),
                )
            with mock.patch("phasebatch.validation_runtime.release_run_result", return_value=True) as release:
                released = runtime.close(timeout=3)
                runtime.close(timeout=3)

        self.assertEqual(released, 2)
        self.assertEqual(release.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == 3 for call in release.call_args_list))


if __name__ == "__main__":
    unittest.main()
