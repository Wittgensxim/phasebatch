import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

from phasebatch.ir_equivalence import EqualityResult
from phasebatch.validation_runtime import (
    ValidationRuntime,
    ValidationTransition,
    ValidationTransitionKey,
)


class ValidationRuntimeTests(unittest.TestCase):
    def test_opt_slots_never_exceed_worker_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=2)
            active = 0
            peak = 0
            lock = threading.Lock()
            two_active = threading.Event()

            def operation() -> str:
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                    if active == 2:
                        two_active.set()
                self.assertTrue(two_active.wait(timeout=1.0))
                with lock:
                    active -= 1
                return "ok"

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda _: runtime.run_with_opt_slot(operation),
                        range(4),
                    )
                )
            snapshot = runtime.snapshot()

        self.assertEqual(results, ["ok"] * 4)
        self.assertEqual(peak, 2)
        self.assertEqual(snapshot.opt_slot_executions, 4)

    def test_nonpositive_worker_budget_clamps_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for configured_workers in (0, -3):
                with self.subTest(configured_workers=configured_workers):
                    runtime = ValidationRuntime(
                        Path(tmp),
                        max_workers=configured_workers,
                    )
                    self.assertEqual(runtime.max_workers, 1)
                    self.assertEqual(runtime.run_with_opt_slot(lambda: "ok"), "ok")

    def test_transition_values_are_immutable(self) -> None:
        key = ValidationTransitionKey("source", "instcombine", "pipeline")
        same_key = ValidationTransitionKey("source", "instcombine", "pipeline")
        transition = ValidationTransition(Path("out.ll"), "result", "computed")

        self.assertEqual({key: "cached"}[same_key], "cached")
        with self.assertRaises(FrozenInstanceError):
            setattr(key, "source_hash", "changed")
        with self.assertRaises(FrozenInstanceError):
            setattr(transition, "source", "changed")

    def test_transition_single_flight_computes_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=4)
            key = ValidationTransitionKey("source", "instcombine", "pipeline")
            callers_ready = threading.Barrier(4)
            compute_calls = 0
            lock = threading.Lock()

            def compute(output_path: Path) -> ValidationTransition:
                nonlocal compute_calls
                with lock:
                    compute_calls += 1
                time.sleep(0.05)
                output_path.write_text("result", encoding="utf-8")
                return ValidationTransition(output_path, "result-hash", "computed")

            def request_transition() -> ValidationTransition:
                callers_ready.wait(timeout=1.0)
                return runtime.get_or_compute_transition(key, compute)

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: request_transition(), range(4)))

        self.assertEqual(compute_calls, 1)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertEqual(results[0].canonical_hash, "result-hash")

    def test_transition_cache_path_is_deterministic_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            key = ValidationTransitionKey("source", "instcombine", "pipeline")
            other_key = ValidationTransitionKey("source", "simplifycfg", "pipeline")

            def capture_path(
                runtime: ValidationRuntime,
                transition_key: ValidationTransitionKey,
            ) -> Path:
                captured: list[Path] = []

                def compute(output_path: Path) -> ValidationTransition:
                    captured.append(output_path)
                    output_path.write_text("result", encoding="utf-8")
                    return ValidationTransition(output_path, "hash", "computed")

                runtime.get_or_compute_transition(transition_key, compute)
                return captured[0]

            first_path = capture_path(ValidationRuntime(state_dir, 1), key)
            repeated_path = capture_path(ValidationRuntime(state_dir, 1), key)
            other_path = capture_path(ValidationRuntime(state_dir, 1), other_key)

        expected_parent = state_dir / "artifacts" / "validation_cache"
        self.assertEqual(first_path.parent, expected_parent)
        self.assertEqual(first_path, repeated_path)
        self.assertNotEqual(first_path, other_path)
        self.assertEqual(first_path.suffix, ".ll")
        self.assertEqual(len(first_path.stem), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in first_path.stem))

    def test_write_keep_marker_covers_validation_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=1)

            marker = runtime.write_keep_marker()

            self.assertEqual(marker, runtime.cache_dir / ".keep_ir_artifacts")
            self.assertTrue(marker.is_file())
            self.assertEqual(runtime.write_keep_marker(), marker)

    def test_transition_exception_wakes_waiters_and_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=2)
            key = ValidationTransitionKey("source", "instcombine", "pipeline")
            compute_started = threading.Event()
            release_failure = threading.Event()
            errors: list[BaseException] = []
            lock = threading.Lock()
            compute_calls = 0

            def failing_compute(output_path: Path) -> ValidationTransition:
                del output_path
                nonlocal compute_calls
                with lock:
                    compute_calls += 1
                compute_started.set()
                self.assertTrue(release_failure.wait(timeout=1.0))
                raise RuntimeError("transition failed")

            def request_transition() -> None:
                try:
                    runtime.get_or_compute_transition(key, failing_compute)
                except BaseException as exc:
                    with lock:
                        errors.append(exc)

            owner = threading.Thread(target=request_transition, daemon=True)
            waiter = threading.Thread(target=request_transition, daemon=True)
            owner.start()
            self.assertTrue(compute_started.wait(timeout=1.0))
            waiter.start()
            time.sleep(0.05)
            release_failure.set()
            owner.join(timeout=1.0)
            waiter.join(timeout=1.0)

            self.assertFalse(owner.is_alive())
            self.assertFalse(waiter.is_alive())
            self.assertEqual(len(errors), 2)
            self.assertTrue(
                all(str(error) == "transition failed" for error in errors)
            )

            def successful_compute(output_path: Path) -> ValidationTransition:
                nonlocal compute_calls
                with lock:
                    compute_calls += 1
                output_path.write_text("retry", encoding="utf-8")
                return ValidationTransition(output_path, "retry-hash", "computed")

            retried = runtime.get_or_compute_transition(key, successful_compute)

        self.assertEqual(compute_calls, 2)
        self.assertEqual(retried.canonical_hash, "retry-hash")

    def test_equivalence_single_flight_normalizes_pair_and_computes_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=4)
            forward_key = ("hash-a", "hash-b", "comparator-v1")
            reverse_key = ("hash-b", "hash-a", "comparator-v1")
            callers_ready = threading.Barrier(4)
            compare_calls = 0
            lock = threading.Lock()

            def compare() -> EqualityResult:
                nonlocal compare_calls
                with lock:
                    compare_calls += 1
                time.sleep(0.05)
                return EqualityResult(
                    equal=True,
                    tier="canonical_hash",
                    can_hard_fold=True,
                    reason="hash_equal",
                )

            def request_equivalence(index: int) -> EqualityResult:
                callers_ready.wait(timeout=1.0)
                key = forward_key if index % 2 == 0 else reverse_key
                return runtime.get_or_compute_equivalence(key, compare)

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(request_equivalence, range(4)))

        self.assertEqual(compare_calls, 1)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertTrue(results[0].can_hard_fold)

    def test_equivalence_exception_wakes_waiters_and_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=2)
            forward_key = ("hash-a", "hash-b", "comparator-v1")
            reverse_key = ("hash-b", "hash-a", "comparator-v1")
            compare_started = threading.Event()
            release_failure = threading.Event()
            errors: list[BaseException] = []
            lock = threading.Lock()
            compare_calls = 0

            def failing_compare() -> EqualityResult:
                nonlocal compare_calls
                with lock:
                    compare_calls += 1
                compare_started.set()
                self.assertTrue(release_failure.wait(timeout=1.0))
                raise RuntimeError("comparison failed")

            def request_equivalence(key: tuple[str, str, str]) -> None:
                try:
                    runtime.get_or_compute_equivalence(key, failing_compare)
                except BaseException as exc:
                    with lock:
                        errors.append(exc)

            owner = threading.Thread(
                target=request_equivalence,
                args=(forward_key,),
                daemon=True,
            )
            waiter = threading.Thread(
                target=request_equivalence,
                args=(reverse_key,),
                daemon=True,
            )
            owner.start()
            self.assertTrue(compare_started.wait(timeout=1.0))
            waiter.start()
            time.sleep(0.05)
            release_failure.set()
            owner.join(timeout=1.0)
            waiter.join(timeout=1.0)

            self.assertFalse(owner.is_alive())
            self.assertFalse(waiter.is_alive())
            self.assertEqual(len(errors), 2)
            self.assertTrue(
                all(str(error) == "comparison failed" for error in errors)
            )

            def successful_compare() -> EqualityResult:
                nonlocal compare_calls
                with lock:
                    compare_calls += 1
                return EqualityResult(
                    equal=False,
                    tier="different",
                    can_hard_fold=False,
                    reason="different",
                )

            retried = runtime.get_or_compute_equivalence(
                forward_key,
                successful_compare,
            )

        self.assertEqual(compare_calls, 2)
        self.assertEqual(retried.tier, "different")

    def test_failed_equivalence_result_is_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=1)
            key = ("hash-a", "hash-b", "comparator-v1")
            compare_calls = 0

            def failed_compare() -> EqualityResult:
                nonlocal compare_calls
                compare_calls += 1
                return EqualityResult(
                    equal=False,
                    tier="failed",
                    can_hard_fold=False,
                    reason="tool_failed",
                )

            first = runtime.get_or_compute_equivalence(key, failed_compare)

            def successful_retry() -> EqualityResult:
                nonlocal compare_calls
                compare_calls += 1
                return EqualityResult(
                    equal=False,
                    tier="different",
                    can_hard_fold=False,
                    reason="llvm_diff_difference",
                )

            second = runtime.get_or_compute_equivalence(key, successful_retry)

        self.assertEqual(first.tier, "failed")
        self.assertEqual(second.tier, "different")
        self.assertEqual(compare_calls, 2)

    def test_seed_transition_reuses_verified_profile_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=1)
            key = ValidationTransitionKey("source", "instcombine", "pipeline")
            profile_ir = Path(tmp) / "profile-instcombine.ll"
            profile_ir.write_text("profile", encoding="utf-8")
            seeded = ValidationTransition(profile_ir, "profile-hash", "profile")

            runtime.seed_transition(key, seeded)

            def unexpected_compute(output_path: Path) -> ValidationTransition:
                raise AssertionError(f"unexpected transition compute: {output_path}")

            result = runtime.get_or_compute_transition(key, unexpected_compute)

        self.assertIs(result, seeded)
        self.assertEqual(result.source, "profile")

    def test_snapshot_reports_explicit_runtime_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ValidationRuntime(Path(tmp), max_workers=2)
            runtime.run_with_opt_slot(lambda: "first")
            runtime.run_with_opt_slot(lambda: "second")

            seeded_key = ValidationTransitionKey("source", "seeded", "pipeline")
            seeded = ValidationTransition(Path(tmp) / "seeded.ll", "seed-hash", "profile")
            runtime.seed_transition(seeded_key, seeded)
            runtime.get_or_compute_transition(
                seeded_key,
                lambda path: ValidationTransition(path, "unused", "computed"),
            )

            computed_key = ValidationTransitionKey(
                "source",
                "computed",
                "pipeline",
            )

            def compute_transition(output_path: Path) -> ValidationTransition:
                output_path.write_text("computed", encoding="utf-8")
                return ValidationTransition(output_path, "computed-hash", "computed")

            runtime.get_or_compute_transition(computed_key, compute_transition)
            runtime.get_or_compute_transition(computed_key, compute_transition)

            equality = EqualityResult(
                equal=True,
                tier="canonical_hash",
                can_hard_fold=True,
                reason="hash_equal",
            )
            runtime.get_or_compute_equivalence(
                ("hash-a", "hash-b", "comparator-v1"),
                lambda: equality,
            )
            runtime.get_or_compute_equivalence(
                ("hash-b", "hash-a", "comparator-v1"),
                lambda: equality,
            )

            snapshot = runtime.snapshot()

        self.assertEqual(snapshot.opt_slot_executions, 2)
        self.assertEqual(snapshot.profile_seed_hits, 1)
        self.assertEqual(snapshot.state_transition_cache_hits, 2)
        self.assertEqual(snapshot.state_transition_cache_misses, 1)
        self.assertEqual(snapshot.state_equivalence_cache_hits, 1)
        self.assertEqual(snapshot.state_equivalence_cache_misses, 1)
