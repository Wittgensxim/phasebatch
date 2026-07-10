import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import phasebatch.opt_backend as opt_backend_module
from phasebatch.opt_backend import WorkerOptBackend, active_opt_backend, opt_backend_session
from phasebatch.opt_worker import WorkerError, WorkerProtocolError, WorkerReply
from phasebatch.runner import run_opt


def _worker_binary() -> Path:
    configured = os.environ.get("PHASEBATCH_OPT_WORKER")
    candidates = [
        Path(configured) if configured else None,
        Path("worker/build/phasebatch-worker.exe"),
        Path("worker/build/phasebatch-worker"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise AssertionError("phasebatch-worker binary has not been built")


class OptBackendTests(unittest.TestCase):
    def test_strict_worker_runtime_error_is_not_wrapped_as_pass_failure(self) -> None:
        backend = mock.Mock()
        backend.fallback_external = False
        backend.run_opt.side_effect = WorkerError("worker protocol failed")

        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch("phasebatch.opt_backend.active_opt_backend", return_value=backend):
            root = Path(tmp)
            with self.assertRaisesRegex(WorkerError, "worker protocol failed"):
                run_opt("opt", root / "input.ll", ["instcombine"], root / "output.ll", 5)

    def test_evicted_borrowed_parent_falls_back_to_materialized_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text(
                "define i32 @f(i32 %x) {\n  %sum = add i32 %x, 0\n  ret i32 %sum\n}\n",
                encoding="utf-8",
            )
            backend = WorkerOptBackend(
                _worker_binary(),
                workers=1,
                max_cached_paths_per_worker=1,
            )
            try:
                parent = backend.run_opt(
                    input_ll,
                    "function(instcombine)",
                    root / "parent.ll",
                    5,
                )
                backend.run_opt(
                    input_ll,
                    "function(simplifycfg)",
                    root / "other.ll",
                    5,
                )
                child = backend.run_opt_from_result(
                    parent,
                    "function(simplifycfg)",
                    root / "child.ll",
                    5,
                )
            finally:
                backend.close()

        self.assertFalse(parent.module_handle_owned)
        self.assertTrue(child.success)
        self.assertNotEqual(child.failure_kind, "unknown_handle")

    def test_materialized_cache_signature_failure_rolls_back_child_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            output_ll = root / "output.ll"
            input_ll.write_text(
                "define i32 @f(i32 %x) {\n  %sum = add i32 %x, 0\n  ret i32 %sum\n}\n",
                encoding="utf-8",
            )
            input_signature = opt_backend_module._path_signature(input_ll.resolve())
            backend = WorkerOptBackend(_worker_binary(), workers=1)
            try:
                with mock.patch(
                    "phasebatch.opt_backend._path_signature",
                    side_effect=[input_signature, OSError("signature failed")],
                ):
                    with self.assertRaisesRegex(OSError, "signature failed"):
                        backend.run_opt(
                            input_ll,
                            "function(instcombine)",
                            output_ll,
                            5,
                        )
                with backend.pool.checkout(timeout=5) as worker:
                    module_count = worker.request("ping", timeout=5).payload["module_count"]
            finally:
                backend.close()

        self.assertEqual(module_count, 1)

    def test_lazy_restart_marks_owned_handle_stale_on_first_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            backend = WorkerOptBackend(_worker_binary(), workers=1)
            try:
                result = backend.run_opt(
                    input_ll,
                    "function(instcombine)",
                    root / "output.ll",
                    5,
                    materialize=False,
                )
                worker = backend.pool._workers[result.worker_id]
                worker._restart()
                released = backend.release_result(result, timeout=5)
            finally:
                backend.close()

        self.assertFalse(released)
        self.assertFalse(result.module_handle_owned)

    def test_silent_worker_exit_retries_once_with_fresh_input_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            output_ll = root / "output.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            backend = WorkerOptBackend(_worker_binary(), workers=1)
            worker = backend.pool._workers[0]
            calls = []

            def fake_request(op, *, timeout, **payload):
                del timeout
                calls.append(op)
                if op == "load":
                    return WorkerReply(
                        worker.worker_id,
                        {"status": "ok", "module_handle": "root", "canonical_hash": "root"},
                    )
                if op == "apply" and calls.count("apply") == 1:
                    worker.generation += 1
                    raise WorkerProtocolError("worker exited before sending a response")
                if op == "apply":
                    Path(payload["materialize_path"]).write_text(
                        "define void @f() { ret void }\n",
                        encoding="utf-8",
                    )
                    return WorkerReply(
                        worker.worker_id,
                        {
                            "status": "ok",
                            "module_handle": "child",
                            "canonical_hash": "child",
                            "features": {},
                        },
                    )
                raise AssertionError(op)

            try:
                with mock.patch.object(worker, "request", side_effect=fake_request):
                    result = backend.run_opt(
                        input_ll,
                        "function(instcombine)",
                        output_ll,
                        5,
                    )
                stats = dict(backend.stats)
            finally:
                backend.close()

        self.assertTrue(result.success)
        self.assertEqual(calls, ["load", "apply", "load", "apply"])
        self.assertEqual(stats["silent_worker_retries"], 1)

    def test_fatal_worker_restart_reloads_input_before_next_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text(
                "define void @f(ptr %p, i64 %n) {\n"
                "entry:\n  br label %loop\n"
                "loop:\n"
                "  %i = phi i64 [ 0, %entry ], [ %next, %loop ]\n"
                "  %v = load i32, ptr %p\n"
                "  %next = add i64 %i, 1\n"
                "  %done = icmp eq i64 %next, %n\n"
                "  br i1 %done, label %exit, label %loop\n"
                "exit:\n  ret void\n}\n",
                encoding="utf-8",
            )
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                backend = active_opt_backend()
                fatal = backend.run_opt(input_ll, "loop-rotate,licm", root / "fatal.ll", 5)
                invalid = backend.run_opt(
                    input_ll,
                    "phasebatch-invalid-pass",
                    root / "invalid.ll",
                    5,
                )
                stats = dict(backend.stats)

        self.assertEqual(fatal.failure_kind, "llvm_fatal")
        self.assertEqual(invalid.failure_kind, "invalid_pipeline")
        self.assertEqual(stats["module_loads"], 2)
        self.assertEqual(stats["restarts"], 1)
        self.assertEqual(stats["llvm_fatal_failures"], 1)

    def test_materialized_result_handle_is_borrowed_from_bounded_path_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                result = run_opt(
                    "unused-opt",
                    input_ll,
                    ["function(instcombine)"],
                    root / "output.ll",
                    5,
                )
                stats = dict(active_opt_backend().stats)

        self.assertTrue(result.module_handle)
        self.assertFalse(result.module_handle_owned)
        self.assertEqual(stats["handle_retains"], 0)

    def test_deferred_result_owns_handle_and_release_is_single_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                backend = active_opt_backend()
                result = backend.run_opt(
                    input_ll,
                    "function(instcombine)",
                    root / "output.ll",
                    5,
                    materialize=False,
                )
                first_release = backend.release_result(result, timeout=5)
                second_release = backend.release_result(result, timeout=5)

        self.assertTrue(result.module_handle)
        self.assertFalse(result.module_handle_owned)
        self.assertTrue(first_release)
        self.assertFalse(second_release)

    def test_worker_backend_preserves_run_opt_file_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            output_ll = root / "output.ll"
            input_ll.write_text(
                "define i32 @f(i32 %x) {\n"
                "entry:\n"
                "  %sum = add i32 %x, 0\n"
                "  ret i32 %sum\n"
                "}\n",
                encoding="utf-8",
            )
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                result = run_opt("unused-opt", input_ll, ["function(instcombine)"], output_ll, 5)
                backend = active_opt_backend()
                stats = dict(backend.stats)

            output_text = output_ll.read_text(encoding="utf-8")

        self.assertTrue(result.success)
        self.assertEqual(result.backend, "worker")
        self.assertEqual(result.worker_id, 0)
        self.assertTrue(result.module_handle)
        self.assertTrue(result.canonical_hash)
        self.assertTrue(result.materialized)
        self.assertIn("ret i32 %x", output_text)
        self.assertEqual(stats["module_loads"], 1)
        self.assertEqual(stats["module_clones"], 1)
        self.assertGreaterEqual(stats["parse_ms"], 0.0)
        self.assertGreaterEqual(stats["clone_ms"], 0.0)
        self.assertGreaterEqual(stats["pipeline_parse_ms"], 0.0)
        self.assertGreaterEqual(stats["pass_ms"], 0.0)
        self.assertGreater(stats["round_trip_ms"], 0.0)

    def test_worker_backend_reuses_loaded_input_on_same_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                first = run_opt("unused-opt", input_ll, ["function(instcombine)"], root / "one.ll", 5)
                second = run_opt("unused-opt", input_ll, ["function(simplifycfg)"], root / "two.ll", 5)
                stats = dict(active_opt_backend().stats)

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(stats["module_loads"], 1)
        self.assertEqual(stats["module_load_cache_hits"], 1)

    def test_path_cache_reloads_same_size_file_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            first_output = root / "first.ll"
            second_output = root / "second.ll"
            first_ir = "define i32 @f() { ret i32 1 }\n"
            second_ir = "define i32 @f() { ret i32 2 }\n"
            self.assertEqual(len(first_ir), len(second_ir))
            fixed_timestamp = 1_700_000_000_000_000_000
            input_ll.write_text(first_ir, encoding="utf-8")
            os.utime(input_ll, ns=(fixed_timestamp, fixed_timestamp))

            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                first = run_opt("unused-opt", input_ll, [], first_output, 5)
                input_ll.write_text(second_ir, encoding="utf-8")
                os.utime(input_ll, ns=(fixed_timestamp, fixed_timestamp))
                second = run_opt("unused-opt", input_ll, [], second_output, 5)
                stats = dict(active_opt_backend().stats)

            second_text = second_output.read_text(encoding="utf-8")

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertIn("ret i32 2", second_text)
        self.assertEqual(stats["module_loads"], 2)
        self.assertEqual(stats["module_load_cache_hits"], 0)

    def test_invalid_pipeline_returns_normal_run_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            output_ll = root / "output.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                result = run_opt("unused-opt", input_ll, ["not-a-real-pass"], output_ll, 5)

        self.assertFalse(result.success)
        self.assertEqual(result.backend, "worker")
        self.assertEqual(result.failure_kind, "invalid_pipeline")
        self.assertIn("not-a-real-pass", result.stderr)
        self.assertFalse(output_ll.exists())

    def test_llvm_fatal_protocol_exit_is_reported_as_pipeline_failure(self) -> None:
        backend = WorkerOptBackend(_worker_binary(), workers=1)
        diagnostic = (
            "LLVM ERROR: Instruction Combining on fannkuch did not reach a fixpoint after 1 iterations.\n"
            "1. Running pass function(instcombine) on module input.ll"
        )
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                backend,
                "_run_opt_once",
                side_effect=WorkerProtocolError(
                    f"worker exited before sending a response: {diagnostic}",
                    diagnostic=diagnostic,
                ),
            ):
                root = Path(tmp)
                result = backend.run_opt(root / "input.ll", "instcombine", root / "output.ll", 5)
                stats = dict(backend.stats)
        finally:
            backend.close()

        self.assertFalse(result.success)
        self.assertEqual(result.failure_kind, "llvm_fatal")
        self.assertEqual(result.backend, "worker")
        self.assertIn("did not reach a fixpoint", result.stderr)
        self.assertEqual(stats["llvm_fatal_failures"], 1)
        self.assertEqual(stats["backend_failures"], 0)

    def test_non_materialized_result_can_be_materialized_on_handle_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            output_ll = root / "deferred" / "output.ll"
            input_ll.write_text(
                "define i32 @f(i32 %x) {\n  %sum = add i32 %x, 0\n  ret i32 %sum\n}\n",
                encoding="utf-8",
            )
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=2):
                backend = active_opt_backend()
                result = backend.run_opt(
                    input_ll,
                    "function(instcombine)",
                    output_ll,
                    5,
                    materialize=False,
                )
                self.assertFalse(output_ll.exists())
                backend.materialize_result(result, output_ll, timeout=5)
                stats = dict(backend.stats)

            output_text = output_ll.read_text(encoding="utf-8")

        self.assertTrue(result.materialized)
        self.assertEqual(result.output_path, output_ll.resolve())
        self.assertIn("ret i32 %x", output_text)
        self.assertEqual(stats["avoided_materializations"], 1)
        self.assertEqual(stats["materializations"], 1)

    def test_child_pipeline_reuses_parent_handle_without_reloading_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text(
                "define i32 @f(i32 %x) {\n  %sum = add i32 %x, 0\n  ret i32 %sum\n}\n",
                encoding="utf-8",
            )
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=2):
                backend = active_opt_backend()
                parent = backend.run_opt(
                    input_ll,
                    "function(instcombine)",
                    root / "parent.ll",
                    5,
                    materialize=False,
                )
                child = backend.run_opt_from_result(
                    parent,
                    "function(simplifycfg)",
                    root / "child.ll",
                    5,
                    materialize=False,
                )
                stats = dict(backend.stats)

        self.assertTrue(child.success)
        self.assertEqual(child.worker_id, parent.worker_id)
        self.assertFalse(child.materialized)
        self.assertEqual(stats["module_loads"], 1)
        self.assertEqual(stats["module_applies"], 2)

    def test_materialized_path_cache_uses_borrowed_result_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            output_ll = root / "output.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
                backend = active_opt_backend()
                result = backend.run_opt(input_ll, "function(instcombine)", output_ll, 5)
                self.assertFalse(backend.release_result(result, timeout=5))
                child = backend.run_opt(output_ll, "function(simplifycfg)", root / "child.ll", 5)
                stats = dict(backend.stats)

        self.assertTrue(child.success)
        self.assertEqual(stats["module_loads"], 1)
        self.assertEqual(stats["handle_retains"], 0)

    def test_path_cache_lru_evicts_and_releases_old_handles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            backend = WorkerOptBackend(
                _worker_binary(),
                workers=1,
                max_cached_paths_per_worker=2,
            )
            try:
                first = backend.run_opt(input_ll, "function(instcombine)", root / "first.ll", 5)
                second = backend.run_opt(input_ll, "function(simplifycfg)", root / "second.ll", 5)
                stats = dict(backend.stats)
            finally:
                backend.close()

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertGreaterEqual(stats["path_cache_evictions"], 1)
        self.assertGreaterEqual(stats["cache_reference_releases"], 1)


    def test_strict_worker_mode_fails_when_binary_is_missing(self) -> None:
        with mock.patch.object(opt_backend_module, "resolve_worker_path", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "phasebatch-worker not found"):
                opt_backend_module.configure_opt_backend("worker", workers=1)

        self.assertIsNone(active_opt_backend())

    def test_backend_session_clears_active_backend(self) -> None:
        self.assertIsNone(active_opt_backend())
        with opt_backend_session("worker", worker_path=_worker_binary(), workers=1):
            self.assertIsNotNone(active_opt_backend())
        self.assertIsNone(active_opt_backend())


if __name__ == "__main__":
    unittest.main()
