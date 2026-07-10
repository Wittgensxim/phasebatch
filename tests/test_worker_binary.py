import os
import tempfile
import unittest
from pathlib import Path

from phasebatch.opt_worker import WorkerProcess


def _worker_binary() -> Path:
    candidates = [
        Path(os.environ.get("PHASEBATCH_OPT_WORKER", "")),
        Path("worker/build/phasebatch-worker.exe"),
        Path("worker/build/phasebatch-worker"),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise AssertionError("phasebatch-worker binary has not been built")


class WorkerBinaryTests(unittest.TestCase):
    def test_ping_reports_protocol_and_llvm_version(self) -> None:
        worker = WorkerProcess([_worker_binary()], worker_id=0)
        try:
            reply = worker.request("ping", timeout=5)
        finally:
            worker.close()

        self.assertEqual(reply.payload["status"], "ok")
        self.assertEqual(reply.payload["protocol_version"], 1)
        self.assertTrue(reply.payload["llvm_version"])

    def test_load_apply_without_materializing_then_materialize(self) -> None:
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
            worker = WorkerProcess([_worker_binary()], worker_id=0)
            try:
                loaded = worker.request("load", timeout=5, path=str(input_ll.resolve()))
                applied = worker.request(
                    "apply",
                    timeout=5,
                    parent_handle=loaded.payload["module_handle"],
                    pipeline="function(instcombine)",
                    verify_each=True,
                )
                self.assertFalse(output_ll.exists())
                materialized = worker.request(
                    "materialize",
                    timeout=5,
                    module_handle=applied.payload["module_handle"],
                    path=str(output_ll.resolve()),
                )
            finally:
                worker.close()
            output_text = output_ll.read_text(encoding="utf-8")

        self.assertEqual(loaded.payload["status"], "ok")
        self.assertEqual(applied.payload["status"], "ok")
        self.assertEqual(materialized.payload["status"], "ok")
        self.assertIn("ret i32 %x", output_text)
        self.assertGreaterEqual(int(applied.payload["features"]["instructions"]), 1)
        self.assertTrue(applied.payload["canonical_hash"])

    def test_repeated_named_type_loads_and_clear_have_stable_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_ll = Path(tmp) / "named.ll"
            input_ll.write_text(
                "%named = type { i32 }\n"
                "@value = global %named zeroinitializer\n",
                encoding="utf-8",
            )
            worker = WorkerProcess([_worker_binary()], worker_id=0)
            try:
                first = worker.request("load", timeout=5, path=str(input_ll.resolve())).payload
                second = worker.request("load", timeout=5, path=str(input_ll.resolve())).payload
                worker.request("clear", timeout=5)
                third = worker.request("load", timeout=5, path=str(input_ll.resolve())).payload
            finally:
                worker.close()

        self.assertEqual(first["canonical_hash"], second["canonical_hash"])
        self.assertEqual(first["canonical_hash"], third["canonical_hash"])
        self.assertEqual(first["module_handle"], second["module_handle"])
        self.assertEqual(first["module_handle"], third["module_handle"])

    def test_invalid_pipeline_is_error_and_worker_stays_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_ll = Path(tmp) / "input.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            worker = WorkerProcess([_worker_binary()], worker_id=0)
            try:
                loaded = worker.request("load", timeout=5, path=str(input_ll.resolve()))
                invalid = worker.request(
                    "apply",
                    timeout=5,
                    parent_handle=loaded.payload["module_handle"],
                    pipeline="definitely-not-a-pass",
                    verify_each=True,
                )
                ping = worker.request("ping", timeout=5)
            finally:
                worker.close()

        self.assertEqual(invalid.payload["status"], "error")
        self.assertEqual(invalid.payload["error_kind"], "invalid_pipeline")
        self.assertIn("definitely-not-a-pass", invalid.payload["error_message"])
        self.assertEqual(ping.payload["status"], "ok")

    def test_apply_materialization_failure_does_not_leak_child_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_ll = root / "input.ll"
            blocked_output = root / "blocked.ll"
            after_release = root / "after-release.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            blocked_output.mkdir()
            worker = WorkerProcess([_worker_binary()], worker_id=0)
            try:
                loaded = worker.request("load", timeout=5, path=str(input_ll.resolve())).payload
                failed = worker.request(
                    "apply",
                    timeout=5,
                    parent_handle=loaded["module_handle"],
                    pipeline="",
                    verify_each=True,
                    materialize_path=str(blocked_output.resolve()),
                ).payload
                worker.request(
                    "release",
                    timeout=5,
                    module_handle=loaded["module_handle"],
                )
                missing = worker.request(
                    "materialize",
                    timeout=5,
                    module_handle=loaded["module_handle"],
                    path=str(after_release.resolve()),
                ).payload
            finally:
                worker.close()

        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error_kind"], "materialize_failed")
        self.assertEqual(missing["status"], "error")
        self.assertEqual(missing["error_kind"], "unknown_handle")

    def test_release_removes_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_ll = Path(tmp) / "input.ll"
            output_ll = Path(tmp) / "released.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            worker = WorkerProcess([_worker_binary()], worker_id=0)
            try:
                loaded = worker.request("load", timeout=5, path=str(input_ll.resolve()))
                released = worker.request(
                    "release",
                    timeout=5,
                    module_handle=loaded.payload["module_handle"],
                )
                missing = worker.request(
                    "materialize",
                    timeout=5,
                    module_handle=loaded.payload["module_handle"],
                    path=str(output_ll.resolve()),
                )
            finally:
                worker.close()

        self.assertEqual(released.payload["status"], "ok")
        self.assertEqual(missing.payload["status"], "error")
        self.assertEqual(missing.payload["error_kind"], "unknown_handle")

    def test_duplicate_handles_are_reference_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_ll = Path(tmp) / "input.ll"
            output_ll = Path(tmp) / "retained.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            worker = WorkerProcess([str(_worker_binary())], worker_id=0)
            try:
                first = worker.request("load", timeout=5, path=str(input_ll)).payload
                second = worker.request("load", timeout=5, path=str(input_ll)).payload
                self.assertEqual(first["module_handle"], second["module_handle"])
                released_once = worker.request(
                    "release",
                    timeout=5,
                    module_handle=first["module_handle"],
                ).payload
                retained = worker.request(
                    "materialize",
                    timeout=5,
                    module_handle=second["module_handle"],
                    path=str(output_ll),
                ).payload
                worker.request("release", timeout=5, module_handle=second["module_handle"])
                missing = worker.request(
                    "materialize",
                    timeout=5,
                    module_handle=second["module_handle"],
                    path=str(output_ll),
                ).payload
            finally:
                worker.close()

        self.assertEqual(released_once["remaining_references"], 1)
        self.assertEqual(retained["status"], "ok")
        self.assertEqual(missing["error_kind"], "unknown_handle")

    def test_retain_adds_an_explicit_handle_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_ll = Path(tmp) / "input.ll"
            output_ll = Path(tmp) / "retained.ll"
            input_ll.write_text("define void @f() { ret void }\n", encoding="utf-8")
            worker = WorkerProcess([str(_worker_binary())], worker_id=0)
            try:
                loaded = worker.request("load", timeout=5, path=str(input_ll)).payload
                retained = worker.request(
                    "retain",
                    timeout=5,
                    module_handle=loaded["module_handle"],
                ).payload
                worker.request("release", timeout=5, module_handle=loaded["module_handle"])
                materialized = worker.request(
                    "materialize",
                    timeout=5,
                    module_handle=loaded["module_handle"],
                    path=str(output_ll),
                ).payload
                worker.request("release", timeout=5, module_handle=loaded["module_handle"])
            finally:
                worker.close()

        self.assertEqual(retained["references"], 2)
        self.assertEqual(materialized["status"], "ok")

    def test_compare_paths_uses_llvm_structural_diff_in_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.ll"
            renamed = root / "renamed.ll"
            different = root / "different.ll"
            left.write_text(
                "define i32 @f(i32 %x) {\n  %left = add i32 %x, 1\n  ret i32 %left\n}\n",
                encoding="utf-8",
            )
            renamed.write_text(
                "define i32 @f(i32 %x) {\n  %right = add i32 %x, 1\n  ret i32 %right\n}\n",
                encoding="utf-8",
            )
            different.write_text(
                "define i32 @f(i32 %x) {\n  %right = add i32 %x, 2\n  ret i32 %right\n}\n",
                encoding="utf-8",
            )
            worker = WorkerProcess([str(_worker_binary())], worker_id=0)
            try:
                equal = worker.request(
                    "compare_paths",
                    timeout=5,
                    left_path=str(left),
                    right_path=str(renamed),
                ).payload
                unequal = worker.request(
                    "compare_paths",
                    timeout=5,
                    left_path=str(left),
                    right_path=str(different),
                ).payload
            finally:
                worker.close()

        self.assertEqual(equal["status"], "ok")
        self.assertTrue(equal["structural_equal"])
        self.assertFalse(unequal["structural_equal"])
        self.assertGreaterEqual(equal["compare_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
