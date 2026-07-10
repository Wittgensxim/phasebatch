import sys
import queue
import threading
import time
import unittest

from phasebatch.opt_worker import (
    WorkerPool,
    WorkerProcess,
    WorkerProtocolError,
    WorkerTimeoutError,
)


FAKE_WORKER = r'''
import json
import sys
import time

for raw in sys.stdin:
    request = json.loads(raw)
    op = request.get("op")
    if op == "malformed":
        print("not-json", flush=True)
        continue
    if op == "exit":
        sys.exit(3)
    if op == "stderr_exit":
        print("old generation fatal", file=sys.stderr, flush=True)
        sys.exit(4)
    if op == "sleep":
        time.sleep(float(request.get("seconds", 0)))
    if op == "stderr":
        print("worker diagnostic", file=sys.stderr, flush=True)
    if op == "shutdown":
        print(json.dumps({"request_id": request["request_id"], "status": "ok"}), flush=True)
        break
    print(json.dumps({
        "request_id": request["request_id"],
        "status": "ok",
        "op": op,
        "value": request.get("value"),
    }), flush=True)
'''


def _fake_command() -> list[str]:
    return [sys.executable, "-u", "-c", FAKE_WORKER]


class WorkerProcessTests(unittest.TestCase):
    def test_protocol_failure_restarts_lazily_and_isolates_stderr_generation(self) -> None:
        worker = WorkerProcess(_fake_command(), worker_id=0)
        try:
            with self.assertRaises(WorkerProtocolError) as caught:
                worker.request("stderr_exit", timeout=2)
            self.assertIn("old generation fatal", str(caught.exception))
            self.assertIsNone(worker._process)

            reply = worker.request("ping", timeout=2)
            self.assertNotIn("old generation fatal", worker.stderr_text)
        finally:
            worker.close()

        self.assertEqual(reply.payload["status"], "ok")
        self.assertEqual(worker.stats["starts"], 2)
        self.assertEqual(worker.stats["restarts"], 1)

    def test_stdout_reader_writes_only_to_its_generation_queue(self) -> None:
        class FakeProcess:
            stdout = iter(["old-response\n"])

        worker = WorkerProcess(_fake_command(), worker_id=0)
        old_queue: queue.Queue[object] = queue.Queue()
        new_queue: queue.Queue[object] = queue.Queue()
        worker._stdout_queue = new_queue

        worker._read_stdout(FakeProcess(), old_queue)

        self.assertEqual(old_queue.get_nowait(), "old-response\n")
        self.assertIsNotNone(old_queue.get_nowait())
        self.assertTrue(new_queue.empty())

    def test_round_trips_request_ids_and_payload(self) -> None:
        worker = WorkerProcess(_fake_command(), worker_id=3)
        try:
            reply = worker.request("ping", timeout=2, value="hello")
        finally:
            worker.close()

        self.assertEqual(reply.worker_id, 3)
        self.assertEqual(reply.payload["request_id"], 1)
        self.assertEqual(reply.payload["value"], "hello")
        self.assertEqual(worker.stats["requests"], 1)

    def test_timeout_restarts_worker_and_next_request_succeeds(self) -> None:
        worker = WorkerProcess(_fake_command(), worker_id=0)
        try:
            with self.assertRaises(WorkerTimeoutError):
                worker.request("sleep", timeout=0.05, seconds=1)
            reply = worker.request("ping", timeout=2)
        finally:
            worker.close()

        self.assertEqual(reply.payload["status"], "ok")
        self.assertEqual(worker.stats["starts"], 2)
        self.assertEqual(worker.stats["restarts"], 1)
        self.assertEqual(worker.stats["timeouts"], 1)

    def test_malformed_response_restarts_worker(self) -> None:
        worker = WorkerProcess(_fake_command(), worker_id=0)
        try:
            with self.assertRaises(WorkerProtocolError):
                worker.request("malformed", timeout=2)
            reply = worker.request("ping", timeout=2)
        finally:
            worker.close()

        self.assertEqual(reply.payload["status"], "ok")
        self.assertEqual(worker.stats["restarts"], 1)

    def test_stderr_is_drained_without_corrupting_stdout_protocol(self) -> None:
        worker = WorkerProcess(_fake_command(), worker_id=0)
        try:
            reply = worker.request("stderr", timeout=2)
            deadline = time.monotonic() + 1
            while "worker diagnostic" not in worker.stderr_text and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            worker.close()

        self.assertEqual(reply.payload["status"], "ok")
        self.assertIn("worker diagnostic", worker.stderr_text)


class WorkerPoolTests(unittest.TestCase):
    def test_checkout_worker_returns_requested_handle_owner(self) -> None:
        pool = WorkerPool(_fake_command(), size=2)
        try:
            with pool.checkout_worker(1, timeout=2) as worker:
                reply = worker.request("ping", timeout=2)
                with pool.checkout(timeout=2) as other:
                    other_reply = other.request("ping", timeout=2)
        finally:
            pool.close()

        self.assertEqual(reply.worker_id, 1)
        self.assertEqual(other_reply.worker_id, 0)

    def test_checkout_keeps_a_request_sequence_on_one_worker(self) -> None:
        pool = WorkerPool(_fake_command(), size=2)
        try:
            with pool.checkout(timeout=2) as worker:
                first = worker.request("ping", timeout=2)
                second = worker.request("ping", timeout=2)
        finally:
            pool.close()

        self.assertEqual(first.worker_id, second.worker_id)

    def test_pool_runs_requests_on_multiple_workers(self) -> None:
        pool = WorkerPool(_fake_command(), size=2)
        replies = []
        lock = threading.Lock()

        def run(value: str) -> None:
            reply = pool.request("sleep", timeout=2, seconds=0.15, value=value)
            with lock:
                replies.append(reply)

        started = time.perf_counter()
        threads = [threading.Thread(target=run, args=(str(index),)) for index in range(2)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            pool.close()
        elapsed = time.perf_counter() - started

        self.assertEqual({reply.worker_id for reply in replies}, {0, 1})
        self.assertLess(elapsed, 0.28)
        self.assertEqual(pool.stats["requests"], 2)


if __name__ == "__main__":
    unittest.main()
