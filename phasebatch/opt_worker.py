from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class WorkerError(RuntimeError):
    pass


class WorkerTimeoutError(WorkerError):
    pass


class WorkerProtocolError(WorkerError):
    def __init__(self, message: str, *, diagnostic: str = "", retryable: bool | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic
        self.retryable = (
            message.startswith("worker exited before sending a response") and not diagnostic
            if retryable is None
            else retryable
        )


@dataclass(frozen=True)
class WorkerReply:
    worker_id: int
    payload: dict[str, Any]


_EOF = object()


class WorkerProcess:
    def __init__(
        self,
        command: Sequence[str | Path],
        *,
        worker_id: int,
        stderr_lines: int = 200,
    ) -> None:
        if not command:
            raise ValueError("worker command must not be empty")
        self.command = [str(part) for part in command]
        self.worker_id = worker_id
        self._stderr = deque(maxlen=max(1, stderr_lines))
        self._request_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_queue: queue.Queue[object] = queue.Queue()
        self._next_request_id = 1
        self._closed = False
        self.generation = 0
        self.stats: dict[str, int] = {
            "starts": 0,
            "restarts": 0,
            "requests": 0,
            "timeouts": 0,
            "protocol_errors": 0,
        }

    @property
    def stderr_text(self) -> str:
        return "".join(self._stderr)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise WorkerError("worker is closed")
            if self._process is not None and self._process.poll() is None:
                return
            if self._process is not None:
                self._stop_locked()
            self._stderr.clear()
            output_queue: queue.Queue[object] = queue.Queue()
            self._stdout_queue = output_queue
            try:
                process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                raise WorkerError(f"failed to start worker: {exc}") from exc
            self._process = process
            self.generation += 1
            self.stats["starts"] += 1
            stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(process, output_queue),
                daemon=True,
            )
            stderr_thread = threading.Thread(target=self._read_stderr, args=(process,), daemon=True)
            self._stdout_thread = stdout_thread
            self._stderr_thread = stderr_thread
            stdout_thread.start()
            stderr_thread.start()

    def request(self, op: str, *, timeout: float, **payload: Any) -> WorkerReply:
        if timeout <= 0:
            raise ValueError("worker timeout must be positive")
        with self._request_lock:
            self.start()
            request_id = self._next_request_id
            self._next_request_id += 1
            request = {"request_id": request_id, "op": op, **payload}
            process = self._require_process()
            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps(request, separators=(",", ":"), ensure_ascii=True) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._protocol_failure(f"worker stdin failed: {exc}")

            self.stats["requests"] += 1
            try:
                raw = self._stdout_queue.get(timeout=timeout)
            except queue.Empty as exc:
                self.stats["timeouts"] += 1
                self._restart()
                raise WorkerTimeoutError(f"worker request {request_id} timed out after {timeout}s") from exc

            if raw is _EOF:
                return self._protocol_failure("worker exited before sending a response")
            if not isinstance(raw, str):
                return self._protocol_failure("worker produced an invalid response object")
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as exc:
                return self._protocol_failure(f"worker returned malformed JSON: {raw.rstrip()}", cause=exc)
            if not isinstance(response, dict):
                return self._protocol_failure("worker response must be a JSON object")
            if response.get("request_id") != request_id:
                return self._protocol_failure(
                    f"worker response ID mismatch: expected {request_id}, got {response.get('request_id')}"
                )
            return WorkerReply(worker_id=self.worker_id, payload=response)

    def close(self) -> None:
        with self._request_lock:
            with self._lifecycle_lock:
                self._closed = True
                self._stop_locked()

    def _read_stdout(
        self,
        process: subprocess.Popen[str],
        output_queue: queue.Queue[object],
    ) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(_EOF)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            self._stderr.append(line)

    def _require_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None or process.poll() is not None:
            raise WorkerError("worker process is not running")
        return process

    def _protocol_failure(self, message: str, *, cause: Exception | None = None):
        self.stats["protocol_errors"] += 1
        detail = self._restart()
        error = WorkerProtocolError(
            f"{message}{': ' + detail if detail else ''}",
            diagnostic=detail,
        )
        if cause is not None:
            raise error from cause
        raise error

    def _restart(self) -> str:
        with self._lifecycle_lock:
            self._stop_locked()
            detail = self.stderr_text.strip()
            if not self._closed:
                self.stats["restarts"] += 1
        return detail

    def _stop_locked(self) -> None:
        process = self._process
        stdout_thread = self._stdout_thread
        stderr_thread = self._stderr_thread
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        current = threading.current_thread()
        for thread in (stdout_thread, stderr_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=1)


class WorkerPool:
    def __init__(self, command: Sequence[str | Path], *, size: int) -> None:
        if size < 1:
            raise ValueError("worker pool size must be positive")
        self._workers = [WorkerProcess(command, worker_id=index) for index in range(size)]
        self._available = deque(range(size))
        self._availability = threading.Condition()
        self._closed = False

    def request(self, op: str, *, timeout: float, **payload: Any) -> WorkerReply:
        with self.checkout(timeout=timeout) as worker:
            return worker.request(op, timeout=timeout, **payload)

    @contextmanager
    def checkout(self, *, timeout: float):
        worker_id = self._acquire_worker(None, timeout)
        try:
            yield self._workers[worker_id]
        finally:
            self._release_worker(worker_id)

    @contextmanager
    def checkout_worker(self, worker_id: int, *, timeout: float):
        if worker_id < 0 or worker_id >= len(self._workers):
            raise ValueError(f"invalid worker id: {worker_id}")
        acquired = self._acquire_worker(worker_id, timeout)
        try:
            yield self._workers[acquired]
        finally:
            self._release_worker(acquired)

    def _acquire_worker(self, requested: int | None, timeout: float) -> int:
        if timeout <= 0:
            raise ValueError("worker pool timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._availability:
            while True:
                if self._closed:
                    raise WorkerError("worker pool is closed")
                if requested is None and self._available:
                    return self._available.popleft()
                if requested is not None and requested in self._available:
                    self._available.remove(requested)
                    return requested
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    label = f"worker {requested}" if requested is not None else "a worker"
                    raise WorkerTimeoutError(f"{label} did not become available within {timeout}s")
                self._availability.wait(remaining)

    def _release_worker(self, worker_id: int) -> None:
        with self._availability:
            if not self._closed:
                self._available.append(worker_id)
            self._availability.notify_all()

    @property
    def stats(self) -> dict[str, int]:
        fields = {field for worker in self._workers for field in worker.stats}
        return {field: sum(worker.stats.get(field, 0) for worker in self._workers) for field in fields}

    def close(self) -> None:
        with self._availability:
            if self._closed:
                return
            self._closed = True
            self._availability.notify_all()
        for worker in self._workers:
            worker.close()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
