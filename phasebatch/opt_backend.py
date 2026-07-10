from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .opt_worker import WorkerError, WorkerPool, WorkerProcess, WorkerProtocolError
from .schema import RunResult


@dataclass(frozen=True)
class _PathHandle:
    signature: tuple[str, int, int, str]
    generation: int
    handle: str


class WorkerOptBackend:
    name = "worker"

    def __init__(
        self,
        worker_path: str | Path,
        *,
        workers: int,
        fallback_external: bool = False,
        max_cached_paths_per_worker: int = 256,
    ) -> None:
        path = Path(worker_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"phasebatch worker not found: {path}")
        self.worker_path = path
        self.fallback_external = fallback_external
        if max_cached_paths_per_worker < 1:
            raise ValueError("path cache capacity must be positive")
        self.max_cached_paths_per_worker = max_cached_paths_per_worker
        self.pool = WorkerPool([path], size=workers)
        self._cache_lock = threading.Lock()
        self._path_handles: list[OrderedDict[str, _PathHandle]] = [
            OrderedDict() for _ in range(workers)
        ]
        self._stats = {
            "module_loads": 0,
            "module_load_cache_hits": 0,
            "module_applies": 0,
            "module_clones": 0,
            "materializations": 0,
            "avoided_materializations": 0,
            "handle_releases": 0,
            "handle_retains": 0,
            "path_cache_evictions": 0,
            "cache_reference_releases": 0,
            "backend_failures": 0,
            "llvm_fatal_failures": 0,
            "fallbacks": 0,
            "silent_worker_retries": 0,
            "orphan_handle_rollbacks": 0,
            "orphan_handle_rollback_failures": 0,
            "structural_comparisons": 0,
            "structural_comparisons_equal": 0,
            "parse_ms": 0.0,
            "clone_ms": 0.0,
            "pipeline_parse_ms": 0.0,
            "pass_ms": 0.0,
            "verify_ms": 0.0,
            "print_ms": 0.0,
            "round_trip_ms": 0.0,
            "compare_ms": 0.0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return {**self.pool.stats, **self._stats}

    def run_opt(
        self,
        input_ll: Path,
        pass_pipeline: str,
        output_ll: Path,
        timeout: float,
        *,
        materialize: bool = True,
    ) -> RunResult:
        started = time.perf_counter()
        input_ll = Path(input_ll).resolve()
        output_ll = Path(output_ll).resolve()
        if materialize:
            output_ll.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.worker_path),
            "apply",
            f"-passes={pass_pipeline}",
            str(input_ll),
            "-o",
            str(output_ll),
        ]
        for attempt in range(2):
            try:
                return self._run_opt_once(
                    input_ll,
                    pass_pipeline,
                    output_ll,
                    timeout,
                    materialize=materialize,
                    command=command,
                    started=started,
                )
            except WorkerProtocolError as exc:
                if _is_llvm_fatal_exit(exc):
                    self._stats["llvm_fatal_failures"] += 1
                    return RunResult(
                        command=command,
                        returncode=1,
                        stdout="",
                        stderr=exc.diagnostic or str(exc),
                        time_ms=_elapsed_ms(started),
                        failure_kind="llvm_fatal",
                        output_path=output_ll,
                        backend="worker",
                        materialized=False,
                    )
                if attempt == 0 and exc.retryable:
                    self._stats["silent_worker_retries"] += 1
                    continue
                self._stats["backend_failures"] += 1
                raise
            except WorkerError:
                self._stats["backend_failures"] += 1
                raise
        raise AssertionError("unreachable worker retry state")

    def _run_opt_once(
        self,
        input_ll: Path,
        pass_pipeline: str,
        output_ll: Path,
        timeout: float,
        *,
        materialize: bool,
        command: list[str],
        started: float,
    ) -> RunResult:
        with self.pool.checkout(timeout=timeout) as worker:
            parent_handle = self._load_input(worker, input_ll, timeout)
            payload = {
                "parent_handle": parent_handle,
                "pipeline": pass_pipeline,
                "verify_each": True,
            }
            if materialize:
                payload["materialize_path"] = str(output_ll)
            request_started = time.perf_counter()
            reply = worker.request("apply", timeout=timeout, **payload).payload
            self._record_reply_metrics(reply, request_started, cloned=True)
            self._stats["module_applies"] += 1
            if reply.get("status") != "ok":
                return self._failure_result(command, output_ll, reply, started, worker.worker_id)
            if materialize and not output_ll.is_file():
                return RunResult(
                    command=command,
                    returncode=1,
                    stdout="",
                    stderr="worker reported success without materializing output",
                    time_ms=_elapsed_ms(started),
                    failure_kind="materialize_failed",
                    output_path=output_ll,
                    backend="worker",
                    worker_id=worker.worker_id,
                )
            handle = str(reply.get("module_handle") or "")
            if materialize:
                self._stats["materializations"] += 1
                self._remember_materialized_path(
                    worker,
                    output_ll,
                    handle,
                    timeout,
                )
            else:
                self._stats["avoided_materializations"] += 1
            return RunResult(
                command=command,
                returncode=0,
                stdout="",
                stderr="",
                time_ms=_elapsed_ms(started),
                output_path=output_ll,
                backend="worker",
                worker_id=worker.worker_id,
                worker_generation=worker.generation,
                module_handle=handle,
                module_handle_owned=not materialize,
                canonical_hash=str(reply.get("canonical_hash") or ""),
                feature_counts=dict(reply.get("features") or {}),
                materialized=materialize,
            )

    def materialize_result(self, result: RunResult, path: Path | None = None, *, timeout: float) -> Path:
        if result.backend != "worker" or result.worker_id is None or not result.module_handle:
            raise ValueError("result does not contain a worker-owned module handle")
        target = path if path is not None else result.output_path
        if target is None:
            raise ValueError("materialization path is required")
        output_path = Path(target)
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pool.checkout_worker(result.worker_id, timeout=timeout) as worker:
            worker.start()
            if result.worker_generation is not None and worker.generation != result.worker_generation:
                result.module_handle_owned = False
                raise WorkerError("module handle is stale after worker restart")
            request_started = time.perf_counter()
            reply = worker.request(
                "materialize",
                timeout=timeout,
                module_handle=result.module_handle,
                path=str(output_path),
            ).payload
            self._record_reply_metrics(reply, request_started)
            if reply.get("status") != "ok" or not output_path.is_file():
                self._stats["backend_failures"] += 1
                raise WorkerError(str(reply.get("error_message") or "worker failed to materialize module"))
            self._stats["materializations"] += 1
            self._remember_path(worker, output_path, result.module_handle, timeout)
        result.output_path = output_path
        result.materialized = True
        return output_path

    def run_opt_from_result(
        self,
        parent: RunResult,
        pass_pipeline: str,
        output_ll: Path,
        timeout: float,
        *,
        materialize: bool = True,
    ) -> RunResult:
        if parent.backend != "worker":
            raise ValueError("parent result does not contain a worker-owned module handle")
        if not parent.module_handle_owned:
            parent_path = Path(parent.output_path) if parent.output_path is not None else None
            if parent.materialized and parent_path is not None and parent_path.is_file():
                return self.run_opt(
                    parent_path,
                    pass_pipeline,
                    output_ll,
                    timeout,
                    materialize=materialize,
                )
            raise ValueError("borrowed parent handle is no longer backed by a materialized path")
        if parent.worker_id is None or not parent.module_handle:
            raise ValueError("parent result does not contain a worker-owned module handle")
        started = time.perf_counter()
        output_ll = Path(output_ll).resolve()
        if materialize:
            output_ll.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.worker_path),
            "apply",
            f"-passes={pass_pipeline}",
            parent.module_handle,
            "-o",
            str(output_ll),
        ]
        try:
            with self.pool.checkout_worker(parent.worker_id, timeout=timeout) as worker:
                worker.start()
                if parent.worker_generation is not None and worker.generation != parent.worker_generation:
                    parent.module_handle_owned = False
                    raise WorkerError("parent module handle is stale after worker restart")
                payload = {
                    "parent_handle": parent.module_handle,
                    "pipeline": pass_pipeline,
                    "verify_each": True,
                }
                if materialize:
                    payload["materialize_path"] = str(output_ll)
                request_started = time.perf_counter()
                reply = worker.request("apply", timeout=timeout, **payload).payload
                self._record_reply_metrics(reply, request_started, cloned=True)
                self._stats["module_applies"] += 1
                if reply.get("status") != "ok":
                    return self._failure_result(command, output_ll, reply, started, worker.worker_id)
                if materialize and not output_ll.is_file():
                    return RunResult(
                        command=command,
                        returncode=1,
                        stdout="",
                        stderr="worker reported success without materializing output",
                        time_ms=_elapsed_ms(started),
                        failure_kind="materialize_failed",
                        output_path=output_ll,
                        backend="worker",
                        worker_id=worker.worker_id,
                        worker_generation=worker.generation,
                    )
                handle = str(reply.get("module_handle") or "")
                if materialize:
                    self._stats["materializations"] += 1
                    self._remember_materialized_path(
                        worker,
                        output_ll,
                        handle,
                        timeout,
                    )
                else:
                    self._stats["avoided_materializations"] += 1
                return RunResult(
                    command=command,
                    returncode=0,
                    stdout="",
                    stderr="",
                    time_ms=_elapsed_ms(started),
                    output_path=output_ll,
                    backend="worker",
                    worker_id=worker.worker_id,
                    worker_generation=worker.generation,
                    module_handle=handle,
                    module_handle_owned=not materialize,
                    canonical_hash=str(reply.get("canonical_hash") or ""),
                    feature_counts=dict(reply.get("features") or {}),
                    materialized=materialize,
                )
        except WorkerError:
            self._stats["backend_failures"] += 1
            raise

    def release_result(self, result: RunResult, *, timeout: float) -> bool:
        if (
            result.backend != "worker"
            or result.worker_id is None
            or not result.module_handle
            or not result.module_handle_owned
        ):
            return False
        with self.pool.checkout_worker(result.worker_id, timeout=timeout) as worker:
            worker.start()
            if result.worker_generation is not None and worker.generation != result.worker_generation:
                result.module_handle_owned = False
                return False
            reply = worker.request(
                "release",
                timeout=timeout,
                module_handle=result.module_handle,
            ).payload
            if reply.get("status") != "ok":
                return False
            result.module_handle_owned = False
            self._stats["handle_releases"] += 1
            return True

    def compare_paths(self, left: Path, right: Path, *, timeout: float) -> bool:
        left = Path(left).resolve()
        right = Path(right).resolve()
        with self.pool.checkout(timeout=timeout) as worker:
            request_started = time.perf_counter()
            reply = worker.request(
                "compare_paths",
                timeout=timeout,
                left_path=str(left),
                right_path=str(right),
            ).payload
            self._record_reply_metrics(reply, request_started)
            self._stats["structural_comparisons"] += 1
            if reply.get("status") != "ok":
                self._stats["backend_failures"] += 1
                raise WorkerError(str(reply.get("error_message") or "worker structural comparison failed"))
            equal = bool(reply.get("structural_equal"))
            if equal:
                self._stats["structural_comparisons_equal"] += 1
            return equal

    def close(self) -> None:
        self.pool.close()

    def _load_input(self, worker: WorkerProcess, path: Path, timeout: float) -> str:
        worker.start()
        signature = _path_signature(path)
        with self._cache_lock:
            cache = self._path_handles[worker.worker_id]
            cached = cache.get(str(path))
            if cached is not None:
                cache.move_to_end(str(path))
        if cached is not None and cached.signature == signature and cached.generation == worker.generation:
            self._stats["module_load_cache_hits"] += 1
            return cached.handle
        request_started = time.perf_counter()
        reply = worker.request("load", timeout=timeout, path=str(path)).payload
        self._record_reply_metrics(reply, request_started)
        self._stats["module_loads"] += 1
        if reply.get("status") != "ok":
            raise WorkerError(str(reply.get("error_message") or "worker failed to load input"))
        handle = str(reply.get("module_handle") or "")
        self._remember_signature(
            worker,
            path,
            signature,
            handle,
            timeout=timeout,
            cache_reference_owned=True,
        )
        return handle

    def _remember_path(
        self,
        worker: WorkerProcess,
        path: Path,
        handle: str,
        timeout: float,
        *,
        cache_reference_owned: bool = False,
    ) -> None:
        self._remember_signature(
            worker,
            path,
            _path_signature(path),
            handle,
            timeout=timeout,
            cache_reference_owned=cache_reference_owned,
        )

    def _remember_materialized_path(
        self,
        worker: WorkerProcess,
        path: Path,
        handle: str,
        timeout: float,
    ) -> None:
        try:
            self._remember_path(
                worker,
                path,
                handle,
                timeout,
                cache_reference_owned=True,
            )
        except BaseException:
            self._rollback_untracked_handle(worker, handle, timeout)
            raise

    def _rollback_untracked_handle(
        self,
        worker: WorkerProcess,
        handle: str,
        timeout: float,
    ) -> None:
        try:
            reply = worker.request("release", timeout=timeout, module_handle=handle).payload
        except WorkerError:
            self._stats["orphan_handle_rollback_failures"] += 1
            return
        if reply.get("status") == "ok":
            self._stats["orphan_handle_rollbacks"] += 1
        else:
            self._stats["orphan_handle_rollback_failures"] += 1

    def _remember_signature(
        self,
        worker: WorkerProcess,
        path: Path,
        signature: tuple[str, int, int, str],
        handle: str,
        *,
        timeout: float,
        cache_reference_owned: bool,
    ) -> None:
        path_key = str(path)
        with self._cache_lock:
            cache = self._path_handles[worker.worker_id]
            existing = cache.get(path_key)
        if (
            existing is not None
            and existing.generation == worker.generation
            and existing.handle == handle
        ):
            if cache_reference_owned:
                self._release_cache_reference(worker, handle, timeout)
            with self._cache_lock:
                cache[path_key] = _PathHandle(signature, worker.generation, handle)
                cache.move_to_end(path_key)
            return

        if not cache_reference_owned:
            reply = worker.request("retain", timeout=timeout, module_handle=handle).payload
            if reply.get("status") != "ok":
                raise WorkerError(str(reply.get("error_message") or "worker failed to retain module"))
            self._stats["handle_retains"] += 1

        releases: list[_PathHandle] = []
        with self._cache_lock:
            cache = self._path_handles[worker.worker_id]
            replaced = cache.pop(path_key, None)
            cache[path_key] = _PathHandle(
                signature=signature,
                generation=worker.generation,
                handle=handle,
            )
            if replaced is not None:
                releases.append(replaced)
            while len(cache) > self.max_cached_paths_per_worker:
                _, evicted = cache.popitem(last=False)
                releases.append(evicted)
                self._stats["path_cache_evictions"] += 1
        for cached in releases:
            if cached.generation == worker.generation:
                self._release_cache_reference(worker, cached.handle, timeout)

    def _release_cache_reference(self, worker: WorkerProcess, handle: str, timeout: float) -> None:
        reply = worker.request("release", timeout=timeout, module_handle=handle).payload
        if reply.get("status") == "ok":
            self._stats["cache_reference_releases"] += 1

    def _record_reply_metrics(self, reply: dict, request_started: float, *, cloned: bool = False) -> None:
        if cloned:
            self._stats["module_clones"] += 1
        self._stats["round_trip_ms"] += _elapsed_ms(request_started)
        for field in (
            "parse_ms",
            "clone_ms",
            "pipeline_parse_ms",
            "pass_ms",
            "verify_ms",
            "print_ms",
            "compare_ms",
        ):
            try:
                self._stats[field] += float(reply.get(field) or 0.0)
            except (TypeError, ValueError):
                continue

    @staticmethod
    def _failure_result(
        command: list[str],
        output_ll: Path,
        reply: dict,
        started: float,
        worker_id: int,
    ) -> RunResult:
        return RunResult(
            command=command,
            returncode=1,
            stdout="",
            stderr=str(reply.get("error_message") or "worker apply failed"),
            time_ms=_elapsed_ms(started),
            failure_kind=str(reply.get("error_kind") or "worker_error"),
            output_path=output_ll,
            backend="worker",
            worker_id=worker_id,
        )


_ACTIVE_BACKEND: WorkerOptBackend | None = None
_BACKEND_CONFIG: dict | None = None
_BACKEND_LOCK = threading.RLock()


def active_opt_backend() -> WorkerOptBackend | None:
    with _BACKEND_LOCK:
        return _ACTIVE_BACKEND


def configure_opt_backend(
    mode: str,
    *,
    worker_path: str | Path | None = None,
    workers: int = 1,
) -> WorkerOptBackend | None:
    global _ACTIVE_BACKEND, _BACKEND_CONFIG
    if mode not in {"external", "worker", "auto"}:
        raise ValueError(f"unknown opt backend: {mode}")
    if workers < 1:
        raise ValueError("opt worker count must be positive")
    with _BACKEND_LOCK:
        previous = _ACTIVE_BACKEND
        _ACTIVE_BACKEND = None
        _BACKEND_CONFIG = None
        if previous is not None:
            previous.close()
        if mode == "external":
            _BACKEND_CONFIG = {
                "requested_mode": mode,
                "backend": "external",
                "worker_path": None,
                "workers": 0,
                "fallback_external": False,
            }
            return None
        resolved = resolve_worker_path(worker_path)
        if resolved is None:
            if mode == "auto":
                _BACKEND_CONFIG = {
                    "requested_mode": mode,
                    "backend": "external",
                    "worker_path": None,
                    "workers": 0,
                    "fallback_external": True,
                }
                return None
            raise FileNotFoundError("phasebatch-worker not found")
        _ACTIVE_BACKEND = WorkerOptBackend(
            resolved,
            workers=workers,
            fallback_external=(mode == "auto"),
        )
        _BACKEND_CONFIG = {
            "requested_mode": mode,
            "backend": "worker",
            "worker_path": str(resolved),
            "workers": workers,
            "fallback_external": mode == "auto",
        }
        return _ACTIVE_BACKEND


def shutdown_opt_backend() -> None:
    global _ACTIVE_BACKEND, _BACKEND_CONFIG
    with _BACKEND_LOCK:
        backend = _ACTIVE_BACKEND
        _ACTIVE_BACKEND = None
        _BACKEND_CONFIG = None
    if backend is not None:
        backend.close()


def opt_backend_metadata() -> dict | None:
    with _BACKEND_LOCK:
        if _BACKEND_CONFIG is None:
            return None
        metadata = dict(_BACKEND_CONFIG)
        if _ACTIVE_BACKEND is not None:
            metadata["stats"] = dict(_ACTIVE_BACKEND.stats)
        return metadata


@contextmanager
def opt_backend_session(
    mode: str,
    *,
    worker_path: str | Path | None = None,
    workers: int = 1,
) -> Iterator[WorkerOptBackend | None]:
    if active_opt_backend() is not None:
        raise RuntimeError("nested opt backend sessions are not supported")
    backend = configure_opt_backend(mode, worker_path=worker_path, workers=workers)
    try:
        yield backend
    finally:
        shutdown_opt_backend()


def resolve_worker_path(value: str | Path | None) -> Path | None:
    candidates = []
    if value:
        candidates.append(Path(value))
    configured = os.environ.get("PHASEBATCH_OPT_WORKER")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("worker/build/phasebatch-worker.exe"),
            Path("worker/build/phasebatch-worker"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _is_llvm_fatal_exit(error: WorkerProtocolError) -> bool:
    diagnostic = error.diagnostic.lstrip()
    return diagnostic.startswith("LLVM ERROR:")


def _path_signature(path: Path) -> tuple[str, int, int, str]:
    for _ in range(3):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if before.st_mtime_ns == after.st_mtime_ns and before.st_size == after.st_size:
            return str(path), after.st_mtime_ns, after.st_size, digest.hexdigest()
    raise OSError(f"file changed while computing worker cache signature: {path}")


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
