from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .schema import RunResult


ROOT_IR_MODES = {"legacy-o0", "inlinable-unoptimized"}


def compile_c_to_ll(
    clang: str,
    src: Path,
    out_ll: Path,
    timeout: int,
    root_ir_mode: str = "legacy-o0",
) -> RunResult:
    if root_ir_mode not in ROOT_IR_MODES:
        raise ValueError(f"unknown root IR mode: {root_ir_mode}")
    out_ll.parent.mkdir(parents=True, exist_ok=True)
    if root_ir_mode == "inlinable-unoptimized":
        frontend_args = ["-O1", "-Xclang", "-disable-llvm-passes"]
    else:
        frontend_args = ["-O0", "-Xclang", "-disable-O0-optnone"]
    command = [clang, *frontend_args, "-S", "-emit-llvm", str(src), "-o", str(out_ll)]
    return _run_command(command, timeout=timeout, output_path=out_ll)


def prepare_input_ir(
    input_path: Path,
    out_dir: Path,
    tools: dict,
    timeout: int,
    root_ir_mode: str = "legacy-o0",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    out_ll = out_dir / "input.ll"

    suffix = input_path.suffix.lower()
    if suffix == ".ll":
        shutil.copyfile(input_path, out_ll)
        return out_ll
    if suffix == ".c":
        result = compile_c_to_ll(
            str(tools["clang"]),
            input_path,
            out_ll,
            timeout,
            root_ir_mode=root_ir_mode,
        )
        if not result.success:
            _write_stderr(out_ll, result)
            raise RuntimeError(f"clang failed for {input_path}: {result.stderr.strip()}")
        return out_ll

    raise RuntimeError(f"unsupported input type '{input_path.suffix}': {input_path}")


def run_opt(
    opt: str,
    input_ll: Path,
    passes: list[str],
    output_ll: Path,
    timeout: int,
    *,
    materialize: bool = True,
) -> RunResult:
    output_ll.parent.mkdir(parents=True, exist_ok=True)
    pass_pipeline = format_pass_pipeline(passes)
    from .opt_backend import active_opt_backend
    from .opt_worker import WorkerError

    backend = active_opt_backend()
    if backend is not None:
        try:
            result = backend.run_opt(
                input_ll,
                pass_pipeline,
                output_ll,
                timeout,
                materialize=materialize,
            )
        except WorkerError:
            if not backend.fallback_external:
                raise
            else:
                result = _run_external_opt(opt, input_ll, pass_pipeline, output_ll, timeout)
                result.backend_fallback = True
                backend._stats["fallbacks"] += 1
        if result.stderr or result.timed_out or result.returncode != 0:
            _write_stderr(output_ll, result)
        return result

    return _run_external_opt(opt, input_ll, pass_pipeline, output_ll, timeout)


def materialize_run_result(result: RunResult, output_ll: Path | None = None, *, timeout: int) -> Path:
    if result.materialized:
        path = Path(output_ll or result.output_path or "")
        if path.is_file():
            return path.resolve()
    from .opt_backend import active_opt_backend

    backend = active_opt_backend()
    if backend is None:
        raise RuntimeError("no active worker backend can materialize this result")
    return backend.materialize_result(result, output_ll, timeout=timeout)


def run_opt_from_result(
    parent: RunResult,
    passes: list[str],
    output_ll: Path,
    timeout: int,
    *,
    materialize: bool = True,
) -> RunResult:
    from .opt_backend import active_opt_backend

    backend = active_opt_backend()
    if backend is None:
        raise RuntimeError("no active worker backend can apply a parent handle")
    return backend.run_opt_from_result(
        parent,
        format_pass_pipeline(passes),
        output_ll,
        timeout,
        materialize=materialize,
    )


def release_run_result(result: RunResult, *, timeout: int) -> bool:
    from .opt_backend import active_opt_backend

    backend = active_opt_backend()
    return backend.release_result(result, timeout=timeout) if backend is not None else False


def worker_handles_enabled() -> bool:
    from .opt_backend import active_opt_backend

    return active_opt_backend() is not None


def _run_external_opt(opt: str, input_ll: Path, pass_pipeline: str, output_ll: Path, timeout: int) -> RunResult:
    command = [
        opt,
        "-S",
        "-verify-each",
        f"-passes={pass_pipeline}",
        str(input_ll),
        "-o",
        str(output_ll),
    ]
    result = _run_command(command, timeout=timeout, output_path=output_ll)
    result.backend = "external"
    if result.stderr or result.timed_out or result.returncode != 0:
        _write_stderr(output_ll, result)
    return result


def format_pass_pipeline(passes: list[str]) -> str:
    cleaned = [p for p in passes if p]
    return ",".join(cleaned)


def _run_command(command: list[str], timeout: int, output_path: Path | None = None) -> RunResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        elapsed = (time.perf_counter() - start) * 1000
        failure_kind = "" if completed.returncode == 0 else "nonzero_exit"
        return RunResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            time_ms=elapsed,
            failure_kind=failure_kind,
            output_path=output_path,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = (time.perf_counter() - start) * 1000
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr) or f"timeout after {timeout}s"
        return RunResult(
            command=command,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            time_ms=elapsed,
            timed_out=True,
            failure_kind="timeout",
            output_path=output_path,
        )


def _write_stderr(output_ll: Path, result: RunResult) -> None:
    stderr_path = output_ll.with_suffix(output_ll.suffix + ".stderr.txt")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    result.stderr_path = stderr_path


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
