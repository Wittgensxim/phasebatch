from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .schema import RunResult


def compile_c_to_ll(clang: str, src: Path, out_ll: Path, timeout: int) -> RunResult:
    out_ll.parent.mkdir(parents=True, exist_ok=True)
    command = [
        clang,
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-S",
        "-emit-llvm",
        str(src),
        "-o",
        str(out_ll),
    ]
    return _run_command(command, timeout=timeout, output_path=out_ll)


def prepare_input_ir(input_path: Path, out_dir: Path, tools: dict, timeout: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    out_ll = out_dir / "input.ll"

    suffix = input_path.suffix.lower()
    if suffix == ".ll":
        shutil.copyfile(input_path, out_ll)
        return out_ll
    if suffix == ".c":
        result = compile_c_to_ll(str(tools["clang"]), input_path, out_ll, timeout)
        if not result.success:
            _write_stderr(out_ll, result)
            raise RuntimeError(f"clang failed for {input_path}: {result.stderr.strip()}")
        return out_ll

    raise RuntimeError(f"unsupported input type '{input_path.suffix}': {input_path}")


def run_opt(opt: str, input_ll: Path, passes: list[str], output_ll: Path, timeout: int) -> RunResult:
    output_ll.parent.mkdir(parents=True, exist_ok=True)
    pass_pipeline = format_pass_pipeline(passes)
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
