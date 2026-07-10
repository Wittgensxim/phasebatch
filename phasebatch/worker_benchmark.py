from __future__ import annotations

import csv
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

from .artifact_cleanup import cleanup_ir_artifacts, mark_ir_artifacts_kept
from .opt_backend import WorkerOptBackend, resolve_worker_path
from .runner import _run_external_opt, prepare_input_ir
from .schema import RunResult
from .tools import collect_toolchain


DEFAULT_WORKLOADS = {
    "no_op": "verify",
    "single": "function(instcombine)",
    "pair": "mem2reg,sroa",
    "validation_shaped": "mem2reg,sroa,instcombine",
}

SAMPLE_FIELDS = [
    "workload",
    "iteration",
    "order",
    "backend",
    "pipeline",
    "success",
    "time_ms",
    "failure_kind",
]

SUMMARY_FIELDS = [
    "workload",
    "iterations",
    "external_successes",
    "worker_successes",
    "external_median_ms",
    "worker_median_ms",
    "external_p95_ms",
    "worker_p95_ms",
    "external_total_ms",
    "worker_total_ms",
    "speedup",
    "module_loads",
    "module_load_cache_hits",
    "cache_hit_rate",
    "worker_process_starts",
    "worker_restarts",
    "acceptance_status",
]

ExecutePipeline = Callable[[str, Path, str, Path, int], RunResult]


def benchmark_opt_worker(
    input_path: Path,
    out_dir: Path,
    *,
    worker_path: Path | str | None = None,
    workers: int = 1,
    iterations: int = 100,
    timeout: int = 30,
    workloads: dict[str, str] | None = None,
    tools: dict | None = None,
    execute: ExecutePipeline | None = None,
    worker_stats: dict | None = None,
    include_startup: bool | None = None,
    keep_ir_artifacts: bool = False,
) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    workloads = dict(workloads or DEFAULT_WORKLOADS)
    if not workloads:
        raise ValueError("at least one benchmark workload is required")

    input_path = Path(input_path).resolve()
    out_dir = Path(out_dir).resolve()
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    owned_executor: _BenchmarkExecutor | None = None
    toolchain = tools
    if execute is None:
        toolchain = toolchain or collect_toolchain()
        owned_executor = _BenchmarkExecutor(
            toolchain,
            worker_path=worker_path,
            workers=workers,
        )
        execute = owned_executor
    if include_startup is None:
        include_startup = owned_executor is not None

    prepared_input = _prepare_input(input_path, artifacts_dir / "input", toolchain, timeout)
    sample_rows: list[dict] = []
    try:
        if include_startup and owned_executor is not None:
            sample_rows.extend(owned_executor.startup_samples(iterations, timeout))
        for workload, pipeline in workloads.items():
            for iteration in range(iterations):
                order = ["external", "worker"] if iteration % 2 == 0 else ["worker", "external"]
                order_label = "external_first" if iteration % 2 == 0 else "worker_first"
                for backend in order:
                    output = artifacts_dir / workload / f"{backend}_{iteration:04d}.ll"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    result = _execute_safely(execute, backend, prepared_input, pipeline, output, timeout)
                    sample_rows.append(
                        {
                            "workload": workload,
                            "iteration": str(iteration),
                            "order": order_label,
                            "backend": backend,
                            "pipeline": pipeline,
                            "success": _bool(result.success),
                            "time_ms": f"{result.time_ms:.3f}",
                            "failure_kind": result.failure_kind,
                        }
                    )
                    if owned_executor is not None and backend == "worker":
                        owned_executor.release(result, timeout)
    finally:
        if owned_executor is not None:
            worker_stats = dict(owned_executor.stats)
            owned_executor.close()

    worker_stats = dict(worker_stats or {})
    pipeline_samples = [row for row in sample_rows if row["workload"] != "startup"]
    summaries = [
        _summary_row(
            workload,
            [row for row in pipeline_samples if row["workload"] == workload],
            worker_stats,
        )
        for workload in workloads
    ]
    total = _summary_row("all_file_compatible", pipeline_samples, worker_stats)
    summaries.append(total)

    samples_csv = out_dir / "worker_benchmark_samples.csv"
    summary_csv = out_dir / "worker_benchmark_summary.csv"
    markdown = out_dir / "worker_benchmark.md"
    _write_csv(samples_csv, SAMPLE_FIELDS, sample_rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summaries)
    _write_markdown(markdown, total, summaries, worker_stats)
    cleanup = mark_ir_artifacts_kept() if keep_ir_artifacts else cleanup_ir_artifacts(out_dir)
    return {
        "acceptance_status": total["acceptance_status"],
        "speedup": total["speedup"],
        "samples": len(sample_rows),
        "worker_benchmark_samples_csv": str(samples_csv),
        "worker_benchmark_summary_csv": str(summary_csv),
        "worker_benchmark_md": str(markdown),
        "worker_stats": worker_stats,
        **cleanup,
    }


class _BenchmarkExecutor:
    def __init__(self, toolchain: dict, *, worker_path: Path | str | None, workers: int) -> None:
        tools = _tool_paths(toolchain)
        self.opt = str(tools["opt"])
        resolved = resolve_worker_path(worker_path)
        if resolved is None:
            raise FileNotFoundError("phasebatch-worker not found")
        self.worker = WorkerOptBackend(resolved, workers=workers)

    def __call__(self, backend: str, input_ll: Path, pipeline: str, output_ll: Path, timeout: int) -> RunResult:
        if backend == "external":
            return _run_external_opt(self.opt, input_ll, pipeline, output_ll, timeout)
        if backend == "worker":
            return self.worker.run_opt(input_ll, pipeline, output_ll, timeout)
        raise ValueError(f"unknown benchmark backend: {backend}")

    def startup_samples(self, iterations: int, timeout: int) -> list[dict]:
        rows = []
        for iteration in range(iterations):
            started = time.perf_counter()
            external = subprocess.run(
                [self.opt, "--version"],
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
            rows.append(
                _startup_row(iteration, "external", (time.perf_counter() - started) * 1000, external.returncode == 0)
            )
            started = time.perf_counter()
            reply = self.worker.pool.request("ping", timeout=timeout).payload
            rows.append(
                _startup_row(iteration, "worker", (time.perf_counter() - started) * 1000, reply.get("status") == "ok")
            )
        return rows

    @property
    def stats(self) -> dict:
        return self.worker.stats

    def release(self, result: RunResult, timeout: int) -> None:
        self.worker.release_result(result, timeout=timeout)

    def close(self) -> None:
        self.worker.close()


def _prepare_input(input_path: Path, out_dir: Path, tools: dict | None, timeout: int) -> Path:
    if input_path.suffix.lower() == ".ll":
        return input_path
    if tools is None:
        raise ValueError("toolchain is required for non-IR benchmark inputs")
    return prepare_input_ir(
        input_path,
        out_dir,
        _tool_paths(tools),
        timeout,
        root_ir_mode="inlinable-unoptimized",
    )


def _execute_safely(
    execute: ExecutePipeline,
    backend: str,
    input_ll: Path,
    pipeline: str,
    output_ll: Path,
    timeout: int,
) -> RunResult:
    try:
        return execute(backend, input_ll, pipeline, output_ll, timeout)
    except Exception as exc:
        return RunResult(
            [backend, pipeline],
            1,
            "",
            str(exc),
            0.0,
            failure_kind="backend_exception",
            output_path=output_ll,
            backend=backend,
        )


def _summary_row(workload: str, rows: list[dict], stats: dict) -> dict:
    external = [float(row["time_ms"]) for row in rows if row["backend"] == "external"]
    worker = [float(row["time_ms"]) for row in rows if row["backend"] == "worker"]
    external_successes = sum(row["backend"] == "external" and row["success"] == "true" for row in rows)
    worker_successes = sum(row["backend"] == "worker" and row["success"] == "true" for row in rows)
    iterations = max(len(external), len(worker))
    external_median = statistics.median(external) if external else 0.0
    worker_median = statistics.median(worker) if worker else 0.0
    speedup = external_median / worker_median if worker_median > 0 else 0.0
    loads = _int(stats.get("module_loads"))
    hits = _int(stats.get("module_load_cache_hits"))
    cache_rate = hits / (loads + hits) if loads + hits else 0.0
    passed = (
        iterations > 0
        and external_successes == len(external)
        and worker_successes == len(worker)
        and speedup >= 3.0
    )
    return {
        "workload": workload,
        "iterations": str(iterations),
        "external_successes": str(external_successes),
        "worker_successes": str(worker_successes),
        "external_median_ms": f"{external_median:.3f}",
        "worker_median_ms": f"{worker_median:.3f}",
        "external_p95_ms": f"{_percentile(external, 0.95):.3f}",
        "worker_p95_ms": f"{_percentile(worker, 0.95):.3f}",
        "external_total_ms": f"{sum(external):.3f}",
        "worker_total_ms": f"{sum(worker):.3f}",
        "speedup": f"{speedup:.3f}",
        "module_loads": str(loads),
        "module_load_cache_hits": str(hits),
        "cache_hit_rate": f"{cache_rate:.6f}",
        "worker_process_starts": str(_int(stats.get("starts"))),
        "worker_restarts": str(_int(stats.get("restarts"))),
        "acceptance_status": "passed" if passed else "failed",
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999)))
    return ordered[index]


def _startup_row(iteration: int, backend: str, elapsed_ms: float, success: bool) -> dict:
    return {
        "workload": "startup",
        "iteration": str(iteration),
        "order": "external_then_worker",
        "backend": backend,
        "pipeline": "--version" if backend == "external" else "ping",
        "success": _bool(success),
        "time_ms": f"{elapsed_ms:.3f}",
        "failure_kind": "" if success else "startup_failed",
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, total: dict, summaries: list[dict], stats: dict) -> None:
    lines = [
        "# LLVM Worker Benchmark",
        "",
        f"- Acceptance: `{total['acceptance_status']}`",
        f"- File-compatible median speedup: `{total['speedup']}x`",
        f"- Cache hit rate: `{total['cache_hit_rate']}`",
        f"- Worker starts/restarts: `{total['worker_process_starts']}/{total['worker_restarts']}`",
        "",
        "| Workload | External median (ms) | Worker median (ms) | P95 speed context | Speedup | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['workload']} | {row['external_median_ms']} | {row['worker_median_ms']} | "
            f"{row['external_p95_ms']} / {row['worker_p95_ms']} | {row['speedup']}x | {row['acceptance_status']} |"
        )
    if stats:
        lines.extend(
            [
                "",
                "## Worker Metrics",
                "",
                f"- Module loads/cache hits: `{_int(stats.get('module_loads'))}/{_int(stats.get('module_load_cache_hits'))}`",
                f"- Applies/materializations avoided: `{_int(stats.get('module_applies'))}/{_int(stats.get('avoided_materializations'))}`",
                f"- Restarts/failures: `{_int(stats.get('restarts'))}/{_int(stats.get('backend_failures'))}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tool_paths(metadata: dict) -> dict[str, str | None]:
    tools = metadata.get("tools", metadata)
    return {
        name: value.get("path") if isinstance(value, dict) else value
        for name, value in tools.items()
    }


def _int(value: object) -> int:
    try:
        return int(float(str(value or 0)))
    except ValueError:
        return 0


def _bool(value: bool) -> str:
    return "true" if value else "false"
