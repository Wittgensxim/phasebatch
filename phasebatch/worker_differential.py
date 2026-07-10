from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .artifact_cleanup import cleanup_ir_artifacts, mark_ir_artifacts_kept
from .ir_equivalence import compare_ir_equivalence, module_safety_fingerprint, safe_canonical_hash
from .opt_backend import WorkerOptBackend, resolve_worker_path
from .pass_config import load_pass_config
from .runner import _run_external_opt, prepare_input_ir
from .schema import RunResult
from .tools import collect_toolchain


@dataclass(frozen=True)
class DifferentialCase:
    case_id: str
    case_kind: str
    pipeline: str


DIFFERENTIAL_FIELDS = [
    "program",
    "input_path",
    "case_id",
    "case_kind",
    "pipeline",
    "observed_activity",
    "external_success",
    "worker_success",
    "status_equal",
    "external_failure_kind",
    "worker_failure_kind",
    "external_diagnostic",
    "worker_diagnostic",
    "failure_parity",
    "external_hash",
    "worker_hash",
    "canonical_hash_equal",
    "external_fingerprint",
    "worker_fingerprint",
    "module_fingerprint_equal",
    "structural_equal",
    "equality_tier",
    "external_time_ms",
    "worker_time_ms",
    "semantic_equal",
    "mismatch_reason",
]

SUMMARY_FIELDS = [
    "status",
    "worker_default_recommended",
    "inputs",
    "total_cases",
    "passed_cases",
    "failed_cases",
    "status_mismatches",
    "failure_parity_mismatches",
    "canonical_hash_mismatches",
    "module_fingerprint_mismatches",
    "structural_equal_cases",
    "external_time_ms",
    "worker_time_ms",
    "worker_speedup",
]

ExecutePipeline = Callable[[str, Path, str, Path, int], RunResult]


def verify_opt_worker(
    inputs: Iterable[Path],
    out_dir: Path,
    pass_config: Path | None = None,
    *,
    cases: list[DifferentialCase] | None = None,
    execute: ExecutePipeline | None = None,
    tools: dict | None = None,
    worker_path: Path | str | None = None,
    workers: int = 1,
    timeout: int = 10,
    max_passes: int | None = None,
    keep_ir_artifacts: bool = False,
) -> dict:
    input_paths = [Path(path).resolve() for path in inputs]
    if not input_paths:
        raise ValueError("at least one differential input is required")
    if cases is None:
        if pass_config is None:
            raise ValueError("pass_config is required when differential cases are not supplied")
        cases = build_differential_cases(Path(pass_config), max_passes=max_passes)
    if not cases:
        raise ValueError("at least one differential case is required")

    out_dir = Path(out_dir).resolve()
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    owned_executor: _ProductionExecutor | None = None
    toolchain = tools
    if execute is None:
        toolchain = toolchain or collect_toolchain()
        owned_executor = _ProductionExecutor(
            toolchain,
            worker_path=worker_path,
            workers=workers,
        )
        execute = owned_executor

    rows: list[dict] = []
    try:
        for input_index, input_path in enumerate(input_paths):
            prepared = _prepare_input(
                input_path,
                artifacts_dir / f"P{input_index:04d}_{_safe_name(input_path.stem)}",
                toolchain,
                timeout,
            )
            program_rows: list[dict] = []
            for case in cases:
                case_dir = prepared.parent / case.case_id
                external_output = case_dir / "external.ll"
                worker_output = case_dir / "worker.ll"
                external = _execute_safely(execute, "external", prepared, case.pipeline, external_output, timeout)
                worker = _execute_safely(execute, "worker", prepared, case.pipeline, worker_output, timeout)
                program_rows.append(
                    _comparison_row(
                        program=input_path.stem,
                        input_path=input_path,
                        case=case,
                        external=external,
                        worker=worker,
                        external_output=external_output,
                        worker_output=worker_output,
                        tools=toolchain,
                    )
                )
            _classify_activity(program_rows)
            rows.extend(program_rows)
    finally:
        if owned_executor is not None:
            owned_executor.close()

    summary = _summarize(rows, len(input_paths))
    differential_csv = out_dir / "worker_differential.csv"
    summary_csv = out_dir / "worker_differential_summary.csv"
    markdown = out_dir / "worker_differential.md"
    _write_csv(differential_csv, DIFFERENTIAL_FIELDS, rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, [summary])
    _write_markdown(markdown, summary, rows)

    cleanup = mark_ir_artifacts_kept() if keep_ir_artifacts else cleanup_ir_artifacts(out_dir)
    return {
        "status": summary["status"],
        "worker_default_recommended": summary["worker_default_recommended"] == "true",
        "rows": len(rows),
        "failed_cases": int(summary["failed_cases"]),
        "worker_differential_csv": str(differential_csv),
        "worker_differential_summary_csv": str(summary_csv),
        "worker_differential_md": str(markdown),
        **cleanup,
    }


def build_differential_cases(pass_config: Path, *, max_passes: int | None = None) -> list[DifferentialCase]:
    pass_specs = load_pass_config(pass_config)
    if max_passes is not None:
        if max_passes < 1:
            raise ValueError("max_passes must be positive")
        pass_specs = pass_specs[:max_passes]
    pipelines = [spec.pipeline or spec.pipeline_candidates[0] for spec in pass_specs]

    specs: list[tuple[str, str]] = [("no_op", "verify")]
    specs.extend(("single", pipeline) for pipeline in pipelines)
    alternate_candidates: list[str] = []
    for spec, primary in zip(pass_specs, pipelines):
        alternate_candidates.extend(
            candidate for candidate in spec.pipeline_candidates if candidate != primary
        )
    specs.extend(("pipeline_candidate", pipeline) for pipeline in alternate_candidates)
    nested = next(
        (pipeline for pipeline in [*pipelines, *alternate_candidates] if "loop(" in pipeline),
        "function(loop(loop-rotate))",
    )
    specs.append(("nested_loop", nested))
    for index in range(len(pipelines) - 1):
        left = pipelines[index]
        right = pipelines[index + 1]
        specs.append(("pair_ab", f"{left},{right}"))
        specs.append(("pair_ba", f"{right},{left}"))
    if pipelines:
        specs.append(("replay", ",".join(pipelines)))
    specs.append(("invalid", "phasebatch-invalid-pass"))
    return [
        DifferentialCase(case_id=f"C{index:04d}", case_kind=kind, pipeline=pipeline)
        for index, (kind, pipeline) in enumerate(specs)
    ]


class _ProductionExecutor:
    def __init__(self, toolchain: dict, *, worker_path: Path | str | None, workers: int) -> None:
        tool_paths = _tool_paths(toolchain)
        self.opt = str(tool_paths["opt"])
        resolved_worker = resolve_worker_path(worker_path)
        if resolved_worker is None:
            raise FileNotFoundError("phasebatch-worker not found")
        self.worker = WorkerOptBackend(resolved_worker, workers=workers)

    def __call__(self, backend: str, input_ll: Path, pipeline: str, output_ll: Path, timeout: int) -> RunResult:
        if backend == "external":
            return _run_external_opt(self.opt, input_ll, pipeline, output_ll, timeout)
        if backend == "worker":
            return self.worker.run_opt(input_ll, pipeline, output_ll, timeout)
        raise ValueError(f"unknown differential backend: {backend}")

    def close(self) -> None:
        self.worker.close()


def _prepare_input(input_path: Path, program_dir: Path, tools: dict | None, timeout: int) -> Path:
    if input_path.suffix.lower() == ".ll":
        return input_path
    if tools is None:
        raise ValueError("toolchain is required for non-IR differential inputs")
    return prepare_input_ir(input_path, program_dir, _tool_paths(tools), timeout)


def _execute_safely(
    execute: ExecutePipeline,
    backend: str,
    input_ll: Path,
    pipeline: str,
    output_ll: Path,
    timeout: int,
) -> RunResult:
    output_ll.parent.mkdir(parents=True, exist_ok=True)
    try:
        return execute(backend, input_ll, pipeline, output_ll, timeout)
    except Exception as exc:
        return RunResult(
            command=[backend, pipeline],
            returncode=1,
            stdout="",
            stderr=str(exc),
            time_ms=0.0,
            failure_kind="backend_exception",
            output_path=output_ll,
            backend=backend,
        )


def _comparison_row(
    *,
    program: str,
    input_path: Path,
    case: DifferentialCase,
    external: RunResult,
    worker: RunResult,
    external_output: Path,
    worker_output: Path,
    tools: dict | None,
) -> dict:
    status_equal = external.success == worker.success
    external_failure = _normalized_failure(external)
    worker_failure = _normalized_failure(worker)
    failure_parity: bool | None = None
    external_hash = ""
    worker_hash = ""
    hash_equal: bool | None = None
    external_fingerprint = ""
    worker_fingerprint = ""
    fingerprint_equal: bool | None = None
    structural_equal: bool | None = None
    equality_tier = ""
    mismatch_reasons: list[str] = []
    observed_activity = "unknown"

    if not status_equal:
        mismatch_reasons.append("status_mismatch")
    elif not external.success:
        failure_parity = external_failure == worker_failure
        if not failure_parity:
            mismatch_reasons.append("failure_kind_mismatch")
        elif external_failure not in {"invalid_pipeline", "llvm_fatal"}:
            mismatch_reasons.append("unexpected_failure")
    else:
        try:
            external_hash = safe_canonical_hash(external_output)
            worker_hash = safe_canonical_hash(worker_output)
            hash_equal = external_hash == worker_hash
            external_fingerprint = module_safety_fingerprint(external_output)
            worker_fingerprint = module_safety_fingerprint(worker_output)
            fingerprint_equal = external_fingerprint == worker_fingerprint
            if hash_equal:
                equality_tier = "canonical_hash"
                if not fingerprint_equal:
                    mismatch_reasons.append("module_fingerprint_mismatch")
            elif fingerprint_equal:
                equivalence = compare_ir_equivalence(
                    external_output,
                    worker_output,
                    tools=tools or {},
                )
                structural_equal = equivalence.equal and equivalence.tier == "structural_diff"
                equality_tier = equivalence.tier
                if not structural_equal:
                    mismatch_reasons.append("canonical_hash_mismatch")
                    mismatch_reasons.append(f"structural_diff_{equivalence.reason}")
            else:
                mismatch_reasons.append("canonical_hash_mismatch")
                mismatch_reasons.append("module_fingerprint_mismatch")
        except OSError as exc:
            mismatch_reasons.append(f"output_read_failed:{exc}")

    semantic_equal = status_equal and not mismatch_reasons
    return {
        "program": program,
        "input_path": str(input_path),
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "pipeline": case.pipeline,
        "observed_activity": observed_activity,
        "external_success": _bool(external.success),
        "worker_success": _bool(worker.success),
        "status_equal": _bool(status_equal),
        "external_failure_kind": external_failure,
        "worker_failure_kind": worker_failure,
        "external_diagnostic": _diagnostic(external),
        "worker_diagnostic": _diagnostic(worker),
        "failure_parity": _optional_bool(failure_parity),
        "external_hash": external_hash,
        "worker_hash": worker_hash,
        "canonical_hash_equal": _optional_bool(hash_equal),
        "external_fingerprint": external_fingerprint,
        "worker_fingerprint": worker_fingerprint,
        "module_fingerprint_equal": _optional_bool(fingerprint_equal),
        "structural_equal": _optional_bool(structural_equal),
        "equality_tier": equality_tier,
        "external_time_ms": f"{external.time_ms:.3f}",
        "worker_time_ms": f"{worker.time_ms:.3f}",
        "semantic_equal": _bool(semantic_equal),
        "mismatch_reason": ";".join(mismatch_reasons),
    }


def _classify_activity(rows: list[dict]) -> None:
    baseline = next(
        (
            row["external_hash"]
            for row in rows
            if row["case_kind"] == "no_op" and row["external_success"] == "true" and row["external_hash"]
        ),
        "",
    )
    if not baseline:
        return
    for row in rows:
        if row["external_success"] != "true" or not row["external_hash"]:
            continue
        row["observed_activity"] = "dormant" if row["external_hash"] == baseline else "active"


def _normalized_failure(result: RunResult) -> str:
    if result.success:
        return ""
    if result.timed_out:
        return "timeout"
    diagnostic = f"{result.failure_kind} {result.stderr}".lower()
    if (
        "invalid_pipeline" in diagnostic
        or "invalid pass" in diagnostic
        or ("unknown" in diagnostic and "pass" in diagnostic)
    ):
        return "invalid_pipeline"
    if "llvm error:" in diagnostic:
        return "llvm_fatal"
    return result.failure_kind or "nonzero_exit"


def _diagnostic(result: RunResult) -> str:
    text = " ".join((result.stderr or result.stdout or "").split())
    return text[:500]


def _summarize(rows: list[dict], inputs: int) -> dict:
    failed = [row for row in rows if row["semantic_equal"] != "true"]
    external_ms = sum(float(row["external_time_ms"]) for row in rows)
    worker_ms = sum(float(row["worker_time_ms"]) for row in rows)
    speedup = external_ms / worker_ms if worker_ms > 0 else 0.0
    passed = not failed
    return {
        "status": "passed" if passed else "failed",
        "worker_default_recommended": _bool(passed),
        "inputs": str(inputs),
        "total_cases": str(len(rows)),
        "passed_cases": str(len(rows) - len(failed)),
        "failed_cases": str(len(failed)),
        "status_mismatches": str(sum(row["status_equal"] != "true" for row in rows)),
        "failure_parity_mismatches": str(sum(row["failure_parity"] == "false" for row in rows)),
        "canonical_hash_mismatches": str(sum(row["canonical_hash_equal"] == "false" for row in rows)),
        "module_fingerprint_mismatches": str(sum(row["module_fingerprint_equal"] == "false" for row in rows)),
        "structural_equal_cases": str(sum(row["structural_equal"] == "true" for row in rows)),
        "external_time_ms": f"{external_ms:.3f}",
        "worker_time_ms": f"{worker_ms:.3f}",
        "worker_speedup": f"{speedup:.3f}",
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict, rows: list[dict]) -> None:
    failures = [row for row in rows if row["semantic_equal"] != "true"]
    lines = [
        "# LLVM Worker Differential Verification",
        "",
        f"- Status: `{summary['status']}`",
        f"- Worker default recommended: `{summary['worker_default_recommended']}`",
        f"- Cases: `{summary['passed_cases']}/{summary['total_cases']}` passed",
        f"- External time: `{summary['external_time_ms']} ms`",
        f"- Worker time: `{summary['worker_time_ms']} ms`",
        f"- Observed speedup: `{summary['worker_speedup']}x`",
        "",
        "## Mismatches",
        "",
    ]
    if failures:
        lines.extend(
            f"- `{row['program']}/{row['case_id']}` `{row['case_kind']}`: {row['mismatch_reason']}"
            for row in failures
        )
    else:
        lines.append("No semantic differentials detected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tool_paths(metadata: dict) -> dict[str, str | None]:
    tools = metadata.get("tools", metadata)
    return {
        name: value.get("path") if isinstance(value, dict) else value
        for name, value in tools.items()
    }


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned or "input"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _optional_bool(value: bool | None) -> str:
    return "" if value is None else _bool(value)
