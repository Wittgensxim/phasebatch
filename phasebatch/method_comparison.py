from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

from .baselines import compare_baselines
from .mainline import expand_inputs


METHOD_COMPARISON_RUN_FIELDS = [
    "program",
    "input_path",
    "optimize_dir",
    "status",
    "error_stage",
    "error_message",
    "optimize_time_ms",
    "baseline_time_ms",
    "total_time_ms",
]

METHOD_COMPARISON_RESULT_FIELDS = [
    "program",
    "method",
    "status",
    "final_ir_inst_count",
    "root_ir_inst_count",
    "ir_inst_delta",
    "ir_inst_reduction_pct",
    "states_evaluated",
    "opt_runs",
    "final_sequence_length",
    "pass_sequence",
    "time_ms",
    "optimizer_total_time_ms",
    "analysis_time_ms",
    "profiling_time_ms",
    "pair_testing_time_ms",
    "batch_validation_time_ms",
    "batch_apply_time_ms",
    "total_opt_invocations",
    "error_message",
]

METHOD_COMPARISON_FAILURE_FIELDS = [
    "program",
    "input_path",
    "stage",
    "method",
    "status",
    "error_message",
]


def run_method_comparison(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    optimizer_mode: str,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    baseline_max_rounds: int,
    random_trials: int,
    seed: int,
    include_default_pipelines: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    out_dir = Path(out_dir)
    passes_path = Path(passes_path)
    input_paths = expand_inputs(inputs, warn)
    if not input_paths:
        raise RuntimeError("no valid .c or .ll inputs remain after expansion")

    out_dir.mkdir(parents=True, exist_ok=True)
    plans = _program_plans(input_paths, out_dir)
    run_rows: list[dict] = []
    result_rows: list[dict] = []
    failure_rows: list[dict] = []

    for program, input_path, program_dir in plans:
        optimize_dir = program_dir / "optimize"
        total_start = time.perf_counter()
        optimize_time_ms = ""
        baseline_time_ms = ""
        status = "success"
        error_stage = ""
        error_message = ""

        try:
            if program_dir.exists():
                if not overwrite:
                    raise RuntimeError(f"output directory already exists: {program_dir}; use --overwrite to rerun")
                _remove_existing_output(program_dir, out_dir)
            program_dir.mkdir(parents=True, exist_ok=True)

            optimize_start = time.perf_counter()
            run_optimizer(
                input_path,
                optimize_dir,
                passes_path,
                mode=optimizer_mode,
                objective=objective,
                max_rounds=max_rounds,
                beam_width=beam_width,
                max_batches_per_state=max_batches_per_state,
                max_states=max_states,
                batch_frontier_policy=batch_frontier_policy,
                validate_batches=validate_batches,
                allow_sampled_batches=False,
                jobs=jobs,
                timeout=timeout,
                max_pairs=max_pairs,
            )
            optimize_time_ms = _format_ms(_elapsed_ms(optimize_start))

            baseline_start = time.perf_counter()
            run_baseline_comparison(
                optimize_dir,
                passes_path,
                objective=objective,
                methods=["all"],
                max_rounds=baseline_max_rounds,
                random_trials=random_trials,
                seed=seed,
                timeout=timeout,
                jobs=jobs,
                include_default_pipelines=include_default_pipelines,
            )
            baseline_time_ms = _format_ms(_elapsed_ms(baseline_start))

            copied_rows = _copy_program_comparison_outputs(program_dir, optimize_dir)
            for row in copied_rows:
                normalized = _result_row_for_program(program, row)
                result_rows.append(normalized)
                if normalized["status"] in {"failed", "unsupported"}:
                    failure_rows.append(
                        {
                            "program": program,
                            "input_path": str(input_path),
                            "stage": "baseline",
                            "method": normalized["method"],
                            "status": normalized["status"],
                            "error_message": normalized["error_message"],
                        }
                    )
        except Exception as exc:
            status = "failed"
            error_stage = error_stage or ("optimize" if not optimize_time_ms else "baseline")
            error_message = str(exc)
            failure_rows.append(
                {
                    "program": program,
                    "input_path": str(input_path),
                    "stage": error_stage,
                    "method": "",
                    "status": "failed",
                    "error_message": error_message,
                }
            )
            if not continue_on_error:
                raise
        finally:
            run_rows.append(
                {
                    "program": program,
                    "input_path": str(input_path),
                    "optimize_dir": str(optimize_dir),
                    "status": status,
                    "error_stage": error_stage,
                    "error_message": error_message,
                    "optimize_time_ms": optimize_time_ms,
                    "baseline_time_ms": baseline_time_ms,
                    "total_time_ms": _format_ms(_elapsed_ms(total_start)),
                }
            )

    _write_csv(out_dir / "method_comparison_runs.csv", METHOD_COMPARISON_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "method_comparison_results.csv", METHOD_COMPARISON_RESULT_FIELDS, result_rows)
    _write_csv(out_dir / "method_comparison_failures.csv", METHOD_COMPARISON_FAILURE_FIELDS, failure_rows)
    summary_path = write_method_comparison_summary(
        out_dir,
        run_rows,
        result_rows,
        failure_rows,
        objective=objective,
        optimizer_mode=optimizer_mode,
        max_rounds=max_rounds,
        random_trials=random_trials,
        seed=seed,
    )
    successes = sum(1 for row in run_rows if row["status"] == "success")
    failures = sum(1 for row in run_rows if row["status"] == "failed")
    return {
        "out_dir": str(out_dir),
        "programs": len(run_rows),
        "successes": successes,
        "failures": failures,
        "method_comparison_runs_csv": str(out_dir / "method_comparison_runs.csv"),
        "method_comparison_results_csv": str(out_dir / "method_comparison_results.csv"),
        "method_comparison_failures_csv": str(out_dir / "method_comparison_failures.csv"),
        "method_comparison_summary_md": str(summary_path),
    }


def run_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(input_path, out_dir, passes_path, **kwargs)


def run_baseline_comparison(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    return compare_baselines(run_dir, passes_path, **kwargs)


def write_method_comparison_summary(
    out_dir: Path,
    run_rows: list[dict],
    result_rows: list[dict],
    failure_rows: list[dict],
    *,
    objective: str,
    optimizer_mode: str,
    max_rounds: int,
    random_trials: int,
    seed: int,
) -> Path:
    out_dir = Path(out_dir)
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = sum(1 for row in run_rows if row.get("status") == "failed")
    lines = [
        "# Method Comparison Summary",
        "",
        "## Overall",
        "",
        f"- total programs: {len(run_rows)}",
        f"- successful programs: {successes}",
        f"- failed programs: {failures}",
        f"- objective: {objective}",
        f"- optimizer mode: {optimizer_mode}",
        f"- max_rounds: {max_rounds}",
        f"- random_trials: {random_trials}",
        f"- seed: {seed}",
        "",
        "## Main Results",
        "",
        *_markdown_table(
            [
                "program",
                "method",
                "status",
                "final IR inst count",
                "delta",
                "reduction %",
                "states evaluated",
                "opt runs",
                "final sequence length",
            ],
            [
                [
                    row.get("program", ""),
                    row.get("method", ""),
                    row.get("status", ""),
                    row.get("final_ir_inst_count", ""),
                    row.get("ir_inst_delta", ""),
                    row.get("ir_inst_reduction_pct", ""),
                    row.get("states_evaluated", ""),
                    row.get("opt_runs", ""),
                    row.get("final_sequence_length", ""),
                ]
                for row in result_rows
            ],
        ),
        "",
        "## Best Method by Program",
        "",
        *_best_method_table(result_rows),
        "",
        "## Batch Optimizer Wins",
        "",
        *_batch_win_lines(result_rows),
        "",
        "## Cost Summary",
        "",
        *_cost_summary_table(result_rows),
        "",
        "## Notes",
        "",
        "IR instruction count is an evaluation objective. It is not used as commutation or independence proof.",
        "",
        "## Failures",
        "",
    ]
    if failure_rows:
        lines.extend(
            _markdown_table(
                ["program", "stage", "method", "status", "error"],
                [
                    [
                        row.get("program", ""),
                        row.get("stage", ""),
                        row.get("method", ""),
                        row.get("status", ""),
                        row.get("error_message", ""),
                    ]
                    for row in failure_rows
                ],
            )
        )
    else:
        lines.append("No failed programs or unsupported default pipelines recorded.")
    lines.append("")
    path = out_dir / "method_comparison_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _copy_program_comparison_outputs(program_dir: Path, optimize_dir: Path) -> list[dict]:
    baseline_csv = optimize_dir / "baseline_results.csv"
    rows = _read_csv(baseline_csv)
    shutil.copyfile(baseline_csv, program_dir / "baseline_results.csv")
    method_md = optimize_dir / "method_comparison.md"
    if method_md.exists():
        shutil.copyfile(method_md, program_dir / "method_comparison.md")
    baselines_dir = optimize_dir / "baselines"
    if baselines_dir.exists():
        target = program_dir / "baselines"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(baselines_dir, target)
    return rows


def _result_row_for_program(program: str, row: dict) -> dict:
    return {
        "program": program,
        "method": row.get("method", ""),
        "status": row.get("status", ""),
        "final_ir_inst_count": row.get("final_ir_inst_count", ""),
        "root_ir_inst_count": row.get("root_ir_inst_count", ""),
        "ir_inst_delta": row.get("ir_inst_delta", ""),
        "ir_inst_reduction_pct": row.get("ir_inst_reduction_pct", ""),
        "states_evaluated": row.get("states_evaluated", ""),
        "opt_runs": row.get("opt_runs", ""),
        "final_sequence_length": row.get("final_sequence_length", ""),
        "pass_sequence": row.get("pass_sequence", ""),
        "time_ms": row.get("time_ms", ""),
        "optimizer_total_time_ms": row.get("optimizer_total_time_ms", ""),
        "analysis_time_ms": row.get("analysis_time_ms", ""),
        "profiling_time_ms": row.get("profiling_time_ms", ""),
        "pair_testing_time_ms": row.get("pair_testing_time_ms", ""),
        "batch_validation_time_ms": row.get("batch_validation_time_ms", ""),
        "batch_apply_time_ms": row.get("batch_apply_time_ms", ""),
        "total_opt_invocations": row.get("total_opt_invocations", ""),
        "error_message": row.get("error_message", ""),
    }


def _program_plans(input_paths: list[Path], out_dir: Path) -> list[tuple[str, Path, Path]]:
    counts: dict[str, int] = {}
    plans = []
    for input_path in input_paths:
        stem = input_path.stem
        index = counts.get(stem, 0)
        counts[stem] = index + 1
        program = stem if index == 0 else f"{stem}_{index}"
        plans.append((program, input_path, Path(out_dir) / program))
    return plans


def _best_method_table(rows: list[dict]) -> list[str]:
    grouped = _rows_by_program(rows)
    table_rows: list[list[str]] = []
    for program, program_rows in grouped.items():
        best = _best_row(program_rows)
        batch = _method_row(program_rows, "batch_optimizer")
        table_rows.append(
            [
                program,
                best.get("method", "") if best else "",
                best.get("final_ir_inst_count", "") if best else "",
                _batch_rank(program_rows),
                batch.get("final_ir_inst_count", "") if batch else "",
                _method_inst(program_rows, "greedy_single_pass"),
                _method_inst(program_rows, "random_single_pass_best"),
                _method_inst(program_rows, "default_O0"),
                _method_inst(program_rows, "default_O2"),
                _method_inst(program_rows, "default_Oz"),
            ]
        )
    return _markdown_table(
        [
            "program",
            "best method",
            "best final IR inst count",
            "batch rank",
            "batch final IR inst count",
            "greedy final IR inst count",
            "random final IR inst count",
            "default_O0 final IR inst count",
            "default_O2",
            "default_Oz",
        ],
        table_rows,
    )


def _batch_win_lines(rows: list[dict]) -> list[str]:
    comparisons = [
        ("default_O0", "batch beats default_O0"),
        ("greedy_single_pass", "batch beats greedy"),
        ("random_single_pass_best", "batch beats random"),
        ("default_O2", "batch beats default_O2 when available"),
        ("default_Oz", "batch beats default_Oz when available"),
    ]
    lines: list[str] = []
    grouped = _rows_by_program(rows)
    for method, label in comparisons:
        wins = ties = losses = 0
        for program_rows in grouped.values():
            batch = _method_row(program_rows, "batch_optimizer")
            other = _method_row(program_rows, method)
            batch_count = _success_count(batch)
            other_count = _success_count(other)
            if batch_count is None or other_count is None:
                continue
            if batch_count < other_count:
                wins += 1
            elif batch_count == other_count:
                ties += 1
            else:
                losses += 1
        lines.append(f"- {label}: {wins}")
        lines.append(f"- ties vs {method}: {ties}")
        lines.append(f"- losses vs {method}: {losses}")
    return lines


def _cost_summary_table(rows: list[dict]) -> list[str]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("method", ""), []).append(row)
    table_rows: list[list[str]] = []
    for method in sorted(grouped):
        method_rows = grouped[method]
        table_rows.append(
            [
                method,
                _avg_field(method_rows, "states_evaluated"),
                _avg_field(method_rows, "opt_runs"),
                _avg_field(method_rows, "final_sequence_length"),
                _avg_field(method_rows, "time_ms"),
                _avg_field(method_rows, "optimizer_total_time_ms"),
                _avg_field(method_rows, "analysis_time_ms"),
                _avg_field(method_rows, "batch_validation_time_ms"),
                _avg_field(method_rows, "batch_apply_time_ms"),
                _avg_field(method_rows, "total_opt_invocations"),
            ]
        )
    return _markdown_table(
        [
            "method",
            "avg states evaluated",
            "avg opt runs",
            "avg final sequence length",
            "avg time ms",
            "avg optimizer total ms",
            "avg analysis ms",
            "avg batch validation ms",
            "avg batch apply ms",
            "avg total opt invocations",
        ],
        table_rows,
    )


def _rows_by_program(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("program", ""), []).append(row)
    return grouped


def _best_row(rows: list[dict]) -> dict | None:
    candidates = []
    for row in rows:
        count = _success_count(row)
        if count is not None:
            candidates.append((count, row.get("method", ""), row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _method_row(rows: list[dict], method: str) -> dict | None:
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def _method_inst(rows: list[dict], method: str) -> str:
    row = _method_row(rows, method)
    if not row:
        return ""
    return row.get("final_ir_inst_count", "") or row.get("status", "")


def _batch_rank(rows: list[dict]) -> str:
    batch = _method_row(rows, "batch_optimizer")
    batch_count = _success_count(batch)
    if batch_count is None:
        return ""
    rank = 1
    for row in rows:
        count = _success_count(row)
        if count is not None and count < batch_count:
            rank += 1
    return str(rank)


def _success_count(row: dict | None) -> int | None:
    if not row or row.get("status") != "success":
        return None
    return _parse_int(row.get("final_ir_inst_count", ""))


def _avg_field(rows: list[dict], field: str) -> str:
    values = []
    for row in rows:
        value = _parse_float(row.get(field, ""))
        if value is not None:
            values.append(value)
    if not values:
        return ""
    return f"{sum(values) / len(values):.2f}"


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _escape_cell(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _remove_existing_output(target: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise RuntimeError(f"refusing to remove output outside run root: {target}")
    shutil.rmtree(resolved_target)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
