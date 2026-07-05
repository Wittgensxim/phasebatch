from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

from .mainline import expand_inputs


V3_LOOP_RUN_FIELDS = [
    "program",
    "input_path",
    "status",
    "valid_passes",
    "invalid_passes",
    "valid_loop_passes",
    "invalid_loop_passes",
    "active_loop_passes_depth0",
    "total_active_passes_depth0",
    "states_reached",
    "transitions",
    "exact_or_budgeted",
    "final_ir_inst_count",
    "optimized_pipeline_length",
    "time_ms",
    "error_message",
]

V3_LOOP_SUMMARY_FIELDS = [
    "program",
    "valid_passes",
    "valid_loop_passes",
    "active_passes_depth0",
    "active_loop_passes_depth0",
    "tested_pairs_depth0",
    "commute_pairs_depth0",
    "sensitive_pairs_depth0",
    "batch_candidates_depth0",
    "certified_batches_depth0",
    "sampled_batches_depth0",
    "skipped_batches_depth0",
    "max_component_size_depth0",
    "states_reached",
    "transitions",
    "final_ir_inst_count",
    "dropped_active_passes",
]


def run_v3_loop_smoke(
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
    validate_batches: bool = False,
    jobs: int = 1,
    timeout: int = 10,
    max_pairs: int | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    out_dir = Path(out_dir)
    passes_path = Path(passes_path)
    input_paths = expand_inputs(inputs, warn)
    if not input_paths:
        raise RuntimeError("no valid .c or .ll inputs remain after expansion")

    if out_dir.exists():
        if overwrite:
            _remove_existing_output(out_dir)
        elif any(out_dir.iterdir()):
            raise RuntimeError(f"output directory already exists: {out_dir}; use --overwrite to rerun")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    for program, input_path, program_dir in _program_plans(input_paths, out_dir):
        row, summary = _run_one_program(
            program,
            input_path,
            program_dir,
            passes_path,
            optimizer_mode=optimizer_mode,
            objective=objective,
            max_rounds=max_rounds,
            beam_width=beam_width,
            max_states=max_states,
            max_batches_per_state=max_batches_per_state,
            batch_frontier_policy=batch_frontier_policy,
            validate_batches=validate_batches,
            jobs=jobs,
            timeout=timeout,
            max_pairs=max_pairs,
            continue_on_error=continue_on_error,
        )
        run_rows.append(row)
        summary_rows.append(summary)
        if row["status"] == "failed" and not continue_on_error:
            _write_outputs(out_dir, run_rows, summary_rows)
            raise RuntimeError(row["error_message"])

    summary_path = _write_outputs(out_dir, run_rows, summary_rows)
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = len(run_rows) - successes
    return {
        "out_dir": str(out_dir),
        "programs_attempted": len(run_rows),
        "successes": successes,
        "failures": failures,
        "v3_loop_runs_csv": str(out_dir / "v3_loop_runs.csv"),
        "v3_loop_summary_csv": str(out_dir / "v3_loop_summary.csv"),
        "v3_loop_summary_md": str(summary_path),
    }


def run_pass_audit(input_path: Path, passes_path: Path, out_dir: Path, **kwargs) -> dict:
    from .pass_audit import audit_passes

    return audit_passes(input_path, passes_path, out_dir, **kwargs)


def run_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(input_path, out_dir, passes_path, **kwargs)


def run_baseline_comparison(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .baselines import compare_baselines

    return compare_baselines(run_dir, passes_path, **kwargs)


def _run_one_program(
    program: str,
    input_path: Path,
    program_dir: Path,
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
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    continue_on_error: bool,
) -> tuple[dict, dict]:
    start = time.perf_counter()
    program_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = program_dir / "audit"
    optimize_dir = program_dir / "optimize"
    errors: list[str] = []
    status = "success"
    audit_metrics = _empty_audit_metrics()
    optimize_metrics = _empty_optimize_metrics()

    try:
        audit_result = run_pass_audit(input_path, passes_path, audit_dir, timeout=timeout, jobs=jobs)
        audit_metrics = _collect_audit_metrics(audit_dir, audit_result)
    except Exception as exc:
        status = "failed"
        errors.append(f"audit: {exc}")
        return _run_row(program, input_path, status, optimizer_mode, audit_metrics, optimize_metrics, _elapsed_ms(start), "; ".join(errors)), _summary_row(program, audit_metrics, optimize_metrics)

    resolved_passes = Path(audit_metrics.get("resolved_passes_yaml", ""))
    if not resolved_passes.exists():
        status = "failed"
        errors.append("optimize: audit produced no resolved_passes.yaml")
    else:
        try:
            run_optimizer(
                input_path,
                optimize_dir,
                resolved_passes,
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
            optimize_metrics = _collect_optimize_metrics(optimize_dir, audit_metrics.get("valid_loop_pass_names", set()))
            _try_run_baselines(program_dir, optimize_dir, resolved_passes, objective, max_rounds, timeout, jobs, errors)
        except Exception as exc:
            status = "failed"
            errors.append(f"optimize: {exc}")
            if not continue_on_error:
                pass

    row = _run_row(program, input_path, status, optimizer_mode, audit_metrics, optimize_metrics, _elapsed_ms(start), "; ".join(errors))
    return row, _summary_row(program, audit_metrics, optimize_metrics)


def _try_run_baselines(
    program_dir: Path,
    optimize_dir: Path,
    resolved_passes: Path,
    objective: str,
    max_rounds: int,
    timeout: int,
    jobs: int,
    errors: list[str],
) -> None:
    try:
        run_baseline_comparison(
            optimize_dir,
            resolved_passes,
            objective=objective,
            methods=["batch"],
            max_rounds=max_rounds,
            random_trials=1,
            seed=0,
            timeout=timeout,
            jobs=jobs,
            include_default_pipelines=False,
        )
        source = optimize_dir / "baselines"
        target = program_dir / "baselines"
        if source.exists() and not target.exists():
            shutil.copytree(source, target)
    except Exception as exc:
        errors.append(f"baseline: {exc}")


def _collect_audit_metrics(audit_dir: Path, audit_result: dict) -> dict:
    rows = _read_csv(audit_dir / "pass_audit.csv")
    valid_rows = [row for row in rows if _truthy(row.get("valid_on_input"))]
    invalid_rows = [row for row in rows if not _truthy(row.get("valid_on_input"))]
    valid_loop_rows = [row for row in valid_rows if row.get("category") == "loop"]
    invalid_loop_rows = [row for row in invalid_rows if row.get("category") == "loop"]
    return {
        "valid_passes": str(audit_result.get("valid_passes", len(valid_rows))),
        "invalid_passes": str(audit_result.get("invalid_passes", len(invalid_rows))),
        "valid_loop_passes": str(len(valid_loop_rows)),
        "invalid_loop_passes": str(len(invalid_loop_rows)),
        "valid_loop_pass_names": {row.get("pass", "") for row in valid_loop_rows},
        "resolved_loop_pipelines": ";".join(
            f"{row.get('pass', '')}={row.get('resolved_pipeline', '')}"
            for row in valid_loop_rows
            if row.get("pass") and row.get("resolved_pipeline")
        ),
        "resolved_passes_yaml": str(audit_result.get("resolved_passes_yaml", audit_dir / "resolved_passes.yaml")),
    }


def _collect_optimize_metrics(optimize_dir: Path, valid_loop_names: set[str]) -> dict:
    root_dir = optimize_dir / "states" / "S0000"
    per_state = _first_row(root_dir / "per_state_summary.csv")
    batch_summary = _first_row(root_dir / "batch_summary.csv")
    coverage = _first_row(root_dir / "coverage_summary.csv")
    chosen = _first_row(optimize_dir / "chosen_path_summary.csv")
    profile_rows = _read_csv(root_dir / "pass_profile.csv")
    correctness_rows = _read_csv(root_dir / "batch_correctness.csv")
    states = _read_csv(optimize_dir / "states.csv")
    transitions = _read_csv(optimize_dir / "batch_state_transitions.csv")

    certified = sum(1 for row in correctness_rows if row.get("correctness_class") == "certified_batch")
    sampled = sum(1 for row in correctness_rows if row.get("correctness_class") == "sampled_batch")
    skipped = sum(1 for row in correctness_rows if row.get("can_execute", "").lower() != "true")
    active_loop = sum(
        1
        for row in profile_rows
        if row.get("pass") in valid_loop_names and _truthy(row.get("success")) and _truthy(row.get("active"))
    )

    return {
        "active_loop_passes_depth0": str(active_loop),
        "total_active_passes_depth0": _first_value(per_state, ["active_passes"], ""),
        "active_passes_depth0": _first_value(per_state, ["active_passes"], ""),
        "tested_pairs_depth0": _first_value(per_state, ["pairs_tested", "pair_rows"], ""),
        "commute_pairs_depth0": _first_value(per_state, ["dynamic_commute", "commute_pairs"], ""),
        "sensitive_pairs_depth0": _first_value(per_state, ["order_sensitive", "order_sensitive_pairs"], ""),
        "unknown_pairs_depth0": _first_value(per_state, ["unknown", "unknown_pairs"], ""),
        "batch_candidates_depth0": _first_value(batch_summary, ["batch_candidates"], ""),
        "certified_batches_depth0": str(certified),
        "sampled_batches_depth0": str(sampled),
        "skipped_batches_depth0": str(skipped),
        "max_component_size_depth0": _first_value(batch_summary, ["max_component_size", "max_conflict_component"], ""),
        "states_reached": str(len(states)),
        "transitions": str(len(transitions)),
        "final_ir_inst_count": _first_value(chosen, ["final_ir_inst_count"], ""),
        "optimized_pipeline_length": str(_pipeline_length(optimize_dir)),
        "dropped_active_passes": _first_value(coverage, ["dropped_active_passes", "dropped"], "0"),
    }


def _run_row(
    program: str,
    input_path: Path,
    status: str,
    optimizer_mode: str,
    audit: dict,
    optimize: dict,
    time_ms: str,
    error_message: str,
) -> dict:
    return {
        "program": program,
        "input_path": str(input_path),
        "status": status,
        "valid_passes": audit.get("valid_passes", ""),
        "invalid_passes": audit.get("invalid_passes", ""),
        "valid_loop_passes": audit.get("valid_loop_passes", ""),
        "invalid_loop_passes": audit.get("invalid_loop_passes", ""),
        "active_loop_passes_depth0": optimize.get("active_loop_passes_depth0", ""),
        "total_active_passes_depth0": optimize.get("total_active_passes_depth0", ""),
        "states_reached": optimize.get("states_reached", ""),
        "transitions": optimize.get("transitions", ""),
        "exact_or_budgeted": optimizer_mode,
        "final_ir_inst_count": optimize.get("final_ir_inst_count", ""),
        "optimized_pipeline_length": optimize.get("optimized_pipeline_length", ""),
        "time_ms": time_ms,
        "error_message": error_message,
    }


def _summary_row(program: str, audit: dict, optimize: dict) -> dict:
    return {
        "program": program,
        "valid_passes": audit.get("valid_passes", ""),
        "valid_loop_passes": audit.get("valid_loop_passes", ""),
        "active_passes_depth0": optimize.get("active_passes_depth0", ""),
        "active_loop_passes_depth0": optimize.get("active_loop_passes_depth0", ""),
        "tested_pairs_depth0": optimize.get("tested_pairs_depth0", ""),
        "commute_pairs_depth0": optimize.get("commute_pairs_depth0", ""),
        "sensitive_pairs_depth0": optimize.get("sensitive_pairs_depth0", ""),
        "unknown_pairs_depth0": optimize.get("unknown_pairs_depth0", ""),
        "batch_candidates_depth0": optimize.get("batch_candidates_depth0", ""),
        "certified_batches_depth0": optimize.get("certified_batches_depth0", ""),
        "sampled_batches_depth0": optimize.get("sampled_batches_depth0", ""),
        "skipped_batches_depth0": optimize.get("skipped_batches_depth0", ""),
        "max_component_size_depth0": optimize.get("max_component_size_depth0", ""),
        "states_reached": optimize.get("states_reached", ""),
        "transitions": optimize.get("transitions", ""),
        "final_ir_inst_count": optimize.get("final_ir_inst_count", ""),
        "optimized_pipeline_length": optimize.get("optimized_pipeline_length", ""),
        "dropped_active_passes": optimize.get("dropped_active_passes", ""),
        "resolved_loop_pipelines": audit.get("resolved_loop_pipelines", ""),
        "invalid_loop_passes": audit.get("invalid_loop_passes", ""),
    }


def _write_outputs(out_dir: Path, run_rows: list[dict], summary_rows: list[dict]) -> Path:
    _write_csv(out_dir / "v3_loop_runs.csv", V3_LOOP_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "v3_loop_summary.csv", V3_LOOP_SUMMARY_FIELDS, summary_rows)
    return _write_summary_md(out_dir, run_rows, summary_rows)


def _write_summary_md(out_dir: Path, run_rows: list[dict], summary_rows: list[dict]) -> Path:
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = len(run_rows) - successes
    valid_loop_counts = [row.get("valid_loop_passes", "") for row in run_rows if row.get("valid_loop_passes", "")]
    valid_loop_count = max((_to_int(value) for value in valid_loop_counts), default=0)
    lines = [
        "# V3 Middle-End / Loop Pass Smoke Summary",
        "",
        "## Overall",
        "",
        f"- programs attempted: {len(run_rows)}",
        f"- successful programs: {successes}",
        f"- failed programs: {failures}",
        f"- valid loop pass count: {valid_loop_count}",
        "",
        "## Loop Pass Resolution",
        "",
        *_markdown_table(
            ["program", "valid loop passes", "invalid loop passes", "resolved pipelines"],
            [
                [
                    row.get("program", ""),
                    row.get("valid_loop_passes", ""),
                    row.get("invalid_loop_passes", ""),
                    row.get("resolved_loop_pipelines", ""),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Depth-0 Relation Summary",
        "",
        *_markdown_table(
            ["program", "active passes", "active loop passes", "pairs", "commute", "sensitive", "unknown"],
            [
                [
                    row.get("program", ""),
                    row.get("active_passes_depth0", ""),
                    row.get("active_loop_passes_depth0", ""),
                    row.get("tested_pairs_depth0", ""),
                    row.get("commute_pairs_depth0", ""),
                    row.get("sensitive_pairs_depth0", ""),
                    row.get("unknown_pairs_depth0", ""),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Batch Summary",
        "",
        *_markdown_table(
            ["program", "batch candidates", "certified", "sampled", "skipped", "max component", "dropped"],
            [
                [
                    row.get("program", ""),
                    row.get("batch_candidates_depth0", ""),
                    row.get("certified_batches_depth0", ""),
                    row.get("sampled_batches_depth0", ""),
                    row.get("skipped_batches_depth0", ""),
                    row.get("max_component_size_depth0", ""),
                    row.get("dropped_active_passes", ""),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Optimization Summary",
        "",
        *_markdown_table(
            ["program", "final IR inst count", "pipeline length", "states reached", "transitions"],
            [
                [
                    row.get("program", ""),
                    row.get("final_ir_inst_count", ""),
                    row.get("optimized_pipeline_length", ""),
                    row.get("states_reached", ""),
                    row.get("transitions", ""),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Notes",
        "",
        "Loop passes may require nested New Pass Manager pipeline syntax. This workflow uses audit-passes to resolve valid local pipeline text.",
        "",
    ]
    path = out_dir / "v3_loop_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _empty_audit_metrics() -> dict:
    return {
        "valid_passes": "",
        "invalid_passes": "",
        "valid_loop_passes": "",
        "invalid_loop_passes": "",
        "valid_loop_pass_names": set(),
        "resolved_loop_pipelines": "",
        "resolved_passes_yaml": "",
    }


def _empty_optimize_metrics() -> dict:
    return {
        "active_loop_passes_depth0": "",
        "total_active_passes_depth0": "",
        "states_reached": "",
        "transitions": "",
        "final_ir_inst_count": "",
        "optimized_pipeline_length": "",
    }


def _program_plans(input_paths: list[Path], out_dir: Path) -> list[tuple[str, Path, Path]]:
    counts: dict[str, int] = {}
    plans: list[tuple[str, Path, Path]] = []
    for input_path in input_paths:
        stem = input_path.stem
        index = counts.get(stem, 0)
        counts[stem] = index + 1
        program = stem if index == 0 else f"{stem}_{index}"
        plans.append((program, input_path, out_dir / program))
    return plans


def _pipeline_length(optimize_dir: Path) -> int:
    for name in ("optimized_pipeline_names.txt", "optimized_pipeline.txt"):
        path = optimize_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return 0
        return len([part for part in text.replace("\n", ",").split(",") if part.strip()])
    return 0


def _remove_existing_output(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if not path.name or resolved == anchor or resolved == Path.cwd().resolve():
        raise RuntimeError(f"refusing to remove unsafe output path: {path}")
    shutil.rmtree(resolved)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _first_value(row: dict, names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _to_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _elapsed_ms(start: float) -> str:
    return f"{(time.perf_counter() - start) * 1000:.3f}"


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


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _escape_cell(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")
