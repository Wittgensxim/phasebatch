from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

from .mainline import expand_inputs


PASSSET_SMOKE_RUN_FIELDS = [
    "program",
    "passset",
    "audit_status",
    "optimize_status",
    "valid_passes",
    "invalid_passes",
    "active_passes_depth0",
    "states_reached",
    "transitions",
    "selected_final_state",
    "root_ir_inst_count",
    "final_ir_inst_count",
    "ir_inst_delta",
    "time_ms",
    "error_message",
]

PASSSET_COMPARISON_FIELDS = [
    "program",
    "metric",
    "v1_value",
    "v2_value",
    "delta",
]

COMPARISON_METRICS = [
    "valid_passes",
    "active_passes_depth0",
    "tested_pairs_depth0",
    "commute_pairs_depth0",
    "order_sensitive_pairs_depth0",
    "batch_candidates_depth0",
    "certified_batches_depth0",
    "sampled_batches_depth0",
    "skipped_batches_depth0",
    "states_reached",
    "transitions",
    "final_ir_inst_count",
    "optimized_pipeline_length",
    "dropped_active_passes",
]


def run_passset_smoke(
    inputs: list[str],
    passsets: list[Path | str],
    out_dir: Path,
    *,
    optimizer_mode: str,
    objective: str,
    max_rounds: int,
    beam_width: int = 8,
    max_states: int = 5000,
    max_batches_per_state: int = 20,
    batch_frontier_policy: str | None = "score",
    validate_batches: bool = False,
    jobs: int = 1,
    timeout: int = 10,
    max_pairs: int | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    out_dir = Path(out_dir)
    passset_paths = [Path(path) for path in passsets]
    if not passset_paths:
        raise RuntimeError("at least one passset config is required")

    input_paths = expand_inputs(inputs, warn)
    if not input_paths:
        raise RuntimeError("no valid .c or .ll inputs remain after expansion")

    if out_dir.exists():
        if overwrite:
            _remove_existing_output(out_dir)
        elif any(out_dir.iterdir()):
            raise RuntimeError(f"output directory already exists: {out_dir}; use --overwrite to rerun")
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = _program_plans(input_paths, out_dir)
    passset_infos = [_passset_info(path, index) for index, path in enumerate(passset_paths)]
    run_rows: list[dict] = []
    metric_rows_by_key: dict[tuple[str, str], dict] = {}

    for program, input_path, program_dir in plans:
        for info in passset_infos:
            passset_dir = program_dir / info["name"]
            row, metrics = _run_one_passset(
                program,
                input_path,
                info,
                passset_dir,
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
            metric_rows_by_key[(program, info["label"])] = metrics
            if row["audit_status"] == "failed" or row["optimize_status"] == "failed":
                if not continue_on_error:
                    _write_outputs(out_dir, run_rows, _comparison_rows(metric_rows_by_key, passset_infos), passset_infos)
                    raise RuntimeError(row["error_message"])

    comparison_rows = _comparison_rows(metric_rows_by_key, passset_infos)
    summary = _write_outputs(out_dir, run_rows, comparison_rows, passset_infos)
    successes = sum(1 for row in run_rows if row.get("audit_status") == "success" and row.get("optimize_status") == "success")
    failures = len(run_rows) - successes
    return {
        "out_dir": str(out_dir),
        "runs": len(run_rows),
        "successes": successes,
        "failures": failures,
        "passset_smoke_runs_csv": str(out_dir / "passset_smoke_runs.csv"),
        "passset_comparison_csv": str(out_dir / "passset_comparison.csv"),
        "passset_smoke_summary_md": str(summary),
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


def _run_one_passset(
    program: str,
    input_path: Path,
    info: dict,
    passset_dir: Path,
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
    passset_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = passset_dir / "audit"
    optimize_dir = passset_dir / "optimize"
    audit_status = "success"
    optimize_status = "success"
    error_parts: list[str] = []
    metrics = _empty_metrics()
    audit_result: dict = {}

    try:
        audit_result = run_pass_audit(input_path, info["path"], audit_dir, timeout=timeout, jobs=jobs)
        metrics["valid_passes"] = _string(audit_result.get("valid_passes", ""))
        metrics["invalid_passes"] = _string(audit_result.get("invalid_passes", ""))
    except Exception as exc:
        audit_status = "failed"
        optimize_status = "not_run"
        error_parts.append(f"audit: {exc}")
        if not continue_on_error:
            return _run_row(program, info, audit_status, "not_run", metrics, "", _elapsed_ms(start), "; ".join(error_parts)), metrics

    resolved_passes = Path(audit_result.get("resolved_passes_yaml", "")) if audit_result.get("resolved_passes_yaml") else None
    if audit_status == "success" and (not resolved_passes or not resolved_passes.exists()):
        optimize_status = "failed"
        error_parts.append("optimize: audit produced no resolved_passes.yaml")
    elif audit_status == "success":
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
            metrics.update(_collect_optimize_metrics(optimize_dir))
            _try_run_baselines(passset_dir, optimize_dir, resolved_passes, objective, max_rounds, timeout, jobs, error_parts)
        except Exception as exc:
            optimize_status = "failed"
            error_parts.append(f"optimize: {exc}")

    row = _run_row(
        program,
        info,
        audit_status,
        optimize_status,
        metrics,
        metrics.get("selected_final_state", ""),
        _elapsed_ms(start),
        "; ".join(part for part in error_parts if part),
    )
    return row, metrics


def _try_run_baselines(
    passset_dir: Path,
    optimize_dir: Path,
    resolved_passes: Path,
    objective: str,
    max_rounds: int,
    timeout: int,
    jobs: int,
    error_parts: list[str],
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
        target = passset_dir / "baselines"
        if source.exists() and not target.exists():
            shutil.copytree(source, target)
    except Exception as exc:
        error_parts.append(f"baseline: {exc}")


def _run_row(
    program: str,
    info: dict,
    audit_status: str,
    optimize_status: str,
    metrics: dict,
    selected_final_state: str,
    time_ms: str,
    error_message: str,
) -> dict:
    return {
        "program": program,
        "passset": info["name"],
        "audit_status": audit_status,
        "optimize_status": optimize_status,
        "valid_passes": metrics.get("valid_passes", ""),
        "invalid_passes": metrics.get("invalid_passes", ""),
        "active_passes_depth0": metrics.get("active_passes_depth0", ""),
        "states_reached": metrics.get("states_reached", ""),
        "transitions": metrics.get("transitions", ""),
        "selected_final_state": selected_final_state,
        "root_ir_inst_count": metrics.get("root_ir_inst_count", ""),
        "final_ir_inst_count": metrics.get("final_ir_inst_count", ""),
        "ir_inst_delta": metrics.get("ir_inst_delta", ""),
        "time_ms": time_ms,
        "error_message": error_message,
    }


def _empty_metrics() -> dict:
    return {metric: "" for metric in COMPARISON_METRICS} | {
        "invalid_passes": "",
        "selected_final_state": "",
        "root_ir_inst_count": "",
        "ir_inst_delta": "",
    }


def _collect_optimize_metrics(optimize_dir: Path) -> dict:
    root_dir = optimize_dir / "states" / "S0000"
    per_state = _first_row(root_dir / "per_state_summary.csv")
    batch_summary = _first_row(root_dir / "batch_summary.csv")
    path_summary = _first_row(optimize_dir / "chosen_path_summary.csv")
    coverage = _first_row(root_dir / "coverage_summary.csv")
    correctness_rows = _read_csv(root_dir / "batch_correctness.csv")
    states = _read_csv(optimize_dir / "states.csv")
    transitions = _read_csv(optimize_dir / "batch_state_transitions.csv")

    certified = sum(1 for row in correctness_rows if row.get("correctness_class") == "certified_batch")
    sampled = sum(1 for row in correctness_rows if row.get("correctness_class") == "sampled_batch")
    skipped = sum(1 for row in correctness_rows if row.get("can_execute", "").lower() != "true")

    root_count = _first_value(path_summary, ["root_ir_inst_count"], "")
    final_count = _first_value(path_summary, ["final_ir_inst_count"], "")
    delta = _first_value(path_summary, ["total_ir_inst_delta", "ir_inst_delta"], "")
    if delta == "" and _is_number(root_count) and _is_number(final_count):
        delta = str(int(float(final_count)) - int(float(root_count)))

    return {
        "active_passes_depth0": _first_value(per_state, ["active_passes"], ""),
        "tested_pairs_depth0": _first_value(per_state, ["pairs_tested", "pair_rows"], ""),
        "commute_pairs_depth0": _first_value(per_state, ["dynamic_commute", "commute_pairs"], ""),
        "order_sensitive_pairs_depth0": _first_value(per_state, ["order_sensitive", "order_sensitive_pairs"], ""),
        "batch_candidates_depth0": _first_value(batch_summary, ["batch_candidates"], ""),
        "certified_batches_depth0": str(certified),
        "sampled_batches_depth0": str(sampled),
        "skipped_batches_depth0": str(skipped),
        "states_reached": str(len(states)),
        "transitions": str(len(transitions)),
        "selected_final_state": _first_value(path_summary, ["selected_final_state"], ""),
        "root_ir_inst_count": root_count,
        "final_ir_inst_count": final_count,
        "ir_inst_delta": delta,
        "optimized_pipeline_length": str(_pipeline_length(optimize_dir)),
        "dropped_active_passes": _first_value(coverage, ["dropped_active_passes", "dropped"], "0"),
    }


def _pipeline_length(optimize_dir: Path) -> int:
    for name in ("optimized_pipeline_names.txt", "optimized_pipeline.txt"):
        path = optimize_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return 0
        normalized = text.replace("\n", ",")
        return len([part for part in (item.strip() for item in normalized.split(",")) if part])
    return 0


def _comparison_rows(metrics_by_key: dict[tuple[str, str], dict], passset_infos: list[dict]) -> list[dict]:
    first_label, second_label = _comparison_labels(passset_infos)
    programs = sorted({program for program, _label in metrics_by_key})
    rows: list[dict] = []
    for program in programs:
        left = metrics_by_key.get((program, first_label), {})
        right = metrics_by_key.get((program, second_label), {})
        for metric in COMPARISON_METRICS:
            left_value = _string(left.get(metric, ""))
            right_value = _string(right.get(metric, ""))
            rows.append(
                {
                    "program": program,
                    "metric": metric,
                    "v1_value": left_value,
                    "v2_value": right_value,
                    "delta": _numeric_delta(left_value, right_value),
                }
            )
    return rows


def _comparison_labels(passset_infos: list[dict]) -> tuple[str, str]:
    labels = [info["label"] for info in passset_infos]
    first = "v1" if "v1" in labels else labels[0]
    second = "v2" if "v2" in labels else (labels[1] if len(labels) > 1 else "")
    return first, second


def _write_outputs(out_dir: Path, run_rows: list[dict], comparison_rows: list[dict], passset_infos: list[dict]) -> Path:
    _write_csv(out_dir / "passset_smoke_runs.csv", PASSSET_SMOKE_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "passset_comparison.csv", PASSSET_COMPARISON_FIELDS, comparison_rows)
    return _write_summary(out_dir, run_rows, comparison_rows, passset_infos)


def _write_summary(out_dir: Path, run_rows: list[dict], comparison_rows: list[dict], passset_infos: list[dict]) -> Path:
    comparison = _comparison_lookup(comparison_rows)
    programs = sorted({row.get("program", "") for row in run_rows})
    passset_names = ", ".join(info["name"] for info in passset_infos)
    failed = [row for row in run_rows if row.get("audit_status") == "failed" or row.get("optimize_status") == "failed"]
    lines = [
        "# Pass Set Smoke Summary",
        "",
        "## Overall",
        "",
        f"- number of programs: {len(programs)}",
        f"- passsets tested: {passset_names}",
        f"- failed runs: {len(failed)}",
        "",
        "## Valid Pass Count",
        "",
        *_markdown_table(
            ["program", "v1 valid", "v2 valid", "v2 invalid"],
            [
                [
                    program,
                    comparison.get((program, "valid_passes"), {}).get("v1_value", ""),
                    comparison.get((program, "valid_passes"), {}).get("v2_value", ""),
                    _run_value(run_rows, program, "v2", "invalid_passes", passset_infos),
                ]
                for program in programs
            ],
        ),
        "",
        "## Depth-0 Relation Changes",
        "",
        *_markdown_table(
            ["program", "v1 active", "v2 active", "v1 pairs", "v2 pairs", "v1 commute", "v2 commute", "v1 sensitive", "v2 sensitive"],
            [
                [
                    program,
                    _metric(comparison, program, "active_passes_depth0", "v1_value"),
                    _metric(comparison, program, "active_passes_depth0", "v2_value"),
                    _metric(comparison, program, "tested_pairs_depth0", "v1_value"),
                    _metric(comparison, program, "tested_pairs_depth0", "v2_value"),
                    _metric(comparison, program, "commute_pairs_depth0", "v1_value"),
                    _metric(comparison, program, "commute_pairs_depth0", "v2_value"),
                    _metric(comparison, program, "order_sensitive_pairs_depth0", "v1_value"),
                    _metric(comparison, program, "order_sensitive_pairs_depth0", "v2_value"),
                ]
                for program in programs
            ],
        ),
        "",
        "## Batch Changes",
        "",
        *_markdown_table(
            ["program", "v1 candidates", "v2 candidates", "v1 certified", "v2 certified", "v1 sampled", "v2 sampled", "dropped"],
            [
                [
                    program,
                    _metric(comparison, program, "batch_candidates_depth0", "v1_value"),
                    _metric(comparison, program, "batch_candidates_depth0", "v2_value"),
                    _metric(comparison, program, "certified_batches_depth0", "v1_value"),
                    _metric(comparison, program, "certified_batches_depth0", "v2_value"),
                    _metric(comparison, program, "sampled_batches_depth0", "v1_value"),
                    _metric(comparison, program, "sampled_batches_depth0", "v2_value"),
                    _metric(comparison, program, "dropped_active_passes", "v2_value"),
                ]
                for program in programs
            ],
        ),
        "",
        "## Final Objective",
        "",
        *_markdown_table(
            ["program", "v1 final IR inst", "v2 final IR inst", "delta"],
            [
                [
                    program,
                    _metric(comparison, program, "final_ir_inst_count", "v1_value"),
                    _metric(comparison, program, "final_ir_inst_count", "v2_value"),
                    _metric(comparison, program, "final_ir_inst_count", "delta"),
                ]
                for program in programs
            ],
        ),
        "",
        "## Notes",
        "",
        "Adding passes may increase active pairs and validation cost. Objective values are evaluation signals, not commutation proof.",
        "",
    ]
    path = out_dir / "passset_smoke_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _run_value(run_rows: list[dict], program: str, label: str, field: str, passset_infos: list[dict]) -> str:
    target_labels = _labels_by_name(passset_infos)
    for row in run_rows:
        if row.get("program") == program and target_labels.get(row.get("passset", "")) == label:
            return row.get(field, "")
    return ""


def _labels_by_name(passset_infos: list[dict]) -> dict[str, str]:
    return {info["name"]: info["label"] for info in passset_infos}


def _metric(comparison: dict[tuple[str, str], dict], program: str, metric: str, field: str) -> str:
    return comparison.get((program, metric), {}).get(field, "")


def _comparison_lookup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row.get("program", ""), row.get("metric", "")): row for row in rows}


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


def _passset_info(path: Path, index: int) -> dict:
    stem = path.stem
    lower = stem.lower()
    if "v1" in lower:
        label = "v1"
    elif "v2" in lower:
        label = "v2"
    elif "v3" in lower:
        label = "v3"
    else:
        label = stem if index > 1 else f"v{index + 1}"
    return {"path": path, "name": stem, "label": label}


def _remove_existing_output(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if not path.name or resolved == anchor or resolved == Path.cwd().resolve():
        raise RuntimeError(f"refusing to remove unsafe output path: {path}")
    shutil.rmtree(resolved)


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _first_value(row: dict, names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return _string(value)
    return default


def _numeric_delta(left: str, right: str) -> str:
    if not (_is_number(left) and _is_number(right)):
        return ""
    delta = float(right) - float(left)
    if delta.is_integer():
        return str(int(delta))
    return f"{delta:.4f}".rstrip("0").rstrip(".")


def _is_number(value: object) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _string(value: object) -> str:
    return "" if value is None else str(value)


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
