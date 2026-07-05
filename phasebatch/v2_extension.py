from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

from .mainline import expand_inputs
from .pass_config import PassSpec, load_pass_config


V2_ADDED_PASSES = [
    ("sccp", "sccp", "scalar", "v2"),
    ("dse", "dse", "memory", "v2"),
    ("memcpyopt", "memcpyopt", "memory", "v2"),
    ("sink", "sink", "cfg", "v2"),
    ("tailcallelim", "tailcallelim", "scalar", "v2"),
]

V2_EXTENSION_RUN_FIELDS = [
    "program",
    "input_path",
    "stage",
    "passset",
    "status",
    "output_dir",
    "valid_passes",
    "invalid_passes",
    "final_ir_inst_count",
    "states_reached",
    "transitions",
    "time_ms",
    "error_message",
]

V2_EXTENSION_COMPARISON_FIELDS = [
    "program",
    "v1_valid_passes",
    "v2_valid_passes",
    "v1_active_depth0",
    "v2_active_depth0",
    "v1_tested_pairs_depth0",
    "v2_tested_pairs_depth0",
    "v1_commute_depth0",
    "v2_commute_depth0",
    "v1_sensitive_depth0",
    "v2_sensitive_depth0",
    "v1_batch_candidates_depth0",
    "v2_batch_candidates_depth0",
    "v1_certified_batches_depth0",
    "v2_certified_batches_depth0",
    "v1_final_ir_inst",
    "v2_final_ir_inst",
    "v2_minus_v1_final_inst",
    "v1_states",
    "v2_states",
    "v1_time_ms",
    "v2_time_ms",
    "v1_dropped_active_passes",
    "v2_dropped_active_passes",
]

V2_EXTENSION_FAILURE_FIELDS = [
    "program",
    "input_path",
    "stage",
    "passset",
    "error_message",
]


def run_v2_extension_study(
    inputs: list[str],
    out_dir: Path,
    v1_passes: Path,
    v2_passes: Path,
    *,
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
    random_trials: int,
    seed: int,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    out_dir = Path(out_dir)
    v1_passes = Path(v1_passes)
    v2_passes = Path(v2_passes)
    _ensure_v2_config(v1_passes, v2_passes)

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
    comparison_rows: list[dict] = []
    failure_rows: list[dict] = []

    for program, input_path, program_dir in _program_plans(input_paths, out_dir):
        program_dir.mkdir(parents=True, exist_ok=True)
        v1_dir = program_dir / "v1"
        v2_dir = program_dir / "v2"
        audit_dir = program_dir / "v2_audit"
        v1_metrics: dict = {}
        v2_metrics: dict = {}

        v1_row, v1_metrics = _run_optimize_stage(
            program,
            input_path,
            "v1",
            v1_dir,
            v1_passes,
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
        )
        run_rows.append(v1_row)
        if v1_row["status"] != "success":
            failure_rows.append(_failure(program, input_path, "optimize", "v1", v1_row["error_message"]))
            if not continue_on_error:
                _write_outputs(out_dir, run_rows, comparison_rows, failure_rows, v1_passes, v2_passes, objective, max_rounds, beam_width, max_states, random_trials, seed)
                raise RuntimeError(v1_row["error_message"])

        audit_row, resolved_v2_passes = _run_v2_audit(program, input_path, audit_dir, v2_passes, timeout=timeout, jobs=jobs)
        run_rows.append(audit_row)
        if audit_row["status"] != "success":
            failure_rows.append(_failure(program, input_path, "audit", "v2", audit_row["error_message"]))
            resolved_v2_passes = v2_passes

        v2_row, v2_metrics = _run_optimize_stage(
            program,
            input_path,
            "v2",
            v2_dir,
            resolved_v2_passes,
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
        )
        if audit_row["status"] == "success":
            v2_row["valid_passes"] = audit_row["valid_passes"] or v2_row["valid_passes"]
            v2_row["invalid_passes"] = audit_row["invalid_passes"] or v2_row["invalid_passes"]
            v2_metrics["valid_passes"] = v2_row["valid_passes"]
            v2_metrics["invalid_passes"] = v2_row["invalid_passes"]
        run_rows.append(v2_row)
        if v2_row["status"] != "success":
            failure_rows.append(_failure(program, input_path, "optimize", "v2", v2_row["error_message"]))
            if not continue_on_error:
                _write_outputs(out_dir, run_rows, comparison_rows, failure_rows, v1_passes, v2_passes, objective, max_rounds, beam_width, max_states, random_trials, seed)
                raise RuntimeError(v2_row["error_message"])

        comparison_rows.append(_comparison_row(program, v1_metrics, v2_metrics))

    summary_path = _write_outputs(out_dir, run_rows, comparison_rows, failure_rows, v1_passes, v2_passes, objective, max_rounds, beam_width, max_states, random_trials, seed)
    programs = sorted({row["program"] for row in comparison_rows})
    successes = sum(1 for program in programs if _program_success(run_rows, program))
    failures = len(programs) - successes
    return {
        "out_dir": str(out_dir),
        "programs": len(programs),
        "successes": successes,
        "failures": failures,
        "v2_extension_runs_csv": str(out_dir / "v2_extension_runs.csv"),
        "v2_extension_comparison_csv": str(out_dir / "v2_extension_comparison.csv"),
        "v2_extension_summary_md": str(summary_path),
        "failures_csv": str(out_dir / "failures.csv"),
    }


def run_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(input_path, out_dir, passes_path, **kwargs)


def run_pass_audit(input_path: Path, passes_path: Path, out_dir: Path, **kwargs) -> dict:
    from .pass_audit import audit_passes

    return audit_passes(input_path, passes_path, out_dir, **kwargs)


def _ensure_v2_config(v1_passes: Path, v2_passes: Path) -> None:
    if v2_passes.exists():
        return
    specs = load_pass_config(v1_passes)
    existing = {spec.name for spec in specs}
    lines = ["passes:"]
    for spec in specs:
        lines.extend(_spec_lines(spec))
    for name, pipeline, category, stage in V2_ADDED_PASSES:
        if name in existing:
            continue
        lines.extend(
            [
                f"  - name: {name}",
                f"    pipeline: {pipeline}",
                f"    category: {category}",
                f"    stage: {stage}",
            ]
        )
    v2_passes.parent.mkdir(parents=True, exist_ok=True)
    v2_passes.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spec_lines(spec: PassSpec) -> list[str]:
    pipeline = spec.pipeline or (spec.pipeline_candidates[0] if spec.pipeline_candidates else spec.name)
    return [
        f"  - name: {spec.name}",
        f"    pipeline: {pipeline}",
        f"    category: {spec.category}",
        f"    stage: {spec.stage}",
    ]


def _run_optimize_stage(
    program: str,
    input_path: Path,
    passset: str,
    out_dir: Path,
    passes_path: Path,
    *,
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
) -> tuple[dict, dict]:
    start = time.perf_counter()
    status = "success"
    error = ""
    try:
        run_optimizer(
            input_path,
            out_dir,
            passes_path,
            mode="budgeted",
            objective=objective,
            max_rounds=max_rounds,
            beam_width=beam_width,
            max_states=max_states,
            max_batches_per_state=max_batches_per_state,
            batch_frontier_policy=batch_frontier_policy,
            validate_batches=validate_batches,
            allow_sampled_batches=False,
            jobs=jobs,
            timeout=timeout,
            max_pairs=max_pairs,
        )
    except Exception as exc:
        status = "failed"
        error = str(exc)
    metrics = _collect_metrics(out_dir)
    row = _run_row(
        program,
        input_path,
        "optimize",
        passset,
        status,
        out_dir,
        metrics,
        _elapsed_ms(start),
        error,
    )
    return row, metrics


def _run_v2_audit(program: str, input_path: Path, out_dir: Path, v2_passes: Path, *, timeout: int, jobs: int) -> tuple[dict, Path]:
    start = time.perf_counter()
    status = "success"
    error = ""
    valid = ""
    invalid = ""
    resolved = v2_passes
    try:
        result = run_pass_audit(input_path, v2_passes, out_dir, timeout=timeout, jobs=jobs)
        valid = _string(result.get("valid_passes", ""))
        invalid = _string(result.get("invalid_passes", ""))
        resolved_text = _string(result.get("resolved_passes_yaml", ""))
        if resolved_text and Path(resolved_text).exists():
            resolved = Path(resolved_text)
    except Exception as exc:
        status = "failed"
        error = str(exc)
    row = {
        "program": program,
        "input_path": str(input_path),
        "stage": "audit",
        "passset": "v2",
        "status": status,
        "output_dir": str(out_dir),
        "valid_passes": valid,
        "invalid_passes": invalid,
        "final_ir_inst_count": "",
        "states_reached": "",
        "transitions": "",
        "time_ms": _elapsed_ms(start),
        "error_message": error,
    }
    return row, resolved


def _collect_metrics(optimize_dir: Path) -> dict:
    root_dir = optimize_dir / "states" / "S0000"
    per_state = _first_row(root_dir / "per_state_summary.csv")
    batch_summary = _first_row(root_dir / "batch_summary.csv")
    path_summary = _first_row(optimize_dir / "chosen_path_summary.csv")
    coverage = _first_row(root_dir / "coverage_summary.csv")
    timing = _first_row(optimize_dir / "optimizer_timing.csv")
    states = _read_csv(optimize_dir / "states.csv")
    transitions = _read_csv(optimize_dir / "batch_state_transitions.csv")
    correctness = _read_csv(root_dir / "batch_correctness.csv")
    return {
        "valid_passes": str(len(_read_csv(optimize_dir / "valid_passes.csv"))),
        "invalid_passes": str(len(_read_csv(optimize_dir / "invalid_passes.csv"))),
        "active_depth0": _value(per_state, "active_passes"),
        "tested_pairs_depth0": _value(per_state, "pairs_tested", "pair_rows"),
        "commute_depth0": _value(per_state, "dynamic_commute", "commute_pairs"),
        "sensitive_depth0": _value(per_state, "order_sensitive", "order_sensitive_pairs"),
        "batch_candidates_depth0": _value(batch_summary, "batch_candidates"),
        "certified_batches_depth0": str(sum(1 for row in correctness if row.get("correctness_class") == "certified_batch")),
        "final_ir_inst": _value(path_summary, "final_ir_inst_count", "final_objective"),
        "states": str(len(states)) if states else "",
        "transitions": str(len(transitions)) if transitions else "",
        "time_ms": _value(timing, "optimizer_total_time_ms"),
        "dropped_active_passes": _value(coverage, "dropped_active_passes", "dropped", default="0"),
    }


def _run_row(
    program: str,
    input_path: Path,
    stage: str,
    passset: str,
    status: str,
    out_dir: Path,
    metrics: dict,
    time_ms: str,
    error: str,
) -> dict:
    return {
        "program": program,
        "input_path": str(input_path),
        "stage": stage,
        "passset": passset,
        "status": status,
        "output_dir": str(out_dir),
        "valid_passes": metrics.get("valid_passes", ""),
        "invalid_passes": metrics.get("invalid_passes", ""),
        "final_ir_inst_count": metrics.get("final_ir_inst", ""),
        "states_reached": metrics.get("states", ""),
        "transitions": metrics.get("transitions", ""),
        "time_ms": metrics.get("time_ms") or time_ms,
        "error_message": error,
    }


def _comparison_row(program: str, v1: dict, v2: dict) -> dict:
    return {
        "program": program,
        "v1_valid_passes": v1.get("valid_passes", ""),
        "v2_valid_passes": v2.get("valid_passes", ""),
        "v1_active_depth0": v1.get("active_depth0", ""),
        "v2_active_depth0": v2.get("active_depth0", ""),
        "v1_tested_pairs_depth0": v1.get("tested_pairs_depth0", ""),
        "v2_tested_pairs_depth0": v2.get("tested_pairs_depth0", ""),
        "v1_commute_depth0": v1.get("commute_depth0", ""),
        "v2_commute_depth0": v2.get("commute_depth0", ""),
        "v1_sensitive_depth0": v1.get("sensitive_depth0", ""),
        "v2_sensitive_depth0": v2.get("sensitive_depth0", ""),
        "v1_batch_candidates_depth0": v1.get("batch_candidates_depth0", ""),
        "v2_batch_candidates_depth0": v2.get("batch_candidates_depth0", ""),
        "v1_certified_batches_depth0": v1.get("certified_batches_depth0", ""),
        "v2_certified_batches_depth0": v2.get("certified_batches_depth0", ""),
        "v1_final_ir_inst": v1.get("final_ir_inst", ""),
        "v2_final_ir_inst": v2.get("final_ir_inst", ""),
        "v2_minus_v1_final_inst": _numeric_delta(v1.get("final_ir_inst", ""), v2.get("final_ir_inst", "")),
        "v1_states": v1.get("states", ""),
        "v2_states": v2.get("states", ""),
        "v1_time_ms": v1.get("time_ms", ""),
        "v2_time_ms": v2.get("time_ms", ""),
        "v1_dropped_active_passes": v1.get("dropped_active_passes", "0"),
        "v2_dropped_active_passes": v2.get("dropped_active_passes", "0"),
    }


def _write_outputs(
    out_dir: Path,
    run_rows: list[dict],
    comparison_rows: list[dict],
    failure_rows: list[dict],
    v1_passes: Path,
    v2_passes: Path,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    random_trials: int,
    seed: int,
) -> Path:
    _write_csv(out_dir / "v2_extension_runs.csv", V2_EXTENSION_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "v2_extension_comparison.csv", V2_EXTENSION_COMPARISON_FIELDS, comparison_rows)
    _write_csv(out_dir / "failures.csv", V2_EXTENSION_FAILURE_FIELDS, failure_rows)
    return _write_summary(out_dir, run_rows, comparison_rows, v1_passes, v2_passes, objective, max_rounds, beam_width, max_states, random_trials, seed)


def _write_summary(
    out_dir: Path,
    run_rows: list[dict],
    comparison_rows: list[dict],
    v1_passes: Path,
    v2_passes: Path,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    random_trials: int,
    seed: int,
) -> Path:
    lines = [
        "# V2 Scalar Pass Set Extension Summary",
        "",
        "## Purpose",
        "",
        "V2 is a scalability extension. It is not intended to replace the Core-v1 case study until stability and cost are evaluated.",
        "",
        f"- v1 passes: {v1_passes}",
        f"- v2 passes: {v2_passes}",
        f"- objective: {objective}",
        f"- max_rounds: {max_rounds}",
        f"- beam_width: {beam_width}",
        f"- max_states: {max_states}",
        f"- random_trials: {random_trials}",
        f"- seed: {seed}",
        "",
        "## Pass Set Difference",
        "",
        *[f"- {name}" for name, _pipeline, _category, _stage in V2_ADDED_PASSES],
        "",
        "## Validity / Audit",
        "",
        *_markdown_table(
            ["program", "v1 valid", "v2 valid", "v2 invalid"],
            [[row["program"], row["v1_valid_passes"], row["v2_valid_passes"], _v2_invalid(run_rows, row["program"])] for row in comparison_rows],
        ),
        "",
        "## Activity and Pair Relations",
        "",
        *_markdown_table(
            ["program", "v1 active", "v2 active", "v1 pairs", "v2 pairs", "v1 commute", "v2 commute", "v1 sensitive", "v2 sensitive"],
            [
                [
                    row["program"],
                    row["v1_active_depth0"],
                    row["v2_active_depth0"],
                    row["v1_tested_pairs_depth0"],
                    row["v2_tested_pairs_depth0"],
                    row["v1_commute_depth0"],
                    row["v2_commute_depth0"],
                    row["v1_sensitive_depth0"],
                    row["v2_sensitive_depth0"],
                ]
                for row in comparison_rows
            ],
        ),
        "",
        "## Batch and Evidence",
        "",
        *_markdown_table(
            ["program", "v1 candidates", "v2 candidates", "v1 certified", "v2 certified", "v1 dropped", "v2 dropped"],
            [
                [
                    row["program"],
                    row["v1_batch_candidates_depth0"],
                    row["v2_batch_candidates_depth0"],
                    row["v1_certified_batches_depth0"],
                    row["v2_certified_batches_depth0"],
                    row["v1_dropped_active_passes"],
                    row["v2_dropped_active_passes"],
                ]
                for row in comparison_rows
            ],
        ),
        "",
        "## Objective and Cost",
        "",
        *_markdown_table(
            ["program", "v1 final inst", "v2 final inst", "delta", "v1 states", "v2 states", "v1 time", "v2 time"],
            [
                [
                    row["program"],
                    row["v1_final_ir_inst"],
                    row["v2_final_ir_inst"],
                    row["v2_minus_v1_final_inst"],
                    row["v1_states"],
                    row["v2_states"],
                    row["v1_time_ms"],
                    row["v2_time_ms"],
                ]
                for row in comparison_rows
            ],
        ),
        "",
        "## Recommendation",
        "",
        *_recommendation_lines(run_rows, comparison_rows),
        "",
        "## Correctness Boundary",
        "",
        "Adding passes changes the explored search space. It does not change the rule that only certified/executable batches may be hard-folded.",
        "",
    ]
    path = out_dir / "v2_extension_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _recommendation_lines(run_rows: list[dict], comparison_rows: list[dict]) -> list[str]:
    if not comparison_rows:
        return ["- investigate v2 failures: no successful v1/v2 comparisons were produced."]
    total = len(comparison_rows)
    v2_successes = sum(1 for row in run_rows if row["passset"] == "v2" and row["stage"] == "optimize" and row["status"] == "success")
    dropped_programs = sum(1 for row in comparison_rows if _int(row.get("v2_dropped_active_passes")) > 0)
    invalid_programs = len({row["program"] for row in run_rows if row["passset"] == "v2" and _int(row.get("invalid_passes")) > 0})
    improved = sum(1 for row in comparison_rows if _float_or_none(row.get("v2_minus_v1_final_inst")) is not None and _float(row["v2_minus_v1_final_inst"]) < 0)
    v1_time = sum(_float(row.get("v1_time_ms")) for row in comparison_rows)
    v2_time = sum(_float(row.get("v2_time_ms")) for row in comparison_rows)
    cost_ratio = v2_time / v1_time if v1_time else 0.0
    lines = []
    if dropped_programs == 0 and v2_successes >= max(1, total // 2 + total % 2):
        lines.append("- keep v2 for larger study: most programs succeeded and v2 dropped active passes are zero.")
    if invalid_programs or dropped_programs:
        lines.append(f"- investigate v2 failures: invalid-pass programs={invalid_programs}, programs with dropped active passes={dropped_programs}.")
    lines.append(f"- v2 improves objective on {improved}/{total} programs.")
    lines.append(f"- v2 cost ratio is {cost_ratio:.2f}x of v1 by summed optimizer time.")
    return lines


def _v2_invalid(run_rows: list[dict], program: str) -> str:
    for row in run_rows:
        if row["program"] == program and row["passset"] == "v2" and row["stage"] == "audit":
            return row.get("invalid_passes", "")
    for row in run_rows:
        if row["program"] == program and row["passset"] == "v2" and row["stage"] == "optimize":
            return row.get("invalid_passes", "")
    return ""


def _program_success(run_rows: list[dict], program: str) -> bool:
    return all(
        row["status"] == "success"
        for row in run_rows
        if row["program"] == program and row["stage"] == "optimize"
    )


def _failure(program: str, input_path: Path, stage: str, passset: str, error: str) -> dict:
    return {"program": program, "input_path": str(input_path), "stage": stage, "passset": passset, "error_message": error}


def _program_plans(input_paths: list[Path], out_dir: Path) -> list[tuple[str, Path, Path]]:
    counts: dict[str, int] = {}
    plans = []
    for path in input_paths:
        count = counts.get(path.stem, 0)
        counts[path.stem] = count + 1
        program = path.stem if count == 0 else f"{path.stem}_{count}"
        plans.append((program, path, out_dir / program))
    return plans


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


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _value(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _escape_cell(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _numeric_delta(left: str, right: str) -> str:
    if _float_or_none(left) is None or _float_or_none(right) is None:
        return ""
    delta = _float(right) - _float(left)
    if delta.is_integer():
        return str(int(delta))
    return f"{delta:.4f}".rstrip("0").rstrip(".")


def _elapsed_ms(start: float) -> str:
    return f"{(time.perf_counter() - start) * 1000:.3f}"


def _string(value: object) -> str:
    return "" if value is None else str(value)


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: object) -> float | None:
    try:
        text = str(value)
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _remove_existing_output(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if not path.name or resolved == anchor or resolved == Path.cwd().resolve():
        raise RuntimeError(f"refusing to remove unsafe output path: {path}")
    shutil.rmtree(resolved)
