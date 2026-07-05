from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path


EXACT_REFERENCE_SELECTION_FIELDS = [
    "program",
    "input_path",
    "category",
    "selection_rank",
    "selection_reason",
    "budgeted_final_inst",
    "budgeted_states",
    "budgeted_time_ms",
    "max_local_reduction_log10",
    "order_sensitive_ratio",
    "dropped_active_passes",
]

EXACT_REFERENCE_RUN_FIELDS = [
    "program",
    "category",
    "input_path",
    "status",
    "exact_status",
    "output_dir",
    "final_ir_inst_count",
    "states_reached",
    "transitions",
    "time_ms",
    "error_message",
]

EXACT_REFERENCE_RESULT_FIELDS = [
    "program",
    "category",
    "budgeted_inst",
    "exact_inst",
    "gap_budgeted_to_exact",
    "budgeted_states",
    "exact_states",
    "state_reduction_pct",
    "budgeted_time_ms",
    "exact_time_ms",
    "time_reduction_pct",
    "greedy_inst",
    "random_best_inst",
    "config_order_inst",
    "exact_vs_greedy",
    "exact_vs_random",
    "budgeted_matched_exact",
]

EXACT_REFERENCE_FAILURE_FIELDS = [
    "program",
    "input_path",
    "stage",
    "error_message",
]


def select_and_run_exact_reference(
    budgeted_study_dir: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    objective: str,
    max_rounds: int,
    max_states: int,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    num_easy: int,
    num_medium: int,
    num_hard: int,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    budgeted_study_dir = Path(budgeted_study_dir)
    out_dir = Path(out_dir)
    passes_path = Path(passes_path)
    if out_dir.exists():
        if overwrite:
            _remove_existing_output(out_dir)
        elif any(out_dir.iterdir()):
            raise RuntimeError(f"output directory already exists: {out_dir}; use --overwrite to rerun")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(budgeted_study_dir)
    if not candidates:
        raise RuntimeError(f"no successful budgeted programs found under {budgeted_study_dir}")

    selected, selection_failures = _select_candidates(
        candidates,
        num_easy=num_easy,
        num_medium=num_medium,
        num_hard=num_hard,
        warn=warn,
    )
    selection_rows = [_selection_row(candidate, index + 1) for index, candidate in enumerate(selected)]
    run_rows: list[dict] = []
    result_rows: list[dict] = []
    failure_rows: list[dict] = selection_failures[:]

    for candidate in selected:
        start = time.perf_counter()
        exact_dir = out_dir / candidate["program"] / "exact_optimize"
        status = "success"
        exact_status = ""
        error_message = ""
        try:
            run_optimizer(
                Path(candidate["input_path"]),
                exact_dir,
                passes_path,
                mode="exact",
                objective=objective,
                max_rounds=max_rounds,
                beam_width=8,
                max_batches_per_state=20,
                max_states=max_states,
                batch_frontier_policy=None,
                validate_batches=validate_batches,
                allow_sampled_batches=False,
                jobs=jobs,
                timeout=timeout,
                max_pairs=max_pairs,
            )
            exact_status = _exact_status(exact_dir)
        except Exception as exc:
            status = "failed"
            exact_status = "failed"
            error_message = str(exc)
            failure_rows.append(_failure(candidate, "exact", error_message))
            if not continue_on_error:
                run_rows.append(_run_row(candidate, exact_dir, status, exact_status, start, error_message))
                _write_outputs(out_dir, selection_rows, run_rows, result_rows, failure_rows, budgeted_study_dir, passes_path, objective, max_rounds, max_states)
                raise
        finally:
            run_row = _run_row(candidate, exact_dir, status, exact_status, start, error_message)
            run_rows.append(run_row)
            result_rows.append(_result_row(candidate, run_row))

    summary_path = _write_outputs(out_dir, selection_rows, run_rows, result_rows, failure_rows, budgeted_study_dir, passes_path, objective, max_rounds, max_states)
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = sum(1 for row in run_rows if row.get("status") == "failed")
    return {
        "out_dir": str(out_dir),
        "selected_programs": len(selection_rows),
        "successes": successes,
        "failures": failures,
        "exact_reference_selection_csv": str(out_dir / "exact_reference_selection.csv"),
        "exact_reference_runs_csv": str(out_dir / "exact_reference_runs.csv"),
        "exact_reference_results_csv": str(out_dir / "exact_reference_results.csv"),
        "exact_reference_summary_md": str(summary_path),
        "failures_csv": str(out_dir / "failures.csv"),
    }


def run_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(input_path, out_dir, passes_path, **kwargs)


def _load_candidates(budgeted_dir: Path) -> list[dict]:
    runs = [row for row in _read_csv(budgeted_dir / "budgeted_study_runs.csv") if row.get("status") == "success"]
    reductions = {row.get("program", ""): row for row in _read_csv(budgeted_dir / "budgeted_study_reduction.csv")}
    evidence = {row.get("program", ""): row for row in _read_csv(budgeted_dir / "budgeted_study_evidence.csv")}
    methods = _method_index(_read_csv(budgeted_dir / "budgeted_study_methods.csv"))

    candidates = []
    for run in runs:
        program = run.get("program", "")
        reduction = reductions.get(program, {})
        ev = evidence.get(program, {})
        tested = _float(_value(reduction, "total_tested_pairs"))
        sensitive = _float(_value(reduction, "total_order_sensitive_pairs"))
        selected_batches = _float(_value(ev, "selected_path_batches"))
        strong_batches = _float(_value(ev, "selected_strong_certificates"))
        strong_ratio = 1.0 if selected_batches <= 0 else strong_batches / selected_batches
        dropped = _int(_value(ev, "dropped_active_passes", "total_dropped_active_passes", default=_value(reduction, "total_dropped_active_passes")))
        candidate = {
            "program": program,
            "input_path": run.get("input_path", ""),
            "budgeted_final_inst": _value(run, "final_ir_inst_count"),
            "budgeted_states": _value(run, "states_reached", default=_value(reduction, "states")),
            "budgeted_time_ms": _value(run, "time_ms"),
            "max_local_reduction_log10": _value(reduction, "max_local_reduction_log10"),
            "order_sensitive_ratio": sensitive / tested if tested > 0 else 0.0,
            "dropped_active_passes": str(dropped),
            "strong_certificate_ratio": strong_ratio,
            "pipeline_exists": _pipeline_exists(run, budgeted_dir),
            "greedy_inst": _method_inst(methods, program, "greedy_single_pass"),
            "random_best_inst": _method_inst(methods, program, "random_single_pass_best"),
            "config_order_inst": _method_inst(methods, program, "config_order_once"),
        }
        candidate["batch_ties_or_beats_greedy"] = _ties_or_beats(candidate["budgeted_final_inst"], candidate["greedy_inst"])
        candidate["batch_loses_to_greedy"] = _loses_to(candidate["budgeted_final_inst"], candidate["greedy_inst"])
        candidate["batch_loses_to_random"] = _loses_to(candidate["budgeted_final_inst"], candidate["random_best_inst"])
        candidates.append(candidate)
    return candidates


def _select_candidates(
    candidates: list[dict],
    *,
    num_easy: int,
    num_medium: int,
    num_hard: int,
    warn=print,
) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    failures: list[dict] = []
    used: set[str] = set()

    def take(category: str, ranked: list[dict], count: int) -> None:
        before = len(selected)
        for candidate in ranked:
            if len(selected) - before >= count:
                break
            if candidate["program"] in used:
                continue
            copy = dict(candidate)
            copy["category"] = category
            copy["selection_reason"] = _selection_reason(category, copy)
            selected.append(copy)
            used.add(candidate["program"])
        if len(selected) - before < count:
            message = f"not enough candidates for {category}: requested {count}, selected {len(selected) - before}"
            warn(f"warning: {message}")
            failures.append({"program": "", "input_path": "", "stage": "selection", "error_message": message})

    easy_ranked = sorted(
        [c for c in candidates if _int(c["dropped_active_passes"]) == 0 and c["strong_certificate_ratio"] >= 0.99 and c["pipeline_exists"]],
        key=lambda c: (_float(c["budgeted_states"]), _float(c["budgeted_time_ms"]), -_float(c["max_local_reduction_log10"]), 0 if c["batch_ties_or_beats_greedy"] else 1, c["program"]),
    )
    hard_ranked = sorted(
        candidates,
        key=lambda c: (
            0 if _int(c["dropped_active_passes"]) == 0 else 1,
            -(1 if c["batch_loses_to_greedy"] or c["batch_loses_to_random"] else 0),
            -_float(c["budgeted_states"]),
            -_float(c["budgeted_time_ms"]),
            -_float(c["order_sensitive_ratio"]),
            -_float(c["max_local_reduction_log10"]),
            c["program"],
        ),
    )
    medium_ranked = _medium_ranked(candidates)

    take("easy", easy_ranked, max(0, num_easy))
    take("medium", medium_ranked, max(0, num_medium))
    take("hard", hard_ranked, max(0, num_hard))

    requested_total = max(0, num_easy) + max(0, num_medium) + max(0, num_hard)
    if len(selected) < min(requested_total, len(candidates)):
        for candidate in sorted(candidates, key=lambda c: c["program"]):
            if len(selected) >= min(requested_total, len(candidates)):
                break
            if candidate["program"] in used:
                continue
            copy = dict(candidate)
            copy["category"] = "medium"
            copy["selection_reason"] = "fill from remaining successful programs"
            selected.append(copy)
            used.add(candidate["program"])
    return selected, failures


def _medium_ranked(candidates: list[dict]) -> list[dict]:
    states_values = [_float(c["budgeted_states"]) for c in candidates]
    time_values = [_float(c["budgeted_time_ms"]) for c in candidates]
    reduction_values = [_float(c["max_local_reduction_log10"]) for c in candidates]
    median_states = _median(states_values)
    median_time = _median(time_values)
    median_reduction = _median(reduction_values)
    max_states = max(states_values or [1]) or 1
    max_time = max(time_values or [1]) or 1
    max_reduction = max(reduction_values or [1]) or 1
    return sorted(
        candidates,
        key=lambda c: (
            0 if _int(c["dropped_active_passes"]) == 0 else 1,
            abs(_float(c["budgeted_states"]) - median_states) / max_states
            + abs(_float(c["budgeted_time_ms"]) - median_time) / max_time
            + abs(_float(c["max_local_reduction_log10"]) - median_reduction) / max_reduction,
            c["program"],
        ),
    )


def _write_outputs(
    out_dir: Path,
    selection_rows: list[dict],
    run_rows: list[dict],
    result_rows: list[dict],
    failure_rows: list[dict],
    budgeted_study_dir: Path,
    passes_path: Path,
    objective: str,
    max_rounds: int,
    max_states: int,
) -> Path:
    _write_csv(out_dir / "exact_reference_selection.csv", EXACT_REFERENCE_SELECTION_FIELDS, selection_rows)
    _write_csv(out_dir / "exact_reference_runs.csv", EXACT_REFERENCE_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "exact_reference_results.csv", EXACT_REFERENCE_RESULT_FIELDS, result_rows)
    _write_csv(out_dir / "failures.csv", EXACT_REFERENCE_FAILURE_FIELDS, failure_rows)
    return _write_summary(out_dir, selection_rows, run_rows, result_rows, failure_rows, budgeted_study_dir, passes_path, objective, max_rounds, max_states)


def _write_summary(
    out_dir: Path,
    selection_rows: list[dict],
    run_rows: list[dict],
    result_rows: list[dict],
    failure_rows: list[dict],
    budgeted_study_dir: Path,
    passes_path: Path,
    objective: str,
    max_rounds: int,
    max_states: int,
) -> Path:
    lines = [
        "# Exact Reference Study",
        "",
        "## Purpose",
        "",
        "Exact mode is used as a reference on selected programs only. It is not run for every benchmark because exact expansion can be expensive.",
        "",
        f"- budgeted study: {budgeted_study_dir}",
        f"- pass set: {passes_path}",
        f"- objective: {objective}",
        f"- max_rounds: {max_rounds}",
        f"- max_states: {max_states}",
        "",
        "## Selection",
        "",
        *_markdown_table(
            ["program", "category", "selection reason", "budgeted states", "budgeted time", "reduction log10"],
            [[row["program"], row["category"], row["selection_reason"], row["budgeted_states"], row["budgeted_time_ms"], row["max_local_reduction_log10"]] for row in selection_rows],
        ),
        "",
        "## Exact Results",
        "",
        *_markdown_table(
            ["program", "category", "exact status", "exact final inst", "exact states", "exact transitions", "exact time"],
            [[row["program"], row["category"], row["exact_status"], row["final_ir_inst_count"], row["states_reached"], row["transitions"], row["time_ms"]] for row in run_rows],
        ),
        "",
        "## Budgeted vs Exact",
        "",
        *_markdown_table(
            ["program", "budgeted", "exact", "gap", "state reduction", "time reduction"],
            [[row["program"], row["budgeted_inst"], row["exact_inst"], row["gap_budgeted_to_exact"], row["state_reduction_pct"], row["time_reduction_pct"]] for row in result_rows],
        ),
        "",
        "## Category Interpretation",
        "",
        "### Easy cases",
        "",
        *_category_lines(selection_rows, result_rows, "easy"),
        "",
        "### Medium cases",
        "",
        *_category_lines(selection_rows, result_rows, "medium"),
        "",
        "### Hard cases",
        "",
        *_category_lines(selection_rows, result_rows, "hard", hard=True),
        "",
        "## Correctness Boundary",
        "",
        "Exact reference is complete only within the current certified batch-state graph, current pass set, compiler version, target, normalization, max_rounds, and max_states.",
        "",
    ]
    if failure_rows:
        lines.extend(["## Warnings / Failures", "", *_failure_lines(failure_rows), ""])
    path = out_dir / "exact_reference_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _selection_row(candidate: dict, rank: int) -> dict:
    return {
        "program": candidate["program"],
        "input_path": candidate["input_path"],
        "category": candidate["category"],
        "selection_rank": str(rank),
        "selection_reason": candidate["selection_reason"],
        "budgeted_final_inst": candidate["budgeted_final_inst"],
        "budgeted_states": candidate["budgeted_states"],
        "budgeted_time_ms": candidate["budgeted_time_ms"],
        "max_local_reduction_log10": candidate["max_local_reduction_log10"],
        "order_sensitive_ratio": _format_float(candidate["order_sensitive_ratio"]),
        "dropped_active_passes": candidate["dropped_active_passes"],
    }


def _run_row(candidate: dict, exact_dir: Path, status: str, exact_status: str, start: float, error_message: str) -> dict:
    states = _read_csv(exact_dir / "states.csv")
    transitions = _read_csv(exact_dir / "batch_state_transitions.csv")
    chosen = _first_row(exact_dir / "chosen_path_summary.csv")
    timing = _first_row(exact_dir / "optimizer_timing.csv")
    return {
        "program": candidate["program"],
        "category": candidate["category"],
        "input_path": candidate["input_path"],
        "status": status,
        "exact_status": exact_status or _exact_status(exact_dir),
        "output_dir": str(exact_dir),
        "final_ir_inst_count": _value(chosen, "final_ir_inst_count", "final_objective"),
        "states_reached": str(len(states)) if states else "",
        "transitions": str(len(transitions)) if transitions else "",
        "time_ms": _value(timing, "optimizer_total_time_ms", default=_format_float((time.perf_counter() - start) * 1000)),
        "error_message": error_message,
    }


def _result_row(candidate: dict, run_row: dict) -> dict:
    budgeted_inst = candidate["budgeted_final_inst"]
    exact_inst = run_row.get("final_ir_inst_count", "")
    budgeted_states = candidate["budgeted_states"]
    exact_states = run_row.get("states_reached", "")
    budgeted_time = candidate["budgeted_time_ms"]
    exact_time = run_row.get("time_ms", "")
    return {
        "program": candidate["program"],
        "category": candidate["category"],
        "budgeted_inst": budgeted_inst,
        "exact_inst": exact_inst,
        "gap_budgeted_to_exact": _delta(budgeted_inst, exact_inst),
        "budgeted_states": budgeted_states,
        "exact_states": exact_states,
        "state_reduction_pct": _reduction_pct(exact_states, budgeted_states),
        "budgeted_time_ms": budgeted_time,
        "exact_time_ms": exact_time,
        "time_reduction_pct": _reduction_pct(exact_time, budgeted_time),
        "greedy_inst": candidate["greedy_inst"],
        "random_best_inst": candidate["random_best_inst"],
        "config_order_inst": candidate["config_order_inst"],
        "exact_vs_greedy": _compare_lower_is_better(exact_inst, candidate["greedy_inst"]),
        "exact_vs_random": _compare_lower_is_better(exact_inst, candidate["random_best_inst"]),
        "budgeted_matched_exact": _bool_text(_float_or_none(budgeted_inst) is not None and _float_or_none(budgeted_inst) == _float_or_none(exact_inst)),
    }


def _selection_reason(category: str, candidate: dict) -> str:
    base = (
        f"states={candidate['budgeted_states']}, time_ms={candidate['budgeted_time_ms']}, "
        f"max_reduction_log10={candidate['max_local_reduction_log10']}, "
        f"order_sensitive_ratio={_format_float(candidate['order_sensitive_ratio'])}"
    )
    if category == "hard" and (candidate["batch_loses_to_greedy"] or candidate["batch_loses_to_random"]):
        return base + ", budgeted loses to a single-pass baseline"
    if category == "easy" and candidate["batch_ties_or_beats_greedy"]:
        return base + ", low-cost and batch ties/beats greedy"
    if category == "medium":
        return base + ", near median budgeted cost/reduction"
    return base


def _category_lines(selection_rows: list[dict], result_rows: list[dict], category: str, hard: bool = False) -> list[str]:
    programs = [row["program"] for row in selection_rows if row.get("category") == category]
    if not programs:
        return ["- No selected programs in this category."]
    lines = []
    by_program = {row["program"]: row for row in result_rows}
    for program in programs:
        result = by_program.get(program, {})
        text = f"- {program}: budgeted={result.get('budgeted_inst', '')}, exact={result.get('exact_inst', '')}, gap={result.get('gap_budgeted_to_exact', '')}."
        if hard:
            text += " Hard cases may need wider beam or exact expansion when budgeted search misses the reference objective."
        lines.append(text)
    return lines


def _failure_lines(rows: list[dict]) -> list[str]:
    return [f"- {row.get('stage', '')}: {row.get('program', '')} {row.get('error_message', '')}".strip() for row in rows]


def _failure(candidate: dict, stage: str, message: str) -> dict:
    return {"program": candidate.get("program", ""), "input_path": candidate.get("input_path", ""), "stage": stage, "error_message": message}


def _method_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row.get("program", ""), row.get("method", "")): row for row in rows}


def _method_inst(index: dict[tuple[str, str], dict], program: str, method: str) -> str:
    row = index.get((program, method), {})
    if row.get("status") and row.get("status") != "success":
        return ""
    return row.get("final_ir_inst_count", "")


def _pipeline_exists(run: dict, budgeted_dir: Path) -> bool:
    if _int(run.get("pipeline_length")) > 0:
        return True
    output_dir = run.get("output_dir", "")
    if output_dir and _nonempty_file(Path(output_dir) / "optimize" / "optimized_pipeline.txt"):
        return True
    program = run.get("program", "")
    return bool(program and _nonempty_file(budgeted_dir / program / "optimize" / "optimized_pipeline.txt"))


def _nonempty_file(path: Path) -> bool:
    return path.exists() and bool(path.read_text(encoding="utf-8", errors="replace").strip())


def _ties_or_beats(left: str, right: str) -> bool:
    lval = _float_or_none(left)
    rval = _float_or_none(right)
    return True if rval is None else lval is not None and lval <= rval


def _loses_to(left: str, right: str) -> bool:
    lval = _float_or_none(left)
    rval = _float_or_none(right)
    return lval is not None and rval is not None and lval > rval


def _compare_lower_is_better(left: str, right: str) -> str:
    lval = _float_or_none(left)
    rval = _float_or_none(right)
    if lval is None or rval is None:
        return ""
    if lval < rval:
        return "win"
    if lval == rval:
        return "tie"
    return "loss"


def _delta(left: str, right: str) -> str:
    lval = _float_or_none(left)
    rval = _float_or_none(right)
    if lval is None or rval is None:
        return ""
    return _format_float(lval - rval)


def _reduction_pct(reference: str, reduced: str) -> str:
    ref = _float_or_none(reference)
    red = _float_or_none(reduced)
    if ref is None or red is None or ref == 0:
        return ""
    return _format_float(((ref - red) / ref) * 100.0)


def _exact_status(exact_dir: Path) -> str:
    path = exact_dir / "exact_status.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


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


def _value(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


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


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except (TypeError, ValueError):
        return 0


def _remove_existing_output(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if not path.name or resolved == anchor or resolved == Path.cwd().resolve():
        raise RuntimeError(f"refusing to remove unsafe output path: {path}")
    shutil.rmtree(resolved)
