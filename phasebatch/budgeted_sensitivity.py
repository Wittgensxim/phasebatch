from __future__ import annotations

import csv
import glob
import re
import shutil
import time
from pathlib import Path


BUDGETED_SENSITIVITY_RUN_FIELDS = [
    "program",
    "input_path",
    "beam_width",
    "max_states",
    "max_rounds",
    "max_batches_per_state",
    "policy",
    "status",
    "output_dir",
    "final_ir_inst_count",
    "root_ir_inst_count",
    "ir_inst_delta",
    "ir_inst_reduction_pct",
    "states_reached",
    "transitions",
    "duplicate_states",
    "pipeline_length",
    "time_ms",
    "stop_reason",
    "error_message",
]

BUDGETED_SENSITIVITY_RESULT_FIELDS = [
    "program",
    "beam_width",
    "max_states",
    "final_ir_inst_count",
    "root_ir_inst_count",
    "ir_inst_delta",
    "ir_inst_reduction_pct",
    "states_reached",
    "transitions",
    "pipeline_length",
    "time_ms",
    "exact_r4_inst",
    "greedy_inst",
    "random_best_inst",
    "config_order_inst",
    "gap_to_exact",
    "gap_to_greedy",
    "gap_to_random",
    "gap_to_config_order",
    "matches_exact",
    "beats_greedy",
    "ties_greedy",
    "loses_to_greedy",
    "beats_random",
    "ties_random",
    "loses_to_random",
]

BUDGETED_SENSITIVITY_BEST_FIELDS = [
    "program",
    "best_beam_width",
    "best_max_states",
    "best_final_ir_inst_count",
    "exact_r4_inst",
    "greedy_inst",
    "random_best_inst",
    "config_order_inst",
    "gap_to_exact",
    "states_reached",
    "time_ms",
    "pipeline_length",
    "matched_exact",
    "beat_greedy",
    "beat_random",
    "output_dir",
]

BUDGETED_SENSITIVITY_FAILURE_FIELDS = [
    "program",
    "beam_width",
    "max_states",
    "stage",
    "error_message",
]


def run_budgeted_sensitivity(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    objective: str,
    max_rounds: int,
    beam_widths: list[int],
    max_states_list: list[int],
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    exact_reference: Path | None = None,
    summarize_components: bool = False,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    out_dir = Path(out_dir)
    passes_path = Path(passes_path)
    beam_widths = _positive_unique(beam_widths, "beam_widths")
    max_states_list = _positive_unique(max_states_list, "max_states_list")

    if out_dir.exists():
        if overwrite:
            _remove_existing_output(out_dir)
        elif any(out_dir.iterdir()):
            raise RuntimeError(f"output directory already exists: {out_dir}; use --overwrite to rerun")
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_inputs, input_failures = _expand_input_plans(inputs, warn)
    reference, reference_available = _load_reference(exact_reference, warn)

    run_rows: list[dict] = []
    result_rows: list[dict] = []
    failure_rows: list[dict] = []

    for failure in input_failures:
        failure_rows.append(
            {
                "program": failure["program"],
                "beam_width": "",
                "max_states": "",
                "stage": "input",
                "error_message": failure["error_message"],
            }
        )
        run_rows.append(
            _run_row(
                program=failure["program"],
                input_path=failure["input_path"],
                beam_width="",
                max_states="",
                max_rounds=max_rounds,
                max_batches_per_state=max_batches_per_state,
                policy=batch_frontier_policy or "",
                status="failed",
                output_dir="",
                metrics={},
                time_ms="0.000",
                stop_reason="input_failed",
                error_message=failure["error_message"],
            )
        )
        if not continue_on_error:
            _write_outputs(out_dir, run_rows, result_rows, [], failure_rows, reference, reference_available, objective, max_rounds, beam_widths, max_states_list, batch_frontier_policy)
            raise RuntimeError(failure["error_message"])

    for program, input_path in valid_inputs:
        for beam_width in beam_widths:
            for max_states in max_states_list:
                run_dir = out_dir / program / f"beam{beam_width}_states{max_states}"
                start = time.perf_counter()
                try:
                    result = run_optimizer(
                        input_path,
                        run_dir,
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
                    metrics = _collect_run_metrics(run_dir, result, start)
                    row = _run_row(
                        program=program,
                        input_path=str(input_path),
                        beam_width=str(beam_width),
                        max_states=str(max_states),
                        max_rounds=max_rounds,
                        max_batches_per_state=max_batches_per_state,
                        policy=batch_frontier_policy or "",
                        status="success",
                        output_dir=str(run_dir),
                        metrics=metrics,
                        time_ms=metrics.get("time_ms", ""),
                        stop_reason=metrics.get("stop_reason", ""),
                        error_message="",
                    )
                    run_rows.append(row)
                    result_rows.append(_result_row(row, reference.get(_program_key(program), {})))
                except Exception as exc:
                    message = str(exc)
                    row = _run_row(
                        program=program,
                        input_path=str(input_path),
                        beam_width=str(beam_width),
                        max_states=str(max_states),
                        max_rounds=max_rounds,
                        max_batches_per_state=max_batches_per_state,
                        policy=batch_frontier_policy or "",
                        status="failed",
                        output_dir=str(run_dir),
                        metrics={},
                        time_ms=f"{_elapsed_ms(start):.3f}",
                        stop_reason="failed",
                        error_message=message,
                    )
                    run_rows.append(row)
                    failure_rows.append(
                        {
                            "program": program,
                            "beam_width": str(beam_width),
                            "max_states": str(max_states),
                            "stage": "optimize",
                            "error_message": message,
                        }
                    )
                    if not continue_on_error:
                        _write_outputs(out_dir, run_rows, result_rows, _best_rows(result_rows), failure_rows, reference, reference_available, objective, max_rounds, beam_widths, max_states_list, batch_frontier_policy)
                        raise

    best_rows = _best_rows(result_rows)
    summary_path = _write_outputs(
        out_dir,
        run_rows,
        result_rows,
        best_rows,
        failure_rows,
        reference,
        reference_available,
        objective,
        max_rounds,
        beam_widths,
        max_states_list,
        batch_frontier_policy,
    )
    component_result = _try_component_summary(_successful_output_dirs(run_rows), out_dir / "components") if summarize_components else {}
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = sum(1 for row in run_rows if row.get("status") == "failed")
    result = {
        "out_dir": str(out_dir),
        "attempted_runs": len(run_rows),
        "successes": successes,
        "failures": failures,
        "budgeted_sensitivity_runs_csv": str(out_dir / "budgeted_sensitivity_runs.csv"),
        "budgeted_sensitivity_results_csv": str(out_dir / "budgeted_sensitivity_results.csv"),
        "budgeted_sensitivity_best_csv": str(out_dir / "budgeted_sensitivity_best.csv"),
        "budgeted_sensitivity_summary_md": str(summary_path),
        "failures_csv": str(out_dir / "failures.csv"),
    }
    if component_result:
        result["component_summary_md"] = component_result.get("component_summary_md", "")
    return result


def run_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(input_path, out_dir, passes_path, **kwargs)


def run_component_summary(run_dirs: list[Path], out_dir: Path) -> dict:
    from .component_summary import summarize_components

    return summarize_components(run_dirs=run_dirs, out_dir=out_dir)


def _successful_output_dirs(run_rows: list[dict]) -> list[Path]:
    return [Path(row.get("output_dir", "")) for row in run_rows if row.get("status") == "success" and row.get("output_dir")]


def _try_component_summary(run_dirs: list[Path], out_dir: Path) -> dict:
    if not run_dirs:
        return {}
    try:
        return run_component_summary(run_dirs, out_dir)
    except Exception:
        return {}


def _collect_run_metrics(run_dir: Path, result: dict, start: float) -> dict:
    chosen = _first_row(run_dir / "chosen_path_summary.csv")
    states = _read_csv(run_dir / "states.csv")
    transitions = _read_csv(run_dir / "batch_state_transitions.csv")
    timing = _first_row(run_dir / "optimizer_timing.csv")
    final_inst = _value(chosen, "final_ir_inst_count", "final_objective")
    root_inst = _value(chosen, "root_ir_inst_count")
    delta = _value(chosen, "total_ir_inst_delta", "ir_inst_delta")
    if delta == "" and _is_number(final_inst) and _is_number(root_inst):
        delta = str(_int(final_inst) - _int(root_inst))
    time_ms = _value(timing, "optimizer_total_time_ms", default=f"{_elapsed_ms(start):.3f}")
    return {
        "final_ir_inst_count": final_inst,
        "root_ir_inst_count": root_inst,
        "ir_inst_delta": delta,
        "ir_inst_reduction_pct": _value(chosen, "ir_inst_reduction_pct", "reduction_pct"),
        "states_reached": str(result.get("states") or len(states)),
        "transitions": str(result.get("batch_transitions") or len(transitions)),
        "duplicate_states": str(sum(1 for row in states if _is_true(row.get("is_duplicate")))),
        "pipeline_length": str(_pipeline_length(run_dir)),
        "time_ms": time_ms,
        "stop_reason": _stop_reason(run_dir),
    }


def _run_row(
    *,
    program: str,
    input_path: str,
    beam_width: str,
    max_states: str,
    max_rounds: int,
    max_batches_per_state: int,
    policy: str,
    status: str,
    output_dir: str,
    metrics: dict,
    time_ms: str,
    stop_reason: str,
    error_message: str,
) -> dict:
    return {
        "program": program,
        "input_path": input_path,
        "beam_width": beam_width,
        "max_states": max_states,
        "max_rounds": str(max_rounds),
        "max_batches_per_state": str(max_batches_per_state),
        "policy": policy,
        "status": status,
        "output_dir": output_dir,
        "final_ir_inst_count": metrics.get("final_ir_inst_count", ""),
        "root_ir_inst_count": metrics.get("root_ir_inst_count", ""),
        "ir_inst_delta": metrics.get("ir_inst_delta", ""),
        "ir_inst_reduction_pct": metrics.get("ir_inst_reduction_pct", ""),
        "states_reached": metrics.get("states_reached", ""),
        "transitions": metrics.get("transitions", ""),
        "duplicate_states": metrics.get("duplicate_states", ""),
        "pipeline_length": metrics.get("pipeline_length", ""),
        "time_ms": time_ms,
        "stop_reason": stop_reason,
        "error_message": error_message,
    }


def _result_row(run_row: dict, reference: dict) -> dict:
    final_inst = run_row.get("final_ir_inst_count", "")
    row = {
        "program": run_row.get("program", ""),
        "beam_width": run_row.get("beam_width", ""),
        "max_states": run_row.get("max_states", ""),
        "final_ir_inst_count": final_inst,
        "root_ir_inst_count": run_row.get("root_ir_inst_count", ""),
        "ir_inst_delta": run_row.get("ir_inst_delta", ""),
        "ir_inst_reduction_pct": run_row.get("ir_inst_reduction_pct", ""),
        "states_reached": run_row.get("states_reached", ""),
        "transitions": run_row.get("transitions", ""),
        "pipeline_length": run_row.get("pipeline_length", ""),
        "time_ms": run_row.get("time_ms", ""),
        "exact_r4_inst": reference.get("exact_r4_inst", ""),
        "greedy_inst": reference.get("greedy_inst", ""),
        "random_best_inst": reference.get("random_best_inst", ""),
        "config_order_inst": reference.get("config_order_inst", ""),
        "output_dir": run_row.get("output_dir", ""),
        "exact_states": reference.get("exact_states", ""),
        "exact_time_ms": reference.get("exact_time_ms", ""),
    }
    row.update(
        {
            "gap_to_exact": _gap(final_inst, row["exact_r4_inst"]),
            "gap_to_greedy": _gap(final_inst, row["greedy_inst"]),
            "gap_to_random": _gap(final_inst, row["random_best_inst"]),
            "gap_to_config_order": _gap(final_inst, row["config_order_inst"]),
            "matches_exact": _bool(_cmp(final_inst, row["exact_r4_inst"]) == 0),
            "beats_greedy": _bool(_cmp(final_inst, row["greedy_inst"]) is not None and _cmp(final_inst, row["greedy_inst"]) < 0),
            "ties_greedy": _bool(_cmp(final_inst, row["greedy_inst"]) == 0),
            "loses_to_greedy": _bool(_cmp(final_inst, row["greedy_inst"]) is not None and _cmp(final_inst, row["greedy_inst"]) > 0),
            "beats_random": _bool(_cmp(final_inst, row["random_best_inst"]) is not None and _cmp(final_inst, row["random_best_inst"]) < 0),
            "ties_random": _bool(_cmp(final_inst, row["random_best_inst"]) == 0),
            "loses_to_random": _bool(_cmp(final_inst, row["random_best_inst"]) is not None and _cmp(final_inst, row["random_best_inst"]) > 0),
        }
    )
    return row


def _best_rows(result_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in result_rows:
        grouped.setdefault(row.get("program", ""), []).append(row)
    best_rows = []
    for program in sorted(grouped):
        best = min(
            grouped[program],
            key=lambda row: (
                _int(row.get("final_ir_inst_count"), 10**12),
                _int(row.get("states_reached"), 10**12),
                _float(row.get("time_ms"), 10**12),
                _int(row.get("beam_width"), 10**12),
                _int(row.get("max_states"), 10**12),
            ),
        )
        best_rows.append(
            {
                "program": program,
                "best_beam_width": best.get("beam_width", ""),
                "best_max_states": best.get("max_states", ""),
                "best_final_ir_inst_count": best.get("final_ir_inst_count", ""),
                "exact_r4_inst": best.get("exact_r4_inst", ""),
                "greedy_inst": best.get("greedy_inst", ""),
                "random_best_inst": best.get("random_best_inst", ""),
                "config_order_inst": best.get("config_order_inst", ""),
                "gap_to_exact": best.get("gap_to_exact", ""),
                "states_reached": best.get("states_reached", ""),
                "time_ms": best.get("time_ms", ""),
                "pipeline_length": best.get("pipeline_length", ""),
                "matched_exact": best.get("matches_exact", ""),
                "beat_greedy": best.get("beats_greedy", ""),
                "beat_random": best.get("beats_random", ""),
                "output_dir": best.get("output_dir", ""),
                "exact_states": best.get("exact_states", ""),
                "exact_time_ms": best.get("exact_time_ms", ""),
                "_gap_to_greedy": best.get("gap_to_greedy", ""),
                "_gap_to_random": best.get("gap_to_random", ""),
                "_gap_to_config_order": best.get("gap_to_config_order", ""),
            }
        )
    return best_rows


def _load_reference(path: Path | None, warn=print) -> tuple[dict[str, dict], bool]:
    if path is None:
        return {}, False
    path = Path(path)
    if not path.exists():
        warn(f"warning: exact reference not found: {path}")
        return {}, False
    rows = _read_csv(path)
    reference: dict[str, dict] = {}
    for row in rows:
        lowered = {_normalize_column(key): value for key, value in row.items()}
        program = lowered.get("program", "")
        if not program:
            continue
        reference[_program_key(program)] = {
            "program": program,
            "exact_r4_inst": _lookup(lowered, ["exact_r4_inst", "exact_r4", "exact", "batch", "batch_optimizer", "final_ir_inst_count"]),
            "greedy_inst": _lookup(lowered, ["greedy_inst", "greedy", "greedy_single_pass"]),
            "random_best_inst": _lookup(lowered, ["random_best_inst", "random_best", "random_single_pass_best", "random"]),
            "config_order_inst": _lookup(lowered, ["config_order_inst", "config_order", "config_once", "config_order_once"]),
            "exact_states": _lookup(lowered, ["exact_states", "batch_states", "states_reached", "states"]),
            "exact_time_ms": _lookup(lowered, ["exact_time_ms", "batch_time_ms", "time_ms"]),
        }
    return reference, bool(reference)


def _write_outputs(
    out_dir: Path,
    run_rows: list[dict],
    result_rows: list[dict],
    best_rows: list[dict],
    failure_rows: list[dict],
    reference: dict[str, dict],
    reference_available: bool,
    objective: str,
    max_rounds: int,
    beam_widths: list[int],
    max_states_list: list[int],
    policy: str | None,
) -> Path:
    _write_csv(out_dir / "budgeted_sensitivity_runs.csv", BUDGETED_SENSITIVITY_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "budgeted_sensitivity_results.csv", BUDGETED_SENSITIVITY_RESULT_FIELDS, result_rows)
    _write_csv(out_dir / "budgeted_sensitivity_best.csv", BUDGETED_SENSITIVITY_BEST_FIELDS, best_rows)
    _write_csv(out_dir / "failures.csv", BUDGETED_SENSITIVITY_FAILURE_FIELDS, failure_rows)
    return _write_summary(out_dir, run_rows, result_rows, best_rows, failure_rows, reference, reference_available, objective, max_rounds, beam_widths, max_states_list, policy)


def _write_summary(
    out_dir: Path,
    run_rows: list[dict],
    result_rows: list[dict],
    best_rows: list[dict],
    failure_rows: list[dict],
    reference: dict[str, dict],
    reference_available: bool,
    objective: str,
    max_rounds: int,
    beam_widths: list[int],
    max_states_list: list[int],
    policy: str | None,
) -> Path:
    programs = sorted({row.get("program", "") for row in run_rows if row.get("program")})
    successful_programs = sorted({row.get("program", "") for row in run_rows if row.get("status") == "success"})
    failed_programs = sorted({row.get("program", "") for row in run_rows if row.get("status") == "failed"} - set(successful_programs))
    lines = [
        "# Budgeted Sensitivity Summary",
        "",
        "## Overall",
        "",
        f"- programs attempted: {len(programs)}",
        f"- successful programs: {len(successful_programs)}",
        f"- failed programs: {len(failed_programs)}",
        f"- max_rounds: {max_rounds}",
        f"- beam_widths: {' '.join(str(value) for value in beam_widths)}",
        f"- max_states_list: {' '.join(str(value) for value in max_states_list)}",
        f"- policy: {policy or ''}",
        f"- objective: {objective}",
        "",
        "## Exact Reference",
        "",
    ]
    if reference_available:
        lines.extend(
            _markdown_table(
                ["program", "exact r4", "greedy", "random best", "config order"],
                [
                    [
                        row.get("program", ""),
                        row.get("exact_r4_inst", ""),
                        row.get("greedy_inst", ""),
                        row.get("random_best_inst", ""),
                        row.get("config_order_inst", ""),
                    ]
                    for row in reference.values()
                ],
            )
        )
        lines.extend(["", "Exact r4 is used as a quality reference, not as a correctness proof."])
    else:
        lines.append("Exact reference was not available.")
    lines.extend(
        [
            "",
            "## Results by Program",
            "",
            *_markdown_table(
                ["program", "exact r4", "best budgeted", "beam", "max states", "gap to exact", "greedy", "random", "states", "time ms"],
                [
                    [
                        row.get("program", ""),
                        row.get("exact_r4_inst", ""),
                        row.get("best_final_ir_inst_count", ""),
                        row.get("best_beam_width", ""),
                        row.get("best_max_states", ""),
                        row.get("gap_to_exact", ""),
                        row.get("greedy_inst", ""),
                        row.get("random_best_inst", ""),
                        row.get("states_reached", ""),
                        row.get("time_ms", ""),
                    ]
                    for row in best_rows
                ],
            ),
            "",
            "## Budgeted vs Exact",
            "",
            *_budgeted_vs_exact_lines(best_rows),
            "",
            "## Budgeted vs Baselines",
            "",
            *_budgeted_vs_baseline_lines(best_rows),
            "",
            "## Sensitivity Table",
            "",
            *_markdown_table(
                ["program", "beam", "max_states", "final inst", "gap exact", "states", "time"],
                [
                    [
                        row.get("program", ""),
                        row.get("beam_width", ""),
                        row.get("max_states", ""),
                        row.get("final_ir_inst_count", ""),
                        row.get("gap_to_exact", ""),
                        row.get("states_reached", ""),
                        row.get("time_ms", ""),
                    ]
                    for row in result_rows
                ],
            ),
            "",
            "## Key Observations",
            "",
            *_observation_lines(result_rows, best_rows),
            "",
            "## Correctness Boundary",
            "",
            "Budgeted mode changes search coverage, not batch correctness. A batch is executable only when its correctness classification allows execution. Objective values are used for path selection and evaluation, not as commutation proof.",
            "",
        ]
    )
    if failure_rows:
        lines.extend(
            [
                "## Failures",
                "",
                *_markdown_table(
                    ["program", "beam", "max states", "stage", "error"],
                    [
                        [
                            row.get("program", ""),
                            row.get("beam_width", ""),
                            row.get("max_states", ""),
                            row.get("stage", ""),
                            row.get("error_message", ""),
                        ]
                        for row in failure_rows
                    ],
                ),
                "",
            ]
        )
    path = out_dir / "budgeted_sensitivity_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _budgeted_vs_exact_lines(best_rows: list[dict]) -> list[str]:
    with_exact = [row for row in best_rows if _is_number(row.get("gap_to_exact"))]
    matches = sum(1 for row in with_exact if _int(row.get("gap_to_exact")) == 0)
    gaps = [_float(row.get("gap_to_exact")) for row in with_exact]
    state_reductions = []
    time_reductions = []
    for row in with_exact:
        if _is_number(row.get("exact_states")) and _is_number(row.get("states_reached")) and _float(row.get("exact_states")) > 0:
            state_reductions.append((_float(row.get("exact_states")) - _float(row.get("states_reached"))) / _float(row.get("exact_states")))
        if _is_number(row.get("exact_time_ms")) and _is_number(row.get("time_ms")) and _float(row.get("exact_time_ms")) > 0:
            time_reductions.append((_float(row.get("exact_time_ms")) - _float(row.get("time_ms"))) / _float(row.get("exact_time_ms")))
    return [
        f"- programs matching exact: {matches}",
        f"- average gap to exact: {_avg(gaps)}",
        f"- max gap to exact: {_fmt_float(max(gaps) if gaps else 0.0)}",
        f"- average state reduction relative to exact: {_pct(_mean(state_reductions)) if state_reductions else 'N/A'}",
        f"- average time reduction relative to exact: {_pct(_mean(time_reductions)) if time_reductions else 'N/A'}",
    ]


def _budgeted_vs_baseline_lines(best_rows: list[dict]) -> list[str]:
    lines = []
    for label, win_field, tie_field, lose_gap in [
        ("greedy", "beat_greedy", None, "_gap_to_greedy"),
        ("random best", "beat_random", None, "_gap_to_random"),
        ("config order once", None, None, "_gap_to_config_order"),
    ]:
        wins = ties = losses = 0
        for row in best_rows:
            gap = row.get(lose_gap, "")
            if not _is_number(gap):
                continue
            value = _int(gap)
            if value < 0:
                wins += 1
            elif value == 0:
                ties += 1
            else:
                losses += 1
        lines.append(f"- budgeted vs {label}: wins={wins} ties={ties} losses={losses}")
    return lines


def _observation_lines(result_rows: list[dict], best_rows: list[dict]) -> list[str]:
    if not result_rows:
        return ["- No successful budgeted runs were available for observation."]
    by_beam: dict[int, list[int]] = {}
    by_cap: dict[int, list[int]] = {}
    for row in result_rows:
        if _is_number(row.get("final_ir_inst_count")):
            by_beam.setdefault(_int(row.get("beam_width")), []).append(_int(row.get("final_ir_inst_count")))
            by_cap.setdefault(_int(row.get("max_states")), []).append(_int(row.get("final_ir_inst_count")))
    lines = []
    if len(by_beam) > 1:
        first_beam = min(by_beam)
        last_beam = max(by_beam)
        lines.append(
            f"- In this run, average final IR count changed from {_avg(by_beam[first_beam])} at beam={first_beam} to {_avg(by_beam[last_beam])} at beam={last_beam}."
        )
    if len(by_cap) > 1:
        first_cap = min(by_cap)
        last_cap = max(by_cap)
        lines.append(
            f"- Under IR instruction count objective, average final IR count changed from {_avg(by_cap[first_cap])} at max_states={first_cap} to {_avg(by_cap[last_cap])} at max_states={last_cap}."
        )
    matched = sum(1 for row in best_rows if row.get("matched_exact") == "true")
    lines.append(f"- Budgeted matched exact reference for {matched} programs with available exact data.")
    hard = [row.get("program", "") for row in best_rows if _is_number(row.get("gap_to_exact")) and _int(row.get("gap_to_exact")) > 0]
    if hard:
        lines.append(f"- Programs with remaining positive exact gap in this run: {', '.join(hard)}.")
    return lines


def _expand_input_plans(inputs: list[str], warn=print) -> tuple[list[tuple[str, Path]], list[dict]]:
    paths: list[Path] = []
    failures: list[dict] = []
    for item in inputs:
        if _has_glob_meta(item):
            matches = sorted(Path(match) for match in glob.glob(item, recursive=True))
            if not matches:
                failures.append(_input_failure(item, f"input glob matched no files: {item}"))
                warn(f"warning: input glob matched no files: {item}")
                continue
            paths.extend(matches)
        else:
            paths.append(Path(item))

    unique: list[tuple[str, Path]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            failures.append(_input_failure(str(path), f"missing input: {path}"))
            warn(f"warning: skipping missing input: {path}")
            continue
        if path.suffix.lower() not in {".c", ".ll"}:
            failures.append(_input_failure(str(path), f"unsupported input type: {path.suffix}"))
            warn(f"warning: skipping unsupported input: {path}")
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        stem = path.stem
        index = counts.get(stem, 0)
        counts[stem] = index + 1
        program = stem if index == 0 else f"{stem}_{index}"
        unique.append((program, path))
    return unique, failures


def _input_failure(input_path: str, message: str) -> dict:
    stem = Path(input_path).stem or "input"
    program = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "input"
    return {"program": program, "input_path": input_path, "error_message": message}


def _positive_unique(values: list[int], name: str) -> list[int]:
    normalized = []
    seen = set()
    for value in values:
        if value <= 0:
            raise RuntimeError(f"{name} values must be positive: {value}")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    if not normalized:
        raise RuntimeError(f"{name} must not be empty")
    return normalized


def _pipeline_length(run_dir: Path) -> int:
    path = run_dir / "optimized_pipeline.txt"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return 0
    return len([part for part in re.split(r"[,\n;]+", text) if part.strip()])


def _stop_reason(run_dir: Path) -> str:
    leaf = _first_row(run_dir / "leaf_states.csv")
    if leaf.get("leaf_reason"):
        return leaf["leaf_reason"]
    exact_status = run_dir / "exact_status.txt"
    if exact_status.exists():
        return exact_status.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def _gap(left: str, right: str) -> str:
    if not (_is_number(left) and _is_number(right)):
        return ""
    return str(_int(left) - _int(right))


def _cmp(left: str, right: str) -> int | None:
    if not (_is_number(left) and _is_number(right)):
        return None
    return _int(left) - _int(right)


def _lookup(row: dict[str, str], names: list[str]) -> str:
    normalized_names = [_normalize_column(name) for name in names]
    for name in normalized_names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _program_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(str(value)).stem.lower())


def _value(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _has_glob_meta(value: str) -> bool:
    return any(char in value for char in "*?[")


def _remove_existing_output(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if not path.name or resolved == anchor or resolved == Path.cwd().resolve():
        raise RuntimeError(f"refusing to remove unsafe output path: {path}")
    shutil.rmtree(resolved)


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


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _is_number(value: object) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg(values: list[int | float]) -> str:
    return _fmt_float(_mean([float(value) for value in values])) if values else "0"


def _fmt_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _bool(value: bool) -> str:
    return "true" if value else "false"
