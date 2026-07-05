from __future__ import annotations

import csv
import glob
import hashlib
import re
import shutil
import time
from collections import Counter
from pathlib import Path


BUDGETED_STUDY_RUN_FIELDS = [
    "program",
    "input_path",
    "status",
    "optimize_status",
    "baseline_status",
    "reduction_status",
    "evidence_status",
    "output_dir",
    "final_ir_inst_count",
    "root_ir_inst_count",
    "ir_inst_delta",
    "ir_inst_reduction_pct",
    "states_reached",
    "transitions",
    "pipeline_length",
    "time_ms",
    "stop_reason",
    "error_message",
]

BUDGETED_STUDY_METHOD_FIELDS = [
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
    "time_ms",
    "error_message",
]

BUDGETED_STUDY_REDUCTION_FIELDS = [
    "program",
    "states",
    "max_depth",
    "total_active_passes",
    "total_tested_pairs",
    "total_commute_pairs",
    "total_order_sensitive_pairs",
    "total_unknown_pairs",
    "total_batch_candidates",
    "total_certified_batches",
    "total_executable_batches",
    "total_executed_transitions",
    "total_skipped_batches",
    "total_dropped_active_passes",
    "avg_local_reduction_log10",
    "max_local_reduction_log10",
]

BUDGETED_STUDY_EVIDENCE_FIELDS = [
    "program",
    "selected_path_batches",
    "selected_strong_certificates",
    "selected_weak_certificates",
    "executed_batches",
    "executed_strong_certificates",
    "executed_weak_certificates",
    "executed_rejected",
    "dropped_active_passes",
    "replay_status",
    "replay_hashes_match",
]

BUDGETED_STUDY_FAILURE_FIELDS = [
    "program",
    "input_path",
    "stage",
    "error_message",
]


def run_core_v1_budgeted_study(
    inputs: list[str],
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
    baseline_methods: list[str] | None,
    random_trials: int,
    seed: int,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    out_dir = Path(out_dir)
    passes_path = Path(passes_path)
    plans, input_failures = _input_plans(inputs, out_dir, warn)
    if not plans and not input_failures:
        raise RuntimeError("no valid .c or .ll inputs remain after expansion")
    if not plans and input_failures and not continue_on_error:
        raise RuntimeError("no valid .c or .ll inputs remain after expansion")

    if out_dir.exists():
        if overwrite:
            _remove_existing_output(out_dir)
        elif any(out_dir.iterdir()):
            raise RuntimeError(f"output directory already exists: {out_dir}; use --overwrite to rerun")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    method_rows: list[dict] = []
    reduction_rows: list[dict] = []
    evidence_rows: list[dict] = []
    failure_rows: list[dict] = []

    for failure in input_failures:
        failure_rows.append(failure)
        if failure.get("_nonfatal"):
            continue
        run_rows.append(_failed_input_run_row(failure))
        if not continue_on_error:
            _write_outputs(out_dir, run_rows, method_rows, reduction_rows, evidence_rows, failure_rows, passes_path, objective, max_rounds, beam_width, max_states, batch_frontier_policy)
            raise RuntimeError(failure["error_message"])

    for program, input_path, program_dir in plans:
        start = time.perf_counter()
        optimize_dir = program_dir / "optimize"
        status = "success"
        optimize_status = "not_run"
        baseline_status = "not_run"
        reduction_status = "not_run"
        evidence_status = "not_run"
        error_message = ""

        try:
            if program_dir.exists():
                if overwrite:
                    _remove_existing_output(program_dir)
                else:
                    raise RuntimeError(f"output directory already exists: {program_dir}; use --overwrite to rerun")
            program_dir.mkdir(parents=True, exist_ok=True)

            run_optimizer(
                input_path,
                optimize_dir,
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
            optimize_status = "success"

            try:
                run_baseline_comparison(
                    optimize_dir,
                    passes_path,
                    objective=objective,
                    methods=_compare_methods(baseline_methods),
                    max_rounds=max_rounds,
                    random_trials=random_trials,
                    seed=seed,
                    timeout=timeout,
                    jobs=jobs,
                    include_default_pipelines=_include_default_pipelines(baseline_methods),
                )
                baseline_status = "success"
                method_rows.extend(_method_rows(program, optimize_dir / "baseline_results.csv"))
                _copy_baselines(program_dir, optimize_dir)
            except Exception as exc:
                baseline_status = "failed"
                failure_rows.append(_failure(program, input_path, "baseline", str(exc)))

            try:
                run_reduction_summary(optimize_dir)
                reduction_status = "success"
                reduction_rows.append(_reduction_row(program, optimize_dir / "reduction_summary.csv"))
                _copy_named_artifacts(program_dir / "reduction", optimize_dir, ["reduction_by_state.csv", "reduction_summary.csv", "reduction_summary.md"])
            except Exception as exc:
                reduction_status = "failed"
                failure_rows.append(_failure(program, input_path, "reduction", str(exc)))

            try:
                run_evidence_pack(optimize_dir)
                evidence_status = "success"
                evidence_rows.append(_evidence_row(program, optimize_dir / "evidence_pack.csv"))
                _copy_named_artifacts(program_dir / "evidence", optimize_dir, ["evidence_pack.csv", "evidence_pack.md", "selected_batch_certificates.csv", "executed_batch_certificates.csv"])
            except Exception as exc:
                evidence_status = "failed"
                failure_rows.append(_failure(program, input_path, "evidence", str(exc)))

        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            if optimize_status == "not_run":
                optimize_status = "failed"
            failure_rows.append(_failure(program, input_path, "optimize", error_message))
            if not continue_on_error:
                run_rows.append(
                    _run_row(
                        program,
                        input_path,
                        status,
                        optimize_status,
                        baseline_status,
                        reduction_status,
                        evidence_status,
                        program_dir,
                        optimize_dir,
                        start,
                        error_message,
                    )
                )
                _write_outputs(out_dir, run_rows, method_rows, reduction_rows, evidence_rows, failure_rows, passes_path, objective, max_rounds, beam_width, max_states, batch_frontier_policy)
                raise
        finally:
            if not run_rows or run_rows[-1].get("program") != program:
                run_rows.append(
                    _run_row(
                        program,
                        input_path,
                        status,
                        optimize_status,
                        baseline_status,
                        reduction_status,
                        evidence_status,
                        program_dir,
                        optimize_dir,
                        start,
                        error_message,
                    )
                )

    summary_path = _write_outputs(out_dir, run_rows, method_rows, reduction_rows, evidence_rows, failure_rows, passes_path, objective, max_rounds, beam_width, max_states, batch_frontier_policy)
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = sum(1 for row in run_rows if row.get("status") == "failed")
    return {
        "out_dir": str(out_dir),
        "programs": len(run_rows),
        "successes": successes,
        "failures": failures,
        "budgeted_study_runs_csv": str(out_dir / "budgeted_study_runs.csv"),
        "budgeted_study_methods_csv": str(out_dir / "budgeted_study_methods.csv"),
        "budgeted_study_reduction_csv": str(out_dir / "budgeted_study_reduction.csv"),
        "budgeted_study_evidence_csv": str(out_dir / "budgeted_study_evidence.csv"),
        "budgeted_study_summary_md": str(summary_path),
        "failures_csv": str(out_dir / "failures.csv"),
    }


def run_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(input_path, out_dir, passes_path, **kwargs)


def run_baseline_comparison(run_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .baselines import compare_baselines

    return compare_baselines(run_dir, passes_path, **kwargs)


def run_reduction_summary(run_dir: Path) -> dict:
    from .reduction_summary import summarize_reduction

    return summarize_reduction(run_dir)


def run_evidence_pack(run_dir: Path) -> dict:
    from .evidence_pack import export_evidence_pack

    return export_evidence_pack(run_dir)


def _write_outputs(
    out_dir: Path,
    run_rows: list[dict],
    method_rows: list[dict],
    reduction_rows: list[dict],
    evidence_rows: list[dict],
    failure_rows: list[dict],
    passes_path: Path,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    policy: str | None,
) -> Path:
    _write_csv(out_dir / "budgeted_study_runs.csv", BUDGETED_STUDY_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "budgeted_study_methods.csv", BUDGETED_STUDY_METHOD_FIELDS, method_rows)
    _write_csv(out_dir / "budgeted_study_reduction.csv", BUDGETED_STUDY_REDUCTION_FIELDS, reduction_rows)
    _write_csv(out_dir / "budgeted_study_evidence.csv", BUDGETED_STUDY_EVIDENCE_FIELDS, evidence_rows)
    _write_csv(out_dir / "failures.csv", BUDGETED_STUDY_FAILURE_FIELDS, failure_rows)
    return _write_summary(out_dir, run_rows, method_rows, reduction_rows, evidence_rows, passes_path, objective, max_rounds, beam_width, max_states, policy)


def _write_summary(
    out_dir: Path,
    run_rows: list[dict],
    method_rows: list[dict],
    reduction_rows: list[dict],
    evidence_rows: list[dict],
    passes_path: Path,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    policy: str | None,
) -> Path:
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = sum(1 for row in run_rows if row.get("status") == "failed")
    lines = [
        "# Core-v1 Budgeted Study Summary",
        "",
        "## Overall",
        "",
        f"- programs attempted: {len(run_rows)}",
        f"- successful programs: {successes}",
        f"- failed programs: {failures}",
        f"- pass set: {passes_path}",
        f"- objective: {objective}",
        f"- max_rounds: {max_rounds}",
        f"- beam_width: {beam_width}",
        f"- max_states: {max_states}",
        f"- policy: {policy or ''}",
        "",
        "## Method Results",
        "",
        *_markdown_table(
            ["program", "batch", "greedy", "random best", "config order", "default O2", "batch rank"],
            [_method_result_row(program, method_rows) for program in _program_order(run_rows)],
        ),
        "",
        "## Budgeted Success Rate",
        "",
        f"- successful optimize runs: {sum(1 for row in run_rows if row.get('optimize_status') == 'success')}",
        f"- failed optimize runs: {sum(1 for row in run_rows if row.get('optimize_status') == 'failed')}",
        f"- replay verified runs: {sum(1 for row in evidence_rows if row.get('replay_hashes_match') == 'true')}",
        f"- runs with dropped active passes: {sum(1 for row in evidence_rows if _int(row.get('dropped_active_passes')) > 0)}",
        f"- runs with only strong selected path certificates: {_strong_selected_runs(evidence_rows)}",
        "",
        "## Search Cost",
        "",
        *_markdown_table(
            ["program", "states reached", "transitions", "time ms", "pipeline length"],
            [
                [
                    row.get("program", ""),
                    row.get("states_reached", ""),
                    row.get("transitions", ""),
                    row.get("time_ms", ""),
                    row.get("pipeline_length", ""),
                ]
                for row in run_rows
            ],
        ),
        "",
        "## Reduction Evidence",
        "",
        *_markdown_table(
            ["program", "tested pairs", "commute", "order-sensitive", "avg reduction log10", "max reduction log10", "dropped"],
            [
                [
                    row.get("program", ""),
                    row.get("total_tested_pairs", ""),
                    row.get("total_commute_pairs", ""),
                    row.get("total_order_sensitive_pairs", ""),
                    row.get("avg_local_reduction_log10", ""),
                    row.get("max_local_reduction_log10", ""),
                    row.get("total_dropped_active_passes", ""),
                ]
                for row in reduction_rows
            ],
        ),
        "",
        "## Evidence Quality",
        "",
        *_markdown_table(
            ["program", "selected path batches", "strong selected", "weak selected", "executed batches", "strong executed", "dropped"],
            [
                [
                    row.get("program", ""),
                    row.get("selected_path_batches", ""),
                    row.get("selected_strong_certificates", ""),
                    row.get("selected_weak_certificates", ""),
                    row.get("executed_batches", ""),
                    row.get("executed_strong_certificates", ""),
                    row.get("dropped_active_passes", ""),
                ]
                for row in evidence_rows
            ],
        ),
        "",
        "## Win/Tie/Loss",
        "",
        *_win_tie_loss_lines(method_rows),
        "",
        "## Key Observations",
        "",
        *_observation_lines(method_rows, evidence_rows, run_rows),
        "",
        "## Correctness Boundary",
        "",
        "Budgeted mode changes search coverage, not batch correctness. A batch is executable only when its correctness classification allows execution. Objective values are used for path selection and evaluation, not as commutation proof.",
        "",
    ]
    path = out_dir / "budgeted_study_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _run_row(
    program: str,
    input_path: Path,
    status: str,
    optimize_status: str,
    baseline_status: str,
    reduction_status: str,
    evidence_status: str,
    program_dir: Path,
    optimize_dir: Path,
    start: float,
    error_message: str,
) -> dict:
    chosen = _first_row(optimize_dir / "chosen_path_summary.csv")
    states = _read_csv(optimize_dir / "states.csv")
    transitions = _read_csv(optimize_dir / "batch_state_transitions.csv")
    timing = _first_row(optimize_dir / "optimizer_timing.csv")
    return {
        "program": program,
        "input_path": str(input_path),
        "status": status,
        "optimize_status": optimize_status,
        "baseline_status": baseline_status,
        "reduction_status": reduction_status,
        "evidence_status": evidence_status,
        "output_dir": str(program_dir),
        "final_ir_inst_count": _value(chosen, "final_ir_inst_count", "final_objective"),
        "root_ir_inst_count": _value(chosen, "root_ir_inst_count"),
        "ir_inst_delta": _value(chosen, "total_ir_inst_delta", "ir_inst_delta"),
        "ir_inst_reduction_pct": _value(chosen, "ir_inst_reduction_pct", "reduction_pct"),
        "states_reached": str(len(states)) if states else "",
        "transitions": str(len(transitions)) if transitions else "",
        "pipeline_length": str(len(_split_pipeline(_read_text(optimize_dir / "optimized_pipeline.txt")))),
        "time_ms": _value(timing, "optimizer_total_time_ms", default=f"{(time.perf_counter() - start) * 1000:.3f}"),
        "stop_reason": _stop_reason(optimize_dir),
        "error_message": error_message,
    }


def _failed_input_run_row(failure: dict) -> dict:
    return {
        "program": failure["program"],
        "input_path": failure["input_path"],
        "status": "failed",
        "optimize_status": "failed",
        "baseline_status": "not_run",
        "reduction_status": "not_run",
        "evidence_status": "not_run",
        "output_dir": "",
        "final_ir_inst_count": "",
        "root_ir_inst_count": "",
        "ir_inst_delta": "",
        "ir_inst_reduction_pct": "",
        "states_reached": "",
        "transitions": "",
        "pipeline_length": "",
        "time_ms": "0.000",
        "stop_reason": "input_failed",
        "error_message": failure["error_message"],
    }


def _method_rows(program: str, path: Path) -> list[dict]:
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
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
                "time_ms": row.get("time_ms") or row.get("optimizer_total_time_ms", ""),
                "error_message": row.get("error_message", ""),
            }
        )
    return rows


def _reduction_row(program: str, path: Path) -> dict:
    row = _first_row(path)
    return {
        "program": program,
        "states": _value(row, "total_states", "states"),
        "max_depth": _value(row, "max_depth"),
        "total_active_passes": _value(row, "total_active_passes"),
        "total_tested_pairs": _value(row, "total_tested_pairs"),
        "total_commute_pairs": _value(row, "total_commute_pairs", "commute_pairs"),
        "total_order_sensitive_pairs": _value(row, "total_order_sensitive_pairs", "order_sensitive_pairs"),
        "total_unknown_pairs": _value(row, "total_unknown_pairs", "unknown_pairs"),
        "total_batch_candidates": _value(row, "total_batch_candidates"),
        "total_certified_batches": _value(row, "total_certified_batches"),
        "total_executable_batches": _value(row, "total_executable_batches"),
        "total_executed_transitions": _value(row, "total_executed_transitions"),
        "total_skipped_batches": _value(row, "total_skipped_batches"),
        "total_dropped_active_passes": _value(row, "total_dropped_active_passes"),
        "avg_local_reduction_log10": _value(row, "avg_local_reduction_log10"),
        "max_local_reduction_log10": _value(row, "max_local_reduction_log10"),
    }


def _evidence_row(program: str, path: Path) -> dict:
    row = _first_row(path)
    return {
        "program": program,
        "selected_path_batches": _value(row, "selected_path_batches"),
        "selected_strong_certificates": _value(row, "selected_strong_certificates"),
        "selected_weak_certificates": _value(row, "selected_weak_certificates"),
        "executed_batches": _value(row, "executed_batches"),
        "executed_strong_certificates": _value(row, "executed_strong_certificates"),
        "executed_weak_certificates": _value(row, "executed_weak_certificates"),
        "executed_rejected": _value(row, "executed_rejected"),
        "dropped_active_passes": _value(row, "dropped_active_passes"),
        "replay_status": _value(row, "replay_status"),
        "replay_hashes_match": _value(row, "replay_hashes_match"),
    }


def _method_result_row(program: str, rows: list[dict]) -> list[str]:
    program_rows = [row for row in rows if row.get("program") == program]
    return [
        program,
        _method_inst(program_rows, "batch_optimizer"),
        _method_inst(program_rows, "greedy_single_pass"),
        _method_inst(program_rows, "random_single_pass_best"),
        _method_inst(program_rows, "config_order_once"),
        _method_inst(program_rows, "default_O2"),
        _batch_rank(program_rows),
    ]


def _win_tie_loss_lines(rows: list[dict]) -> list[str]:
    lines = []
    for method in ["greedy_single_pass", "random_single_pass_best", "config_order_once", "default_O2"]:
        wins = ties = losses = 0
        for program in sorted({row.get("program", "") for row in rows}):
            program_rows = [row for row in rows if row.get("program") == program]
            batch = _success_count(_method_row(program_rows, "batch_optimizer"))
            other = _success_count(_method_row(program_rows, method))
            if batch is None or other is None:
                continue
            if batch < other:
                wins += 1
            elif batch == other:
                ties += 1
            else:
                losses += 1
        lines.append(f"- batch vs {method}: wins={wins} ties={ties} losses={losses}")
    return lines


def _observation_lines(method_rows: list[dict], evidence_rows: list[dict], run_rows: list[dict]) -> list[str]:
    if not run_rows:
        return ["- No programs were available for observation."]
    greedy_line = _win_tie_loss_lines(method_rows)[0] if method_rows else "- batch vs greedy_single_pass: wins=0 ties=0 losses=0"
    dropped = sum(_int(row.get("dropped_active_passes")) for row in evidence_rows)
    selected = sum(_int(row.get("selected_path_batches")) for row in evidence_rows)
    strong_selected = sum(_int(row.get("selected_strong_certificates")) for row in evidence_rows)
    max_time = max((_float(row.get("time_ms")) for row in run_rows), default=0.0)
    return [
        f"- In this run, {greedy_line.removeprefix('- ')}.",
        f"- Dropped active passes are {'zero' if dropped == 0 else str(dropped)} across programs with evidence output.",
        f"- Strong selected-path certificate ratio is {strong_selected}/{selected}.",
        f"- Maximum recorded optimizer time is {max_time:.3f} ms in this study.",
    ]


def _input_plans(inputs: list[str], out_dir: Path, warn=print) -> tuple[list[tuple[str, Path, Path]], list[dict]]:
    paths: list[Path] = []
    failures: list[dict] = []
    for item in inputs:
        if _has_glob_meta(item):
            matches = sorted(Path(match) for match in glob.glob(item, recursive=True))
            if not matches:
                warn(f"warning: input glob matched no files: {item}")
                failure = _failure(_safe_program_name(Path(item).stem or "input"), Path(item), "input_glob", f"input glob matched no files: {item}")
                failure["_nonfatal"] = "true"
                failures.append(failure)
                continue
            paths.extend(matches)
        else:
            paths.append(Path(item))

    seen: set[str] = set()
    stem_counts: Counter = Counter()
    plans = []
    for path in paths:
        if not path.exists():
            warn(f"warning: missing input: {path}")
            failures.append(_failure(_safe_program_name(path.stem or "input"), path, "input", f"missing input: {path}"))
            continue
        if path.suffix.lower() not in {".c", ".ll"}:
            warn(f"warning: unsupported input type: {path}")
            failures.append(_failure(_safe_program_name(path.stem or "input"), path, "input", f"unsupported input type: {path.suffix}"))
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        stem = _safe_program_name(path.stem)
        stem_counts[stem] += 1
        program = stem if stem_counts[stem] == 1 else f"{stem}_{_short_hash(path)}"
        plans.append((program, path, out_dir / program))
    return plans, failures


def _compare_methods(methods: list[str] | None) -> list[str]:
    tokens = _method_tokens(methods)
    if not tokens or "all" in tokens:
        return ["all"]
    normalized = []
    aliases = {"batch": "batch", "greedy": "greedy", "random": "random", "default": "default", "config": "config"}
    for token in tokens:
        normalized.append(aliases.get(token, token))
    if "config" not in normalized and "config_order_once" not in normalized:
        normalized.append("config")
    return normalized


def _include_default_pipelines(methods: list[str] | None) -> bool:
    tokens = _method_tokens(methods)
    return not tokens or "all" in tokens or "default" in tokens or "default_O2" in tokens or "default_Oz" in tokens


def _method_tokens(methods: list[str] | None) -> set[str]:
    if not methods:
        return {"all"}
    return {part.strip() for method in methods for part in str(method).split(",") if part.strip()}


def _copy_baselines(program_dir: Path, optimize_dir: Path) -> None:
    source = optimize_dir / "baselines"
    if source.exists():
        target = program_dir / "baselines"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def _copy_named_artifacts(target_dir: Path, source_dir: Path, names: list[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = source_dir / name
        if source.exists():
            shutil.copyfile(source, target_dir / name)


def _program_order(run_rows: list[dict]) -> list[str]:
    return [row.get("program", "") for row in run_rows if row.get("program")]


def _method_row(rows: list[dict], method: str) -> dict | None:
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def _method_inst(rows: list[dict], method: str) -> str:
    row = _method_row(rows, method)
    if not row:
        return ""
    return row.get("final_ir_inst_count") or row.get("status", "")


def _batch_rank(rows: list[dict]) -> str:
    batch = _success_count(_method_row(rows, "batch_optimizer"))
    if batch is None:
        return ""
    rank = 1
    for row in rows:
        count = _success_count(row)
        if count is not None and count < batch:
            rank += 1
    return str(rank)


def _strong_selected_runs(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if _int(row.get("selected_path_batches")) > 0
        and _int(row.get("selected_path_batches")) == _int(row.get("selected_strong_certificates"))
    )


def _success_count(row: dict | None) -> int | None:
    if not row or row.get("status") != "success":
        return None
    value = row.get("final_ir_inst_count", "")
    if value == "":
        return None
    return _int(value)


def _stop_reason(optimize_dir: Path) -> str:
    for row in _read_csv(optimize_dir / "leaf_states.csv"):
        if row.get("selected_as_final") == "true":
            return row.get("leaf_reason", "")
    return ""


def _failure(program: str, input_path: Path, stage: str, message: str) -> dict:
    return {"program": program, "input_path": str(input_path), "stage": stage, "error_message": message}


def _safe_program_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "program")).strip("_") or "program"


def _short_hash(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]


def _remove_existing_output(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if not path.name or resolved == anchor or resolved == Path.cwd().resolve():
        raise RuntimeError(f"refusing to remove unsafe output path: {path}")
    shutil.rmtree(resolved)


def _split_pipeline(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").replace("\n", "").replace(";", ",").split(",") if part.strip()]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


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


def _has_glob_meta(value: str) -> bool:
    return any(char in value for char in "*?[]")


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
