from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

from .batch_objective import count_ir_instructions
from .normalizer import canonical_hash
from .pass_config import PassRegistry, load_pass_registry, resolve_pipeline_sequence
from .profiler import profile_passes, validate_passes
from .runner import run_opt
from .schema import BASELINE_RESULT_FIELDS, RANDOM_BASELINE_TRIAL_FIELDS, RunResult


SUPPORTED_OBJECTIVES = {"ir-inst-count"}
SUPPORTED_COMPARE_METHODS = {
    "all",
    "default",
    "default_O0",
    "default_O2",
    "default_Oz",
    "batch",
    "batch_optimizer",
    "root",
    "optimized",
    "optimized_pipeline",
    "config",
    "config_order_once",
    "greedy",
    "greedy_single_pass",
    "random",
    "random_single_pass_best",
    "llvm-defaults",
}
DEFAULT_PIPELINES = {
    "default_O2": "default<O2>",
    "default_Oz": "default<Oz>",
}
GREEDY_PATH_FIELDS = [
    "round",
    "input_ir_path",
    "selected_pass",
    "output_ir_path",
    "input_hash",
    "output_hash",
    "input_ir_inst_count",
    "output_ir_inst_count",
    "inst_delta",
    "active_passes",
    "tested_passes",
    "opt_runs_this_round",
    "stop_reason",
]
RANDOM_BEST_PATH_FIELDS = [
    "trial",
    "round",
    "input_ir_path",
    "selected_pass",
    "output_ir_path",
    "input_hash",
    "output_hash",
    "input_ir_inst_count",
    "output_ir_inst_count",
    "inst_delta",
    "active_passes",
    "tested_passes",
    "opt_runs_this_round",
    "stop_reason",
]
LLVM_DEFAULT_PIPELINES = {
    "llvm_default_O1": "default<O1>",
    "llvm_default_O2": "default<O2>",
    "llvm_default_O3": "default<O3>",
    "llvm_default_Oz": "default<Oz>",
}


def compare_baselines(
    run_dir: Path,
    passes_path: Path,
    *,
    objective: str = "ir-inst-count",
    methods: list[str] | None = None,
    max_rounds: int = 2,
    random_trials: int = 20,
    seed: int = 0,
    timeout: int = 10,
    jobs: int = 1,
    greedy_allow_nonimproving: bool = False,
    include_default_pipelines: bool = False,
    include_llvm_defaults: bool = False,
) -> dict:
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    selected_methods = _normalize_methods(methods)

    run_dir = Path(run_dir)
    passes_path = Path(passes_path)
    baseline_dir = run_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(run_dir / "metadata.json")
    tools = _tool_paths(metadata)
    if "opt" not in tools:
        raise RuntimeError(f"metadata.json under {run_dir} does not contain an opt tool path")

    root_ir = run_dir / "states" / "S0000" / "input.ll"
    if not root_ir.exists():
        raise RuntimeError(f"missing root IR: {root_ir}")

    pass_registry = load_pass_registry(passes_path)
    configured_passes = pass_registry.names()
    valid_passes, skipped_passes, valid_note = _valid_passes_in_config_order(
        run_dir,
        configured_passes,
        root_ir,
        tools,
        timeout,
        pass_registry=pass_registry,
    )
    root_count = count_ir_instructions(root_ir)
    program = run_dir.name
    rows: list[dict] = []
    trial_rows: list[dict] = []
    include_default_pipelines = include_default_pipelines or include_llvm_defaults

    rows.append(_default_o0_row(program, root_ir, baseline_dir, root_count))
    if include_default_pipelines and _method_enabled(selected_methods, "default"):
        for method, pipeline in DEFAULT_PIPELINES.items():
            rows.append(_default_pipeline_row(program, method, pipeline, root_ir, baseline_dir, tools, root_count, timeout))

    if _method_enabled(selected_methods, "root"):
        root_out = baseline_dir / "root.ll"
        shutil.copyfile(root_ir, root_out)
        rows.append(
            _result_row(
                program=program,
                method="root",
                status="success",
                final_ir_path=root_out,
                root_count=root_count,
                pass_sequence=[],
                pass_invocations=0,
                states_analyzed=0,
                opt_runs=0,
                time_ms=0.0,
                stop_reason="",
                error_message="",
            )
        )
    if _method_enabled(selected_methods, "batch_optimizer"):
        rows.append(_batch_optimizer_row(program, run_dir, root_ir, root_count))

    if _method_enabled(selected_methods, "optimized_pipeline"):
        rows.append(_optimized_pipeline_row(program, root_ir, baseline_dir, tools, root_count, timeout))
    if _method_enabled(selected_methods, "config_order_once"):
        rows.append(
            _run_pass_sequence_row(
                program=program,
                method="config_order_once",
                root_ir=root_ir,
                output_path=baseline_dir / "config_order_once.ll",
                tools=tools,
                passes=valid_passes,
                root_count=root_count,
                timeout=timeout,
                error_note=_join_notes([valid_note, _skipped_note(skipped_passes)]),
                pass_registry=pass_registry,
            )
        )
    if _method_enabled(selected_methods, "greedy_single_pass"):
        rows.append(
            run_greedy_single_pass_baseline(
                root_ir,
                valid_passes,
                tools,
                baseline_dir / "greedy_single_pass",
                objective=objective,
                max_rounds=max_rounds,
                timeout=timeout,
                allow_nonimproving=greedy_allow_nonimproving,
                pass_registry=pass_registry,
            )
        )
    if _method_enabled(selected_methods, "random_single_pass_best"):
        random_row = run_random_single_pass_baseline(
            root_ir,
            valid_passes,
            tools,
            baseline_dir / "random_single_pass",
            objective=objective,
            max_rounds=max_rounds,
            random_trials=random_trials,
            seed=seed,
            timeout=timeout,
            pass_registry=pass_registry,
        )
        trial_rows = random_row.pop("_trial_rows", [])
        rows.append(random_row)

    if "llvm-defaults" in selected_methods:
        for method, pipeline in LLVM_DEFAULT_PIPELINES.items():
            rows.append(_llvm_default_row(program, method, pipeline, root_ir, baseline_dir, tools, root_count, timeout))

    _write_csv(run_dir / "baseline_results.csv", BASELINE_RESULT_FIELDS, rows)
    _write_csv(run_dir / "random_baseline_trials.csv", RANDOM_BASELINE_TRIAL_FIELDS, trial_rows)
    method_comparison = _write_method_comparison(run_dir, rows)
    return {
        "run_dir": str(run_dir),
        "objective": objective,
        "rows": len(rows),
        "random_trials": len(trial_rows),
        "baseline_results_csv": str(run_dir / "baseline_results.csv"),
        "random_baseline_trials_csv": str(run_dir / "random_baseline_trials.csv"),
        "baselines_dir": str(baseline_dir),
        "method_comparison_md": str(method_comparison),
    }


def run_opt_raw_pipeline(opt: str, input_ll: Path, pipeline: str, output_ll: Path, timeout: int) -> RunResult:
    return run_opt(opt, input_ll, [pipeline], output_ll, timeout)


def _run_opt_safely(opt: str, input_ll: Path, passes: list[str], output_ll: Path, timeout: int) -> RunResult:
    try:
        return run_opt(opt, input_ll, passes, output_ll, timeout)
    except OSError as exc:
        return RunResult(
            command=[opt],
            returncode=-1,
            stdout="",
            stderr=str(exc),
            time_ms=0.0,
            failure_kind="failed_to_start",
            output_path=output_ll,
        )


def _default_o0_row(program: str, root_ir: Path, baseline_dir: Path, root_count: int) -> dict:
    output_path = baseline_dir / "default_O0" / "default_O0.ll"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root_ir, output_path)
    return _result_row(
        program=program,
        method="default_O0",
        status="success",
        final_ir_path=output_path,
        root_count=root_count,
        pass_sequence=[],
        pass_invocations=0,
        states_analyzed=1,
        opt_runs=0,
        time_ms=0.0,
        stop_reason="",
        error_message="",
    )


def _default_pipeline_row(
    program: str,
    method: str,
    pipeline: str,
    root_ir: Path,
    baseline_dir: Path,
    tools: dict[str, str],
    root_count: int,
    timeout: int,
) -> dict:
    start = time.perf_counter()
    output_path = baseline_dir / method / f"{method}.ll"
    result = run_opt_raw_pipeline(tools["opt"], root_ir, pipeline, output_path, timeout)
    status = "success" if result.success and output_path.exists() else "unsupported"
    return _result_row(
        program=program,
        method=method,
        status=status,
        final_ir_path=output_path if status == "success" else None,
        root_count=root_count,
        pass_sequence=[pipeline],
        pass_invocations=1,
        states_analyzed=1,
        opt_runs=1,
        time_ms=(time.perf_counter() - start) * 1000,
        stop_reason="",
        error_message="" if status == "success" else (_error_text(result) or "unsupported default pipeline"),
    )


def _batch_optimizer_row(program: str, run_dir: Path, root_ir: Path, root_count: int) -> dict:
    run_dir = Path(run_dir)
    final_path = run_dir / "final.ll"
    pipeline_text = _normalized_pipeline_text((run_dir / "optimized_pipeline.txt").read_text(encoding="utf-8") if (run_dir / "optimized_pipeline.txt").exists() else "")
    passes = _split_optimized_pipeline(pipeline_text)
    status = "success" if final_path.exists() else "failed"
    costs = _batch_optimizer_costs(run_dir)
    row = _result_row(
        program=program,
        method="batch_optimizer",
        status=status,
        final_ir_path=final_path if final_path.exists() else None,
        root_count=root_count,
        pass_sequence=passes,
        pass_invocations=len(passes),
        states_analyzed=_count_nonduplicate_states(run_dir / "states.csv"),
        opt_runs=_count_optimizer_opt_runs(run_dir),
        time_ms=_parse_float(costs.get("optimizer_total_time_ms")) or 0.0,
        stop_reason="",
        error_message="" if status == "success" else f"missing final.ll under {run_dir}",
    )
    row["pass_sequence"] = pipeline_text
    row["final_sequence_length"] = str(len(passes))
    row.update(costs)
    return row


def _optimized_pipeline_row(
    program: str,
    root_ir: Path,
    baseline_dir: Path,
    tools: dict[str, str],
    root_count: int,
    timeout: int,
) -> dict:
    pipeline_path = baseline_dir.parent / "optimized_pipeline.txt"
    passes = _split_optimized_pipeline(pipeline_path.read_text(encoding="utf-8") if pipeline_path.exists() else "")
    return _run_pass_sequence_row(
        program=program,
        method="optimized_pipeline",
        root_ir=root_ir,
        output_path=baseline_dir / "optimized_pipeline.ll",
        tools=tools,
        passes=passes,
        root_count=root_count,
        timeout=timeout,
        error_note="",
    )


def _run_pass_sequence_row(
    *,
    program: str,
    method: str,
    root_ir: Path,
    output_path: Path,
    tools: dict[str, str],
    passes: list[str],
    root_count: int,
    timeout: int,
    error_note: str,
    pass_registry: PassRegistry | None = None,
) -> dict:
    start = time.perf_counter()
    if not passes:
        shutil.copyfile(root_ir, output_path)
        return _result_row(
            program=program,
            method=method,
            status="success",
            final_ir_path=output_path,
            root_count=root_count,
            pass_sequence=[],
            pass_invocations=0,
            states_analyzed=0,
            opt_runs=0,
            time_ms=(time.perf_counter() - start) * 1000,
            stop_reason="",
            error_message=error_note,
        )

    result = _run_opt_safely(tools["opt"], root_ir, resolve_pipeline_sequence(passes, pass_registry), output_path, timeout)
    status = "success" if result.success and output_path.exists() else "failed"
    error = _join_notes([error_note, _error_text(result) if status != "success" else ""])
    return _result_row(
        program=program,
        method=method,
        status=status,
        final_ir_path=output_path if output_path.exists() else None,
        root_count=root_count,
        pass_sequence=passes,
        pass_invocations=len(passes),
        states_analyzed=0,
        opt_runs=1,
        time_ms=(time.perf_counter() - start) * 1000,
        stop_reason="",
        error_message=error,
    )


def run_greedy_single_pass_baseline(
    root_ir: Path,
    valid_passes: list[str],
    tools: dict[str, str],
    out_dir: Path,
    *,
    objective: str = "ir-inst-count",
    max_rounds: int,
    timeout: int,
    allow_nonimproving: bool = False,
    pass_registry: PassRegistry | None = None,
) -> dict:
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")

    root_ir = Path(root_ir)
    out_dir = Path(out_dir)
    if (out_dir / "states" / "S0000" / "input.ll").exists():
        out_dir = out_dir / "baselines" / "greedy_single_pass"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _infer_run_dir_from_greedy_out_dir(out_dir)
    program = run_dir.name if run_dir is not None else out_dir.name

    start = time.perf_counter()
    current_ir = root_ir
    root_count = count_ir_instructions(root_ir)
    sequence: list[str] = []
    path_rows: list[dict] = []
    states_evaluated = 0
    opt_runs = 0
    status = "success"
    error_message = ""
    stop_reason = "max_rounds_reached" if max_rounds <= 0 else ""

    try:
        for round_index in range(max(0, max_rounds)):
            input_hash = canonical_hash(current_ir)
            input_count = count_ir_instructions(current_ir)
            round_dir = out_dir / "rounds" / f"R{round_index:04d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            active: list[dict] = []
            opt_runs_this_round = 0

            for pass_index, pass_name in enumerate(valid_passes):
                output = round_dir / f"{pass_index:04d}_{_safe_pass_stem(pass_name)}.ll"
                result = _run_opt_safely(tools["opt"], current_ir, resolve_pipeline_sequence([pass_name], pass_registry), output, timeout)
                opt_runs += 1
                opt_runs_this_round += 1
                if not result.success or not output.exists():
                    continue
                output_hash = canonical_hash(output)
                if output_hash == input_hash:
                    continue
                output_count = count_ir_instructions(output)
                active.append(
                    {
                        "pass": pass_name,
                        "pass_index": pass_index,
                        "output_path": output,
                        "output_hash": output_hash,
                        "output_count": output_count,
                        "inst_delta": output_count - input_count,
                        "immediate_reduction": input_count - output_count,
                    }
                )

            if not active:
                stop_reason = "no_active_passes"
                path_rows.append(
                    _greedy_path_row(
                        round_index=round_index,
                        input_ir=current_ir,
                        selected=None,
                        input_hash=input_hash,
                        input_count=input_count,
                        active_passes=0,
                        tested_passes=len(valid_passes),
                        opt_runs_this_round=opt_runs_this_round,
                        stop_reason=stop_reason,
                    )
                )
                break

            chosen = min(
                active,
                key=lambda row: (
                    row["output_count"],
                    -row["immediate_reduction"],
                    row["pass_index"],
                    row["pass"],
                ),
            )
            if not allow_nonimproving and chosen["output_count"] >= input_count:
                stop_reason = "no_improving_pass"
                path_rows.append(
                    _greedy_path_row(
                        round_index=round_index,
                        input_ir=current_ir,
                        selected=chosen,
                        input_hash=input_hash,
                        input_count=input_count,
                        active_passes=len(active),
                        tested_passes=len(valid_passes),
                        opt_runs_this_round=opt_runs_this_round,
                        stop_reason=stop_reason,
                    )
                )
                break

            path_rows.append(
                _greedy_path_row(
                    round_index=round_index,
                    input_ir=current_ir,
                    selected=chosen,
                    input_hash=input_hash,
                    input_count=input_count,
                    active_passes=len(active),
                    tested_passes=len(valid_passes),
                    opt_runs_this_round=opt_runs_this_round,
                    stop_reason="",
                )
            )
            sequence.append(chosen["pass"])
            current_ir = chosen["output_path"]
            states_evaluated += 1
            stop_reason = "max_rounds_reached"
    except Exception as exc:  # pragma: no cover - defensive path for real tool failures.
        status = "failed"
        stop_reason = "failed"
        error_message = str(exc)

    _write_csv(out_dir / "greedy_path.csv", GREEDY_PATH_FIELDS, path_rows)
    output_path = out_dir / "greedy_final.ll"
    if status == "success" and Path(current_ir).exists():
        shutil.copyfile(current_ir, output_path)
        final_path: Path | None = output_path
    else:
        final_path = None
    row = _result_row(
        program=program,
        method="greedy_single_pass",
        status=status,
        final_ir_path=final_path,
        root_count=root_count,
        pass_sequence=sequence,
        pass_invocations=len(sequence),
        states_analyzed=states_evaluated,
        opt_runs=opt_runs,
        time_ms=(time.perf_counter() - start) * 1000,
        stop_reason=stop_reason,
        error_message=error_message,
    )
    _write_greedy_summary(out_dir / "greedy_summary.md", row, max_rounds, allow_nonimproving)
    if run_dir is not None:
        _upsert_baseline_result(run_dir / "baseline_results.csv", row)
    return row


def run_random_single_pass_baseline(
    root_ir: Path,
    valid_passes: list[str],
    tools: dict,
    out_dir: Path,
    *,
    max_rounds: int,
    random_trials: int,
    seed: int,
    timeout: int,
    objective: str = "ir-inst-count",
    pass_registry: PassRegistry | None = None,
) -> dict:
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")

    root_ir = Path(root_ir)
    out_dir = Path(out_dir)
    if (out_dir / "states" / "S0000" / "input.ll").exists():
        out_dir = out_dir / "baselines" / "random_single_pass"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _infer_run_dir_from_baseline_out_dir(out_dir, "random_single_pass")
    program = run_dir.name if run_dir is not None else out_dir.name
    root_count = count_ir_instructions(root_ir)
    start = time.perf_counter()

    trial_rows: list[dict] = []
    trial_paths: dict[int, list[dict]] = {}
    trial_results: list[dict] = []

    for trial_index in range(max(0, random_trials)):
        rng = random.Random(seed + trial_index)
        trial_dir = out_dir / "trials" / f"T{trial_index:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        current_ir = root_ir
        sequence: list[str] = []
        states_evaluated = 1
        opt_runs = 0
        status = "success"
        error_message = ""
        stop_reason = "max_rounds_reached" if max_rounds <= 0 else ""
        path_rows: list[dict] = []

        try:
            for round_index in range(max(0, max_rounds)):
                input_hash = canonical_hash(current_ir)
                input_count = count_ir_instructions(current_ir)
                round_dir = trial_dir / "rounds" / f"R{round_index:04d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                active: list[dict] = []
                opt_runs_this_round = 0

                for pass_index, pass_name in enumerate(valid_passes):
                    output = round_dir / f"{pass_index:04d}_{_safe_pass_stem(pass_name)}.ll"
                    result = _run_opt_safely(str(tools["opt"]), current_ir, resolve_pipeline_sequence([pass_name], pass_registry), output, timeout)
                    opt_runs += 1
                    opt_runs_this_round += 1
                    if not result.success or not output.exists():
                        continue
                    output_hash = canonical_hash(output)
                    if output_hash == input_hash:
                        continue
                    output_count = count_ir_instructions(output)
                    active.append(
                        {
                            "pass": pass_name,
                            "pass_index": pass_index,
                            "output_path": output,
                            "output_hash": output_hash,
                            "output_count": output_count,
                        }
                    )

                if not active:
                    stop_reason = "no_active_passes"
                    path_rows.append(
                        _random_best_path_row(
                            trial_index=trial_index,
                            round_index=round_index,
                            input_ir=current_ir,
                            selected=None,
                            input_hash=input_hash,
                            input_count=input_count,
                            active_passes=0,
                            tested_passes=len(valid_passes),
                            opt_runs_this_round=opt_runs_this_round,
                            stop_reason=stop_reason,
                        )
                    )
                    break

                chosen = rng.choice(active)
                path_rows.append(
                    _random_best_path_row(
                        trial_index=trial_index,
                        round_index=round_index,
                        input_ir=current_ir,
                        selected=chosen,
                        input_hash=input_hash,
                        input_count=input_count,
                        active_passes=len(active),
                        tested_passes=len(valid_passes),
                        opt_runs_this_round=opt_runs_this_round,
                        stop_reason="",
                    )
                )
                sequence.append(chosen["pass"])
                current_ir = chosen["output_path"]
                states_evaluated += 1
                stop_reason = "max_rounds_reached"
        except Exception as exc:  # pragma: no cover - defensive path for real tool failures.
            status = "failed"
            stop_reason = "failed"
            error_message = str(exc)

        final_path = trial_dir / "final.ll"
        if status == "success" and Path(current_ir).exists():
            shutil.copyfile(current_ir, final_path)
        final_count = count_ir_instructions(final_path) if final_path.exists() else None
        final_hash = canonical_hash(final_path) if final_path.exists() else ""
        trial_row = {
            "trial": str(trial_index),
            "status": status,
            "final_ir_path": str(final_path) if final_path.exists() else "",
            "final_ir_hash": final_hash,
            "final_ir_inst_count": "" if final_count is None else str(final_count),
            "root_ir_inst_count": str(root_count),
            "ir_inst_delta": "" if final_count is None else str(final_count - root_count),
            "ir_inst_reduction_pct": "" if final_count is None else _format_float(_reduction_pct(root_count, final_count)),
            "pass_sequence": _join_sequence(sequence),
            "final_sequence_length": str(len(sequence)),
            "states_evaluated": str(states_evaluated),
            "opt_runs": str(opt_runs),
            "stop_reason": stop_reason,
            "time_ms": "0.000",
            "error_message": error_message,
        }
        trial_rows.append(trial_row)
        trial_paths[trial_index] = path_rows
        if status == "success" and final_count is not None:
            trial_results.append(
                {
                    "trial": trial_index,
                    "final_count": final_count,
                    "final_path": final_path,
                    "sequence": sequence,
                    "states_evaluated": states_evaluated,
                    "opt_runs": opt_runs,
                    "stop_reason": stop_reason,
                    "status": status,
                    "error_message": error_message,
                }
            )

    _write_csv(out_dir / "random_trials.csv", RANDOM_BASELINE_TRIAL_FIELDS, trial_rows)

    if trial_results:
        best = min(
            trial_results,
            key=lambda row: (row["final_count"], len(row["sequence"]), row["trial"]),
        )
        best_path = out_dir / "random_best_final.ll"
        shutil.copyfile(best["final_path"], best_path)
        best_trial = best["trial"]
        _write_csv(out_dir / "random_best_path.csv", RANDOM_BEST_PATH_FIELDS, trial_paths.get(best_trial, []))
        row = _result_row(
            program=program,
            method="random_single_pass_best",
            status=best["status"],
            final_ir_path=best_path,
            root_count=root_count,
            pass_sequence=list(best["sequence"]),
            pass_invocations=len(best["sequence"]),
            states_analyzed=best["states_evaluated"],
            opt_runs=best["opt_runs"],
            time_ms=(time.perf_counter() - start) * 1000,
            stop_reason=best["stop_reason"],
            error_message=best["error_message"],
        )
        row["best_trial"] = str(best_trial)
    else:
        best_path = out_dir / "random_best_final.ll"
        shutil.copyfile(root_ir, best_path)
        _write_csv(out_dir / "random_best_path.csv", RANDOM_BEST_PATH_FIELDS, [])
        row = _result_row(
            program=program,
            method="random_single_pass_best",
            status="failed" if random_trials <= 0 else "success",
            final_ir_path=best_path,
            root_count=root_count,
            pass_sequence=[],
            pass_invocations=0,
            states_analyzed=1 if random_trials > 0 else 0,
            opt_runs=0,
            time_ms=(time.perf_counter() - start) * 1000,
            stop_reason="no_random_trials" if random_trials <= 0 else "no_successful_trials",
            error_message="no random trials requested" if random_trials <= 0 else "",
        )
        row["best_trial"] = ""

    _write_random_summary(out_dir / "random_summary.md", row, random_trials, seed, max_rounds)
    if run_dir is not None:
        _upsert_baseline_result(run_dir / "baseline_results.csv", row)
    row["_trial_rows"] = trial_rows
    return row


def _random_single_pass_rows(
    program: str,
    root_ir: Path,
    baseline_dir: Path,
    tools: dict[str, str],
    valid_passes: list[str],
    root_count: int,
    *,
    max_rounds: int,
    random_trials: int,
    seed: int,
    timeout: int,
    jobs: int,
) -> tuple[dict, list[dict]]:
    start = time.perf_counter()
    trial_rows: list[dict] = []
    trial_outputs: list[tuple[int, Path, list[str], str]] = []
    total_opt_runs = 0
    total_states_analyzed = 0

    for trial in range(max(0, random_trials)):
        rng = random.Random(seed + trial)
        current_ir = root_ir
        sequence: list[str] = []
        status = "success"
        error_message = ""
        try:
            for round_index in range(max(0, max_rounds)):
                profile_dir = baseline_dir / "random_profiles" / f"T{trial:04d}" / f"R{round_index:04d}"
                profile_dir.mkdir(parents=True, exist_ok=True)
                rows = profile_passes(
                    current_ir,
                    valid_passes,
                    tools,
                    profile_dir,
                    jobs,
                    timeout,
                    program=program,
                    state_id=f"RANDOM_T{trial:04d}_R{round_index:04d}",
                    depth=round_index,
                    parent_state_id="",
                    transition_pass=sequence[-1] if sequence else "",
                )
                total_states_analyzed += 1
                total_opt_runs += len(valid_passes)
                active = _active_outputs(rows)
                if not active:
                    break
                chosen_pass, chosen_ir, _chosen_count = rng.choice(active)
                sequence.append(chosen_pass)
                current_ir = chosen_ir
        except Exception as exc:  # pragma: no cover - defensive path for real tool failures.
            status = "failed"
            error_message = str(exc)

        final_count = count_ir_instructions(current_ir) if status == "success" and Path(current_ir).exists() else None
        if final_count is not None:
            trial_outputs.append((final_count, Path(current_ir), sequence, status))
        trial_rows.append(
            {
                "trial": str(trial),
                "status": status,
                "final_ir_inst_count": "" if final_count is None else str(final_count),
                "ir_inst_delta": "" if final_count is None else str(final_count - root_count),
                "ir_inst_reduction_pct": "" if final_count is None else _format_float(_reduction_pct(root_count, final_count)),
                "pass_sequence": _join_sequence(sequence),
                "error_message": error_message,
            }
        )

    if not trial_outputs:
        output_path = baseline_dir / "random_single_pass_best.ll"
        shutil.copyfile(root_ir, output_path)
        row = _result_row(
            program=program,
            method="random_single_pass_best",
            status="failed" if random_trials <= 0 else "success",
            final_ir_path=output_path,
            root_count=root_count,
            pass_sequence=[],
            pass_invocations=0,
            states_analyzed=total_states_analyzed,
            opt_runs=total_opt_runs,
            time_ms=(time.perf_counter() - start) * 1000,
            stop_reason="no_random_trials" if random_trials <= 0 else "",
            error_message="no random trials requested" if random_trials <= 0 else "",
        )
        return row, trial_rows

    best_count, best_ir, best_sequence, best_status = min(
        trial_outputs,
        key=lambda item: (item[0], _join_sequence(item[2])),
    )
    output_path = baseline_dir / "random_single_pass_best.ll"
    shutil.copyfile(best_ir, output_path)
    return (
        _result_row(
            program=program,
            method="random_single_pass_best",
            status=best_status,
            final_ir_path=output_path,
            root_count=root_count,
            pass_sequence=best_sequence,
            pass_invocations=len(best_sequence),
            states_analyzed=total_states_analyzed,
            opt_runs=total_opt_runs,
            time_ms=(time.perf_counter() - start) * 1000,
            stop_reason="",
            error_message="",
        ),
        trial_rows,
    )


def _llvm_default_row(
    program: str,
    method: str,
    pipeline: str,
    root_ir: Path,
    baseline_dir: Path,
    tools: dict[str, str],
    root_count: int,
    timeout: int,
) -> dict:
    start = time.perf_counter()
    output_path = baseline_dir / f"{method}.ll"
    result = run_opt_raw_pipeline(tools["opt"], root_ir, pipeline, output_path, timeout)
    status = "success" if result.success and output_path.exists() else "unsupported"
    return _result_row(
        program=program,
        method=method,
        status=status,
        final_ir_path=output_path if output_path.exists() else None,
        root_count=root_count,
        pass_sequence=[pipeline],
        pass_invocations=1,
        states_analyzed=0,
        opt_runs=1,
        time_ms=(time.perf_counter() - start) * 1000,
        stop_reason="",
        error_message="" if status == "success" else _error_text(result),
    )


def _active_outputs(rows: list[dict]) -> list[tuple[str, Path, int]]:
    active: list[tuple[str, Path, int]] = []
    for row in rows:
        output = Path(row.get("output_path", ""))
        if row.get("success") == "true" and row.get("active") == "true" and output.exists():
            active.append((row.get("pass", ""), output, count_ir_instructions(output)))
    return active


def _greedy_path_row(
    *,
    round_index: int,
    input_ir: Path,
    selected: dict | None,
    input_hash: str,
    input_count: int,
    active_passes: int,
    tested_passes: int,
    opt_runs_this_round: int,
    stop_reason: str,
) -> dict:
    output_path = selected.get("output_path") if selected else None
    output_count = selected.get("output_count") if selected else None
    output_hash = selected.get("output_hash") if selected else ""
    return {
        "round": str(round_index),
        "input_ir_path": str(input_ir),
        "selected_pass": selected.get("pass", "") if selected else "",
        "output_ir_path": "" if output_path is None else str(output_path),
        "input_hash": input_hash,
        "output_hash": output_hash,
        "input_ir_inst_count": str(input_count),
        "output_ir_inst_count": "" if output_count is None else str(output_count),
        "inst_delta": "" if output_count is None else str(output_count - input_count),
        "active_passes": str(active_passes),
        "tested_passes": str(tested_passes),
        "opt_runs_this_round": str(opt_runs_this_round),
        "stop_reason": stop_reason,
    }


def _random_best_path_row(
    *,
    trial_index: int,
    round_index: int,
    input_ir: Path,
    selected: dict | None,
    input_hash: str,
    input_count: int,
    active_passes: int,
    tested_passes: int,
    opt_runs_this_round: int,
    stop_reason: str,
) -> dict:
    row = _greedy_path_row(
        round_index=round_index,
        input_ir=input_ir,
        selected=selected,
        input_hash=input_hash,
        input_count=input_count,
        active_passes=active_passes,
        tested_passes=tested_passes,
        opt_runs_this_round=opt_runs_this_round,
        stop_reason=stop_reason,
    )
    return {"trial": str(trial_index), **row}


def _result_row(
    *,
    program: str,
    method: str,
    status: str,
    final_ir_path: Path | None,
    root_count: int,
    pass_sequence: list[str],
    pass_invocations: int,
    states_analyzed: int,
    opt_runs: int,
    time_ms: float,
    stop_reason: str,
    error_message: str,
) -> dict:
    final_count = count_ir_instructions(final_ir_path) if final_ir_path and final_ir_path.exists() else None
    final_hash = canonical_hash(final_ir_path) if final_ir_path and final_ir_path.exists() else ""
    return {
        "program": program,
        "method": method,
        "status": status,
        "final_ir_path": "" if not final_ir_path else str(final_ir_path),
        "final_ir_hash": final_hash,
        "final_ir_inst_count": "" if final_count is None else str(final_count),
        "root_ir_inst_count": str(root_count),
        "ir_inst_delta": "" if final_count is None else str(final_count - root_count),
        "ir_inst_reduction_pct": "" if final_count is None else _format_float(_reduction_pct(root_count, final_count)),
        "pass_sequence": _join_sequence(pass_sequence),
        "final_sequence_length": str(pass_invocations),
        "states_evaluated": str(states_analyzed),
        "opt_runs": str(opt_runs),
        "time_ms": f"{time_ms:.3f}",
        "stop_reason": stop_reason,
        "error_message": error_message,
    }


def _valid_passes_in_config_order(
    run_dir: Path,
    configured_passes: list[str],
    root_ir: Path,
    tools: dict[str, str],
    timeout: int,
    pass_registry: PassRegistry | None = None,
) -> tuple[list[str], list[str], str]:
    valid_rows = _read_csv(run_dir / "valid_passes.csv")
    if not valid_rows:
        valid, invalid_rows = validate_passes(root_ir, configured_passes, tools, run_dir, timeout, pass_registry=pass_registry)
        invalid = [row.get("pass", "") for row in invalid_rows if row.get("pass")]
        return valid, invalid, "valid_passes.csv missing; validated configured pass list on root IR"
    valid_set = {row.get("pass", "") for row in valid_rows if row.get("pass")}
    valid = [pass_name for pass_name in configured_passes if pass_name in valid_set]
    skipped = [pass_name for pass_name in configured_passes if pass_name not in valid_set]
    return valid, skipped, ""


def _write_greedy_summary(path: Path, row: dict, max_rounds: int, allow_nonimproving: bool) -> None:
    pass_sequence = row.get("pass_sequence", "")
    root_count = _parse_int(row.get("root_ir_inst_count", ""))
    final_count = _parse_int(row.get("final_ir_inst_count", ""))
    reduction = row.get("ir_inst_reduction_pct", "")
    lines = [
        "# Greedy Single-Pass Baseline Summary",
        "",
        "## Method",
        "",
        "At each round, this baseline runs every valid pass once on the current IR,",
        "keeps active passes whose canonical output hash changes, and chooses the",
        "active pass with the smallest output IR instruction count.",
        "",
        "## Configuration",
        "",
        f"- max_rounds: {max_rounds}",
        f"- allow_nonimproving: {str(allow_nonimproving).lower()}",
        "",
        "## Result",
        "",
        f"- root IR instructions: {'' if root_count is None else root_count}",
        f"- final IR instructions: {'' if final_count is None else final_count}",
        f"- instruction delta: {row.get('ir_inst_delta', '')}",
        f"- reduction pct: {reduction}",
        f"- pass_sequence: {pass_sequence}",
        f"- final_sequence_length: {row.get('final_sequence_length', '')}",
        f"- states_evaluated: {row.get('states_evaluated', '')}",
        f"- opt_runs: {row.get('opt_runs', '')}",
        f"- stop_reason: {row.get('stop_reason', '')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_random_summary(path: Path, row: dict, random_trials: int, seed: int, max_rounds: int) -> None:
    lines = [
        "# Random Single-Pass Baseline Summary",
        "",
        f"- method: {row.get('method', 'random_single_pass_best')}",
        f"- random_trials: {random_trials}",
        f"- seed: {seed}",
        f"- max_rounds: {max_rounds}",
        f"- best_trial: {row.get('best_trial', '')}",
        f"- final_ir_inst_count: {row.get('final_ir_inst_count', '')}",
        f"- root_ir_inst_count: {row.get('root_ir_inst_count', '')}",
        f"- ir_inst_delta: {row.get('ir_inst_delta', '')}",
        f"- reduction_pct: {row.get('ir_inst_reduction_pct', '')}",
        f"- best_pass_sequence: {row.get('pass_sequence', '')}",
        f"- final_sequence_length: {row.get('final_sequence_length', '')}",
        f"- states_evaluated: {row.get('states_evaluated', '')}",
        f"- opt_runs: {row.get('opt_runs', '')}",
        f"- stop_reason: {row.get('stop_reason', '')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _upsert_baseline_result(path: Path, row: dict) -> None:
    rows = [existing for existing in _read_csv(path) if existing.get("method") != row.get("method")]
    rows.append(row)
    _write_csv(path, BASELINE_RESULT_FIELDS, rows)


def _infer_run_dir_from_greedy_out_dir(out_dir: Path) -> Path | None:
    return _infer_run_dir_from_baseline_out_dir(out_dir, "greedy_single_pass")


def _infer_run_dir_from_baseline_out_dir(out_dir: Path, expected_name: str) -> Path | None:
    out_dir = Path(out_dir)
    if out_dir.name == expected_name and out_dir.parent.name == "baselines":
        return out_dir.parent.parent
    if out_dir.name == "baselines":
        return out_dir.parent
    return None


def _safe_pass_stem(pass_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pass_name)
    return safe.strip("._-") or "pass"


def _count_nonduplicate_states(path: Path) -> int:
    rows = _read_csv(path)
    if not rows:
        return 1
    count = 0
    for row in rows:
        is_duplicate = row.get("is_duplicate", "").strip().lower()
        if is_duplicate not in {"true", "1", "yes"}:
            count += 1
    return count


def _count_optimizer_opt_runs(run_dir: Path) -> int:
    events = _read_csv(Path(run_dir) / "optimizer_events.csv")
    if events:
        interesting = {"apply_batch", "analyze_state"}
        return sum(1 for row in events if row.get("event_type") in interesting)
    transitions = _read_csv(Path(run_dir) / "state_dag.csv")
    return len(transitions)


def _batch_optimizer_costs(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    timing = _first_row(run_dir / "optimizer_timing.csv")
    state_dirs = _optimizer_state_dirs(run_dir)
    profiling_time = pair_time = analysis_time = 0.0
    batch_validation_time = 0.0
    validation_opt_invocations = 0
    profile_opt_invocations = 0
    pair_opt_invocations = 0

    for state_dir in state_dirs:
        summary = _first_row(state_dir / "per_state_summary.csv")
        profiling_time += _parse_float(summary.get("profile_time_ms")) or 0.0
        pair_time += _parse_float(summary.get("pair_time_ms")) or 0.0
        analysis_time += _parse_float(summary.get("total_time_ms")) or 0.0
        profile_opt_invocations += len(_read_csv(state_dir / "pass_profile.csv"))
        for row in _read_csv(state_dir / "pair_relation.csv"):
            if row.get("dynamic_relation") == "not_tested" or row.get("failure_kind") == "max_pairs":
                continue
            if row.get("pair_test_opt_runs") not in {"", None}:
                pair_opt_invocations += int(_parse_float(row.get("pair_test_opt_runs")) or 0)
            else:
                pair_opt_invocations += 2
        for row in _read_csv(state_dir / "batch_validation.csv"):
            batch_validation_time += _parse_float(row.get("time_ms")) or 0.0
            validation_opt_invocations += _parse_int(row.get("tested_orders")) or 0

    batch_apply_time = timing.get("batch_apply_time_ms", "")
    batch_apply_invocations = _parse_int(timing.get("batch_apply_opt_invocations", ""))
    if batch_apply_invocations is None:
        batch_apply_invocations = len(_read_csv(run_dir / "batch_state_transitions.csv"))

    optimizer_total = timing.get("optimizer_total_time_ms", "")
    if _parse_float(optimizer_total) is None:
        optimizer_total_value = analysis_time + batch_validation_time
        batch_apply_value = _parse_float(batch_apply_time)
        if batch_apply_value is not None:
            optimizer_total_value += batch_apply_value
        optimizer_total = _format_ms(optimizer_total_value)

    total_opt_invocations = (
        _pass_validation_invocations(run_dir)
        + profile_opt_invocations
        + pair_opt_invocations
        + validation_opt_invocations
        + batch_apply_invocations
    )
    return {
        "optimizer_total_time_ms": _format_existing_ms(optimizer_total),
        "analysis_time_ms": _format_ms(analysis_time),
        "profiling_time_ms": _format_ms(profiling_time),
        "pair_testing_time_ms": _format_ms(pair_time),
        "batch_validation_time_ms": _format_ms(batch_validation_time),
        "batch_apply_time_ms": _format_existing_ms(batch_apply_time),
        "total_opt_invocations": str(total_opt_invocations),
    }


def _optimizer_state_dirs(run_dir: Path) -> list[Path]:
    rows = _read_csv(run_dir / "states.csv")
    dirs: list[Path] = []
    if rows:
        for row in rows:
            if _is_true(row.get("is_duplicate")):
                continue
            state_dir = row.get("state_dir", "")
            if state_dir:
                dirs.append(Path(state_dir))
            elif row.get("state_id"):
                dirs.append(run_dir / "states" / row["state_id"])
    else:
        states_root = run_dir / "states"
        if states_root.exists():
            dirs.extend(path for path in states_root.iterdir() if path.is_dir())
    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for state_dir in dirs:
        try:
            key = state_dir.resolve()
        except OSError:
            key = state_dir
        if key not in seen:
            seen.add(key)
            unique_dirs.append(state_dir)
    return unique_dirs


def _pass_validation_invocations(run_dir: Path) -> int:
    return len(_read_csv(run_dir / "valid_passes.csv")) + len(_read_csv(run_dir / "invalid_passes.csv"))


def _write_method_comparison(run_dir: Path, rows: list[dict]) -> Path:
    path = Path(run_dir) / "method_comparison.md"
    best = _best_method_row(rows)
    table_rows: list[list[str]] = []
    for row in rows:
        method = row.get("method", "")
        display = f"**{method}**" if best and method == best.get("method") else method
        table_rows.append(
            [
                display,
                row.get("status", ""),
                row.get("final_ir_inst_count", ""),
                row.get("ir_inst_delta", ""),
                row.get("ir_inst_reduction_pct", ""),
                row.get("states_evaluated", ""),
                row.get("opt_runs", ""),
                row.get("final_sequence_length", ""),
            ]
        )
    batch = _row_by_method(rows, "batch_optimizer")
    lines = [
        "# Method Comparison",
        "",
        "Objective values are evaluation only: objective is not proof of commutation, batch correctness, or pass independence.",
        "",
        *_markdown_table(
            [
                "method",
                "status",
                "final IR inst count",
                "delta",
                "reduction %",
                "states evaluated",
                "opt runs",
                "final sequence length",
            ],
            table_rows,
        ),
        "",
        "under the IR instruction count objective in this run:",
        f"- best method by final IR instruction count: {_method_count_text(best)}",
        f"- batch_optimizer beats greedy_single_pass: {_beats_text(batch, _row_by_method(rows, 'greedy_single_pass'))}",
        f"- batch_optimizer beats random_single_pass_best: {_beats_text(batch, _row_by_method(rows, 'random_single_pass_best'))}",
        f"- batch_optimizer beats default_O0: {_beats_text(batch, _row_by_method(rows, 'default_O0'))}",
        f"- batch_optimizer beats default_O2: {_beats_text(batch, _row_by_method(rows, 'default_O2'))}",
        f"- batch_optimizer beats default_Oz: {_beats_text(batch, _row_by_method(rows, 'default_Oz'))}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |")
    return lines


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _best_method_row(rows: list[dict]) -> dict | None:
    candidates = []
    for row in rows:
        if row.get("status") != "success":
            continue
        count = _parse_int(row.get("final_ir_inst_count", ""))
        if count is None:
            continue
        candidates.append((count, row.get("method", ""), row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _row_by_method(rows: list[dict], method: str) -> dict | None:
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def _method_count_text(row: dict | None) -> str:
    if not row:
        return "N/A"
    return f"{row.get('method', '')} ({row.get('final_ir_inst_count', '')})"


def _beats_text(left: dict | None, right: dict | None) -> str:
    if not left or not right:
        return "N/A"
    left_count = _parse_int(left.get("final_ir_inst_count", ""))
    right_count = _parse_int(right.get("final_ir_inst_count", ""))
    if left_count is None or right_count is None:
        return "N/A"
    if left_count < right_count:
        return "yes"
    if left_count == right_count:
        return "tie"
    return "no"


def _normalize_methods(methods: list[str] | None) -> set[str]:
    requested = {"all"} if not methods else {
        part.strip()
        for method in methods
        for part in method.split(",")
        if part.strip()
    }
    unknown = requested - SUPPORTED_COMPARE_METHODS
    if unknown:
        raise ValueError("unsupported baseline method(s): " + ", ".join(sorted(unknown)))
    if "all" in requested:
        return {"all"}
    normalized: set[str] = set()
    aliases = {
        "optimized": "optimized_pipeline",
        "config": "config_order_once",
        "default_O0": "default",
        "default_O2": "default",
        "default_Oz": "default",
        "greedy": "greedy_single_pass",
        "random": "random_single_pass_best",
        "batch": "batch_optimizer",
    }
    for method in requested:
        normalized.add(aliases.get(method, method))
    return normalized


def _method_enabled(methods: set[str], method: str) -> bool:
    return "all" in methods or method in methods


def _skipped_note(skipped_passes: list[str]) -> str:
    if not skipped_passes:
        return ""
    return "skipped invalid passes: " + ";".join(skipped_passes)


def _split_optimized_pipeline(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", "").replace(";", ",").split(",") if part.strip()]


def _normalized_pipeline_text(text: str) -> str:
    return ",".join(_split_optimized_pipeline(text))


def _join_sequence(passes: list[str]) -> str:
    return ";".join(passes)


def _join_notes(notes: list[str]) -> str:
    return "; ".join(note for note in notes if note)


def _tool_paths(metadata: dict) -> dict[str, str]:
    return {
        name: details["path"]
        for name, details in metadata.get("tools", {}).items()
        if details.get("path")
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _reduction_pct(root_count: int, final_count: int) -> float:
    if root_count <= 0:
        return 0.0
    return ((root_count - final_count) / root_count) * 100.0


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _format_existing_ms(value: object) -> str:
    parsed = _parse_float(value)
    return "" if parsed is None else _format_ms(parsed)


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _error_text(result: RunResult) -> str:
    return (result.stderr or result.failure_kind or "failed").strip()


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
