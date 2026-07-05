from __future__ import annotations

import csv
import shutil
import time
from collections import Counter
from pathlib import Path

from .mainline import expand_inputs


REDUCTION_STUDY_RUN_FIELDS = [
    "program",
    "input_path",
    "status",
    "optimize_status",
    "reduction_status",
    "evidence_status",
    "states_reached",
    "transitions",
    "time_ms",
    "error_message",
]

REDUCTION_STUDY_SUMMARY_FIELDS = [
    "program",
    "total_states",
    "max_depth",
    "total_active_passes",
    "total_tested_pairs",
    "commute_pairs",
    "order_sensitive_pairs",
    "unknown_pairs",
    "total_batch_candidates",
    "total_certified_batches",
    "total_executable_batches",
    "total_executed_transitions",
    "total_skipped_batches",
    "total_dropped_active_passes",
    "avg_local_reduction_log10",
    "max_local_reduction_log10",
    "selected_path_steps",
    "final_pipeline_length",
    "replay_hashes_match",
]

EVIDENCE_QUALITY_SUMMARY_FIELDS = [
    "program",
    "selected_path_batches",
    "selected_strong_certificates",
    "selected_weak_certificates",
    "selected_rejected",
    "executed_batches",
    "executed_strong_certificates",
    "executed_weak_certificates",
    "executed_rejected",
    "dropped_active_passes",
    "replay_status",
    "replay_hashes_match",
]


def run_reduction_study(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    optimizer_mode: str,
    objective: str,
    max_rounds: int,
    max_states: int,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    summarize_components: bool = False,
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
    evidence_rows: list[dict] = []
    component_run_dirs: list[Path] = []
    plans = _program_plans(input_paths, out_dir)

    for program, input_path, program_dir in plans:
        start = time.perf_counter()
        optimize_dir = program_dir / "optimize"
        status = "success"
        optimize_status = "not_run"
        reduction_status = "not_run"
        evidence_status = "not_run"
        error_message = ""
        states_reached = ""
        transitions = ""

        try:
            program_dir.mkdir(parents=True, exist_ok=True)
            run_optimizer(
                input_path,
                optimize_dir,
                passes_path,
                mode=optimizer_mode,
                objective=objective,
                max_rounds=max_rounds,
                max_states=max_states,
                max_batches_per_state=20,
                validate_batches=validate_batches,
                allow_sampled_batches=False,
                jobs=jobs,
                timeout=timeout,
                max_pairs=max_pairs,
            )
            optimize_status = "success"

            reduction_result = run_reduction_summary(optimize_dir)
            reduction_status = "success"

            replay = _try_replay(optimize_dir, timeout)

            evidence_result = run_evidence_pack(optimize_dir)
            evidence_status = "success"

            _copy_program_artifacts(program_dir, optimize_dir)
            reduction = _first_row(optimize_dir / "reduction_summary.csv")
            reduction_by_state = _read_csv(optimize_dir / "reduction_by_state.csv")
            evidence = _first_row(optimize_dir / "evidence_pack.csv")
            replay = _first_row(optimize_dir / "pipeline_replay.csv") or replay

            summary_row = _study_summary_row(program, reduction, reduction_by_state, replay)
            evidence_row = _evidence_row(program, evidence, replay)
            summary_rows.append(summary_row)
            evidence_rows.append(evidence_row)
            component_run_dirs.append(optimize_dir)
            states_reached = summary_row.get("total_states", "") or str(reduction_result.get("states", ""))
            transitions = summary_row.get("total_executed_transitions", "") or str(evidence_result.get("executed_batches", ""))
        except Exception as exc:
            status = "failed"
            if optimize_status == "not_run":
                optimize_status = "failed"
            elif reduction_status == "not_run":
                reduction_status = "failed"
            elif evidence_status == "not_run":
                evidence_status = "failed"
            error_message = str(exc)
            if not continue_on_error:
                run_rows.append(
                    _run_row(
                        program,
                        input_path,
                        status,
                        optimize_status,
                        reduction_status,
                        evidence_status,
                        states_reached,
                        transitions,
                        start,
                        error_message,
                    )
                )
                _write_outputs(out_dir, run_rows, summary_rows, evidence_rows, passes_path, optimizer_mode, max_rounds)
                raise
        finally:
            if not run_rows or run_rows[-1].get("program") != program:
                run_rows.append(
                    _run_row(
                        program,
                        input_path,
                        status,
                        optimize_status,
                        reduction_status,
                        evidence_status,
                        states_reached,
                        transitions,
                        start,
                        error_message,
                    )
                )

    summary_path = _write_outputs(out_dir, run_rows, summary_rows, evidence_rows, passes_path, optimizer_mode, max_rounds)
    component_result = _try_component_summary(component_run_dirs, out_dir / "components") if summarize_components else {}
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = sum(1 for row in run_rows if row.get("status") == "failed")
    result = {
        "out_dir": str(out_dir),
        "programs": len(run_rows),
        "successes": successes,
        "failures": failures,
        "reduction_study_runs_csv": str(out_dir / "reduction_study_runs.csv"),
        "reduction_study_summary_csv": str(out_dir / "reduction_study_summary.csv"),
        "evidence_quality_summary_csv": str(out_dir / "evidence_quality_summary.csv"),
        "reduction_study_summary_md": str(summary_path),
    }
    if component_result:
        result["component_summary_md"] = component_result.get("component_summary_md", "")
    return result


def run_optimizer(input_path: Path, out_dir: Path, passes_path: Path, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(input_path, out_dir, passes_path, **kwargs)


def run_reduction_summary(run_dir: Path) -> dict:
    from .reduction_summary import summarize_reduction

    return summarize_reduction(run_dir)


def run_evidence_pack(run_dir: Path) -> dict:
    from .evidence_pack import export_evidence_pack

    return export_evidence_pack(run_dir)


def run_component_summary(run_dirs: list[Path], out_dir: Path) -> dict:
    from .component_summary import summarize_components

    return summarize_components(run_dirs=run_dirs, out_dir=out_dir)


def _try_component_summary(run_dirs: list[Path], out_dir: Path) -> dict:
    if not run_dirs:
        return {}
    try:
        return run_component_summary(run_dirs, out_dir)
    except Exception:
        return {}


def run_replay(run_dir: Path, timeout: int = 10) -> dict:
    from .pipeline_replay import replay_optimized_pipeline

    return replay_optimized_pipeline(run_dir, timeout=timeout)


def _try_replay(optimize_dir: Path, timeout: int) -> dict:
    try:
        return run_replay(optimize_dir, timeout=timeout)
    except Exception:
        return {}


def _copy_program_artifacts(program_dir: Path, optimize_dir: Path) -> None:
    for name in [
        "reduction_by_state.csv",
        "reduction_summary.csv",
        "reduction_summary.md",
        "evidence_pack.csv",
        "evidence_pack.md",
    ]:
        source = optimize_dir / name
        if source.exists():
            shutil.copyfile(source, program_dir / name)


def _study_summary_row(program: str, reduction: dict, reduction_by_state: list[dict], replay: dict) -> dict:
    return {
        "program": program,
        "total_states": _value(reduction, "total_states"),
        "max_depth": _value(reduction, "max_depth"),
        "total_active_passes": _value(reduction, "total_active_passes"),
        "total_tested_pairs": _value(reduction, "total_tested_pairs"),
        "commute_pairs": _value(reduction, "total_commute_pairs", "commute_pairs"),
        "order_sensitive_pairs": _value(reduction, "total_order_sensitive_pairs", "order_sensitive_pairs"),
        "unknown_pairs": _value(reduction, "total_unknown_pairs", "unknown_pairs", default=str(_sum(reduction_by_state, "unknown_pairs"))),
        "total_batch_candidates": _value(reduction, "total_batch_candidates"),
        "total_certified_batches": _value(reduction, "total_certified_batches"),
        "total_executable_batches": _value(reduction, "total_executable_batches"),
        "total_executed_transitions": _value(reduction, "total_executed_transitions"),
        "total_skipped_batches": _value(reduction, "total_skipped_batches"),
        "total_dropped_active_passes": _value(reduction, "total_dropped_active_passes"),
        "avg_local_reduction_log10": _value(reduction, "avg_local_reduction_log10"),
        "max_local_reduction_log10": _value(reduction, "max_local_reduction_log10"),
        "selected_path_steps": _value(reduction, "selected_path_steps"),
        "final_pipeline_length": _value(reduction, "final_pipeline_length"),
        "replay_hashes_match": replay.get("hashes_match") or replay.get("replay_hashes_match", ""),
    }


def _evidence_row(program: str, evidence: dict, replay: dict) -> dict:
    return {
        "program": program,
        "selected_path_batches": _value(evidence, "selected_path_batches"),
        "selected_strong_certificates": _value(evidence, "selected_strong_certificates"),
        "selected_weak_certificates": _value(evidence, "selected_weak_certificates"),
        "selected_rejected": _value(evidence, "selected_rejected"),
        "executed_batches": _value(evidence, "executed_batches"),
        "executed_strong_certificates": _value(evidence, "executed_strong_certificates"),
        "executed_weak_certificates": _value(evidence, "executed_weak_certificates"),
        "executed_rejected": _value(evidence, "executed_rejected"),
        "dropped_active_passes": _value(evidence, "dropped_active_passes"),
        "replay_status": evidence.get("replay_status") or replay.get("replay_status", ""),
        "replay_hashes_match": evidence.get("replay_hashes_match") or replay.get("hashes_match") or replay.get("replay_hashes_match", ""),
    }


def _write_outputs(
    out_dir: Path,
    run_rows: list[dict],
    summary_rows: list[dict],
    evidence_rows: list[dict],
    passes_path: Path,
    optimizer_mode: str,
    max_rounds: int,
) -> Path:
    _write_csv(out_dir / "reduction_study_runs.csv", REDUCTION_STUDY_RUN_FIELDS, run_rows)
    _write_csv(out_dir / "reduction_study_summary.csv", REDUCTION_STUDY_SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "evidence_quality_summary.csv", EVIDENCE_QUALITY_SUMMARY_FIELDS, evidence_rows)
    return _write_summary(out_dir, run_rows, summary_rows, evidence_rows, passes_path, optimizer_mode, max_rounds)


def _write_summary(
    out_dir: Path,
    run_rows: list[dict],
    summary_rows: list[dict],
    evidence_rows: list[dict],
    passes_path: Path,
    optimizer_mode: str,
    max_rounds: int,
) -> Path:
    successes = sum(1 for row in run_rows if row.get("status") == "success")
    failures = sum(1 for row in run_rows if row.get("status") == "failed")
    lines = [
        "# Reduction Study Summary",
        "",
        "## Overall",
        "",
        f"- total programs: {len(run_rows)}",
        f"- successful programs: {successes}",
        f"- failed programs: {failures}",
        f"- pass set: {passes_path}",
        f"- mode: {optimizer_mode}",
        f"- max rounds: {max_rounds}",
        "",
        "## Search-Space Reduction",
        "",
        *_markdown_table(
            ["program", "states", "active pairs", "batch candidates", "certified batches", "avg reduction log10", "max reduction log10"],
            [
                [
                    row.get("program", ""),
                    row.get("total_states", ""),
                    row.get("total_tested_pairs", ""),
                    row.get("total_batch_candidates", ""),
                    row.get("total_certified_batches", ""),
                    row.get("avg_local_reduction_log10", ""),
                    row.get("max_local_reduction_log10", ""),
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Evidence Quality",
        "",
        *_markdown_table(
            ["program", "executed batches", "strong certs", "weak certs", "rejected", "dropped", "replay"],
            [
                [
                    row.get("program", ""),
                    row.get("executed_batches", ""),
                    row.get("executed_strong_certificates", ""),
                    row.get("executed_weak_certificates", ""),
                    row.get("executed_rejected", ""),
                    row.get("dropped_active_passes", ""),
                    row.get("replay_hashes_match", "") or row.get("replay_status", ""),
                ]
                for row in evidence_rows
            ],
        ),
        "",
        "## State-Aware Behavior",
        "",
        *_markdown_table(
            ["program", "max depth", "relation flips", "enable/suppress", "states reached"],
            [_state_aware_row(out_dir, row) for row in summary_rows],
        ),
        "",
        "## Key Observations",
        "",
        *_observation_lines(summary_rows, evidence_rows),
        "",
        "## Correctness Boundary",
        "",
        "Reduction claims are state-local. They apply only to reached states, current pass set, current compiler, current target, and current normalization.",
        "",
    ]
    path = out_dir / "reduction_study_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _state_aware_row(out_dir: Path, summary: dict) -> list[str]:
    program = summary.get("program", "")
    optimize_dir = out_dir / program / "optimize"
    flips = _read_csv(optimize_dir / "relation_flip.csv")
    enable = _read_csv(optimize_dir / "enable_suppress.csv")
    enable_counts = Counter(row.get("relation", "") for row in enable)
    return [
        program,
        summary.get("max_depth", ""),
        str(len(flips)) if flips else "N/A",
        _enable_summary(enable_counts) if enable else "N/A",
        summary.get("total_states", ""),
    ]


def _observation_lines(summary_rows: list[dict], evidence_rows: list[dict]) -> list[str]:
    if not summary_rows and not evidence_rows:
        return ["- No successful programs were available for reduction observations."]
    dropped = _sum(evidence_rows, "dropped_active_passes")
    weak = _sum(evidence_rows, "executed_weak_certificates")
    rejected = _sum(evidence_rows, "executed_rejected")
    strong = _sum(evidence_rows, "executed_strong_certificates")
    executed = _sum(evidence_rows, "executed_batches")
    replay_missing = sum(1 for row in evidence_rows if not row.get("replay_hashes_match"))
    max_reduction = max((_float(row.get("max_local_reduction_log10")) for row in summary_rows), default=0.0)
    min_reduction = min((_float(row.get("avg_local_reduction_log10")) for row in summary_rows), default=0.0)
    lines = [
        f"- Dropped active passes are {'zero' if dropped == 0 else str(dropped)} in this run.",
        f"- Executed batch evidence includes {strong} strong certificates out of {executed} executed batches.",
        f"- Weak certificates: {weak}; rejected executed batches: {rejected}.",
        f"- Observed reduction log10 range is roughly {min_reduction:.3f} to {max_reduction:.3f}.",
    ]
    if replay_missing:
        lines.append(f"- Pipeline replay is missing for {replay_missing} successful programs.")
    else:
        lines.append("- Pipeline replay was recorded for all successful programs.")
    return lines


def _run_row(
    program: str,
    input_path: Path,
    status: str,
    optimize_status: str,
    reduction_status: str,
    evidence_status: str,
    states_reached: str,
    transitions: str,
    start: float,
    error_message: str,
) -> dict:
    return {
        "program": program,
        "input_path": str(input_path),
        "status": status,
        "optimize_status": optimize_status,
        "reduction_status": reduction_status,
        "evidence_status": evidence_status,
        "states_reached": states_reached,
        "transitions": transitions,
        "time_ms": f"{(time.perf_counter() - start) * 1000:.3f}",
        "error_message": error_message,
    }


def _program_plans(input_paths: list[Path], out_dir: Path) -> list[tuple[str, Path, Path]]:
    counts: dict[str, int] = {}
    plans = []
    for input_path in input_paths:
        stem = input_path.stem
        index = counts.get(stem, 0)
        counts[stem] = index + 1
        program = stem if index == 0 else f"{stem}_{index}"
        plans.append((program, input_path, out_dir / program))
    return plans


def _remove_existing_output(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if not path.name or resolved == anchor or resolved == Path.cwd().resolve():
        raise RuntimeError(f"refusing to remove unsafe output path: {path}")
    shutil.rmtree(resolved)


def _enable_summary(counts: Counter) -> str:
    parts = []
    for name in ["enable", "suppress", "effect_changed"]:
        if counts.get(name, 0):
            parts.append(f"{name}={counts[name]}")
    return ", ".join(parts) if parts else "0"


def _value(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _sum(rows: list[dict], field: str) -> int:
    return sum(_int(row.get(field)) for row in rows)


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
