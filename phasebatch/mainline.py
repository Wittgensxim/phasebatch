from __future__ import annotations

import csv
import glob
import shutil
import time
from pathlib import Path

from .batch_objective import eval_batch_objectives
from .mainline_summary import generate_mainline_summary

MAINLINE_RUN_FIELDS = [
    "program",
    "input_path",
    "output_dir",
    "status",
    "error_message",
    "total_time_ms",
]

MAINLINE_MISSING_OUTPUT_FIELDS = [
    "program",
    "expected_file",
    "status",
]

AGGREGATE_SPECS = [
    ("aggregate_by_depth.csv", "mainline_aggregate_states.csv"),
    ("aggregate_batch_summary.csv", "mainline_aggregate_batches.csv"),
    ("aggregate_coverage_summary.csv", "mainline_aggregate_coverage.csv"),
    ("aggregate_overlap_summary.csv", "mainline_aggregate_overlap.csv"),
]


def expand_inputs(inputs: list[str], warn=print) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if _has_glob_meta(item):
            matches = sorted(Path(match) for match in glob.glob(item, recursive=True))
            if not matches:
                warn(f"warning: input glob matched no files: {item}")
                continue
            paths.extend(matches)
        else:
            paths.append(Path(item))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            warn(f"warning: skipping missing input: {path}")
            continue
        if path.suffix.lower() not in {".c", ".ll"}:
            warn(f"warning: skipping unsupported input: {path}")
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def run_mainline(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    *,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_batches_per_state: int,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
    eval_objective: str | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
    warn=print,
) -> dict:
    out_dir = Path(out_dir)
    input_paths = expand_inputs(inputs, warn)
    if not input_paths:
        raise RuntimeError("no valid .c or .ll inputs remain after expansion")

    out_dir.mkdir(parents=True, exist_ok=True)
    plans = _program_plans(input_paths, out_dir)
    run_rows: list[dict] = []

    for program, input_path, program_out in plans:
        start = time.perf_counter()
        error_message = ""
        status = "success"
        try:
            if program_out.exists():
                if not overwrite:
                    raise RuntimeError(f"output directory already exists: {program_out}; use --overwrite to rerun")
                _remove_existing_output(program_out, out_dir)

            run_explore_batches(
                input_path,
                program_out,
                passes_path,
                jobs=jobs,
                timeout=timeout,
                max_pairs=max_pairs,
                max_depth=max_depth,
                max_component_size=max_component_size,
                max_batch_candidates=max_batch_candidates,
                max_batches_per_state=max_batches_per_state,
                max_frontier_states=max_frontier_states,
                batch_frontier_policy=batch_frontier_policy,
                validate_batches=validate_batches,
                allow_sampled_batches=allow_sampled_batches,
            )
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            if not continue_on_error:
                raise
        finally:
            run_rows.append(
                {
                    "program": program,
                    "input_path": str(input_path),
                    "output_dir": str(program_out),
                    "status": status,
                    "error_message": error_message,
                    "total_time_ms": _elapsed_ms(start),
                }
            )

    _write_csv(out_dir / "mainline_runs.csv", MAINLINE_RUN_FIELDS, run_rows)
    missing_rows = _write_aggregate_outputs(out_dir, run_rows, warn)
    _write_csv(out_dir / "mainline_missing_outputs.csv", MAINLINE_MISSING_OUTPUT_FIELDS, missing_rows)
    summary = generate_mainline_summary(out_dir)
    objective_result = None
    if eval_objective:
        objective_result = eval_batch_objectives(out_dir, objective=eval_objective, recursive=True)
    successes = sum(1 for row in run_rows if row["status"] == "success")
    failures = sum(1 for row in run_rows if row["status"] == "failed")
    result = {
        "out_dir": str(out_dir),
        "programs": len(run_rows),
        "successes": successes,
        "failures": failures,
        "mainline_runs_csv": str(out_dir / "mainline_runs.csv"),
        "mainline_missing_outputs_csv": str(out_dir / "mainline_missing_outputs.csv"),
        "mainline_aggregate_states_csv": str(out_dir / "mainline_aggregate_states.csv"),
        "mainline_aggregate_batches_csv": str(out_dir / "mainline_aggregate_batches.csv"),
        "mainline_aggregate_coverage_csv": str(out_dir / "mainline_aggregate_coverage.csv"),
        "mainline_aggregate_overlap_csv": str(out_dir / "mainline_aggregate_overlap.csv"),
        "mainline_summary_md": str(summary),
    }
    if objective_result:
        result.update(
            {
                "aggregate_objective_signal_csv": objective_result.get("aggregate_objective_signal_csv", ""),
                "objective_summary_md": objective_result.get("objective_summary_md", ""),
                "objective_rows": objective_result.get("rows", 0),
            }
        )
    return result


def run_explore_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_batches_per_state: int = 20,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
) -> dict:
    from .batch_explorer import explore_batches

    return explore_batches(
        input_path,
        out_dir,
        passes_path,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        max_depth=max_depth,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
        max_batches_per_state=max_batches_per_state,
        max_frontier_states=max_frontier_states,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        allow_sampled_batches=allow_sampled_batches,
    )


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


def _write_aggregate_outputs(out_dir: Path, run_rows: list[dict], warn=print) -> list[dict]:
    missing_rows: list[dict] = []
    for source_name, output_name in AGGREGATE_SPECS:
        all_rows: list[dict] = []
        fieldnames: list[str] = []
        for run in run_rows:
            program = run.get("program", "")
            expected = Path(run.get("output_dir", "")) / source_name
            if run.get("status") != "success" or not expected.exists():
                status = "missing" if run.get("status") == "success" else "skipped_failed_program"
                missing_rows.append({"program": program, "expected_file": source_name, "status": status})
                if run.get("status") == "success":
                    warn(f"warning: missing expected output for {program}: {expected}")
                continue
            missing_rows.append({"program": program, "expected_file": source_name, "status": "present"})
            rows = _read_csv(expected)
            if not rows:
                continue
            for row in rows:
                row = dict(row)
                row["program"] = row.get("program") or program
                all_rows.append(row)
                for field in row:
                    if field not in fieldnames:
                        fieldnames.append(field)
        if all_rows and "program" in fieldnames:
            fieldnames = ["program", *[field for field in fieldnames if field != "program"]]
        elif not fieldnames:
            fieldnames = ["program"]
        _write_csv(out_dir / output_name, fieldnames, all_rows)
    return missing_rows


def _remove_existing_output(target: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise RuntimeError(f"refusing to remove output outside run root: {target}")
    shutil.rmtree(resolved_target)


def _has_glob_meta(value: str) -> bool:
    return any(char in value for char in "*?[]")


def _elapsed_ms(start: float) -> str:
    return f"{(time.perf_counter() - start) * 1000:.2f}"


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
