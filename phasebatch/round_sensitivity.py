from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .schema import ROUND_SENSITIVITY_FIELDS


def run_round_sensitivity(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    rounds: list[int],
    optimizer_mode: str,
    objective: str,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str | None,
    validate_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    overwrite: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    input_path = Path(input_path)
    passes_path = Path(passes_path)
    normalized_rounds = _normalize_rounds(rounds)
    if not normalized_rounds:
        raise RuntimeError("at least one positive max_rounds value is required")
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for max_rounds in normalized_rounds:
        round_dir = out_dir / f"round_{max_rounds}"
        if round_dir.exists():
            raise RuntimeError(f"round output already exists: {round_dir}; use --overwrite to rerun")
        _optimize_batches(
            input_path,
            round_dir,
            passes_path,
            mode=optimizer_mode,
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
        run_dirs.append(round_dir)

    result = generate_round_sensitivity(run_dirs, out_dir, input_label=str(input_path), passes_label=str(passes_path))
    result.update({"rounds": len(run_dirs)})
    return result


def _optimize_batches(*args, **kwargs) -> dict:
    from .optimizer import optimize_batches

    return optimize_batches(*args, **kwargs)


def generate_round_sensitivity(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    input_label: str = "",
    passes_label: str = "",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_round_row(Path(run_dir)) for run_dir in run_dirs]
    rows.sort(key=lambda row: _int(row.get("max_rounds")))

    csv_path = out_dir / "round_sensitivity.csv"
    md_path = out_dir / "round_sensitivity.md"
    _write_csv(csv_path, ROUND_SENSITIVITY_FIELDS, rows)
    _write_summary(md_path, rows, input_label=input_label, passes_label=passes_label)
    return {
        "round_sensitivity_csv": str(csv_path),
        "round_sensitivity_md": str(md_path),
        "rows": len(rows),
    }


def _round_row(run_dir: Path) -> dict:
    max_rounds = _max_rounds_for_run(run_dir)
    chosen = _first_row(run_dir / "chosen_path_summary.csv")
    leaf = _selected_leaf(run_dir)
    timing = _first_row(run_dir / "optimizer_timing.csv")
    exact_status = _first_line(run_dir / "exact_status.txt")
    selected_final_state = chosen.get("selected_final_state") or _final_state_value(run_dir, "selected_final_state") or leaf.get("state_id", "")
    final_inst = chosen.get("final_ir_inst_count") or _final_state_value(run_dir, "final_objective")
    pipeline = _pipeline_text(run_dir)
    leaf_reason = leaf.get("leaf_reason", "")
    active_info = _remaining_active_pass_info(run_dir, selected_final_state)
    executable_info = _remaining_executable_batch_info(run_dir, selected_final_state, active_info)
    truncated = _is_truncated(exact_status, leaf_reason)
    stop_reason = leaf_reason or exact_status
    status = "success" if final_inst else "failed"
    error_message = "" if status == "success" else "missing final objective"
    return {
        "max_rounds": str(max_rounds),
        "run_dir": str(run_dir),
        "status": status,
        "states_reached": str(len(_read_csv(run_dir / "states.csv"))),
        "transitions": str(len(_read_csv(run_dir / "batch_state_transitions.csv"))),
        "final_inst": final_inst,
        "pipeline_len": str(_pipeline_len(pipeline, chosen)),
        "exact_status": exact_status,
        "selected_final_state": selected_final_state,
        "selected_leaf_reason": leaf_reason,
        "truncated": _bool(truncated),
        "selected_final_state_is_terminal": _bool(_is_terminal_state(leaf_reason, active_info)),
        "selected_final_state_stop_reason": stop_reason,
        "selected_final_state_truncated": _bool(truncated),
        "remaining_active_pass_count": active_info["count"],
        "remaining_active_passes": active_info["passes"],
        "remaining_executable_batch_count": executable_info["count"],
        "remaining_executable_batches": executable_info["batches"],
        "optimizer_total_time_ms": timing.get("optimizer_total_time_ms", ""),
        "optimized_pipeline": pipeline,
        "error_message": error_message,
    }


def _write_summary(path: Path, rows: list[dict], *, input_label: str, passes_label: str) -> None:
    lines = [
        "# Round Sensitivity Report",
        "",
        f"- input: {input_label}",
        f"- passes: {passes_label}",
        "",
        "This report tracks the batch-state exact search convergence curve across max_rounds values. It is not a blind depth increase; it shows whether additional certified rounds still expose objective improvement.",
        "",
        "## By Round",
        "",
        *_markdown_table(
            [
                "max_rounds",
                "states",
                "transitions",
                "final inst",
                "pipeline len",
                "exact status",
                "selected final state",
                "leaf reason",
                "truncated",
                "terminal",
                "remaining active passes",
                "remaining executable batches",
            ],
            [
                [
                    row.get("max_rounds", ""),
                    row.get("states_reached", ""),
                    row.get("transitions", ""),
                    row.get("final_inst", ""),
                    row.get("pipeline_len", ""),
                    row.get("exact_status", ""),
                    row.get("selected_final_state", ""),
                    row.get("selected_leaf_reason", ""),
                    row.get("truncated", ""),
                    row.get("selected_final_state_is_terminal", ""),
                    row.get("remaining_active_passes", ""),
                    row.get("remaining_executable_batches", ""),
                ]
                for row in rows
            ],
        ),
        "",
        "## Interpretation",
        "",
        *_interpretation_lines(rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _interpretation_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["- No round sensitivity rows were generated."]
    first = rows[0]
    best = min(
        (row for row in rows if _int(row.get("final_inst")) > 0),
        key=lambda row: (_int(row.get("final_inst")), _int(row.get("max_rounds"))),
        default=None,
    )
    lines = [
        f"- First measured round: max_rounds={first.get('max_rounds')} final_inst={first.get('final_inst')}.",
    ]
    if best:
        lines.append(
            f"- Best measured round: max_rounds={best.get('max_rounds')} final_inst={best.get('final_inst')} selected_state={best.get('selected_final_state')}."
        )
    if any(row.get("truncated") == "true" for row in rows):
        lines.append("- At least one measured round is truncated, so more certified rounds may still change the final objective.")
    for row in rows:
        if row.get("selected_final_state_truncated") == "true" and row.get("remaining_active_passes"):
            lines.append(
                f"- max_rounds={row.get('max_rounds')} selected {row.get('selected_final_state')} by budget stop ({row.get('selected_final_state_stop_reason')}); remaining active passes: {row.get('remaining_active_passes')}."
            )
    if rows[-1].get("truncated") == "false":
        lines.append("- The deepest measured round selected a non-truncated final state, suggesting the curve has reached a local fixed point for this pass set and configuration.")
    return lines


def _max_rounds_for_run(run_dir: Path) -> int:
    summary = run_dir / "optimize_summary.md"
    if summary.exists():
        for line in summary.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("- max_rounds:"):
                return _int(line.split(":", 1)[1].strip())
    if run_dir.name.startswith("round_"):
        return _int(run_dir.name.split("_", 1)[1])
    return 0


def _selected_leaf(run_dir: Path) -> dict:
    for row in _read_csv(run_dir / "leaf_states.csv"):
        if row.get("selected_as_final") == "true":
            return row
    return {}


def _is_truncated(exact_status: str, leaf_reason: str) -> bool:
    if exact_status and exact_status not in {"exact_complete", "not_applicable"}:
        return True
    return leaf_reason in {"max_rounds_reached", "state_cap_reached", "exact_incomplete", "beam_pruned"}


def _is_terminal_state(leaf_reason: str, active_info: dict) -> bool:
    if leaf_reason in {"no_active_passes", "no_executable_batches"}:
        return True
    if active_info.get("known") == "true" and active_info.get("count") == "0":
        return True
    return False


def _remaining_active_pass_info(run_dir: Path, state_id: str) -> dict:
    if not state_id:
        return {"known": "false", "count": "", "passes": ""}
    path = run_dir / "states" / state_id / "pass_profile.csv"
    if not path.exists():
        return {"known": "false", "count": "", "passes": ""}
    passes = [
        row.get("pass", "")
        for row in _read_csv(path)
        if row.get("success") == "true" and row.get("active") == "true" and row.get("pass")
    ]
    return {"known": "true", "count": str(len(passes)), "passes": ";".join(passes)}


def _remaining_executable_batch_info(run_dir: Path, state_id: str, active_info: dict) -> dict:
    if not state_id:
        return {"count": "", "batches": ""}
    state_dir = run_dir / "states" / state_id
    correctness_path = state_dir / "batch_correctness.csv"
    if not correctness_path.exists():
        if _int(active_info.get("count")) > 0:
            return {"count": "", "batches": "not_evaluated_at_terminal_depth"}
        return {"count": "0", "batches": ""}
    executable_rows = [row for row in _read_csv(correctness_path) if row.get("can_execute") == "true"]
    return {
        "count": str(len(executable_rows)),
        "batches": " | ".join(_batch_label(row) for row in executable_rows),
    }


def _batch_label(row: dict) -> str:
    batch_id = row.get("batch_id", "")
    batch_passes = row.get("batch_passes", "")
    if batch_id and batch_passes:
        return f"{batch_id}:{batch_passes}"
    return batch_passes or batch_id


def _pipeline_text(run_dir: Path) -> str:
    path = run_dir / "optimized_pipeline.txt"
    if not path.exists():
        return ""
    return ",".join(_split_pipeline(path.read_text(encoding="utf-8", errors="replace")))


def _pipeline_len(pipeline: str, chosen: dict) -> int:
    chosen_value = _int(chosen.get("total_pass_invocations"))
    if chosen_value:
        return chosen_value
    return len(_split_pipeline(pipeline))


def _split_pipeline(text: str) -> list[str]:
    return [part.strip() for part in str(text).replace("\n", "").replace(";", ",").split(",") if part.strip()]


def _final_state_value(run_dir: Path, key: str) -> str:
    path = run_dir / "final_state.txt"
    if not path.exists():
        return ""
    prefix = f"{key}="
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return ""


def _first_line(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
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


def _normalize_rounds(rounds: list[int]) -> list[int]:
    normalized: list[int] = []
    for value in rounds:
        number = _int(value)
        if number <= 0:
            continue
        if number not in normalized:
            normalized.append(number)
    return normalized


def _int(value: object) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def _bool(value: bool) -> str:
    return "true" if value else "false"
