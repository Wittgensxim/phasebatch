from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .equality_summary import equality_tier_markdown, equality_tier_summary_for_run
from .schema import FINAL_SUMMARY_INDEX_FIELDS


MISSING_BASELINE_WARNING = "Baseline results are missing. Run compare-baselines or use --run-baselines."
OBJECTIVE_REMINDER = "Objective signals are used only for path selection and evaluation. They are not used as commutation proof."
CORRECTNESS_BOUNDARY = (
    "The batch reduction layer only treats a batch as hard-foldable when its correctness evidence supports it, "
    "such as all_permutations_same validation. Objective values are used only for search ranking, path selection, "
    "and evaluation; they are not independence or commutation proof."
)
TRUE_RELATION_FLIPS = {"commute_to_sensitive", "sensitive_to_commute", "known_to_unknown", "unknown_to_known", "other_flip"}
PAIR_AVAILABILITY_CHANGES = {"active_pair_to_missing", "missing_to_active_pair"}


def generate_final_summary(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    metadata = _read_json(run_dir / "metadata.json")
    optimize_summary = _read_key_value_summary(run_dir / "optimize_summary.md")
    chosen_path = _read_csv(run_dir / "chosen_path.csv")
    chosen_summary = _first_row(run_dir / "chosen_path_summary.csv")
    baselines = _read_csv(run_dir / "baseline_results.csv")
    enable_suppress = _read_csv(run_dir / "enable_suppress.csv")
    relation_flip = _read_csv(run_dir / "relation_flip.csv")
    replay = _first_row(run_dir / "pipeline_replay.csv")
    validation_cost_rows = _read_csv(run_dir / "batch_validation_ladder_summary.csv")
    exact_status = _read_exact_status(run_dir / "exact_status.txt") or _first_value(optimize_summary, ["exact_status"])
    optimized_pipeline = _read_text(run_dir / "optimized_pipeline.txt").strip()

    warnings = _warnings(run_dir, baselines)
    lines = ["# Final Optimization Summary", ""]
    if warnings:
        lines.extend(_warning_section(warnings))
    lines.extend(_run_configuration_section(metadata, optimize_summary, exact_status))
    lines.extend(_final_result_section(run_dir, chosen_summary, optimized_pipeline))
    lines.extend(equality_tier_markdown(equality_tier_summary_for_run(run_dir)))
    lines.append("")
    lines.extend(_validation_cost_section(validation_cost_rows))
    lines.extend(_pipeline_replay_section(replay))
    lines.extend(_chosen_path_section(chosen_path))
    lines.extend(_executable_reason_section(chosen_path))
    lines.extend(_state_changes_section(chosen_path, enable_suppress, relation_flip))
    lines.extend(_objective_signal_section(chosen_path))
    lines.extend(_baseline_section(baselines))
    lines.extend(_artifact_section(run_dir))
    lines.extend(_correctness_boundary_section())

    summary_path = run_dir / "final_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_final_summary_index(run_dir, chosen_summary, optimized_pipeline, baselines, replay)
    return summary_path


def _warnings(run_dir: Path, baselines: list[dict]) -> list[str]:
    warnings = []
    for name in ["chosen_path.csv", "chosen_path_summary.csv", "optimized_pipeline.txt"]:
        if not (run_dir / name).exists():
            warnings.append(f"missing input file: {name}")
    if not baselines:
        warnings.append(MISSING_BASELINE_WARNING)
    return warnings


def _run_configuration_section(metadata: dict, optimize_summary: dict, exact_status: str) -> list[str]:
    selected_mode = _first_value(optimize_summary, ["selected_mode", "mode"]) or str(metadata.get("mode", ""))
    requested_mode = _first_value(optimize_summary, ["requested_mode"]) or str(metadata.get("mode", ""))
    lines = [
        "## 1. Run Configuration",
        "",
        f"- input: {metadata.get('input', '')}",
        f"- requested mode: {requested_mode}",
        f"- selected mode: {selected_mode}",
        f"- objective: {metadata.get('objective', _first_value(optimize_summary, ['objective']))}",
        f"- exact_status: {exact_status}",
    ]
    if selected_mode == "rolling-exact":
        window_cap = metadata.get("max_rolling_windows", _first_value(optimize_summary, ["max_rolling_windows"]))
        lines.extend(
            [
                f"- exact_scope: {metadata.get('exact_scope', _first_value(optimize_summary, ['exact_scope']))}",
                f"- rolling_window_depth: {metadata.get('rolling_window_depth', _first_value(optimize_summary, ['rolling_window_depth']))}",
                f"- rolling_frontier_width: {metadata.get('rolling_frontier_width', _first_value(optimize_summary, ['rolling_frontier_width']))}",
                f"- max_rolling_windows: {window_cap}" + (" (until closure)" if str(window_cap) == "0" else ""),
                f"- rolling_windows_completed: {metadata.get('rolling_windows_completed', _first_value(optimize_summary, ['rolling_windows_completed']))}",
                f"- rolling_committed_depth: {metadata.get('rolling_committed_depth', _first_value(optimize_summary, ['rolling_committed_depth']))}",
                f"- rolling_closure_reason: {metadata.get('rolling_closure_reason', _first_value(optimize_summary, ['rolling_closure_reason']))}",
                f"- rolling_frontier_pruned: {metadata.get('rolling_frontier_pruned', _first_value(optimize_summary, ['rolling_frontier_pruned']))}",
                f"- rolling_frontier_states_pruned: {metadata.get('rolling_frontier_states_pruned', _first_value(optimize_summary, ['rolling_frontier_states_pruned']))}",
                f"- global_search_complete: {metadata.get('global_search_complete', _first_value(optimize_summary, ['global_search_complete']))}",
            ]
        )
        budget_keys = ["max_states"]
    else:
        lines.append(f"- max_rounds: {metadata.get('max_rounds', _first_value(optimize_summary, ['max_rounds']))}")
        budget_keys = [
            "beam_width",
            "max_states",
            "max_batches_per_state",
            "budgeted_validation_strategy",
            "batch_selection_policy",
            "frontier_selection_policy",
            "selection_seed",
        ]
    lines.append(f"- pass config path: {metadata.get('pass_config', '')}")
    for key in budget_keys:
        value = metadata.get(key, _first_value(optimize_summary, [key]))
        if value not in (None, ""):
            lines.append(f"- {key}: {value}")
    tools = metadata.get("tools", {})
    if tools:
        lines.append("- LLVM tools:")
        for name in sorted(tools):
            tool = tools.get(name) or {}
            path = tool.get("path", "")
            version = _first_line(tool.get("version", ""))
            lines.append(f"  - {name}: {path} ({version})")
    lines.append("")
    return lines


def _final_result_section(run_dir: Path, summary: dict, optimized_pipeline: str) -> list[str]:
    root_count = _first_value(summary, ["root_ir_inst_count"])
    final_count = _first_value(summary, ["final_ir_inst_count"])
    delta = _first_value(summary, ["total_ir_inst_delta"])
    reduction = _first_value(summary, ["total_ir_inst_reduction_pct", "reduction_pct"])
    return [
        "## 2. Final Result",
        "",
        f"- root IR instruction count: {root_count}",
        f"- final IR instruction count: {final_count}",
        f"- total IR instruction delta: {delta}",
        f"- total reduction percentage: {reduction}",
        f"- selected final state: {_first_value(summary, ['selected_final_state'])}",
        f"- path length: {_first_value(summary, ['path_steps'])}",
        f"- total pass invocations: {_first_value(summary, ['total_pass_invocations'])}",
        f"- final.ll path: {run_dir / 'final.ll'}",
        f"- optimized_pipeline.txt path: {run_dir / 'optimized_pipeline.txt'}",
        f"- optimized pipeline: {optimized_pipeline or '(root state selected; no batch pipeline)'}",
        "",
    ]


def _pipeline_replay_section(replay: dict) -> list[str]:
    if not replay:
        return []
    lines = [
        "## Final Pipeline Replay Verification",
        "",
        f"- replay status: {replay.get('replay_status', '')}",
        f"- hashes match: {replay.get('hashes_match', '')}",
        f"- replay hash: {replay.get('replay_hash', '')}",
        f"- final hash: {replay.get('final_hash', '')}",
        f"- replayed_final.ll path: {replay.get('replay_output_path', '')}",
    ]
    if replay.get("hashes_match") != "true":
        lines.append("- **WARNING** final pipeline replay did not reproduce final.ll.")
    error = replay.get("error_message", "")
    if error:
        lines.append(f"- error: {error}")
    lines.append("")
    return lines


def _validation_cost_section(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    total = lambda field: sum(_to_int_or_none(row.get(field)) or 0 for row in rows)
    validation_time = sum(_to_float(row.get("validation_time_ms")) for row in rows)
    return [
        "## Validation Cost",
        "",
        f"- validation opt invocations: {total('validation_opt_invocations')}",
        "- validation pass invocations: "
        f"{total('validation_pass_invocations_baseline')} baseline, "
        f"{total('validation_pass_invocations_actual')} actual, "
        f"{total('validation_pass_invocations_saved')} saved",
        f"- profile reuse hits: {total('validation_profile_reuse_hits')}",
        "- state cache hits: "
        f"{total('validation_state_transition_cache_hits')} transitions, "
        f"{total('validation_state_equivalence_cache_hits')} equivalence comparisons",
        f"- validation time ms: {validation_time:.3f}",
        "",
    ]


def _chosen_path_section(rows: list[dict]) -> list[str]:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get("step", ""),
                row.get("parent_state_id", ""),
                row.get("batch_id", ""),
                row.get("batch_passes") or row.get("canonical_order", ""),
                row.get("validation_status", ""),
                row.get("correctness_class", ""),
                row.get("child_state_id", ""),
                f"{row.get('parent_active_passes', '')} -> {row.get('child_active_passes', '')}",
                f"{row.get('ir_inst_before', '')} -> {row.get('ir_inst_after', '')}",
                row.get("ir_inst_delta", ""),
            ]
        )
    return [
        "## 3. Chosen Batch Path",
        "",
        *_markdown_table(
            [
                "step",
                "parent state",
                "batch id",
                "batch passes",
                "validation",
                "correctness",
                "child state",
                "active passes before -> after",
                "IR inst before -> after",
                "delta",
            ],
            table_rows,
        ),
        "",
    ]


def _executable_reason_section(rows: list[dict]) -> list[str]:
    lines = ["## 4. Why Each Batch Was Executable", ""]
    if not rows:
        return [*lines, "No batch was selected; the root state is the final state.", ""]
    for row in rows:
        correctness_class = row.get("correctness_class", "")
        lines.extend(
            [
                f"- Step {row.get('step', '')}, batch {row.get('batch_id', '')}:",
                f"  - correctness_class: {correctness_class}",
                f"  - validation_status: {row.get('validation_status', '')}",
                f"  - can_hard_fold: {row.get('can_hard_fold', '')}",
                f"  - can_execute: {row.get('can_execute', '')}",
                f"  - reason: {_execution_reason(correctness_class)}",
            ]
        )
    lines.append("")
    return lines


def _state_changes_section(rows: list[dict], enable_suppress: list[dict], relation_flip: list[dict]) -> list[str]:
    lines = ["## 5. State Changes Along the Path", ""]
    if not rows:
        return [*lines, "No transition was selected.", ""]
    for row in rows:
        parent = row.get("parent_state_id", "")
        child = row.get("child_state_id", "")
        interaction = _interaction_counts(parent, child, enable_suppress, relation_flip)
        lines.extend(
            [
                f"### Step {row.get('step', '')}: {parent} -> {child}",
                "",
                f"- parent_active_passes: {row.get('parent_active_passes', '')}",
                f"- child_active_passes: {row.get('child_active_passes', '')}",
                f"- parent_tested_pairs: {row.get('parent_tested_pairs', '')}",
                f"- child_tested_pairs: {row.get('child_tested_pairs', '')}",
                f"- parent_commute_pairs: {row.get('parent_commute_pairs', '')}",
                f"- child_commute_pairs: {row.get('child_commute_pairs', '')}",
                f"- parent_order_sensitive_pairs: {row.get('parent_order_sensitive_pairs', '')}",
                f"- child_order_sensitive_pairs: {row.get('child_order_sensitive_pairs', '')}",
            ]
        )
        if interaction:
            lines.extend(
                [
                    f"- enable count: {interaction['enable']}",
                    f"- suppress count: {interaction['suppress']}",
                    f"- effect_changed count: {interaction['effect_changed']}",
                    f"- true relation flips: {interaction['true_relation_flips']}",
                    f"- pair availability changes: {interaction['pair_availability_changes']}",
                ]
            )
        else:
            lines.append("- enable/suppress and relation flip data: not available")
        lines.append("")
    return lines


def _objective_signal_section(rows: list[dict]) -> list[str]:
    table_rows = [
        [
            row.get("step", ""),
            row.get("parent_state_id", ""),
            row.get("child_state_id", ""),
            row.get("ir_inst_before", ""),
            row.get("ir_inst_after", ""),
            row.get("ir_inst_delta", ""),
            row.get("ir_inst_reduction_pct", ""),
        ]
        for row in rows
    ]
    return [
        "## 6. Objective Signal Along the Path",
        "",
        OBJECTIVE_REMINDER,
        "",
        *_markdown_table(["step", "parent", "child", "IR before", "IR after", "delta", "reduction %"], table_rows),
        "",
    ]


def _baseline_section(rows: list[dict]) -> list[str]:
    lines = ["## 7. Baseline Comparison", ""]
    if not rows:
        return [*lines, MISSING_BASELINE_WARNING, ""]

    best_all = _best_row(rows, include_optimized=True)
    best_baseline = _best_row(rows, include_optimized=False)
    optimized = _row_by_method(rows, "optimized_pipeline")
    table_rows = []
    for row in rows:
        method = row.get("method", "")
        display = f"**{method}**" if best_all and method == best_all.get("method") else method
        table_rows.append(
            [
                display,
                row.get("status", ""),
                row.get("final_ir_inst_count", ""),
                row.get("ir_inst_delta", ""),
                row.get("ir_inst_reduction_pct", ""),
                row.get("final_sequence_length", "") or row.get("pass_invocations", ""),
                row.get("time_ms", ""),
            ]
        )
    lines.extend(
        _markdown_table(
            ["method", "status", "final IR inst", "delta", "reduction %", "pass invocations", "time ms"],
            table_rows,
        )
    )
    lines.extend(
        [
            "",
            "In this run, under IR instruction count objective:",
            f"- best method by final IR inst count: {_method_count_text(best_all)}",
            f"- best baseline by final IR inst count: {_method_count_text(best_baseline)}",
            f"- optimized_pipeline beats greedy: {_beats_text(optimized, _row_by_method(rows, 'greedy_single_pass'))}",
            f"- optimized_pipeline beats random best: {_beats_text(optimized, _row_by_method(rows, 'random_single_pass_best'))}",
            f"- optimized_pipeline beats config_order_once: {_beats_text(optimized, _row_by_method(rows, 'config_order_once'))}",
            "",
        ]
    )
    return lines


def _artifact_section(run_dir: Path) -> list[str]:
    names = [
        "chosen_path.csv",
        "chosen_path_summary.csv",
        "optimized_batches.txt",
        "optimized_pipeline.txt",
        "final.ll",
        "baseline_results.csv",
        "pipeline_replay.csv",
        "batch_validation_ladder_summary.csv",
        "replayed_final.ll",
        "state_dag.csv",
        "leaf_states.csv",
        "optimize_summary.md",
    ]
    lines = ["## 8. Reproducibility Artifacts", ""]
    for name in names:
        path = run_dir / name
        status = "present" if path.exists() else "missing"
        lines.append(f"- {name}: {path} ({status})")
    lines.append("")
    return lines


def _correctness_boundary_section() -> list[str]:
    return [
        "## 9. Correctness Boundary",
        "",
        CORRECTNESS_BOUNDARY,
        "",
        "- sampled batches are heuristic only",
        "- rejected/failed/unvalidated batches are not executable by default",
        "- conclusions are state-local and compiler-version-specific",
        "",
    ]


def _write_final_summary_index(run_dir: Path, summary: dict, optimized_pipeline: str, baselines: list[dict], replay: dict) -> None:
    optimized = _row_by_method(baselines, "optimized_pipeline")
    best_baseline = _best_row(baselines, include_optimized=False)
    row = {
        "program": _first_value(summary, ["program"]) or run_dir.name,
        "final_state": _first_value(summary, ["selected_final_state"]),
        "root_ir_inst_count": _first_value(summary, ["root_ir_inst_count"]),
        "final_ir_inst_count": _first_value(summary, ["final_ir_inst_count"]),
        "reduction_pct": _first_value(summary, ["total_ir_inst_reduction_pct", "reduction_pct"]),
        "path_steps": _first_value(summary, ["path_steps"]),
        "pass_invocations": _first_value(summary, ["total_pass_invocations", "pass_invocations"]),
        "optimized_pipeline": optimized_pipeline,
        "best_baseline_method": best_baseline.get("method", "") if best_baseline else "",
        "best_baseline_inst_count": best_baseline.get("final_ir_inst_count", "") if best_baseline else "",
        "optimized_beats_greedy": _beats_text(optimized, _row_by_method(baselines, "greedy_single_pass")),
        "optimized_beats_random": _beats_text(optimized, _row_by_method(baselines, "random_single_pass_best")),
        "optimized_beats_config_order": _beats_text(optimized, _row_by_method(baselines, "config_order_once")),
        "replay_status": replay.get("replay_status", ""),
        "replay_hashes_match": replay.get("hashes_match", ""),
    }
    _write_csv(run_dir / "final_summary_index.csv", FINAL_SUMMARY_INDEX_FIELDS, [row])


def _interaction_counts(parent: str, child: str, enable_suppress: list[dict], relation_flip: list[dict]) -> dict[str, int]:
    matching_enable = [
        row for row in enable_suppress
        if row.get("parent_state_id") == parent and row.get("child_state_id") == child
    ]
    matching_flips = [
        row for row in relation_flip
        if row.get("parent_state_id") == parent and row.get("child_state_id") == child
    ]
    if not matching_enable and not matching_flips:
        return {}
    relation_counts = Counter(row.get("relation", "") for row in matching_enable)
    flip_counts = Counter(row.get("flip_kind", "") for row in matching_flips)
    return {
        "enable": relation_counts.get("enable", 0),
        "suppress": relation_counts.get("suppress", 0),
        "effect_changed": relation_counts.get("effect_changed", 0),
        "true_relation_flips": sum(flip_counts.get(kind, 0) for kind in TRUE_RELATION_FLIPS),
        "pair_availability_changes": sum(flip_counts.get(kind, 0) for kind in PAIR_AVAILABILITY_CHANGES),
    }


def _execution_reason(correctness_class: str) -> str:
    return {
        "certified_batch": "all tested permutations produced identical canonical IR",
        "sampled_batch": "sampled permutations matched; heuristic only",
        "rejected_batch": "not executable",
        "unvalidated_batch": "not executable",
        "failed_batch": "not executable",
        "unknown_batch": "not executable",
    }.get(correctness_class, "not available")


def _best_row(rows: list[dict], *, include_optimized: bool) -> dict | None:
    candidates = []
    for row in rows:
        if row.get("status") != "success":
            continue
        if not include_optimized and row.get("method") == "optimized_pipeline":
            continue
        count = _to_int_or_none(row.get("final_ir_inst_count"))
        if count is None:
            continue
        candidates.append((count, row.get("method", ""), row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _row_by_method(rows: list[dict], method: str) -> dict:
    return next((row for row in rows if row.get("method") == method), {})


def _method_count_text(row: dict | None) -> str:
    if not row:
        return "N/A"
    return f"{row.get('method', '')} ({row.get('final_ir_inst_count', '')})"


def _beats_text(optimized: dict, baseline: dict) -> str:
    opt_count = _to_int_or_none(optimized.get("final_ir_inst_count"))
    baseline_count = _to_int_or_none(baseline.get("final_ir_inst_count"))
    if opt_count is None or baseline_count is None:
        return "N/A"
    return "true" if opt_count < baseline_count else "false"


def _warning_section(warnings: list[str]) -> list[str]:
    return ["## Warnings", "", *[f"- {warning}" for warning in warnings], ""]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(_cell(cell) for cell in row)} |" for row in rows)
    return lines


def _cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


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


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _read_exact_status(path: Path) -> str:
    text = _read_text(path).strip()
    return text.splitlines()[0] if text else ""


def _read_key_value_summary(path: Path) -> dict:
    values = {}
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _first_value(row: dict, names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _first_line(value: object) -> str:
    return str(value or "").splitlines()[0] if value else ""


def _to_int_or_none(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0
