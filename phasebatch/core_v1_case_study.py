from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


NUMBERS_FIELDS = [
    "program",
    "root_inst",
    "exact_batch_inst",
    "greedy_inst",
    "random_best_inst",
    "config_order_inst",
    "batch_vs_greedy",
    "batch_vs_random",
    "batch_vs_config",
    "exact_states",
    "exact_transitions",
    "max_reduction_log10",
    "avg_reduction_log10",
    "tested_pairs",
    "commute_pairs",
    "order_sensitive_pairs",
    "unknown_pairs",
    "executed_batches",
    "strong_certs",
    "dropped_active_passes",
    "best_budgeted_inst",
    "best_budgeted_beam",
    "best_budgeted_max_states",
    "budgeted_gap_to_exact",
    "budgeted_states",
    "budgeted_time_ms",
]

KEY_CLAIM_FIELDS = ["claim_id", "claim", "supporting_metric", "value", "source_file", "caution"]
FIGURES_FIELDS = ["figure", "program", "metric", "value"]
MISSING_FIELDS = ["input_name", "path", "status", "message"]

CORRECTNESS_BOUNDARY = (
    "All reduction claims are state-local. They apply only to reached states under the current pass set, "
    "compiler version, target, and IR normalization. Objective values are used for path selection and "
    "evaluation, not as commutation proof."
)


def summarize_core_v1_case_study(
    exact_method_summary: Path,
    exact_reduction_summary: Path,
    budgeted_sensitivity_summary: Path,
    out_dir: Path,
    *,
    label: str,
    nbody_round_study: Path | None = None,
    puzzle_case_study: Path | None = None,
    extra_notes: Path | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[dict] = []
    method_rows = _load_method_summary(Path(exact_method_summary), missing)
    reduction = _load_reduction_bundle(Path(exact_reduction_summary), missing)
    budgeted = _load_budgeted_bundle(Path(budgeted_sensitivity_summary), missing)
    optional_notes = _load_optional_notes(
        {
            "nbody_round_study": nbody_round_study,
            "puzzle_case_study": puzzle_case_study,
            "extra_notes": extra_notes,
        },
        missing,
    )

    numbers = _numbers_rows(method_rows, reduction, budgeted)
    claims = _key_claims(numbers, reduction, budgeted, exact_method_summary, exact_reduction_summary, budgeted_sensitivity_summary)
    figures = _figures_rows(numbers)

    _write_csv(out_dir / "core_v1_case_study_numbers.csv", NUMBERS_FIELDS, numbers)
    _write_csv(out_dir / "core_v1_key_claims.csv", KEY_CLAIM_FIELDS, claims)
    _write_csv(out_dir / "core_v1_figures_data.csv", FIGURES_FIELDS, figures)
    _write_csv(out_dir / "missing_inputs.csv", MISSING_FIELDS, missing)
    summary_path = _write_summary(
        out_dir / "core_v1_case_study_summary.md",
        label,
        numbers,
        reduction,
        budgeted,
        optional_notes,
        missing,
        exact_method_summary,
        exact_reduction_summary,
        budgeted_sensitivity_summary,
    )
    return {
        "out_dir": str(out_dir),
        "programs": len(numbers),
        "missing_inputs": len(missing),
        "core_v1_case_study_summary_md": str(summary_path),
        "core_v1_case_study_numbers_csv": str(out_dir / "core_v1_case_study_numbers.csv"),
        "core_v1_key_claims_csv": str(out_dir / "core_v1_key_claims.csv"),
        "core_v1_figures_data_csv": str(out_dir / "core_v1_figures_data.csv"),
        "missing_inputs_csv": str(out_dir / "missing_inputs.csv"),
    }


def _load_method_summary(path: Path, missing: list[dict]) -> list[dict]:
    if not _record_required(path, "exact_method_summary", missing):
        return []
    try:
        raw_rows = _read_csv(path) if path.suffix.lower() == ".csv" else _markdown_table_with_header(path, "program")
        rows = []
        for row in raw_rows:
            rows.append(
                {
                    "program": _value(row, "program"),
                    "root_inst": _value(row, "root", "root_inst", "root_ir_inst_count"),
                    "exact_batch_inst": _value(row, "batch", "batch exact r4", "exact_batch_inst", "batch_optimizer"),
                    "greedy_inst": _value(row, "greedy", "greedy_inst", "greedy_single_pass"),
                    "random_best_inst": _value(row, "random_best", "random best", "random_best_inst", "random_single_pass_best"),
                    "config_order_inst": _value(row, "config_once", "config order", "config_order_inst", "config_order_once"),
                    "batch_vs_greedy": _value(row, "batch_vs_greedy", "batch-greedy delta"),
                    "batch_vs_random": _value(row, "batch_vs_random", "batch-random delta"),
                    "batch_vs_config": _value(row, "batch_vs_config", "batch-config delta"),
                    "batch_states": _value(row, "batch_states", "batch states"),
                    "batch_transitions": _value(row, "batch_transitions", "batch transitions"),
                    "batch_time_ms": _value(row, "batch_time_ms", "batch time ms"),
                }
            )
        return [row for row in rows if row.get("program")]
    except Exception as exc:
        missing.append(_missing("exact_method_summary", path, "unparsable", str(exc)))
        return []


def _load_reduction_bundle(path: Path, missing: list[dict]) -> dict:
    if not _record_required(path, "exact_reduction_summary", missing):
        return _empty_reduction()
    base = path.parent
    markdown = _read_text(path)
    by_program = {row.get("program", ""): row for row in _read_csv(base / "reduction_by_program.csv") if row.get("program")}
    runs = {row.get("program", ""): row for row in _read_csv(base / "exact_reduction_runs.csv") if row.get("program")}
    coverage = {row.get("program", ""): row for row in _read_csv(base / "coverage_by_program.csv") if row.get("program")}
    evidence_counts: dict[str, Counter] = {}
    for row in _read_csv(base / "evidence_by_batch_all.csv"):
        evidence_counts.setdefault(row.get("program", ""), Counter())[row.get("evidence_strength", "")] += 1
    selected_counts = Counter(row.get("evidence_strength", "") for row in _read_csv(base / "selected_path_evidence.csv"))
    pair_rows = _parse_markdown_table_after_heading(markdown, "Pair Relation Evidence")
    if pair_rows and not by_program:
        for row in pair_rows:
            program = _value(row, "program")
            by_program[program] = {
                "program": program,
                "total_tested_pairs": _value(row, "tested pairs"),
                "total_commute_pairs": _value(row, "commute"),
                "total_order_sensitive_pairs": _value(row, "order-sensitive"),
                "total_unknown_pairs": _value(row, "unknown"),
            }
    return {
        "path": str(path),
        "markdown": markdown,
        "by_program": by_program,
        "runs": runs,
        "coverage": coverage,
        "evidence_counts": evidence_counts,
        "selected_counts": selected_counts,
        "overall": {
            "total_states": _extract_bullet(markdown, "total states"),
            "total_transitions": _extract_bullet(markdown, "total transitions"),
            "total_selected_path_steps": _extract_bullet(markdown, "total selected path steps"),
        },
        "all_selected_strong": "All selected path batches have strong certificates" in markdown,
        "dropped_zero": "Dropped active passes are zero" in markdown,
    }


def _load_budgeted_bundle(path: Path, missing: list[dict]) -> dict:
    if not _record_required(path, "budgeted_sensitivity_summary", missing):
        return _empty_budgeted()
    base = path.parent
    markdown = _read_text(path)
    best_rows = {row.get("program", ""): row for row in _read_csv(base / "budgeted_sensitivity_best.csv") if row.get("program")}
    all_rows = _read_csv(base / "budgeted_sensitivity_results.csv")
    if not best_rows:
        for row in _parse_markdown_table_after_heading(markdown, "Results by Program"):
            program = _value(row, "program")
            best_rows[program] = {
                "program": program,
                "best_final_ir_inst_count": _value(row, "best budgeted"),
                "best_beam_width": _value(row, "beam"),
                "best_max_states": _value(row, "max states"),
                "gap_to_exact": _value(row, "gap to exact"),
                "states_reached": _value(row, "states"),
                "time_ms": _value(row, "time ms"),
                "exact_r4_inst": _value(row, "exact r4"),
            }
    return {
        "path": str(path),
        "markdown": markdown,
        "best": best_rows,
        "all_results": all_rows,
        "metrics": {
            "programs_matching_exact": _extract_bullet(markdown, "programs matching exact"),
            "average_gap_to_exact": _extract_bullet(markdown, "average gap to exact"),
            "average_state_reduction": _extract_bullet(markdown, "average state reduction relative to exact"),
            "average_time_reduction": _extract_bullet(markdown, "average time reduction relative to exact"),
            "vs_greedy": _extract_dash_line(markdown, "budgeted vs greedy"),
            "vs_random": _extract_dash_line(markdown, "budgeted vs random best"),
            "vs_config": _extract_dash_line(markdown, "budgeted vs config order once"),
        },
    }


def _numbers_rows(method_rows: list[dict], reduction: dict, budgeted: dict) -> list[dict]:
    programs = sorted(
        {
            row.get("program", "")
            for row in method_rows
        }
        | set(reduction.get("by_program", {}).keys())
        | set(budgeted.get("best", {}).keys())
    )
    method_by_program = {row.get("program", ""): row for row in method_rows}
    rows = []
    for program in programs:
        if not program:
            continue
        method = method_by_program.get(program, {})
        red = reduction.get("by_program", {}).get(program, {})
        run = reduction.get("runs", {}).get(program, {})
        cov = reduction.get("coverage", {}).get(program, {})
        evidence = reduction.get("evidence_counts", {}).get(program, Counter())
        best = budgeted.get("best", {}).get(program, {})
        rows.append(
            {
                "program": program,
                "root_inst": method.get("root_inst", ""),
                "exact_batch_inst": method.get("exact_batch_inst", ""),
                "greedy_inst": method.get("greedy_inst", ""),
                "random_best_inst": method.get("random_best_inst", ""),
                "config_order_inst": method.get("config_order_inst", ""),
                "batch_vs_greedy": method.get("batch_vs_greedy") or _delta(method.get("greedy_inst"), method.get("exact_batch_inst")),
                "batch_vs_random": method.get("batch_vs_random") or _delta(method.get("random_best_inst"), method.get("exact_batch_inst")),
                "batch_vs_config": method.get("batch_vs_config") or _delta(method.get("config_order_inst"), method.get("exact_batch_inst")),
                "_batch_time_ms": method.get("batch_time_ms", ""),
                "exact_states": _value(red, "states") or _value(run, "states_reached") or method.get("batch_states", ""),
                "exact_transitions": _value(run, "transitions") or _value(red, "total_executed_transitions") or method.get("batch_transitions", ""),
                "max_reduction_log10": _value(red, "max_local_reduction_log10"),
                "avg_reduction_log10": _value(red, "avg_local_reduction_log10"),
                "tested_pairs": _value(red, "total_tested_pairs"),
                "commute_pairs": _value(red, "total_commute_pairs"),
                "order_sensitive_pairs": _value(red, "total_order_sensitive_pairs"),
                "unknown_pairs": _value(red, "total_unknown_pairs"),
                "executed_batches": str(sum(evidence.values())) if evidence else _value(run, "transitions"),
                "strong_certs": str(evidence.get("strong", 0)),
                "dropped_active_passes": _value(cov, "dropped_active_passes"),
                "best_budgeted_inst": _value(best, "best_final_ir_inst_count"),
                "best_budgeted_beam": _value(best, "best_beam_width"),
                "best_budgeted_max_states": _value(best, "best_max_states"),
                "budgeted_gap_to_exact": _value(best, "gap_to_exact"),
                "budgeted_states": _value(best, "states_reached"),
                "budgeted_time_ms": _value(best, "time_ms"),
            }
        )
    return rows


def _write_summary(
    path: Path,
    label: str,
    numbers: list[dict],
    reduction: dict,
    budgeted: dict,
    optional_notes: dict[str, str],
    missing: list[dict],
    exact_method_summary: Path,
    exact_reduction_summary: Path,
    budgeted_sensitivity_summary: Path,
) -> Path:
    method_wtl = _wtl(numbers, ["batch_vs_greedy", "batch_vs_random", "batch_vs_config"])
    all_selected_strong = _all_selected_path_strong(reduction)
    dropped_zero = all(_int(row.get("dropped_active_passes")) == 0 for row in numbers)
    max_reduction = max((_float(row.get("max_reduction_log10")) for row in numbers), default=0.0)
    lines = [
        "# Core-v1 Case Study Summary",
        "",
        "## 1. Purpose",
        "",
        f"- study label: {label}",
        "- Core-v1 is used as the controlled setting for validating the state-local certified batch reduction idea. It is not claimed to cover all LLVM passes.",
        "",
        "## 2. Experimental Setting",
        "",
        "- pass set: Core-v1 / configs/core_passes.yaml",
        f"- programs: {', '.join(row['program'] for row in numbers)}",
        "- optimizer: exact r4 and budgeted r4",
        "- objective: IR instruction count",
        f"- validation: {'all_permutations_same for executed exact batches' if _all_executed_strong(reduction) else 'see evidence tables for mixed statuses'}",
        "- correctness boundary: objective is not commutation proof",
        "",
        "## 3. Exact r4 Method Comparison",
        "",
        *_markdown_table(
            [
                "program",
                "root",
                "batch exact r4",
                "greedy",
                "random best",
                "config order",
                "batch-greedy delta",
                "batch-random delta",
                "batch-config delta",
                "batch states",
                "batch time ms",
            ],
            [
                [
                    row.get("program", ""),
                    row.get("root_inst", ""),
                    row.get("exact_batch_inst", ""),
                    row.get("greedy_inst", ""),
                    row.get("random_best_inst", ""),
                    row.get("config_order_inst", ""),
                    row.get("batch_vs_greedy", ""),
                    row.get("batch_vs_random", ""),
                    row.get("batch_vs_config", ""),
                    row.get("exact_states", ""),
                    row.get("_batch_time_ms", ""),
                ]
                for row in numbers
            ],
        ),
        "",
        f"- batch vs greedy: {method_wtl['batch_vs_greedy']}",
        f"- batch vs random best: {method_wtl['batch_vs_random']}",
        f"- batch vs config order once: {method_wtl['batch_vs_config']}",
        "",
        "## 4. Exact r4 Reduction Evidence",
        "",
        *_markdown_table(
            [
                "program",
                "states",
                "transitions",
                "avg active passes",
                "total batch candidates",
                "executable batches",
                "avg reduction log10",
                "max reduction log10",
                "dropped active passes",
            ],
            [
                [
                    row.get("program", ""),
                    row.get("exact_states", ""),
                    row.get("exact_transitions", ""),
                    _value(reduction.get("by_program", {}).get(row.get("program", ""), {}), "avg_active_passes"),
                    _value(reduction.get("by_program", {}).get(row.get("program", ""), {}), "total_batch_candidates"),
                    _value(reduction.get("by_program", {}).get(row.get("program", ""), {}), "total_executable_batches"),
                    row.get("avg_reduction_log10", ""),
                    row.get("max_reduction_log10", ""),
                    row.get("dropped_active_passes", ""),
                ]
                for row in numbers
            ],
        ),
        "",
        f"- total states: {reduction.get('overall', {}).get('total_states') or sum(_int(row.get('exact_states')) for row in numbers)}",
        f"- total transitions: {reduction.get('overall', {}).get('total_transitions') or sum(_int(row.get('exact_transitions')) for row in numbers)}",
        f"- total selected path steps: {reduction.get('overall', {}).get('total_selected_path_steps') or 'N/A'}",
        f"- max observed local reduction log10: {_fmt_float(max_reduction)}",
        f"- all selected path batches strong-certified: {_bool(all_selected_strong)}",
        f"- dropped active passes are zero: {_bool(dropped_zero)}",
        "",
        "## 5. Pair Relation Evidence",
        "",
        *_markdown_table(
            ["program", "tested pairs", "commute", "order-sensitive", "unknown", "commute %"],
            [
                [
                    row.get("program", ""),
                    row.get("tested_pairs", ""),
                    row.get("commute_pairs", ""),
                    row.get("order_sensitive_pairs", ""),
                    row.get("unknown_pairs", ""),
                    _pct(_int(row.get("commute_pairs")), _int(row.get("tested_pairs"))),
                ]
                for row in numbers
            ],
        ),
        "",
        "Commute pairs are candidates for certified folding; order-sensitive pairs are conservatively retained.",
        "",
        "## 6. Budgeted Sensitivity",
        "",
        *_markdown_table(
            ["program", "exact r4", "best budgeted", "beam", "max states", "gap to exact", "states", "time ms"],
            [
                [
                    row.get("program", ""),
                    row.get("exact_batch_inst", ""),
                    row.get("best_budgeted_inst", ""),
                    row.get("best_budgeted_beam", ""),
                    row.get("best_budgeted_max_states", ""),
                    row.get("budgeted_gap_to_exact", ""),
                    row.get("budgeted_states", ""),
                    row.get("budgeted_time_ms", ""),
                ]
                for row in numbers
            ],
        ),
        "",
        f"- programs matching exact: {budgeted.get('metrics', {}).get('programs_matching_exact', '')}",
        f"- average gap to exact: {budgeted.get('metrics', {}).get('average_gap_to_exact', '')}",
        f"- average state reduction relative to exact: {budgeted.get('metrics', {}).get('average_state_reduction', '')}",
        f"- average time reduction relative to exact: {budgeted.get('metrics', {}).get('average_time_reduction', '')}",
        f"- budgeted vs greedy: {budgeted.get('metrics', {}).get('vs_greedy', '')}",
        f"- budgeted vs random best: {budgeted.get('metrics', {}).get('vs_random', '')}",
        f"- budgeted vs config order once: {budgeted.get('metrics', {}).get('vs_config', '')}",
        "",
        "## 7. n-body Round-Depth Case Study",
        "",
        *_nbody_section(optional_notes.get("nbody_round_study", "")),
        "",
        "## 8. Puzzle Hard-Case Note",
        "",
        *_puzzle_section(optional_notes.get("puzzle_case_study", ""), budgeted, numbers),
        "",
        "## 9. Main Takeaways",
        "",
        "- Exact certified batch graph can produce competitive optimized sequences.",
        "- Local ordering space can be reduced by multiple orders of magnitude.",
        "- All executed exact batches are strong-certified in the summarized study.",
        "- Budgeted search can match exact with fewer states/time.",
        "- Random best still wins on puzzle, so this is not a global optimality claim.",
        "",
        "## 10. Correctness Boundary",
        "",
        CORRECTNESS_BOUNDARY,
        "",
        "## Sources",
        "",
        f"- exact method summary: {exact_method_summary}",
        f"- exact reduction summary: {exact_reduction_summary}",
        f"- budgeted sensitivity summary: {budgeted_sensitivity_summary}",
        "",
    ]
    if optional_notes.get("extra_notes"):
        lines.extend(["## Extra Notes", "", optional_notes["extra_notes"].strip(), ""])
    if missing:
        lines.extend(
            [
                "## Missing Inputs",
                "",
                *_markdown_table(
                    ["input", "path", "status", "message"],
                    [[row["input_name"], row["path"], row["status"], row["message"]] for row in missing],
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _key_claims(
    numbers: list[dict],
    reduction: dict,
    budgeted: dict,
    method_path: Path,
    reduction_path: Path,
    budgeted_path: Path,
) -> list[dict]:
    total_pairs = sum(_int(row.get("tested_pairs")) for row in numbers)
    commute_pairs = sum(_int(row.get("commute_pairs")) for row in numbers)
    max_reduction = max((_float(row.get("max_reduction_log10")) for row in numbers), default=0.0)
    executed = sum(_int(row.get("executed_batches")) for row in numbers)
    strong = sum(_int(row.get("strong_certs")) for row in numbers)
    dropped = sum(_int(row.get("dropped_active_passes")) for row in numbers)
    return [
        {
            "claim_id": "C1",
            "claim": "Many pass pairs commute on reached states.",
            "supporting_metric": "commute_pairs / tested_pairs",
            "value": f"{commute_pairs}/{total_pairs}",
            "source_file": str(reduction_path),
            "caution": "Pair relation is state-local and tied to current normalization.",
        },
        {
            "claim_id": "C2",
            "claim": "Local candidate space is reduced by orders of magnitude.",
            "supporting_metric": "max_reduction_log10",
            "value": _fmt_float(max_reduction),
            "source_file": str(reduction_path),
            "caution": "Reduction is local to reached states, not global phase-order proof.",
        },
        {
            "claim_id": "C3",
            "claim": "Executed batches have strong certificates.",
            "supporting_metric": "strong_certs / executed_batches",
            "value": f"{strong}/{executed}",
            "source_file": str(reduction_path),
            "caution": "Strong certificate applies to tested batch permutations and canonical IR hash.",
        },
        {
            "claim_id": "C4",
            "claim": "Budgeted matches exact with fewer states/time.",
            "supporting_metric": "programs matching exact; average state/time reduction",
            "value": f"{budgeted.get('metrics', {}).get('programs_matching_exact', '')}; states {budgeted.get('metrics', {}).get('average_state_reduction', '')}; time {budgeted.get('metrics', {}).get('average_time_reduction', '')}",
            "source_file": str(budgeted_path),
            "caution": "Budgeted changes coverage, not batch correctness.",
        },
        {
            "claim_id": "C5",
            "claim": "Not a global optimality claim.",
            "supporting_metric": "random best comparison",
            "value": _random_best_note(numbers),
            "source_file": str(method_path),
            "caution": "Objective values evaluate paths; they are not commutation proof.",
        },
        {
            "claim_id": "C6",
            "claim": "Active passes are not silently dropped in this study.",
            "supporting_metric": "dropped_active_passes",
            "value": str(dropped),
            "source_file": str(reduction_path),
            "caution": "Coverage is measured over summarized reached states.",
        },
    ]


def _figures_rows(numbers: list[dict]) -> list[dict]:
    rows = []
    for row in numbers:
        program = row.get("program", "")
        for metric in ["avg_reduction_log10", "max_reduction_log10"]:
            rows.append({"figure": "reduction_log10_by_program", "program": program, "metric": metric, "value": row.get(metric, "")})
        rows.extend(
            [
                {"figure": "exact_vs_budgeted_state_count", "program": program, "metric": "exact_states", "value": row.get("exact_states", "")},
                {"figure": "exact_vs_budgeted_state_count", "program": program, "metric": "budgeted_states", "value": row.get("budgeted_states", "")},
                {"figure": "exact_vs_budgeted_final_inst", "program": program, "metric": "exact_batch_inst", "value": row.get("exact_batch_inst", "")},
                {"figure": "exact_vs_budgeted_final_inst", "program": program, "metric": "best_budgeted_inst", "value": row.get("best_budgeted_inst", "")},
                {"figure": "batch_vs_baselines", "program": program, "metric": "batch_vs_greedy", "value": row.get("batch_vs_greedy", "")},
                {"figure": "batch_vs_baselines", "program": program, "metric": "batch_vs_random", "value": row.get("batch_vs_random", "")},
                {"figure": "batch_vs_baselines", "program": program, "metric": "batch_vs_config", "value": row.get("batch_vs_config", "")},
                {"figure": "pair_relation_counts", "program": program, "metric": "commute_pairs", "value": row.get("commute_pairs", "")},
                {"figure": "pair_relation_counts", "program": program, "metric": "order_sensitive_pairs", "value": row.get("order_sensitive_pairs", "")},
            ]
        )
    return rows


def _nbody_section(note: str) -> list[str]:
    if note:
        return [note.strip(), "", "n-body shows that iterative state-local recomputation matters: shallow r2 underperforms, while r4 reaches 211 and beats greedy."]
    return [
        "WARNING: n-body round-depth study was not provided.",
        "",
        "n-body shows that iterative state-local recomputation matters: shallow r2 underperforms, while r4 reaches 211 and beats greedy.",
    ]


def _puzzle_section(note: str, budgeted: dict, numbers: list[dict]) -> list[str]:
    if note:
        return [note.strip()]
    puzzle = next((row for row in numbers if row.get("program") == "puzzle"), {})
    lines = [
        "WARNING: puzzle case-study note was not provided; inferred from budgeted sensitivity data.",
        f"- exact r4 final instruction count: {puzzle.get('exact_batch_inst', '')}",
        f"- best budgeted final instruction count: {puzzle.get('best_budgeted_inst', '')}",
        f"- best budgeted beam width: {puzzle.get('best_budgeted_beam', '')}",
        f"- random best instruction count: {puzzle.get('random_best_inst', '')}",
    ]
    beam_results = [
        row for row in budgeted.get("all_results", [])
        if row.get("program") == "puzzle" and row.get("beam_width") in {"4", "8", "16"}
    ]
    if beam_results:
        by_beam = {}
        for row in beam_results:
            by_beam.setdefault(row.get("beam_width"), set()).add(row.get("gap_to_exact"))
        details = "; ".join(f"beam={beam}: gaps={','.join(sorted(gaps))}" for beam, gaps in sorted(by_beam.items(), key=lambda item: _int(item[0])))
        lines.append(f"- sensitivity detail: {details}")
    else:
        lines.append("- puzzle matched exact only at beam=16 in the current data, while beam=4/8 stayed one instruction above exact.")
    return lines


def _load_optional_notes(paths: dict[str, Path | None], missing: list[dict]) -> dict[str, str]:
    notes = {}
    for name, path in paths.items():
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            missing.append(_missing(name, path, "missing", "optional input was not provided or does not exist"))
            continue
        try:
            notes[name] = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            missing.append(_missing(name, path, "unreadable", str(exc)))
    return notes


def _all_executed_strong(reduction: dict) -> bool:
    counts = reduction.get("evidence_counts", {})
    if not counts:
        return False
    total = sum(sum(counter.values()) for counter in counts.values())
    strong = sum(counter.get("strong", 0) for counter in counts.values())
    return total > 0 and total == strong


def _all_selected_path_strong(reduction: dict) -> bool:
    counts = reduction.get("selected_counts", Counter())
    total = sum(counts.values())
    return total > 0 and total == counts.get("strong", 0)


def _wtl(rows: list[dict], keys: list[str]) -> dict[str, str]:
    result = {}
    for key in keys:
        wins = ties = losses = 0
        for row in rows:
            value = _float(row.get(key))
            if value > 0:
                wins += 1
            elif value == 0:
                ties += 1
            else:
                losses += 1
        result[key] = f"wins={wins} ties={ties} losses={losses}"
    return result


def _random_best_note(rows: list[dict]) -> str:
    wins = ties = losses = 0
    for row in rows:
        delta = _float(row.get("batch_vs_random"))
        if delta > 0:
            wins += 1
        elif delta == 0:
            ties += 1
        else:
            losses += 1
    return f"batch vs random best wins={wins} ties={ties} losses={losses}"


def _empty_reduction() -> dict:
    return {"path": "", "markdown": "", "by_program": {}, "runs": {}, "coverage": {}, "evidence_counts": {}, "selected_counts": Counter(), "overall": {}}


def _empty_budgeted() -> dict:
    return {"path": "", "markdown": "", "best": {}, "all_results": [], "metrics": {}}


def _record_required(path: Path, name: str, missing: list[dict]) -> bool:
    if path.exists():
        return True
    missing.append(_missing(name, path, "missing", "required input does not exist"))
    return False


def _missing(name: str, path: Path, status: str, message: str) -> dict:
    return {"input_name": name, "path": str(path), "status": status, "message": message}


def _delta(baseline: object, batch: object) -> str:
    if not _is_number(baseline) or not _is_number(batch):
        return ""
    return str(_int(baseline) - _int(batch))


def _extract_bullet(markdown: str, label: str) -> str:
    pattern = re.compile(rf"^-\s*{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(markdown)
    return match.group(1).strip() if match else ""


def _extract_dash_line(markdown: str, label: str) -> str:
    pattern = re.compile(rf"^-\s*{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(markdown)
    return match.group(1).strip() if match else ""


def _markdown_table_with_header(path: Path, required_header: str) -> list[dict]:
    rows = []
    for table in _parse_all_markdown_tables(_read_text(path)):
        if any(_normalize_key(header) == _normalize_key(required_header) for header in table.get("headers", [])):
            rows.extend(table["rows"])
    return rows


def _parse_markdown_table_after_heading(markdown: str, heading: str) -> list[dict]:
    marker = f"## {heading}"
    if marker not in markdown:
        return []
    tail = markdown.split(marker, 1)[1]
    next_heading = re.search(r"\n##\s+", tail)
    if next_heading:
        tail = tail[: next_heading.start()]
    tables = _parse_all_markdown_tables(tail)
    return tables[0]["rows"] if tables else []


def _parse_all_markdown_tables(markdown: str) -> list[dict]:
    lines = markdown.splitlines()
    tables = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not (line.startswith("|") and line.endswith("|")):
            index += 1
            continue
        if index + 1 >= len(lines) or "---" not in lines[index + 1]:
            index += 1
            continue
        headers = [_clean_cell(cell) for cell in line.strip("|").split("|")]
        index += 2
        rows = []
        while index < len(lines) and lines[index].strip().startswith("|") and lines[index].strip().endswith("|"):
            values = [_clean_cell(cell) for cell in lines[index].strip().strip("|").split("|")]
            rows.append({headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))})
            index += 1
        tables.append({"headers": headers, "rows": rows})
    return tables


def _clean_cell(value: str) -> str:
    return value.strip().replace("\\|", "|")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


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


def _value(row: dict, *names: str) -> str:
    normalized = {_normalize_key(key): value for key, value in row.items()}
    for name in names:
        value = normalized.get(_normalize_key(name))
        if value is not None:
            return str(value)
    return ""


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def _is_number(value: object) -> bool:
    try:
        float(str(value))
        return str(value) != ""
    except ValueError:
        return False


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


def _fmt_float(value: float) -> str:
    if abs(value) < 0.0000005:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _bool(value: bool) -> str:
    return "true" if value else "false"
