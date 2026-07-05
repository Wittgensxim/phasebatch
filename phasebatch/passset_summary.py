from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PASSSET_COMPARISON_MATRIX_FIELDS = [
    "program",
    "passset",
    "valid_passes",
    "invalid_passes",
    "active_passes_depth0",
    "tested_pairs_depth0",
    "commute_pairs_depth0",
    "sensitive_pairs_depth0",
    "unknown_pairs_depth0",
    "batch_candidates_depth0",
    "certified_batches_depth0",
    "sampled_batches_depth0",
    "skipped_batches_depth0",
    "dropped_active_passes",
    "states_reached",
    "transitions",
    "final_ir_inst_count",
    "optimized_pipeline_length",
    "total_time_ms",
]

PASSSET_FAILURE_FIELDS = [
    "input_path",
    "source_kind",
    "program",
    "passset",
    "status",
    "error_message",
]


def summarize_passsets(inputs: list[Path | str], out_dir: Path, *, warn=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows: list[dict] = []
    internal_rows: list[dict] = []
    failure_rows: list[dict] = []

    for raw_input in inputs:
        source = Path(raw_input)
        if not source.exists():
            message = f"missing input directory: {source}"
            warn(f"warning: {message}")
            failure_rows.append(_failure_row(source, "", "", "", "missing", message))
            continue
        if not source.is_dir():
            message = f"input is not a directory: {source}"
            warn(f"warning: {message}")
            failure_rows.append(_failure_row(source, "", "", "", "not_directory", message))
            continue

        if (source / "passset_smoke_runs.csv").exists():
            rows, failures = _read_passset_smoke_root(source)
            internal_rows.extend(rows)
            failure_rows.extend(failures)
        elif (source / "v3_loop_runs.csv").exists():
            rows, failures = _read_v3_loop_root(source)
            internal_rows.extend(rows)
            failure_rows.extend(failures)
        else:
            direct = _read_direct_run_dir(source)
            if direct:
                internal_rows.append(direct)
            else:
                message = f"unrecognized passset output directory: {source}"
                warn(f"warning: {message}")
                failure_rows.append(_failure_row(source, "", "", "", "unrecognized", message))

    matrix_rows = [{field: row.get(field, "") for field in PASSSET_COMPARISON_MATRIX_FIELDS} for row in internal_rows]
    _write_csv(out_dir / "passset_comparison_matrix.csv", PASSSET_COMPARISON_MATRIX_FIELDS, matrix_rows)
    _write_csv(out_dir / "passset_failures.csv", PASSSET_FAILURE_FIELDS, failure_rows)
    report = _write_report(out_dir, internal_rows, failure_rows)
    return {
        "out_dir": str(out_dir),
        "matrix_rows": len(matrix_rows),
        "failures": len(failure_rows),
        "passset_comparison_report_md": str(report),
        "passset_comparison_matrix_csv": str(out_dir / "passset_comparison_matrix.csv"),
        "passset_failures_csv": str(out_dir / "passset_failures.csv"),
    }


def _read_passset_smoke_root(root: Path) -> tuple[list[dict], list[dict]]:
    run_rows = _read_csv(root / "passset_smoke_runs.csv")
    failures: list[dict] = []
    rows: list[dict] = []
    for run in run_rows:
        program = run.get("program", "")
        passset_name = run.get("passset", "")
        passset = _normalize_passset(passset_name)
        passset_dir = root / program / passset_name
        row = _collect_matrix_row(
            program=program,
            passset=passset,
            audit_dir=passset_dir / "audit",
            optimize_dir=passset_dir / "optimize",
            fallback={
                "valid_passes": run.get("valid_passes", ""),
                "invalid_passes": run.get("invalid_passes", ""),
                "active_passes_depth0": run.get("active_passes_depth0", ""),
                "states_reached": run.get("states_reached", ""),
                "transitions": run.get("transitions", ""),
                "final_ir_inst_count": run.get("final_ir_inst_count", ""),
                "total_time_ms": run.get("time_ms", ""),
                "_root_ir_inst_count": run.get("root_ir_inst_count", ""),
            },
        )
        rows.append(row)
        if run.get("audit_status") != "success" or run.get("optimize_status") != "success":
            failures.append(
                _failure_row(
                    root,
                    "passset_smoke",
                    program,
                    passset,
                    "failed",
                    run.get("error_message", "") or f"audit={run.get('audit_status')} optimize={run.get('optimize_status')}",
                )
            )
    return rows, failures


def _read_v3_loop_root(root: Path) -> tuple[list[dict], list[dict]]:
    run_rows = _read_csv(root / "v3_loop_runs.csv")
    summary_by_program = {row.get("program", ""): row for row in _read_csv(root / "v3_loop_summary.csv")}
    rows: list[dict] = []
    failures: list[dict] = []
    for run in run_rows:
        program = run.get("program", "")
        summary = summary_by_program.get(program, {})
        program_dir = root / program
        fallback = {
            "valid_passes": run.get("valid_passes", "") or summary.get("valid_passes", ""),
            "invalid_passes": run.get("invalid_passes", ""),
            "active_passes_depth0": summary.get("active_passes_depth0", "") or run.get("total_active_passes_depth0", ""),
            "tested_pairs_depth0": summary.get("tested_pairs_depth0", ""),
            "commute_pairs_depth0": summary.get("commute_pairs_depth0", ""),
            "sensitive_pairs_depth0": summary.get("sensitive_pairs_depth0", ""),
            "batch_candidates_depth0": summary.get("batch_candidates_depth0", ""),
            "certified_batches_depth0": summary.get("certified_batches_depth0", ""),
            "sampled_batches_depth0": summary.get("sampled_batches_depth0", ""),
            "skipped_batches_depth0": summary.get("skipped_batches_depth0", ""),
            "dropped_active_passes": summary.get("dropped_active_passes", ""),
            "states_reached": run.get("states_reached", "") or summary.get("states_reached", ""),
            "transitions": run.get("transitions", "") or summary.get("transitions", ""),
            "final_ir_inst_count": run.get("final_ir_inst_count", "") or summary.get("final_ir_inst_count", ""),
            "optimized_pipeline_length": run.get("optimized_pipeline_length", ""),
            "total_time_ms": run.get("time_ms", ""),
            "_valid_loop_passes": run.get("valid_loop_passes", "") or summary.get("valid_loop_passes", ""),
            "_invalid_loop_passes": run.get("invalid_loop_passes", ""),
            "_active_loop_passes_depth0": run.get("active_loop_passes_depth0", "") or summary.get("active_loop_passes_depth0", ""),
            "_max_component_size_depth0": summary.get("max_component_size_depth0", ""),
        }
        row = _collect_matrix_row(
            program=program,
            passset="v3",
            audit_dir=program_dir / "audit",
            optimize_dir=program_dir / "optimize",
            fallback=fallback,
        )
        row["_valid_loop_passes"] = fallback.get("_valid_loop_passes", row.get("_valid_loop_passes", ""))
        row["_invalid_loop_passes"] = fallback.get("_invalid_loop_passes", row.get("_invalid_loop_passes", ""))
        row["_active_loop_passes_depth0"] = fallback.get("_active_loop_passes_depth0", row.get("_active_loop_passes_depth0", ""))
        row["_max_component_size_depth0"] = fallback.get("_max_component_size_depth0", row.get("_max_component_size_depth0", ""))
        rows.append(row)
        if run.get("status") != "success":
            failures.append(_failure_row(root, "v3_loop_smoke", program, "v3", "failed", run.get("error_message", "")))
    return rows, failures


def _read_direct_run_dir(root: Path) -> dict | None:
    optimize_dir = root / "optimize" if (root / "optimize").exists() else root
    if not (optimize_dir / "states" / "S0000").exists():
        return None
    audit_dir = root / "audit"
    return _collect_matrix_row(
        program=root.name,
        passset=_normalize_passset(root.name),
        audit_dir=audit_dir,
        optimize_dir=optimize_dir,
        fallback={},
    )


def _collect_matrix_row(program: str, passset: str, audit_dir: Path, optimize_dir: Path, fallback: dict) -> dict:
    audit = _audit_metrics(audit_dir)
    optimize = _optimize_metrics(optimize_dir)
    row = {
        "program": program,
        "passset": passset,
        "valid_passes": _first_nonempty(fallback.get("valid_passes"), audit.get("valid_passes")),
        "invalid_passes": _first_nonempty(fallback.get("invalid_passes"), audit.get("invalid_passes")),
        "active_passes_depth0": _first_nonempty(optimize.get("active_passes_depth0"), fallback.get("active_passes_depth0")),
        "tested_pairs_depth0": _first_nonempty(optimize.get("tested_pairs_depth0"), fallback.get("tested_pairs_depth0")),
        "commute_pairs_depth0": _first_nonempty(optimize.get("commute_pairs_depth0"), fallback.get("commute_pairs_depth0")),
        "sensitive_pairs_depth0": _first_nonempty(optimize.get("sensitive_pairs_depth0"), fallback.get("sensitive_pairs_depth0")),
        "unknown_pairs_depth0": _first_nonempty(optimize.get("unknown_pairs_depth0"), fallback.get("unknown_pairs_depth0")),
        "batch_candidates_depth0": _first_nonempty(optimize.get("batch_candidates_depth0"), fallback.get("batch_candidates_depth0")),
        "certified_batches_depth0": _first_nonempty(optimize.get("certified_batches_depth0"), fallback.get("certified_batches_depth0")),
        "sampled_batches_depth0": _first_nonempty(optimize.get("sampled_batches_depth0"), fallback.get("sampled_batches_depth0")),
        "skipped_batches_depth0": _first_nonempty(optimize.get("skipped_batches_depth0"), fallback.get("skipped_batches_depth0")),
        "dropped_active_passes": _first_nonempty(optimize.get("dropped_active_passes"), fallback.get("dropped_active_passes")),
        "states_reached": _first_nonempty(optimize.get("states_reached"), fallback.get("states_reached")),
        "transitions": _first_nonempty(optimize.get("transitions"), fallback.get("transitions")),
        "final_ir_inst_count": _first_nonempty(optimize.get("final_ir_inst_count"), fallback.get("final_ir_inst_count")),
        "optimized_pipeline_length": _first_nonempty(optimize.get("optimized_pipeline_length"), fallback.get("optimized_pipeline_length")),
        "total_time_ms": _first_nonempty(fallback.get("total_time_ms"), optimize.get("total_time_ms")),
        "_root_ir_inst_count": _first_nonempty(optimize.get("_root_ir_inst_count"), fallback.get("_root_ir_inst_count")),
        "_categories": audit.get("categories", ""),
        "_configured_passes": _first_nonempty(audit.get("configured_passes"), _sum_strings(fallback.get("valid_passes"), fallback.get("invalid_passes"))),
        "_valid_loop_passes": audit.get("valid_loop_passes", fallback.get("_valid_loop_passes", "")),
        "_invalid_loop_passes": audit.get("invalid_loop_passes", fallback.get("_invalid_loop_passes", "")),
    }
    return row


def _audit_metrics(audit_dir: Path) -> dict:
    rows = _read_csv(audit_dir / "pass_audit.csv")
    invalid_rows = _read_csv(audit_dir / "invalid_passes.csv")
    valid_rows = [row for row in rows if _truthy(row.get("valid_on_input", "true"))]
    if rows:
        invalid_count = len([row for row in rows if not _truthy(row.get("valid_on_input", "true"))])
        valid_count = len(valid_rows)
    else:
        invalid_count = len(invalid_rows)
        valid_count = 0
    categories = sorted({row.get("category", "") for row in rows if row.get("category")})
    loop_valid = [row for row in valid_rows if row.get("category") == "loop"]
    loop_invalid = [row for row in rows if row.get("category") == "loop" and not _truthy(row.get("valid_on_input", "true"))]
    return {
        "valid_passes": str(valid_count) if rows else "",
        "invalid_passes": str(invalid_count) if rows or invalid_rows else "",
        "configured_passes": str(len(rows)) if rows else "",
        "categories": ",".join(categories),
        "valid_loop_passes": str(len(loop_valid)) if rows else "",
        "invalid_loop_passes": str(len(loop_invalid)) if rows else "",
    }


def _optimize_metrics(optimize_dir: Path) -> dict:
    root_dir = optimize_dir / "states" / "S0000"
    per_state = _first_row(root_dir / "per_state_summary.csv")
    batch_summary = _first_row(root_dir / "batch_summary.csv")
    coverage = _first_row(root_dir / "coverage_summary.csv")
    chosen = _first_row(optimize_dir / "chosen_path_summary.csv")
    correctness = _read_csv(root_dir / "batch_correctness.csv")
    states = _read_csv(optimize_dir / "states.csv")
    transitions = _read_csv(optimize_dir / "batch_state_transitions.csv")
    certified = sum(1 for row in correctness if row.get("correctness_class") == "certified_batch")
    sampled = sum(1 for row in correctness if row.get("correctness_class") == "sampled_batch")
    skipped = sum(1 for row in correctness if row.get("can_execute", "").lower() != "true")
    return {
        "active_passes_depth0": _first_value(per_state, ["active_passes"], ""),
        "tested_pairs_depth0": _first_value(per_state, ["pairs_tested", "pair_rows"], ""),
        "commute_pairs_depth0": _first_value(per_state, ["dynamic_commute", "commute_pairs"], ""),
        "sensitive_pairs_depth0": _first_value(per_state, ["order_sensitive", "order_sensitive_pairs"], ""),
        "unknown_pairs_depth0": _first_value(per_state, ["unknown", "unknown_pairs"], ""),
        "batch_candidates_depth0": _first_value(batch_summary, ["batch_candidates"], ""),
        "certified_batches_depth0": str(certified) if correctness else "",
        "sampled_batches_depth0": str(sampled) if correctness else "",
        "skipped_batches_depth0": str(skipped) if correctness else "",
        "dropped_active_passes": _first_value(coverage, ["dropped_active_passes", "dropped"], ""),
        "states_reached": str(len(states)) if states else "",
        "transitions": str(len(transitions)) if transitions else "",
        "final_ir_inst_count": _first_value(chosen, ["final_ir_inst_count"], ""),
        "optimized_pipeline_length": str(_pipeline_length(optimize_dir)),
        "_root_ir_inst_count": _first_value(chosen, ["root_ir_inst_count"], ""),
    }


def _write_report(out_dir: Path, rows: list[dict], failures: list[dict]) -> Path:
    grouped = _group_by_passset(rows)
    recommendation = _recommendation(grouped, failures)
    lines = [
        "# Pass Set Comparison Report",
        "",
        "## Purpose",
        "",
        "- v1 is the stable regression pass set.",
        "- v2 adds scalar/memory/CFG passes.",
        "- v3 adds loop-middle-end passes.",
        "- This report measures scalability and coverage, not proof of global optimality.",
        "",
        "## Pass Set Sizes",
        "",
        *_markdown_table(
            ["passset", "configured passes", "valid passes", "invalid passes", "categories"],
            [
                [
                    passset,
                    _max_field(passset_rows, "_configured_passes"),
                    _max_field(passset_rows, "valid_passes"),
                    _max_field(passset_rows, "invalid_passes"),
                    _merged_categories(passset_rows),
                ]
                for passset, passset_rows in grouped.items()
            ],
        ),
        "",
        "## Coverage and Activity",
        "",
        *_markdown_table(
            ["passset", "avg active passes depth0", "avg tested pairs depth0", "avg commute", "avg sensitive", "avg unknown"],
            [
                [
                    passset,
                    _avg_field(passset_rows, "active_passes_depth0"),
                    _avg_field(passset_rows, "tested_pairs_depth0"),
                    _avg_field(passset_rows, "commute_pairs_depth0"),
                    _avg_field(passset_rows, "sensitive_pairs_depth0"),
                    _avg_field(passset_rows, "unknown_pairs_depth0"),
                ]
                for passset, passset_rows in grouped.items()
            ],
        ),
        "",
        "## Batch Reduction",
        "",
        *_markdown_table(
            ["passset", "avg batch candidates", "avg certified batches", "avg sampled batches", "avg skipped batches", "avg dropped active passes"],
            [
                [
                    passset,
                    _avg_field(passset_rows, "batch_candidates_depth0"),
                    _avg_field(passset_rows, "certified_batches_depth0"),
                    _avg_field(passset_rows, "sampled_batches_depth0"),
                    _avg_field(passset_rows, "skipped_batches_depth0"),
                    _avg_field(passset_rows, "dropped_active_passes"),
                ]
                for passset, passset_rows in grouped.items()
            ],
        ),
        "",
        "## Search / Optimizer Cost",
        "",
        *_markdown_table(
            ["passset", "avg states reached", "avg transitions", "avg pipeline length", "avg time ms"],
            [
                [
                    passset,
                    _avg_field(passset_rows, "states_reached"),
                    _avg_field(passset_rows, "transitions"),
                    _avg_field(passset_rows, "optimized_pipeline_length"),
                    _avg_field(passset_rows, "total_time_ms"),
                ]
                for passset, passset_rows in grouped.items()
            ],
        ),
        "",
        "## Objective Signal",
        "",
        *_markdown_table(
            ["passset", "avg final IR inst count", "avg reduction vs root if available"],
            [
                [
                    passset,
                    _avg_field(passset_rows, "final_ir_inst_count"),
                    _avg_reduction(passset_rows),
                ]
                for passset, passset_rows in grouped.items()
            ],
        ),
        "",
        "Objective values are evaluation signals only and are not used as commutation proof.",
        "",
        "## V2 vs V1",
        "",
        *_v2_vs_v1_bullets(grouped),
        "",
        "## V3 vs V2",
        "",
        *_v3_vs_v2_bullets(grouped),
        "",
        "## Recommendation",
        "",
        f"- {recommendation}",
        "",
    ]
    if failures:
        lines.extend(
            [
                "## Failures",
                "",
                *_markdown_table(
                    ["input", "source", "program", "passset", "status", "error"],
                    [
                        [
                            row.get("input_path", ""),
                            row.get("source_kind", ""),
                            row.get("program", ""),
                            row.get("passset", ""),
                            row.get("status", ""),
                            row.get("error_message", ""),
                        ]
                        for row in failures
                    ],
                ),
                "",
            ]
        )
    path = out_dir / "passset_comparison_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _v2_vs_v1_bullets(grouped: dict[str, list[dict]]) -> list[str]:
    if "v1" not in grouped or "v2" not in grouped:
        return ["- v1/v2 data is incomplete in this report."]
    return [
        _compare_bullet("v2 increased valid passes", grouped["v1"], grouped["v2"], "valid_passes"),
        _compare_bullet("v2 increased active passes", grouped["v1"], grouped["v2"], "active_passes_depth0"),
        _compare_bullet("v2 increased pair tests", grouped["v1"], grouped["v2"], "tested_pairs_depth0"),
        _compare_bullet("v2 increased certified batches", grouped["v1"], grouped["v2"], "certified_batches_depth0"),
        f"- dropped active passes remained zero: {_all_zero(grouped['v2'], 'dropped_active_passes')}",
    ]


def _v3_vs_v2_bullets(grouped: dict[str, list[dict]]) -> list[str]:
    if "v3" not in grouped:
        return ["- v3 data is not present in this report."]
    v3_rows = grouped["v3"]
    return [
        f"- loop passes resolved successfully: {_sum_field(v3_rows, '_valid_loop_passes') > 0 and _sum_field(v3_rows, '_invalid_loop_passes') == 0}",
        f"- active loop passes were observed: {_sum_field(v3_rows, '_active_loop_passes_depth0') > 0}",
        f"- v3 caused validation failures or large components: {_sum_field(v3_rows, 'skipped_batches_depth0') > 0 or _max_numeric(v3_rows, '_max_component_size_depth0') > 20}",
        "- v3 should be used for loop-heavy benchmarks only.",
    ]


def _recommendation(grouped: dict[str, list[dict]], failures: list[dict]) -> str:
    v3_rows = grouped.get("v3", [])
    if v3_rows:
        if _sum_field(v3_rows, "invalid_passes") > 0 or _sum_field(v3_rows, "dropped_active_passes") > 0:
            return "investigate invalid loop pass resolution and limit v3 until invalid/dropped counts are resolved."
        if _sum_field(v3_rows, "skipped_batches_depth0") > 0 or _max_numeric(v3_rows, "_max_component_size_depth0") > 30:
            return "reduce v3 if validation cost explodes."
        return "use v3 only for loop-heavy case studies; keep v2 as the main experimental pass set."
    if grouped.get("v2"):
        return "keep v2 as main experimental pass set."
    return "collect v1/v2/v3 smoke outputs before choosing a main pass set."


def _compare_bullet(label: str, left_rows: list[dict], right_rows: list[dict], field: str) -> str:
    left = _avg_numeric(left_rows, field)
    right = _avg_numeric(right_rows, field)
    if left is None or right is None:
        return f"- {label}: insufficient data"
    return f"- {label}: {right > left} ({_fmt(left)} -> {_fmt(right)})"


def _group_by_passset(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    order = {"v1": 0, "v2": 1, "v3": 2}
    for row in rows:
        grouped[row.get("passset", "")].append(row)
    return dict(sorted(grouped.items(), key=lambda item: (order.get(item[0], 99), item[0])))


def _normalize_passset(value: str) -> str:
    lower = value.lower()
    if "v1" in lower:
        return "v1"
    if "v2" in lower:
        return "v2"
    if "v3" in lower:
        return "v3"
    return value


def _avg_field(rows: list[dict], field: str) -> str:
    value = _avg_numeric(rows, field)
    return "" if value is None else _fmt(value)


def _avg_numeric(rows: list[dict], field: str) -> float | None:
    values = [_to_float(row.get(field, "")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _avg_reduction(rows: list[dict]) -> str:
    values = []
    for row in rows:
        root = _to_float(row.get("_root_ir_inst_count", ""))
        final = _to_float(row.get("final_ir_inst_count", ""))
        if root and final is not None:
            values.append((root - final) / root * 100)
    if not values:
        return ""
    return f"{sum(values) / len(values):.2f}%"


def _max_field(rows: list[dict], field: str) -> str:
    value = _max_numeric(rows, field)
    return "" if value == 0 and not any(str(row.get(field, "")).strip() for row in rows) else str(value)


def _max_numeric(rows: list[dict], field: str) -> int:
    values = []
    for row in rows:
        value = _to_float(row.get(field, ""))
        if value is not None:
            values.append(int(value))
    return max(values, default=0)


def _sum_field(rows: list[dict], field: str) -> int:
    total = 0
    for row in rows:
        value = _to_float(row.get(field, ""))
        if value is not None:
            total += int(value)
    return total


def _all_zero(rows: list[dict], field: str) -> bool:
    values = [_to_float(row.get(field, "")) for row in rows]
    values = [value for value in values if value is not None]
    return bool(values) and all(value == 0 for value in values)


def _merged_categories(rows: list[dict]) -> str:
    categories: set[str] = set()
    for row in rows:
        categories.update(part for part in row.get("_categories", "").split(",") if part)
    return ",".join(sorted(categories))


def _sum_strings(*values: object) -> str:
    total = 0
    found = False
    for value in values:
        number = _to_float(value)
        if number is not None:
            total += int(number)
            found = True
    return str(total) if found else ""


def _first_nonempty(*values: object) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _to_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _fmt(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _first_row(path: Path) -> dict:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _first_value(row: dict, names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _pipeline_length(optimize_dir: Path) -> int:
    for name in ("optimized_pipeline_names.txt", "optimized_pipeline.txt"):
        path = optimize_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return 0
        return len([part for part in text.replace("\n", ",").split(",") if part.strip()])
    return 0


def _failure_row(path: Path, source_kind: str, program: str, passset: str, status: str, error_message: str) -> dict:
    return {
        "input_path": str(path),
        "source_kind": source_kind,
        "program": program,
        "passset": passset,
        "status": status,
        "error_message": error_message,
    }


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
