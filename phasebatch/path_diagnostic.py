from __future__ import annotations

import csv
import json
from pathlib import Path

from .batch_objective import count_ir_instructions
from .runner import run_opt
from .schema import MISSED_PASS_DIAGNOSTIC_FIELDS, PIPELINE_COMPARISON_FIELDS, PREFIX_EVAL_FIELDS


METHOD_ORDER = [
    "batch_optimizer",
    "greedy_single_pass",
    "random_single_pass_best",
    "config_order_once",
    "default_O0",
    "default_O2",
]


def diagnose_paths(run_dir: Path, baseline_dir: Path | None = None, timeout: int = 10) -> dict:
    run_dir = Path(run_dir)
    baseline_dir = Path(baseline_dir) if baseline_dir is not None else run_dir
    baselines = _baseline_rows(run_dir, baseline_dir)
    comparison_rows = _pipeline_comparison_rows(baselines)

    root_ir = _root_ir(run_dir)
    root_count = count_ir_instructions(root_ir) if root_ir.exists() else 0
    opt = _opt_path(run_dir)

    batch_sequence = _optimized_pipeline(run_dir, baselines)
    greedy_sequence = _greedy_sequence(run_dir, baseline_dir, baselines)
    random_sequence = _random_sequence(run_dir, baseline_dir, baselines)

    batch_prefix = _prefix_rows(run_dir, "batch_optimizer", batch_sequence, root_ir, root_count, opt, timeout)
    greedy_prefix = _prefix_rows(run_dir, "greedy_single_pass", greedy_sequence, root_ir, root_count, opt, timeout)
    random_prefix = _prefix_rows(run_dir, "random_single_pass_best", random_sequence, root_ir, root_count, opt, timeout)
    missed_rows = _missed_pass_rows(run_dir, greedy_sequence, batch_sequence)

    comparison_path = run_dir / "pipeline_comparison.csv"
    batch_prefix_path = run_dir / "prefix_eval_batch.csv"
    greedy_prefix_path = run_dir / "prefix_eval_greedy.csv"
    random_prefix_path = run_dir / "prefix_eval_random.csv"
    missed_path = run_dir / "missed_pass_diagnostic.csv"
    md_path = run_dir / "path_diagnostic.md"

    _write_csv(comparison_path, PIPELINE_COMPARISON_FIELDS, comparison_rows)
    _write_csv(batch_prefix_path, PREFIX_EVAL_FIELDS, batch_prefix)
    _write_csv(greedy_prefix_path, PREFIX_EVAL_FIELDS, greedy_prefix)
    _write_csv(random_prefix_path, PREFIX_EVAL_FIELDS, random_prefix)
    _write_csv(missed_path, MISSED_PASS_DIAGNOSTIC_FIELDS, missed_rows)
    _write_markdown(md_path, comparison_rows, batch_prefix, greedy_prefix, random_prefix, missed_rows)
    return {
        "path_diagnostic_md": str(md_path),
        "pipeline_comparison_csv": str(comparison_path),
        "prefix_eval_batch_csv": str(batch_prefix_path),
        "prefix_eval_greedy_csv": str(greedy_prefix_path),
        "prefix_eval_random_csv": str(random_prefix_path),
        "missed_pass_diagnostic_csv": str(missed_path),
        "methods": len(comparison_rows),
    }


def _pipeline_comparison_rows(baselines: list[dict]) -> list[dict]:
    by_method = {row.get("method", ""): row for row in baselines}
    rows = []
    for method in METHOD_ORDER:
        row = by_method.get(method)
        if not row:
            continue
        rows.append({field: row.get(field, "") for field in PIPELINE_COMPARISON_FIELDS})
    return rows


def _prefix_rows(
    run_dir: Path,
    method: str,
    sequence: list[str],
    root_ir: Path,
    root_count: int,
    opt: str,
    timeout: int,
) -> list[dict]:
    if not sequence:
        return []
    rows = [
        {
            "method": method,
            "prefix_len": "0",
            "pass_prefix": "",
            "ir_inst_count": str(root_count),
            "inst_delta_from_root": "0",
            "status": "success" if root_ir.exists() else "failed",
            "error_message": "" if root_ir.exists() else f"missing root IR: {root_ir}",
        }
    ]
    prefix_dir = run_dir / "path_diagnostic_prefixes" / method
    prefix_dir.mkdir(parents=True, exist_ok=True)
    for prefix_len in range(1, len(sequence) + 1):
        prefix = sequence[:prefix_len]
        output_ll = prefix_dir / f"prefix_{prefix_len:04d}.ll"
        try:
            result = run_opt(opt, root_ir, prefix, output_ll, timeout)
            if result.success and output_ll.exists():
                count = count_ir_instructions(output_ll)
                rows.append(_prefix_row(method, prefix, count, root_count, "success", ""))
            else:
                rows.append(_prefix_row(method, prefix, "", root_count, "failed", _result_error(result)))
        except Exception as exc:
            rows.append(_prefix_row(method, prefix, "", root_count, "failed", str(exc)))
    return rows


def _prefix_row(method: str, prefix: list[str], count: int | str, root_count: int, status: str, error: str) -> dict:
    delta = ""
    if count != "":
        delta = str(int(count) - root_count)
    return {
        "method": method,
        "prefix_len": str(len(prefix)),
        "pass_prefix": ";".join(prefix),
        "ir_inst_count": str(count),
        "inst_delta_from_root": delta,
        "status": status,
        "error_message": error,
    }


def _missed_pass_rows(run_dir: Path, greedy_sequence: list[str], batch_sequence: list[str]) -> list[dict]:
    if not greedy_sequence:
        return []
    root_state = run_dir / "states" / "S0000"
    candidates = _read_csv(root_state / "batch_candidates.csv")
    correctness = _read_csv(root_state / "batch_correctness.csv")
    pair_rows = _read_csv(root_state / "pair_relation.csv")
    candidate_passes = _passes_by_class(candidates, correctness)
    batch_set = set(batch_sequence)
    rows = []
    for index, pass_name in enumerate(greedy_sequence):
        in_batch = pass_name in batch_set
        in_candidate = pass_name in candidate_passes["any"]
        in_certified = pass_name in candidate_passes["certified_batch"]
        in_sampled = pass_name in candidate_passes["sampled_batch"]
        in_rejected = pass_name in candidate_passes["rejected_batch"]
        in_sensitive = _appears_in_order_sensitive_pair(pass_name, pair_rows)
        rows.append(
            {
                "greedy_step": str(index),
                "greedy_pass": pass_name,
                "appears_in_batch_pipeline": _bool(in_batch),
                "appears_in_any_root_batch_candidate": _bool(in_candidate),
                "appears_in_any_certified_root_batch": _bool(in_certified),
                "appears_in_any_sampled_root_batch": _bool(in_sampled),
                "appears_in_any_rejected_root_batch": _bool(in_rejected),
                "appears_in_any_order_sensitive_pair": _bool(in_sensitive),
                "diagnostic_reason": _diagnostic_reason(
                    in_batch=in_batch,
                    in_candidate=in_candidate,
                    in_certified=in_certified,
                    in_sampled=in_sampled,
                    in_rejected=in_rejected,
                    in_sensitive=in_sensitive,
                ),
            }
        )
    return rows


def _write_markdown(
    path: Path,
    comparison_rows: list[dict],
    batch_prefix: list[dict],
    greedy_prefix: list[dict],
    random_prefix: list[dict],
    missed_rows: list[dict],
) -> None:
    divergence = _first_divergence(batch_prefix, greedy_prefix)
    lines = [
        "# Path Diagnostic",
        "",
        "## Method Comparison",
        "",
        *_markdown_table(
            ["method", "status", "final inst", "delta", "sequence length"],
            [
                [
                    row.get("method", ""),
                    row.get("status", ""),
                    row.get("final_ir_inst_count", ""),
                    row.get("ir_inst_delta", ""),
                    row.get("final_sequence_length", ""),
                ]
                for row in comparison_rows
            ],
        ),
        "",
        "## Prefix Curves",
        "",
        *_prefix_curve_table(batch_prefix, greedy_prefix, random_prefix),
        "",
        "## Where Greedy Diverges",
        "",
    ]
    if divergence:
        lines.extend(
            [
                f"- first prefix where greedy is lower: {divergence['prefix_len']}",
                f"- greedy prefix: {divergence['greedy_prefix']}",
                f"- batch prefix: {divergence['batch_prefix']}",
                f"- greedy inst: {divergence['greedy_inst']}",
                f"- batch inst: {divergence['batch_inst']}",
            ]
        )
    else:
        lines.append("- No prefix where greedy has a lower IR instruction count than batch was observed.")
    lines.extend(
        [
            "",
            "## Greedy Pass Coverage in Batch Search",
            "",
            *_markdown_table(
                ["step", "pass", "in batch pipeline", "root candidate", "certified", "sampled", "order-sensitive", "reason"],
                [
                    [
                        row.get("greedy_step", ""),
                        row.get("greedy_pass", ""),
                        row.get("appears_in_batch_pipeline", ""),
                        row.get("appears_in_any_root_batch_candidate", ""),
                        row.get("appears_in_any_certified_root_batch", ""),
                        row.get("appears_in_any_sampled_root_batch", ""),
                        row.get("appears_in_any_order_sensitive_pair", ""),
                        row.get("diagnostic_reason", ""),
                    ]
                    for row in missed_rows
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- This diagnostic identifies likely reasons; it does not prove global optimality.",
            "- A greedy win may indicate insufficient rounds, conservative certification, or missing batch candidates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _first_divergence(batch_prefix: list[dict], greedy_prefix: list[dict]) -> dict:
    batch_by_len = {row.get("prefix_len", ""): row for row in batch_prefix if row.get("status") == "success"}
    greedy_by_len = {row.get("prefix_len", ""): row for row in greedy_prefix if row.get("status") == "success"}
    for prefix_len in sorted(set(batch_by_len) & set(greedy_by_len), key=lambda value: int(value)):
        if prefix_len == "0":
            continue
        batch_inst = _int(batch_by_len[prefix_len].get("ir_inst_count"))
        greedy_inst = _int(greedy_by_len[prefix_len].get("ir_inst_count"))
        if greedy_inst < batch_inst:
            return {
                "prefix_len": prefix_len,
                "batch_prefix": batch_by_len[prefix_len].get("pass_prefix", ""),
                "greedy_prefix": greedy_by_len[prefix_len].get("pass_prefix", ""),
                "batch_inst": str(batch_inst),
                "greedy_inst": str(greedy_inst),
            }
    return {}


def _prefix_curve_table(batch_prefix: list[dict], greedy_prefix: list[dict], random_prefix: list[dict]) -> list[str]:
    batch = {row.get("prefix_len", ""): row.get("ir_inst_count", "") for row in batch_prefix}
    greedy = {row.get("prefix_len", ""): row.get("ir_inst_count", "") for row in greedy_prefix}
    random = {row.get("prefix_len", ""): row.get("ir_inst_count", "") for row in random_prefix}
    keys = sorted(set(batch) | set(greedy) | set(random), key=lambda value: int(value or 0))
    return _markdown_table(["prefix_len", "batch inst", "greedy inst", "random inst"], [[key, batch.get(key, ""), greedy.get(key, ""), random.get(key, "")] for key in keys])


def _passes_by_class(candidates: list[dict], correctness_rows: list[dict]) -> dict[str, set[str]]:
    correctness_by_batch = {row.get("batch_id", ""): row for row in correctness_rows}
    result = {
        "any": set(),
        "certified_batch": set(),
        "sampled_batch": set(),
        "rejected_batch": set(),
    }
    for candidate in candidates:
        batch_id = candidate.get("batch_id", "")
        passes = set(_split_sequence(candidate.get("batch_passes", "")))
        result["any"].update(passes)
        correctness_class = correctness_by_batch.get(batch_id, {}).get("correctness_class", "")
        if correctness_class in result:
            result[correctness_class].update(passes)
    return result


def _diagnostic_reason(
    *,
    in_batch: bool,
    in_candidate: bool,
    in_certified: bool,
    in_sampled: bool,
    in_rejected: bool,
    in_sensitive: bool,
) -> str:
    if in_batch:
        return "pass included in batch pipeline"
    if in_sampled:
        return "pass only appears in sampled batch"
    if in_rejected:
        return "pass blocked by rejected batch"
    if in_sensitive:
        return "pass blocked by order-sensitive component"
    if in_certified or in_candidate:
        return "pass available but not selected by batch path"
    if not in_candidate:
        return "pass not active in root state"
    return "insufficient data"


def _baseline_rows(run_dir: Path, baseline_dir: Path) -> list[dict]:
    for path in [run_dir / "baseline_results.csv", baseline_dir / "baseline_results.csv"]:
        rows = _read_csv(path)
        if rows:
            return rows
    return []


def _optimized_pipeline(run_dir: Path, baselines: list[dict]) -> list[str]:
    path = run_dir / "optimized_pipeline.txt"
    if path.exists():
        return _split_sequence(path.read_text(encoding="utf-8", errors="replace"))
    return _sequence_for_method(baselines, "batch_optimizer")


def _greedy_sequence(run_dir: Path, baseline_dir: Path, baselines: list[dict]) -> list[str]:
    path = run_dir / "baselines" / "greedy_single_pass" / "greedy_path.csv"
    if not path.exists():
        path = baseline_dir / "baselines" / "greedy_single_pass" / "greedy_path.csv"
    rows = _read_csv(path)
    if rows:
        return [row.get("selected_pass", "") for row in rows if row.get("selected_pass")]
    return _sequence_for_method(baselines, "greedy_single_pass")


def _random_sequence(run_dir: Path, baseline_dir: Path, baselines: list[dict]) -> list[str]:
    path = run_dir / "baselines" / "random_single_pass" / "random_best_path.csv"
    if not path.exists():
        path = baseline_dir / "baselines" / "random_single_pass" / "random_best_path.csv"
    rows = _read_csv(path)
    if rows:
        return [row.get("selected_pass", "") for row in rows if row.get("selected_pass")]
    return _sequence_for_method(baselines, "random_single_pass_best")


def _sequence_for_method(baselines: list[dict], method: str) -> list[str]:
    for row in baselines:
        if row.get("method") == method:
            return _split_sequence(row.get("pass_sequence", ""))
    return []


def _root_ir(run_dir: Path) -> Path:
    for path in [run_dir / "states" / "S0000" / "input.ll", run_dir / "input.ll"]:
        if path.exists():
            return path
    return run_dir / "states" / "S0000" / "input.ll"


def _opt_path(run_dir: Path) -> str:
    metadata = _read_json(run_dir / "metadata.json")
    return str(metadata.get("tools", {}).get("opt", {}).get("path") or "opt")


def _appears_in_order_sensitive_pair(pass_name: str, pair_rows: list[dict]) -> bool:
    for row in pair_rows:
        relation = row.get("final_relation", "")
        if relation != "final_order_sensitive":
            continue
        if pass_name in {row.get("pass_a", ""), row.get("pass_b", "")}:
            return True
    return False


def _result_error(result) -> str:
    return getattr(result, "stderr", "") or getattr(result, "failure_kind", "") or "opt failed"


def _split_sequence(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").replace("\n", "").replace(";", ",").split(",") if part.strip()]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def _bool(value: bool) -> str:
    return "true" if value else "false"
