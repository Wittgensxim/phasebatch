from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

from .deterministic_io import write_csv, write_json
from .extractors import run_extractor, select_incremental_instruction, select_pair
from .models import (
    ExtractionLevel,
    FrozenDataset,
    RuntimeRecord,
    SelectionDecision,
    TransitionFeature,
)
from .timing import nearest_rank_percentile, paired_incremental_rows


def merge_three_decisions(
    decisions: list[SelectionDecision] | tuple[SelectionDecision, ...],
) -> SelectionDecision:
    if len(decisions) != 3:
        raise ValueError(f"expected three source decisions, got {len(decisions)}")
    if any(decision.status == "unknown" for decision in decisions):
        reasons = sorted(
            {decision.reason for decision in decisions if decision.status == "unknown"}
        )
        return SelectionDecision("unknown", "source_unknown:" + ";".join(reasons))
    statuses = {decision.status for decision in decisions}
    if len(statuses) != 1:
        return SelectionDecision("unknown", "selection_unstable_across_sources")
    status = decisions[0].status
    reasons = sorted({decision.reason for decision in decisions})
    return SelectionDecision(status, ";".join(reasons))


def build_screen_snapshot(
    dataset: FrozenDataset,
    path: Path,
    *,
    progress=None,  # noqa: ANN001
) -> dict:
    """Extract all three sources and persist a timing-free deterministic screen."""

    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_screen_snapshot(payload)
        return payload

    runs = []
    validations: list[dict[str, object]] = []
    for source_repetition in (1, 2, 3):
        run = run_extractor(
            dataset,
            source_repetition,
            ExtractionLevel.INSTRUCTION_ONLY,
        )
        runs.append(run)
        validation: dict[str, object] = {
            "source_repetition": source_repetition,
            "artifact_error_count": len(run.artifact_errors),
            "collision_count": len(run.collisions),
        }
        for level, expected in (
            (ExtractionLevel.FUNC_ONLY, 30),
            (ExtractionLevel.BLOCK_ONLY, 46),
            (ExtractionLevel.EFFECT_ONLY, 47),
        ):
            decisions = [
                select_pair(
                    run.features[(pair.program, pair.action_a_id)],
                    run.features[(pair.program, pair.action_b_id)],
                    level,
                )
                for pair in dataset.pairs
            ]
            actual = sum(decision.status == "selected" for decision in decisions)
            validation[f"{level.value.lower()}_selected"] = actual
            if actual != expected:
                raise ValueError(
                    f"legacy gate mismatch from instruction source: {level.value}:"
                    f"source={source_repetition}:expected={expected}:actual={actual}"
                )
        validations.append(validation)
        if progress is not None:
            progress(source_repetition, run)

    pair_costs = {
        pair.observation_id: median(
            attempt.pair_cost_ms[pair.observation_id] for attempt in dataset.attempts
        )
        for pair in dataset.pairs
    }
    rows: list[dict[str, object]] = []
    for pair in dataset.pairs:
        features_a = [
            run.features[(pair.program, pair.action_a_id)] for run in runs
        ]
        features_b = [
            run.features[(pair.program, pair.action_b_id)] for run in runs
        ]
        source_decisions = [
            select_incremental_instruction(left, right)
            for left, right in zip(features_a, features_b, strict=True)
        ]
        transition_stable = len({_feature_signature(value) for value in features_a}) == 1
        transition_stable = transition_stable and len(
            {_feature_signature(value) for value in features_b}
        ) == 1
        if pair.h_effect_selected:
            merged = SelectionDecision("not_selected", "not_screened_existing_h_effect")
            incremental_selected = False
            cumulative_selected = True
            selection_origin = "H_effect"
        elif not transition_stable:
            merged = SelectionDecision("unknown", "transition_unstable_across_sources")
            incremental_selected = False
            cumulative_selected = False
            selection_origin = "unknown"
        else:
            merged = merge_three_decisions(source_decisions)
            incremental_selected = merged.status == "selected"
            cumulative_selected = incremental_selected
            selection_origin = "H_inst_incremental" if incremental_selected else merged.status
        row: dict[str, object] = {
            "observation_id": pair.observation_id,
            "program": pair.program,
            "action_a_id": pair.action_a_id,
            "action_b_id": pair.action_b_id,
            "action_a_name": pair.action_a_name,
            "action_b_name": pair.action_b_name,
            "action_a_pipeline": pair.action_a_pipeline,
            "action_b_pipeline": pair.action_b_pipeline,
            "dynamic_relation": pair.dynamic_relation,
            "dynamic_cost_median_ms": pair_costs[pair.observation_id],
            "h_func_selected": pair.h_func_selected,
            "h_block_selected": pair.h_block_selected,
            "h_effect_selected": pair.h_effect_selected,
            "incremental_status": merged.status,
            "incremental_selected": incremental_selected,
            "incremental_reason": merged.reason,
            "transition_stable_across_sources": transition_stable,
            "cumulative_selected": cumulative_selected,
            "selection_origin": selection_origin,
        }
        for index, (decision, left, right) in enumerate(
            zip(source_decisions, features_a, features_b, strict=True), start=1
        ):
            row[f"source_{index}_status"] = decision.status
            row[f"source_{index}_reason"] = decision.reason
            row[f"source_{index}_a_instruction_token_count"] = len(
                left.instruction_tokens
            )
            row[f"source_{index}_b_instruction_token_count"] = len(
                right.instruction_tokens
            )
            row[f"source_{index}_intersection_count"] = len(
                left.instruction_tokens & right.instruction_tokens
            )
            row[f"source_{index}_a_wildcards"] = ";".join(left.wildcard_reasons)
            row[f"source_{index}_b_wildcards"] = ";".join(right.wildcard_reasons)
        rows.append(row)

    collisions = [
        {
            "source_repetition": run.source_repetition,
            "fingerprint": collision.fingerprint,
            "function": collision.function,
            "block": collision.block,
            "effect_class": collision.effect_class,
            "canonical_forms": list(collision.canonical_forms),
        }
        for run in runs
        for collision in run.collisions
    ]
    artifact_errors = [dict(row) for run in runs for row in run.artifact_errors]
    payload = {
        "schema_version": "instruction-granularity-screen-v1",
        "pair_count": len(rows),
        "program_count": 49,
        "action_count": 14,
        "transition_count": 686,
        "source_validations": validations,
        "pairs": rows,
        "collisions": collisions,
        "artifact_errors": artifact_errors,
    }
    _validate_screen_snapshot(payload)
    write_json(path, payload)
    return payload


def generate_aggregate_outputs(
    dataset: FrozenDataset,
    snapshot: dict,
    runtime_records: tuple[RuntimeRecord, ...],
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = list(snapshot["pairs"])
    if len(pair_rows) != 1411:
        raise ValueError("screen snapshot must retain all 1411 rows")
    screen_by_id = {row["observation_id"]: row for row in pair_rows}
    if len(screen_by_id) != 1411:
        raise ValueError("duplicate screen pair rows")

    measured = tuple(record for record in runtime_records if record.phase == "measured")
    if len(measured) != 120:
        raise ValueError(f"expected 120 measured level rows, got {len(measured)}")
    paired = paired_incremental_rows(measured)
    instruction_deltas = [
        row.delta_ms for row in paired if row.comparison == "INSTRUCTION-EFFECT"
    ]
    instruction_incremental_extraction_cost = median(instruction_deltas)

    cost_by_id = {
        row["observation_id"]: float(row["dynamic_cost_median_ms"])
        for row in pair_rows
    }
    total_dynamic_cost = sum(cost_by_id.values())
    pair_by_id = dataset.pair_by_id

    def selected_rows(attribute: str) -> list[dict]:
        if attribute == "H_inst":
            return [row for row in pair_rows if row["cumulative_selected"]]
        field = {
            "H_func": "h_func_selected",
            "H_block": "h_block_selected",
            "H_effect": "h_effect_selected",
        }[attribute]
        return [row for row in pair_rows if row[field]]

    coverage_rows: list[dict[str, object]] = []
    coverage_by_level: dict[str, dict[str, object]] = {}
    for heuristic in ("H_func", "H_block", "H_effect", "H_inst"):
        selected = selected_rows(heuristic)
        outcomes = Counter(row["dynamic_relation"] for row in selected)
        selected_count = len(selected)
        commute = outcomes["dynamic_commute"]
        unsafe = outcomes["dynamic_order_sensitive"] + outcomes["failed"]
        screen_unknown = (
            sum(
                row["incremental_status"] == "unknown"
                for row in pair_rows
                if not row["h_effect_selected"]
            )
            if heuristic == "H_inst"
            else 0
        )
        coverage = {
            "heuristic": heuristic,
            "selected_count": selected_count,
            "selected_commute": commute,
            "selected_order_sensitive": outcomes["dynamic_order_sensitive"],
            "selected_failed": outcomes["failed"],
            "screen_unknown_count": screen_unknown,
            "precision": _ratio(commute, selected_count),
            "commute_recall": _ratio(commute, 833),
            "pair_coverage": _ratio(selected_count, 1411),
            "selected_dynamic_cost_ms": sum(
                cost_by_id[row["observation_id"]] for row in selected
            ),
            "cost_weighted_coverage": _ratio(
                sum(cost_by_id[row["observation_id"]] for row in selected),
                total_dynamic_cost,
            ),
            "unsafe_count": unsafe,
        }
        coverage_by_level[heuristic] = coverage
        coverage_rows.append(_format_row(coverage))

    incremental_candidates = [row for row in pair_rows if not row["h_effect_selected"]]
    incremental_selected = [row for row in incremental_candidates if row["incremental_selected"]]
    incremental_outcomes = Counter(
        row["dynamic_relation"] for row in incremental_selected
    )
    incremental_unknown = sum(
        row["incremental_status"] == "unknown" for row in incremental_candidates
    )
    incremental_dynamic_cost = sum(
        cost_by_id[row["observation_id"]] for row in incremental_selected
    )
    incremental_true_cost = sum(
        cost_by_id[row["observation_id"]]
        for row in incremental_selected
        if row["dynamic_relation"] == "dynamic_commute"
    )
    incremental_unsafe_cost = sum(
        cost_by_id[row["observation_id"]]
        for row in incremental_selected
        if row["dynamic_relation"] != "dynamic_commute"
    )
    incremental_summary = {
        "incremental_selected_count": len(incremental_selected),
        "incremental_commute": incremental_outcomes["dynamic_commute"],
        "incremental_order_sensitive": incremental_outcomes[
            "dynamic_order_sensitive"
        ],
        "incremental_failed": incremental_outcomes["failed"],
        "incremental_unknown": incremental_unknown,
        "incremental_precision": _ratio(
            incremental_outcomes["dynamic_commute"], len(incremental_selected)
        ),
        "incremental_commute_recall": _ratio(
            incremental_outcomes["dynamic_commute"], 833
        ),
        "incremental_pair_coverage": _ratio(len(incremental_selected), 1411),
        "incremental_candidate_pool_coverage": _ratio(
            len(incremental_selected), 1364
        ),
        "incremental_dynamic_cost_ms": incremental_dynamic_cost,
        "incremental_true_commute_cost_ms": incremental_true_cost,
        "incremental_unsafe_cost_ms": incremental_unsafe_cost,
        "instruction_incremental_extraction_cost_ms": instruction_incremental_extraction_cost,
        "instruction_marginal_value": incremental_true_cost
        - incremental_unsafe_cost
        - instruction_incremental_extraction_cost,
    }
    h_inst = coverage_by_level["H_inst"]
    cumulative_summary = {
        "cumulative_selected": h_inst["selected_count"],
        "cumulative_commute": h_inst["selected_commute"],
        "cumulative_unsafe": h_inst["unsafe_count"],
        "cumulative_order_sensitive": h_inst["selected_order_sensitive"],
        "cumulative_failed": h_inst["selected_failed"],
        "cumulative_precision": h_inst["precision"],
        "cumulative_coverage_of_833_commute": h_inst["commute_recall"],
        "cumulative_pair_coverage": h_inst["pair_coverage"],
        "cumulative_selected_dynamic_cost_ms": h_inst["selected_dynamic_cost_ms"],
        "cumulative_cost_weighted_coverage": h_inst["cost_weighted_coverage"],
    }

    pair_fields = (
        "observation_id",
        "program",
        "action_a_id",
        "action_b_id",
        "action_a_name",
        "action_b_name",
        "action_a_pipeline",
        "action_b_pipeline",
        "dynamic_relation",
        "dynamic_cost_median_ms",
        "h_func_selected",
        "h_block_selected",
        "h_effect_selected",
        "incremental_status",
        "incremental_selected",
        "incremental_reason",
        "transition_stable_across_sources",
        "source_1_status",
        "source_1_reason",
        "source_1_a_instruction_token_count",
        "source_1_b_instruction_token_count",
        "source_1_intersection_count",
        "source_1_a_wildcards",
        "source_1_b_wildcards",
        "source_2_status",
        "source_2_reason",
        "source_2_a_instruction_token_count",
        "source_2_b_instruction_token_count",
        "source_2_intersection_count",
        "source_2_a_wildcards",
        "source_2_b_wildcards",
        "source_3_status",
        "source_3_reason",
        "source_3_a_instruction_token_count",
        "source_3_b_instruction_token_count",
        "source_3_intersection_count",
        "source_3_a_wildcards",
        "source_3_b_wildcards",
        "cumulative_selected",
        "selection_origin",
    )
    write_csv(
        output_dir / "instruction_screen_pairs.csv",
        pair_fields,
        (_format_row(row) for row in pair_rows),
    )
    write_csv(
        output_dir / "granularity_coverage_summary.csv",
        tuple(coverage_rows[0]),
        coverage_rows,
    )
    write_csv(
        output_dir / "instruction_incremental_summary.csv",
        tuple(incremental_summary),
        [_format_row(incremental_summary)],
    )
    write_csv(
        output_dir / "instruction_cumulative_summary.csv",
        tuple(cumulative_summary),
        [_format_row(cumulative_summary)],
    )

    runtime_summary = _runtime_summary_rows(measured)
    write_csv(
        output_dir / "extraction_runtime_summary.csv",
        tuple(runtime_summary[0]),
        runtime_summary,
    )
    incremental_cost_rows = _incremental_cost_rows(paired)
    write_csv(
        output_dir / "extraction_incremental_cost.csv",
        tuple(incremental_cost_rows[0]),
        incremental_cost_rows,
    )

    by_program = _group_screen_rows(pair_rows, "program")
    write_csv(
        output_dir / "instruction_by_program.csv",
        tuple(by_program[0]),
        by_program,
    )
    by_pass_pair = _by_pass_pair_rows(pair_rows)
    write_csv(
        output_dir / "instruction_by_pass_pair.csv",
        tuple(by_pass_pair[0]),
        by_pass_pair,
    )
    collision_rows = _collision_rows(snapshot)
    collision_fields = (
        "source_repetition",
        "fingerprint",
        "function",
        "block",
        "effect_class",
        "canonical_variant_count",
        "canonical_forms_json",
    )
    write_csv(
        output_dir / "fingerprint_collision_diagnostics.csv",
        collision_fields,
        collision_rows,
    )
    failure_rows = _failure_rows(snapshot, pair_rows)
    failure_fields = (
        "scope",
        "source_repetition",
        "program",
        "observation_id",
        "action_id",
        "path",
        "status",
        "reason",
    )
    write_csv(output_dir / "failure_ledger.csv", failure_fields, failure_rows)

    metrics = {
        "schema_version": "instruction-granularity-aggregate-v1",
        "hard_counts": {
            "pairs": 1411,
            "programs": 49,
            "actions": 14,
            "commute": 833,
            "order_sensitive": 569,
            "failed": 9,
            "transitions": 686,
        },
        "coverage": coverage_by_level,
        "incremental": incremental_summary,
        "cumulative": cumulative_summary,
        "runtime_summary": runtime_summary,
        "incremental_cost": incremental_cost_rows,
        "failure_ledger_count": len(failure_rows),
        "collision_count": len(collision_rows),
        "total_dynamic_cost_ms": total_dynamic_cost,
    }
    write_json(output_dir.parent / "raw" / "aggregate_metrics.json", metrics)
    return metrics


def _validate_screen_snapshot(payload: dict) -> None:
    if payload.get("schema_version") != "instruction-granularity-screen-v1":
        raise ValueError("invalid instruction screen schema")
    if (
        payload.get("pair_count"),
        payload.get("program_count"),
        payload.get("action_count"),
        payload.get("transition_count"),
    ) != (1411, 49, 14, 686):
        raise ValueError("instruction screen hard counts failed")
    rows = payload.get("pairs")
    if not isinstance(rows, list) or len(rows) != 1411:
        raise ValueError("instruction screen pair rows missing")
    if len({row["observation_id"] for row in rows}) != 1411:
        raise ValueError("instruction screen observation IDs not unique")
    effect = [row for row in rows if row["h_effect_selected"]]
    if len(effect) != 47:
        raise ValueError("H_effect preservation gate failed")


def _feature_signature(feature: TransitionFeature) -> tuple:
    return (
        tuple(sorted(feature.functions)),
        tuple(sorted(feature.blocks)),
        tuple(sorted(feature.effect_tokens)),
        tuple(sorted(feature.instruction_tokens)),
        feature.wildcard_reasons,
        tuple(sorted(feature.observed_opcodes)),
    )


def _runtime_summary_rows(records: tuple[RuntimeRecord, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    components = (
        "artifact_read_ms",
        "parse_ms",
        "feature_build_ms",
        "pair_selection_ms",
        "total_extraction_ms",
    )
    for level in ExtractionLevel:
        values = sorted(
            (record for record in records if record.level == level),
            key=lambda record: record.measured_repetition,
        )
        if len(values) != 30:
            raise ValueError(f"expected 30 measured records for {level.value}")
        row: dict[str, object] = {"level": level.value, "measured_repetitions": 30}
        for component in components:
            samples = [getattr(record, component) for record in values]
            prefix = component.removesuffix("_ms")
            row[f"first_{prefix}_ms"] = _fmt(samples[0])
            row[f"median_{prefix}_ms"] = _fmt(median(samples))
            row[f"p90_{prefix}_ms"] = _fmt(
                nearest_rank_percentile(samples, 0.9)
            )
            row[f"min_{prefix}_ms"] = _fmt(min(samples))
            row[f"max_{prefix}_ms"] = _fmt(max(samples))
        totals = [record.total_extraction_ms for record in values]
        row["median_per_transition_ms"] = _fmt(median(totals) / 686)
        row["p90_per_transition_ms"] = _fmt(
            nearest_rank_percentile(totals, 0.9) / 686
        )
        row["median_per_pair_ms"] = _fmt(median(totals) / 1411)
        row["p90_per_pair_ms"] = _fmt(
            nearest_rank_percentile(totals, 0.9) / 1411
        )
        rows.append(row)
    return rows


def _incremental_cost_rows(paired) -> list[dict[str, object]]:  # noqa: ANN001
    rows: list[dict[str, object]] = []
    for comparison in ("BLOCK-FUNC", "EFFECT-BLOCK", "INSTRUCTION-EFFECT"):
        selected = [row for row in paired if row.comparison == comparison]
        values = [row.delta_ms for row in selected]
        rows.append(
            {
                "comparison": comparison,
                "paired_repetitions": len(values),
                "first_delta_ms": _fmt(values[0]),
                "median_delta_ms": _fmt(median(values)),
                "p90_delta_ms": _fmt(nearest_rank_percentile(values, 0.9)),
                "min_delta_ms": _fmt(min(values)),
                "max_delta_ms": _fmt(max(values)),
                "paired_values_json": json.dumps(
                    [round(value, 6) for value in values], separators=(",", ":")
                ),
            }
        )
    return rows


def _group_screen_rows(rows: list[dict], field: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result: list[dict[str, object]] = []
    for key in sorted(grouped):
        values = grouped[key]
        incremental = [row for row in values if row["incremental_selected"]]
        cumulative = [row for row in values if row["cumulative_selected"]]
        result.append(
            {
                field: key,
                "pair_rows": len(values),
                "commute_rows": sum(
                    row["dynamic_relation"] == "dynamic_commute" for row in values
                ),
                "order_sensitive_rows": sum(
                    row["dynamic_relation"] == "dynamic_order_sensitive"
                    for row in values
                ),
                "failed_rows": sum(
                    row["dynamic_relation"] == "failed" for row in values
                ),
                "h_effect_selected": sum(row["h_effect_selected"] for row in values),
                "incremental_selected": len(incremental),
                "incremental_commute": sum(
                    row["dynamic_relation"] == "dynamic_commute"
                    for row in incremental
                ),
                "incremental_unsafe": sum(
                    row["dynamic_relation"] != "dynamic_commute"
                    for row in incremental
                ),
                "incremental_unknown": sum(
                    row["incremental_status"] == "unknown" for row in values
                ),
                "cumulative_selected": len(cumulative),
                "cumulative_commute": sum(
                    row["dynamic_relation"] == "dynamic_commute" for row in cumulative
                ),
                "cumulative_unsafe": sum(
                    row["dynamic_relation"] != "dynamic_commute" for row in cumulative
                ),
            }
        )
    return result


def _by_pass_pair_rows(rows: list[dict]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["action_a_id"],
            row["action_b_id"],
            row["action_a_name"],
            row["action_b_name"],
        )
        grouped[key].append(row)
    result: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda value: (value[2], value[3], value[0], value[1])):
        values = grouped[key]
        incremental = [row for row in values if row["incremental_selected"]]
        cumulative = [row for row in values if row["cumulative_selected"]]
        result.append(
            {
                "action_a_id": key[0],
                "action_b_id": key[1],
                "action_a_name": key[2],
                "action_b_name": key[3],
                "pass_pair": f"{key[2]}__{key[3]}",
                "pair_rows": len(values),
                "h_effect_selected": sum(row["h_effect_selected"] for row in values),
                "incremental_selected": len(incremental),
                "incremental_commute": sum(
                    row["dynamic_relation"] == "dynamic_commute"
                    for row in incremental
                ),
                "incremental_unsafe": sum(
                    row["dynamic_relation"] != "dynamic_commute"
                    for row in incremental
                ),
                "incremental_unknown": sum(
                    row["incremental_status"] == "unknown" for row in values
                ),
                "cumulative_selected": len(cumulative),
                "cumulative_commute": sum(
                    row["dynamic_relation"] == "dynamic_commute" for row in cumulative
                ),
                "cumulative_unsafe": sum(
                    row["dynamic_relation"] != "dynamic_commute" for row in cumulative
                ),
            }
        )
    return result


def _collision_rows(snapshot: dict) -> list[dict[str, object]]:
    return [
        {
            "source_repetition": row["source_repetition"],
            "fingerprint": row["fingerprint"],
            "function": row["function"],
            "block": row["block"],
            "effect_class": row["effect_class"],
            "canonical_variant_count": len(row["canonical_forms"]),
            "canonical_forms_json": json.dumps(
                row["canonical_forms"], ensure_ascii=False, separators=(",", ":")
            ),
        }
        for row in snapshot.get("collisions", [])
    ]


def _failure_rows(snapshot: dict, pair_rows: list[dict]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in snapshot.get("artifact_errors", []):
        result.append(
            {
                "scope": row.get("scope", "artifact"),
                "source_repetition": row.get("source_repetition", ""),
                "program": row.get("program", ""),
                "observation_id": "",
                "action_id": row.get("action_id", ""),
                "path": row.get("path", ""),
                "status": "unknown",
                "reason": row.get("reason", "artifact_validation_failed"),
            }
        )
    for row in pair_rows:
        if row["incremental_status"] == "unknown" and not row["h_effect_selected"]:
            result.append(
                {
                    "scope": "instruction_screen",
                    "source_repetition": "1;2;3",
                    "program": row["program"],
                    "observation_id": row["observation_id"],
                    "action_id": f"{row['action_a_id']};{row['action_b_id']}",
                    "path": "",
                    "status": "unknown",
                    "reason": row["incremental_reason"],
                }
            )
        if row["dynamic_relation"] == "failed":
            result.append(
                {
                    "scope": "dynamic_label",
                    "source_repetition": "1;2;3",
                    "program": row["program"],
                    "observation_id": row["observation_id"],
                    "action_id": f"{row['action_a_id']};{row['action_b_id']}",
                    "path": "",
                    "status": "failed",
                    "reason": "frozen_dynamic_failed",
                }
            )
    return sorted(
        result,
        key=lambda row: (
            str(row["scope"]),
            str(row["program"]),
            str(row["observation_id"]),
            str(row["action_id"]),
        ),
    )


def _format_row(row: dict) -> dict:
    return {
        key: _fmt(value) if isinstance(value, float) else value
        for key, value in row.items()
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
