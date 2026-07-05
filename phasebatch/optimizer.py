from __future__ import annotations

import csv
import hashlib
import math
import shutil
import time
from pathlib import Path

from .batch_correctness import classify_batch_correctness
from .batch_objective import count_ir_instructions
from .batcher import build_batch_family, validate_batch_candidates
from .cli import analyze_state
from .coverage import build_coverage_report
from .normalizer import canonical_hash
from .pass_config import PassRegistry, load_pass_registry, resolve_pipeline_sequence
from .profiler import validate_passes
from .runner import prepare_input_ir, run_opt
from .schema import STATE_FIELDS
from .tools import collect_toolchain, write_metadata


STATE_DAG_FIELDS = [
    "program",
    "source_state_id",
    "target_state_id",
    "source_hash",
    "target_hash",
    "transition_kind",
    "batch_id",
    "batch_passes",
    "canonical_order",
    "validation_status",
    "correctness_class",
    "is_duplicate",
    "duplicate_of",
]

OPT_BATCH_STATE_TRANSITION_FIELDS = [
    "program",
    "parent_state_id",
    "child_state_id",
    "batch_id",
    "batch_passes",
    "batch_size",
    "validation_status",
    "correctness_class",
    "parent_hash",
    "child_hash",
    "is_duplicate",
    "duplicate_of",
]

LEAF_STATE_FIELDS = [
    "program",
    "state_id",
    "depth",
    "state_hash",
    "objective_kind",
    "objective_value",
    "is_leaf",
    "leaf_reason",
    "path_length",
    "pass_invocations",
    "selected_as_final",
]

CHOSEN_PATH_FIELDS = [
    "step",
    "round",
    "parent_state_id",
    "parent_depth",
    "parent_state_hash",
    "batch_id",
    "batch_passes",
    "batch_size",
    "canonical_order",
    "validation_status",
    "correctness_class",
    "can_hard_fold",
    "can_execute",
    "child_state_id",
    "child_depth",
    "child_state_hash",
    "is_duplicate_transition",
    "duplicate_of",
    "parent_ir_path",
    "child_ir_path",
    "parent_active_passes",
    "child_active_passes",
    "parent_tested_pairs",
    "child_tested_pairs",
    "parent_commute_pairs",
    "child_commute_pairs",
    "parent_order_sensitive_pairs",
    "child_order_sensitive_pairs",
    "parent_unknown_pairs",
    "child_unknown_pairs",
    "ir_inst_before",
    "ir_inst_after",
    "ir_inst_delta",
    "ir_inst_reduction_pct",
    "selection_reason",
]

CHOSEN_PATH_SUMMARY_FIELDS = [
    "program",
    "selected_final_state",
    "path_steps",
    "total_pass_invocations",
    "unique_pass_types",
    "root_ir_inst_count",
    "final_ir_inst_count",
    "total_ir_inst_delta",
    "total_ir_inst_reduction_pct",
    "all_batches_certified",
    "any_sampled_batch",
    "any_rejected_batch",
    "any_unvalidated_batch",
    "replay_verified",
]

FRONTIER_SCORE_FIELDS = [
    "round",
    "state_id",
    "parent_state_id",
    "last_batch_id",
    "depth",
    "objective_value",
    "active_passes",
    "batch_candidates",
    "last_batch_size",
    "validation_status",
    "correctness_class",
    "enable_count_from_parent",
    "effect_changed_count_from_parent",
    "root_inst_count",
    "parent_inst_count",
    "child_inst_count",
    "parent_gain",
    "objective_score",
    "future_potential_score",
    "evidence_quality_score",
    "novelty_score",
    "cost_score",
    "risk_penalty",
    "final_state_score",
    "pareto_kept",
    "policy",
    "rank",
    "selection_bucket",
    "selected_for_frontier",
    "selection_reason",
]

BATCH_CANDIDATE_SCORE_FIELDS = [
    "program",
    "state_id",
    "state_hash",
    "batch_id",
    "batch_passes",
    "batch_size",
    "correctness_class",
    "validation_status",
    "coverage_score",
    "batch_size_score",
    "reduction_score",
    "evidence_score",
    "diversity_score",
    "risk_penalty",
    "final_batch_score",
    "selected_for_execution",
    "selection_reason",
]

OPTIMIZER_EVENT_FIELDS = [
    "event_id",
    "round",
    "state_id",
    "event_type",
    "message",
]

OPTIMIZER_TIMING_FIELDS = [
    "program",
    "optimizer_total_time_ms",
    "analysis_time_ms",
    "profiling_time_ms",
    "pair_testing_time_ms",
    "batch_validation_time_ms",
    "batch_apply_time_ms",
    "total_opt_invocations",
    "batch_apply_opt_invocations",
]


def optimize_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    mode: str,
    objective: str,
    max_rounds: int,
    beam_width: int = 8,
    max_batches_per_state: int,
    validate_batches: bool,
    allow_sampled_batches: bool,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_states: int = 2000,
    batch_frontier_policy: str | None = None,
    batch_selection_policy: str | None = None,
    frontier_selection_policy: str | None = None,
    selection_seed: int = 0,
    exact_fail_on_incomplete: bool = True,
    run_baselines: bool = False,
    verify_final_pipeline: bool = True,
) -> dict:
    optimizer_start = time.perf_counter()
    if objective != "ir-inst-count":
        raise ValueError(f"unsupported objective: {objective}")
    if mode not in {"budgeted", "exact", "auto"}:
        raise NotImplementedError(f"optimize-batches mode '{mode}' is not implemented yet")
    batch_selection_policy, frontier_selection_policy = _resolve_budgeted_policies(
        batch_frontier_policy,
        batch_selection_policy,
        frontier_selection_policy,
    )
    for name, policy in (("batch selection", batch_selection_policy), ("frontier selection", frontier_selection_policy)):
        if policy not in {"score", "largest-batch", "certified-first", "objective", "diverse"}:
            raise ValueError(f"unknown {name} policy: {policy}")
    if mode == "exact" and allow_sampled_batches:
        raise ValueError("Exact mode does not allow sampled batches.")

    context = _prepare_run(
        input_path=Path(input_path),
        out_dir=Path(out_dir),
        passes_path=Path(passes_path),
        mode=mode,
        objective=objective,
        max_rounds=max_rounds,
        beam_width=beam_width,
        max_batches_per_state=max_batches_per_state,
        validate_batches=validate_batches,
        allow_sampled_batches=allow_sampled_batches,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        max_states=max_states,
        batch_frontier_policy=batch_frontier_policy or "",
        batch_selection_policy=batch_selection_policy,
        frontier_selection_policy=frontier_selection_policy,
        selection_seed=selection_seed,
        exact_fail_on_incomplete=exact_fail_on_incomplete,
        verify_final_pipeline=verify_final_pipeline,
    )
    context["timing"] = {
        "optimizer_start": optimizer_start,
        "batch_apply_time_ms": 0.0,
        "batch_apply_opt_invocations": 0,
    }
    context["requested_mode"] = mode
    context["auto_reason"] = ""
    if mode == "auto":
        selected_mode, auto_reason = _choose_auto_mode(context)
        if selected_mode == "exact" and allow_sampled_batches:
            selected_mode = "budgeted"
            auto_reason = "sampled batches are allowed, so exact proof mode is not applicable"
        context["mode"] = selected_mode
        context["auto_reason"] = auto_reason
    if context["mode"] == "budgeted":
        result = _run_budgeted(context)
    else:
        result = _run_exact(context)
    if run_baselines:
        from .baselines import compare_baselines

        baseline_result = compare_baselines(
            Path(result["out_dir"]),
            Path(passes_path),
            objective=objective,
            max_rounds=max_rounds,
            random_trials=20,
            seed=0,
            timeout=timeout,
            jobs=jobs,
        )
        result.update(
            {
                "baseline_results_csv": baseline_result["baseline_results_csv"],
                "random_baseline_trials_csv": baseline_result["random_baseline_trials_csv"],
                "baselines_dir": baseline_result["baselines_dir"],
            }
        )
    from .pipeline_replay import replay_optimized_pipeline, update_replay_status_artifacts

    if verify_final_pipeline:
        replay_result = replay_optimized_pipeline(Path(result["out_dir"]), timeout=timeout)
        replay_verified = "true" if replay_result.get("hashes_match") == "true" else "false"
        result["pipeline_replay_csv"] = replay_result.get("pipeline_replay_csv", "")
        result["replay_status"] = replay_result.get("replay_status", "")
        result["replay_hashes_match"] = replay_result.get("hashes_match", "")
        result["replayed_final_ll"] = replay_result.get("replay_output_path", "")
        update_replay_status_artifacts(Path(result["out_dir"]), replay_result, replay_verified)
    else:
        result["replay_status"] = "not_run"
        result["replay_hashes_match"] = "not_run"
        update_replay_status_artifacts(Path(result["out_dir"]), None, "not_run")

    from .final_summary import generate_final_summary

    final_summary = generate_final_summary(Path(result["out_dir"]))
    result["final_summary"] = str(final_summary)
    result["final_summary_index"] = str(Path(result["out_dir"]) / "final_summary_index.csv")
    timing_path = _write_optimizer_timing(Path(result["out_dir"]), context, elapsed_ms=(time.perf_counter() - optimizer_start) * 1000)
    result["optimizer_timing_csv"] = str(timing_path)
    return result


def _resolve_budgeted_policies(
    legacy_policy: str | None,
    batch_selection_policy: str | None,
    frontier_selection_policy: str | None,
) -> tuple[str, str]:
    if legacy_policy:
        if batch_selection_policy is None:
            batch_selection_policy = legacy_policy
        if frontier_selection_policy is None:
            frontier_selection_policy = legacy_policy
    return batch_selection_policy or "score", frontier_selection_policy or "score"


def _prepare_run(**kwargs) -> dict:
    input_path: Path = kwargs["input_path"]
    out_dir: Path = kwargs["out_dir"]
    passes_path: Path = kwargs["passes_path"]
    out_dir.mkdir(parents=True, exist_ok=True)
    states_dir = out_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    program = out_dir.name

    pass_registry = load_pass_registry(passes_path)
    configured_passes = pass_registry.names()
    metadata = collect_toolchain()
    metadata.update(
        {
            "input": str(input_path),
            "out_dir": str(out_dir),
            "pass_config": str(passes_path),
            "mode": kwargs["mode"],
            "objective": kwargs["objective"],
            "max_rounds": kwargs["max_rounds"],
            "beam_width": kwargs["beam_width"],
            "max_batches_per_state": kwargs["max_batches_per_state"],
            "max_states": kwargs["max_states"],
            "batch_frontier_policy": kwargs["batch_frontier_policy"],
            "batch_selection_policy": kwargs["batch_selection_policy"],
            "frontier_selection_policy": kwargs["frontier_selection_policy"],
            "selection_seed": kwargs["selection_seed"],
            "validate_batches": kwargs["validate_batches"],
            "allow_sampled_batches": kwargs["allow_sampled_batches"],
            "exact_fail_on_incomplete": kwargs["exact_fail_on_incomplete"],
            "verify_final_pipeline": kwargs["verify_final_pipeline"],
            "jobs": kwargs["jobs"],
            "timeout": kwargs["timeout"],
            "max_pairs": kwargs["max_pairs"],
            "optimizer_version": "exact-1",
        }
    )
    write_metadata(out_dir, metadata)
    tools = _tool_paths(metadata)
    tools["_pass_registry"] = pass_registry

    prepared_ir = prepare_input_ir(input_path, out_dir, tools, kwargs["timeout"])
    valid_passes, invalid_rows = validate_passes(
        prepared_ir,
        configured_passes,
        tools,
        out_dir,
        kwargs["timeout"],
        pass_registry=pass_registry,
    )

    root_dir = states_dir / "S0000"
    root_dir.mkdir(parents=True, exist_ok=True)
    root_ir = root_dir / "input.ll"
    shutil.copyfile(prepared_ir, root_ir)
    root_hash = canonical_hash(root_ir)
    _analyze(
        root_ir,
        root_dir,
        tools,
        valid_passes=valid_passes,
        invalid_rows=invalid_rows,
        configured_pass_count=len(configured_passes),
        jobs=kwargs["jobs"],
        timeout=kwargs["timeout"],
        max_pairs=kwargs["max_pairs"],
        program=program,
        state_id="S0000",
        depth=0,
        parent_state_id="",
        transition_pass="",
    )
    root_row = _state_row_from_summary(
        root_dir,
        program=program,
        state_id="S0000",
        state_hash=root_hash,
        depth=0,
        parent_state_id="",
        transition_pass="",
        ir_path=root_ir,
        is_duplicate=False,
        duplicate_of="",
    )
    return {
        **kwargs,
        "program": program,
        "configured_passes": configured_passes,
        "pass_registry": pass_registry,
        "tools": tools,
        "valid_passes": valid_passes,
        "invalid_rows": invalid_rows,
        "configured_pass_count": len(configured_passes),
        "states_dir": states_dir,
        "root_dir": root_dir,
        "root_ir": root_ir,
        "root_hash": root_hash,
        "root_row": root_row,
    }


def _choose_auto_mode(context: dict) -> tuple[str, str]:
    if not context["validate_batches"]:
        return "budgeted", "batch validation is disabled, so exact feasibility is uncertain"

    root_dir: Path = context["root_dir"]
    batch_info = _build_validate_classify(root_dir, context, allow_sampled_batches=False)
    summary = _first_row(root_dir / "batch_summary.csv")
    candidates = _int(summary.get("batch_candidates"))
    if candidates == 0:
        candidates = _int(batch_info.get("batch_candidates"))
    unresolved = _int(summary.get("unresolved_components")) > 0
    truncated = _is_true(summary.get("truncated")) or bool(batch_info.get("truncated"))
    components = _read_csv(root_dir / "batch_components.csv")
    unresolved = unresolved or any((not _is_true(row.get("is_exact"))) or bool(row.get("unresolved_reason")) for row in components)
    correctness = _read_csv(root_dir / "batch_correctness.csv")
    has_uncertified = any(row.get("correctness_class") != "certified_batch" for row in correctness)
    estimated_total_states = 1 + candidates

    if context["allow_sampled_batches"]:
        return "budgeted", "sampled batches are allowed, so exact proof mode is not applicable"
    if truncated:
        return "budgeted", "root batch candidates were truncated"
    if unresolved:
        return "budgeted", "root has unresolved conflict components"
    if has_uncertified:
        return "budgeted", "root has candidates without all-permutation certificates"
    if candidates > context["max_batches_per_state"]:
        return "budgeted", "root batch candidate count exceeds max_batches_per_state"
    if estimated_total_states > context["max_states"]:
        return "budgeted", "estimated root expansion exceeds max_states"
    return "exact", "root feasibility check found certified, untruncated batches within configured bounds"


def _run_budgeted(context: dict) -> dict:
    program = context["program"]
    tools = context["tools"]
    timeout = context["timeout"]
    max_rounds = context["max_rounds"]
    max_states = context["max_states"]
    beam_width = context["beam_width"]
    max_batches_per_state = context["max_batches_per_state"]
    batch_policy = context["batch_selection_policy"]
    frontier_policy = context["frontier_selection_policy"]

    state_rows: list[dict] = [context["root_row"]]
    state_rows_by_id: dict[str, dict] = {"S0000": context["root_row"]}
    dag_rows: list[dict] = []
    transition_rows: list[dict] = []
    parent_by_child: dict[str, dict] = {}
    objective_by_state: dict[str, int] = {"S0000": count_ir_instructions(context["root_ir"])}
    path_info_by_state: dict[str, dict] = {"S0000": _root_path_info()}
    state_input_by_id: dict[str, Path] = {"S0000": context["root_ir"]}
    hash_to_state_id: dict[str, str] = {context["root_hash"]: "S0000"}
    canonical_rows_by_id: dict[str, dict] = {"S0000": context["root_row"]}
    child_info_by_state: dict[str, dict] = {}
    leaf_reasons: dict[str, str] = {}
    frontier_score_rows: list[dict] = []
    event_rows: list[dict] = []
    frontier = ["S0000"]
    next_state_number = 1
    incumbent_id = "S0000"
    budget_exhausted = False
    stop_reason = "max_rounds_reached"
    expanded_states: set[str] = set()
    any_executable = False

    for round_index in range(max_rounds):
        if not frontier:
            stop_reason = "frontier_empty"
            break
        all_children: list[str] = []
        for parent_id in frontier:
            if _unique_state_count(state_rows) >= max_states:
                budget_exhausted = True
                stop_reason = "max_states_reached"
                _event(event_rows, round_index, parent_id, "budget_exhausted", "maximum unique state budget reached")
                break

            parent_row = state_rows_by_id[parent_id]
            parent_dir = Path(parent_row["state_dir"])
            parent_input = state_input_by_id[parent_id]
            expanded_states.add(parent_id)
            _event(event_rows, round_index, parent_id, "build_batches", "building and classifying batch candidates")
            _build_validate_classify(parent_dir, context, allow_sampled_batches=context["allow_sampled_batches"])
            if context["validate_batches"]:
                _event(event_rows, round_index, parent_id, "validate_batches", "batch validation completed")

            active_passes = _int(_first_row(parent_dir / "per_state_summary.csv").get("active_passes"))
            if active_passes == 0:
                leaf_reasons[parent_id] = "no_active_passes"
                _event(event_rows, round_index, parent_id, "no_active_passes", "state has no active passes")
                continue

            candidates = _read_csv(parent_dir / "batch_candidates.csv")
            correctness_by_batch = {
                row.get("batch_id", ""): row
                for row in _read_csv(parent_dir / "batch_correctness.csv")
                if row.get("batch_id")
            }
            executable = [
                (candidate, correctness_by_batch.get(candidate.get("batch_id", ""), {}))
                for candidate in candidates
                if correctness_by_batch.get(candidate.get("batch_id", ""), {}).get("can_execute") == "true"
            ]
            batch_score_rows = _score_batch_candidates(parent_dir, candidates, correctness_by_batch, context)
            if not executable:
                _mark_selected_batch_scores(parent_dir, batch_score_rows, set(), {})
                leaf_reasons[parent_id] = "no_executable_batches"
                _event(event_rows, round_index, parent_id, "no_executable_batches", "no safe executable batches for this state")
                continue
            any_executable = True

            selected_batches = _select_budgeted_batches(
                executable,
                policy=batch_policy,
                limit=max_batches_per_state,
                score_rows=batch_score_rows,
                state_id=parent_id,
                selection_seed=context["selection_seed"],
            )
            selected_keys = {(candidate.get("batch_id", ""), correctness.get("correctness_class", "")) for candidate, correctness in selected_batches}
            selected_reasons = {
                candidate.get("batch_id", ""): f"selected_by_{batch_policy}"
                for candidate, _correctness in selected_batches
            }
            _mark_selected_batch_scores(parent_dir, batch_score_rows, selected_keys, selected_reasons)

            for candidate, correctness in selected_batches:
                if _unique_state_count(state_rows) >= max_states:
                    budget_exhausted = True
                    stop_reason = "max_states_reached"
                    leaf_reasons[parent_id] = "state_cap_reached"
                    _event(event_rows, round_index, parent_id, "budget_exhausted", "maximum unique state budget reached")
                    break

                order = _split_order(candidate.get("canonical_order") or candidate.get("batch_passes"))
                if not order:
                    continue
                batch_id = candidate.get("batch_id", f"B{next_state_number:04d}")
                child_artifact = _successor_artifact(parent_dir, f"R{round_index:04d}_{batch_id}")
                _event(event_rows, round_index, parent_id, "apply_batch", f"applying {batch_id}")
                result = run_opt(tools["opt"], parent_input, resolve_pipeline_sequence(order, context["pass_registry"]), child_artifact, timeout)
                _record_batch_apply(context, result)
                if not result.success or not child_artifact.exists():
                    _event(event_rows, round_index, parent_id, "error", f"batch {batch_id} failed to produce child IR")
                    continue

                child_hash = canonical_hash(child_artifact)
                duplicate_of = hash_to_state_id.get(child_hash, "")
                is_duplicate = bool(duplicate_of)
                child_id = f"S{next_state_number:04d}"
                next_state_number += 1
                child_dir = context["states_dir"] / child_id
                child_input = _materialize_state_input(child_dir, child_artifact)
                canonical_order = ";".join(order)

                if is_duplicate:
                    child_row = _duplicate_state_row(
                        canonical_rows_by_id[duplicate_of],
                        state_id=child_id,
                        depth=round_index + 1,
                        parent_state_id=parent_id,
                        transition_pass=canonical_order,
                        ir_path=child_input,
                        duplicate_of=duplicate_of,
                    )
                    _event(event_rows, round_index, child_id, "duplicate_state", f"duplicates {duplicate_of}")
                else:
                    _event(event_rows, round_index, child_id, "analyze_state", "analyzing new child state")
                    _analyze_new_state(context, child_input, child_dir, child_id, round_index + 1, parent_id, canonical_order)
                    _build_validate_classify(child_dir, context, allow_sampled_batches=context["allow_sampled_batches"])
                    _write_unselected_batch_scores(child_dir, context)
                    child_row = _state_row_from_summary(
                        child_dir,
                        program=program,
                        state_id=child_id,
                        state_hash=child_hash,
                        depth=round_index + 1,
                        parent_state_id=parent_id,
                        transition_pass=canonical_order,
                        ir_path=child_input,
                        is_duplicate=False,
                        duplicate_of="",
                    )
                    hash_to_state_id[child_hash] = child_id
                    canonical_rows_by_id[child_id] = child_row
                    all_children.append(child_id)

                state_rows.append(child_row)
                state_rows_by_id[child_id] = child_row
                state_input_by_id[child_id] = child_input
                objective_by_state[child_id] = count_ir_instructions(child_input)
                path_info_by_state[child_id] = _extend_path_info(path_info_by_state[parent_id], candidate, correctness, order)
                transition = _transition_row(program, parent_row, child_row, candidate, correctness, child_hash, is_duplicate, duplicate_of)
                transition_rows.append(transition)
                dag_rows.append(_dag_row(program, parent_row, child_row, candidate, correctness, child_hash, is_duplicate, duplicate_of))
                parent_by_child[child_id] = _edge_for_path(
                    transition,
                    canonical_order,
                    objective_by_state[parent_id],
                    objective_by_state[child_id],
                )
                enable_effect_counts = _enable_effect_counts(parent_dir, child_dir, context["valid_passes"])
                child_info_by_state[child_id] = {
                    "parent_state_id": parent_id,
                    "last_batch_id": batch_id,
                    "last_batch_size": candidate.get("batch_size", ""),
                    "batch_passes": candidate.get("batch_passes", ""),
                    "component_choices": candidate.get("component_choices", ""),
                    "active_pass_signature": _active_pass_signature(child_dir),
                    "validation_status": correctness.get("validation_status", ""),
                    "correctness_class": correctness.get("correctness_class", ""),
                    **enable_effect_counts,
                }
                if _is_better_state(child_id, incumbent_id, objective_by_state, path_info_by_state):
                    incumbent_id = child_id
                    _event(event_rows, round_index, child_id, "update_incumbent", f"objective={objective_by_state[child_id]}")

            if budget_exhausted:
                break

        selected_frontier, score_rows = _select_budgeted_frontier(
            all_children,
            round_index=round_index,
            policy=frontier_policy,
            beam_width=beam_width,
            state_rows_by_id=state_rows_by_id,
            objective_by_state=objective_by_state,
            child_info_by_state=child_info_by_state,
            context=context,
        )
        frontier_score_rows.extend(score_rows)
        _event(event_rows, round_index, "", "select_frontier", f"selected {len(selected_frontier)} of {len(all_children)} child states")
        frontier = selected_frontier
        if budget_exhausted:
            break
        if not frontier:
            stop_reason = "frontier_empty" if any_executable else "no_executable_batches"
            break

    if budget_exhausted:
        stop_reason = "max_states_reached"
    elif not any_executable:
        stop_reason = "no_executable_batches"
    elif frontier:
        stop_reason = "max_rounds_reached"

    for row in state_rows:
        state_id = row["state_id"]
        if state_id not in leaf_reasons and state_id not in expanded_states:
            leaf_reasons[state_id] = "max_rounds_reached" if _int(row.get("depth")) >= max_rounds else "beam_pruned"

    return _finish_run(
        context,
        state_rows=state_rows,
        dag_rows=dag_rows,
        transition_rows=transition_rows,
        parent_by_child=parent_by_child,
        objective_by_state=objective_by_state,
        path_info_by_state=path_info_by_state,
        state_input_by_id=state_input_by_id,
        leaf_reasons=leaf_reasons,
        exact_status="not_applicable",
        exact_reasons=[],
        duplicate_transitions=sum(1 for row in dag_rows if row.get("is_duplicate") == "true"),
        frontier_score_rows=frontier_score_rows,
        event_rows=event_rows,
        budget_exhausted=budget_exhausted,
        stop_reason=stop_reason,
    )


def _run_exact(context: dict) -> dict:
    out_dir: Path = context["out_dir"]
    program = context["program"]
    tools = context["tools"]
    timeout = context["timeout"]
    max_rounds = context["max_rounds"]
    max_states = context["max_states"]
    exact_fail_on_incomplete = context["exact_fail_on_incomplete"]

    state_rows: list[dict] = [context["root_row"]]
    state_rows_by_id: dict[str, dict] = {"S0000": context["root_row"]}
    dag_rows: list[dict] = []
    transition_rows: list[dict] = []
    parent_by_child: dict[str, dict] = {}
    objective_by_state: dict[str, int] = {"S0000": count_ir_instructions(context["root_ir"])}
    path_info_by_state: dict[str, dict] = {"S0000": _root_path_info()}
    state_input_by_id: dict[str, Path] = {"S0000": context["root_ir"]}
    hash_to_state_id: dict[str, str] = {context["root_hash"]: "S0000"}
    leaf_reasons: dict[str, str] = {}
    exact_reasons: list[str] = []
    frontier = ["S0000"]
    next_state_number = 1
    stop_expansion = False

    for round_index in range(max_rounds):
        if stop_expansion:
            break
        next_frontier: list[str] = []
        for parent_id in frontier:
            parent_row = state_rows_by_id[parent_id]
            parent_dir = Path(parent_row["state_dir"])
            parent_input = state_input_by_id[parent_id]
            batch_info = _build_validate_classify(parent_dir, context, allow_sampled_batches=False)
            state_reasons = _exact_incomplete_reasons(parent_dir, batch_info)
            for reason in state_reasons:
                _add_unique(exact_reasons, reason)

            if state_reasons and exact_fail_on_incomplete:
                leaf_reasons[parent_id] = "exact_incomplete"
                stop_expansion = True
                break

            active_passes = _int(_first_row(parent_dir / "per_state_summary.csv").get("active_passes"))
            if active_passes == 0:
                leaf_reasons[parent_id] = "no_active_passes"
                continue

            candidates = _read_csv(parent_dir / "batch_candidates.csv")
            correctness_by_batch = {
                row.get("batch_id", ""): row
                for row in _read_csv(parent_dir / "batch_correctness.csv")
                if row.get("batch_id")
            }
            executable = [
                (candidate, correctness_by_batch.get(candidate.get("batch_id", ""), {}))
                for candidate in candidates
                if _is_exact_executable(correctness_by_batch.get(candidate.get("batch_id", ""), {}))
            ]
            if not executable:
                leaf_reasons[parent_id] = "no_executable_batches"
                continue

            for candidate, correctness in executable:
                order = _split_order(candidate.get("canonical_order") or candidate.get("batch_passes"))
                if not order:
                    continue
                batch_id = candidate.get("batch_id", "")
                child_artifact = _successor_artifact(parent_dir, f"R{round_index:04d}_{batch_id}")
                result = run_opt(tools["opt"], parent_input, resolve_pipeline_sequence(order, context["pass_registry"]), child_artifact, timeout)
                _record_batch_apply(context, result)
                if not result.success or not child_artifact.exists():
                    continue

                child_hash = canonical_hash(child_artifact)
                duplicate_of = hash_to_state_id.get(child_hash, "")
                is_duplicate = bool(duplicate_of)
                if is_duplicate:
                    child_id = duplicate_of
                    child_row = state_rows_by_id[child_id]
                    child_input = state_input_by_id[child_id]
                else:
                    if len(state_rows) >= max_states:
                        _add_unique(exact_reasons, "state_cap_exceeded")
                        leaf_reasons[parent_id] = "state_cap_reached"
                        stop_expansion = True
                        break
                    child_id = f"S{next_state_number:04d}"
                    next_state_number += 1
                    child_dir = context["states_dir"] / child_id
                    child_input = _materialize_state_input(child_dir, child_artifact)
                    canonical_order = ";".join(order)
                    _analyze_new_state(context, child_input, child_dir, child_id, round_index + 1, parent_id, canonical_order)
                    child_row = _state_row_from_summary(
                        child_dir,
                        program=program,
                        state_id=child_id,
                        state_hash=child_hash,
                        depth=round_index + 1,
                        parent_state_id=parent_id,
                        transition_pass=canonical_order,
                        ir_path=child_input,
                        is_duplicate=False,
                        duplicate_of="",
                    )
                    state_rows.append(child_row)
                    state_rows_by_id[child_id] = child_row
                    state_input_by_id[child_id] = child_input
                    objective_by_state[child_id] = count_ir_instructions(child_input)
                    hash_to_state_id[child_hash] = child_id
                    next_frontier.append(child_id)

                canonical_order = ";".join(order)
                transition = _transition_row(program, parent_row, child_row, candidate, correctness, child_hash, is_duplicate, duplicate_of)
                transition_rows.append(transition)
                dag_rows.append(_dag_row(program, parent_row, child_row, candidate, correctness, child_hash, is_duplicate, duplicate_of))

                candidate_path = _extend_path_info(path_info_by_state[parent_id], candidate, correctness, order)
                if child_id not in path_info_by_state or _path_is_better(candidate_path, path_info_by_state[child_id]):
                    path_info_by_state[child_id] = candidate_path
                    parent_by_child[child_id] = _edge_for_path(
                        transition,
                        canonical_order,
                        objective_by_state[parent_id],
                        objective_by_state[child_id],
                    )
        frontier = next_frontier

    if not stop_expansion:
        for state_id in frontier:
            leaf_reasons.setdefault(state_id, "max_rounds_reached")

    status = _exact_status(exact_reasons, continued=not exact_fail_on_incomplete and bool(exact_reasons))
    return _finish_run(
        context,
        state_rows=state_rows,
        dag_rows=dag_rows,
        transition_rows=transition_rows,
        parent_by_child=parent_by_child,
        objective_by_state=objective_by_state,
        path_info_by_state=path_info_by_state,
        state_input_by_id=state_input_by_id,
        leaf_reasons=leaf_reasons,
        exact_status=status,
        exact_reasons=exact_reasons,
        duplicate_transitions=sum(1 for row in dag_rows if row.get("is_duplicate") == "true"),
    )


def _finish_run(
    context: dict,
    *,
    state_rows: list[dict],
    dag_rows: list[dict],
    transition_rows: list[dict],
    parent_by_child: dict[str, dict],
    objective_by_state: dict[str, int],
    path_info_by_state: dict[str, dict],
    state_input_by_id: dict[str, Path],
    leaf_reasons: dict[str, str],
    exact_status: str,
    exact_reasons: list[str],
    duplicate_transitions: int,
    frontier_score_rows: list[dict] | None = None,
    event_rows: list[dict] | None = None,
    budget_exhausted: bool = False,
    stop_reason: str = "max_rounds_reached",
) -> dict:
    out_dir: Path = context["out_dir"]
    frontier_score_rows = frontier_score_rows or []
    event_rows = event_rows or []
    state_rows_by_id = {row.get("state_id", ""): row for row in state_rows}
    selected_state_id = _select_best_state(state_rows, objective_by_state, path_info_by_state)
    selected_input = state_input_by_id[selected_state_id]
    shutil.copyfile(selected_input, out_dir / "final.ll")

    leaf_rows = _leaf_rows(
        context["program"],
        state_rows,
        context["objective"],
        objective_by_state,
        path_info_by_state,
        selected_state_id,
        leaf_reasons,
    )
    chosen_path_rows = _chosen_path_rows(
        selected_state_id,
        parent_by_child,
        state_rows_by_id=state_rows_by_id,
        state_input_by_id=state_input_by_id,
        objective_by_state=objective_by_state,
    )
    chosen_path_summary_rows = [
        _chosen_path_summary_row(
            context["program"],
            selected_state_id,
            chosen_path_rows,
            objective_by_state["S0000"],
            objective_by_state[selected_state_id],
        )
    ]
    optimized_pipeline_names = _flatten_pipeline_names(chosen_path_rows)
    optimized_pipeline = _flatten_pipeline(chosen_path_rows, context.get("pass_registry"))
    _write_csv(out_dir / "states.csv", STATE_FIELDS, state_rows)
    _write_csv(out_dir / "state_dag.csv", STATE_DAG_FIELDS, dag_rows)
    _write_csv(out_dir / "batch_state_transitions.csv", OPT_BATCH_STATE_TRANSITION_FIELDS, transition_rows)
    _write_csv(out_dir / "leaf_states.csv", LEAF_STATE_FIELDS, leaf_rows)
    _write_csv(out_dir / "chosen_path.csv", CHOSEN_PATH_FIELDS, chosen_path_rows)
    _write_csv(out_dir / "chosen_path_summary.csv", CHOSEN_PATH_SUMMARY_FIELDS, chosen_path_summary_rows)
    _write_csv(out_dir / "frontier_scores.csv", FRONTIER_SCORE_FIELDS, frontier_score_rows)
    _write_csv(out_dir / "optimizer_events.csv", OPTIMIZER_EVENT_FIELDS, event_rows)
    _write_optimized_batches(out_dir / "optimized_batches.txt", chosen_path_rows)
    (out_dir / "optimized_pipeline.txt").write_text(optimized_pipeline + ("\n" if optimized_pipeline else ""), encoding="utf-8")
    (out_dir / "optimized_pipeline_names.txt").write_text(
        optimized_pipeline_names + ("\n" if optimized_pipeline_names else ""),
        encoding="utf-8",
    )
    _write_optimized_pipeline_readable(out_dir / "optimized_pipeline_readable.txt", chosen_path_rows, optimized_pipeline, optimized_pipeline_names)
    _write_final_state(out_dir / "final_state.txt", state_rows_by_id[selected_state_id], state_input_by_id[selected_state_id], objective_by_state[selected_state_id])
    _write_path_artifacts(
        out_dir / "path_artifacts.md",
        selected_state_id,
        chosen_path_rows,
        chosen_path_summary_rows[0],
        optimized_pipeline,
        optimized_pipeline_names,
    )
    _write_exact_status(out_dir / "exact_status.txt", exact_status, exact_reasons)
    _write_summary(
        out_dir / "optimize_summary.md",
        input_path=context["input_path"],
        requested_mode=context.get("requested_mode", context["mode"]),
        mode=context["mode"],
        auto_reason=context.get("auto_reason", ""),
        objective=context["objective"],
        max_rounds=context["max_rounds"],
        beam_width=context["beam_width"],
        max_states=context["max_states"],
        max_batches_per_state=context["max_batches_per_state"],
        batch_frontier_policy=context["batch_frontier_policy"],
        batch_selection_policy=context["batch_selection_policy"],
        frontier_selection_policy=context["frontier_selection_policy"],
        selection_seed=context["selection_seed"],
        states=state_rows,
        transitions=transition_rows,
        duplicate_transitions=duplicate_transitions,
        leaf_rows=leaf_rows,
        chosen_path_rows=chosen_path_rows,
        selected_state_id=selected_state_id,
        root_objective=objective_by_state["S0000"],
        final_objective=objective_by_state[selected_state_id],
        optimized_pipeline=optimized_pipeline,
        exact_status=exact_status,
        exact_reasons=exact_reasons,
        budget_exhausted=budget_exhausted,
        stop_reason=stop_reason,
    )

    duplicate_states = sum(1 for row in state_rows if row.get("is_duplicate") == "true")
    return {
        "program": context["program"],
        "out_dir": str(out_dir),
        "requested_mode": context.get("requested_mode", context["mode"]),
        "selected_mode": context["mode"],
        "auto_reason": context.get("auto_reason", ""),
        "batch_selection_policy": context["batch_selection_policy"],
        "frontier_selection_policy": context["frontier_selection_policy"],
        "states": len(state_rows),
        "unique_states": len(state_rows) - duplicate_states,
        "duplicate_states": duplicate_states,
        "duplicate_transitions": duplicate_transitions,
        "batch_transitions": len(transition_rows),
        "selected_final_state": selected_state_id,
        "final_objective_value": objective_by_state[selected_state_id],
        "exact_status": exact_status,
        "exact_incomplete_reasons": ";".join(exact_reasons),
        "budget_exhausted": budget_exhausted,
        "stop_reason": stop_reason,
        "final_ll": str(out_dir / "final.ll"),
        "optimized_pipeline": str(out_dir / "optimized_pipeline.txt"),
        "chosen_path_csv": str(out_dir / "chosen_path.csv"),
        "optimize_summary": str(out_dir / "optimize_summary.md"),
    }


def _record_batch_apply(context: dict, result) -> None:
    timing = context.setdefault("timing", {})
    timing["batch_apply_time_ms"] = _float(timing.get("batch_apply_time_ms")) + _float(getattr(result, "time_ms", 0.0))
    timing["batch_apply_opt_invocations"] = _int(timing.get("batch_apply_opt_invocations")) + 1


def _write_optimizer_timing(out_dir: Path, context: dict, *, elapsed_ms: float) -> Path:
    timing = context.get("timing", {})
    state_dirs = _optimizer_state_dirs(out_dir)
    analysis_time = profiling_time = pair_time = batch_validation_time = 0.0
    profile_opt_invocations = pair_opt_invocations = validation_opt_invocations = 0

    for state_dir in state_dirs:
        summary = _first_row(state_dir / "per_state_summary.csv")
        analysis_time += _float(summary.get("total_time_ms"))
        profiling_time += _float(summary.get("profile_time_ms"))
        pair_time += _float(summary.get("pair_time_ms"))
        profile_opt_invocations += len(_read_csv(state_dir / "pass_profile.csv"))
        for row in _read_csv(state_dir / "pair_relation.csv"):
            if row.get("dynamic_relation") == "not_tested" or row.get("failure_kind") == "max_pairs":
                continue
            pair_opt_invocations += 2
        for row in _read_csv(state_dir / "batch_validation.csv"):
            batch_validation_time += _float(row.get("time_ms"))
            validation_opt_invocations += _int(row.get("tested_orders"))

    batch_apply_time = _float(timing.get("batch_apply_time_ms"))
    batch_apply_invocations = _int(timing.get("batch_apply_opt_invocations"))
    total_opt_invocations = (
        len(_read_csv(out_dir / "valid_passes.csv"))
        + len(_read_csv(out_dir / "invalid_passes.csv"))
        + profile_opt_invocations
        + pair_opt_invocations
        + validation_opt_invocations
        + batch_apply_invocations
    )
    row = {
        "program": context["program"],
        "optimizer_total_time_ms": f"{elapsed_ms:.3f}",
        "analysis_time_ms": f"{analysis_time:.3f}",
        "profiling_time_ms": f"{profiling_time:.3f}",
        "pair_testing_time_ms": f"{pair_time:.3f}",
        "batch_validation_time_ms": f"{batch_validation_time:.3f}",
        "batch_apply_time_ms": f"{batch_apply_time:.3f}",
        "total_opt_invocations": str(total_opt_invocations),
        "batch_apply_opt_invocations": str(batch_apply_invocations),
    }
    path = out_dir / "optimizer_timing.csv"
    _write_csv(path, OPTIMIZER_TIMING_FIELDS, [row])
    return path


def _optimizer_state_dirs(out_dir: Path) -> list[Path]:
    rows = _read_csv(out_dir / "states.csv")
    dirs: list[Path] = []
    if rows:
        for row in rows:
            if _is_true(row.get("is_duplicate")):
                continue
            state_dir = row.get("state_dir", "")
            if state_dir:
                dirs.append(Path(state_dir))
            elif row.get("state_id"):
                dirs.append(out_dir / "states" / row["state_id"])
    else:
        states_root = out_dir / "states"
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


def _build_validate_classify(state_dir: Path, context: dict, *, allow_sampled_batches: bool) -> dict:
    result = build_batch_family(state_dir)
    if context["validate_batches"]:
        result.update(
            validate_batch_candidates(
                state_dir,
                context["tools"],
                timeout=context["timeout"],
                jobs=context["jobs"],
            )
        )
    correctness_rows = classify_batch_correctness(state_dir, allow_sampled_batches=allow_sampled_batches)
    build_coverage_report(state_dir)
    result["correctness_rows"] = correctness_rows
    return result


def _exact_incomplete_reasons(state_dir: Path, batch_info: dict) -> list[str]:
    state_id = state_dir.name
    reasons: list[str] = []
    summary = _first_row(state_dir / "batch_summary.csv")
    if _is_true(summary.get("truncated")) or bool(batch_info.get("truncated")):
        reasons.append(f"truncated_batch_candidates:{state_id}")
    components = _read_csv(state_dir / "batch_components.csv")
    if any((not _is_true(row.get("is_exact"))) or bool(row.get("unresolved_reason")) for row in components):
        reasons.append(f"unresolved_components:{state_id}")
    coverage = _first_row(state_dir / "coverage_summary.csv")
    if _int(coverage.get("dropped_active_passes")) > 0:
        reasons.append(f"dropped_active_passes:{state_id}")

    candidates = _read_csv(state_dir / "batch_candidates.csv")
    correctness = _read_csv(state_dir / "batch_correctness.csv")
    correctness_by_id = {row.get("batch_id", ""): row for row in correctness if row.get("batch_id")}
    missing = [row.get("batch_id", "") for row in candidates if row.get("batch_id", "") not in correctness_by_id]
    if missing:
        reasons.append(f"missing_correctness_rows:{state_id}")
    if any(correctness_by_id.get(row.get("batch_id", ""), {}).get("validation_status") in {"", "not_validated"} for row in candidates):
        reasons.append(f"missing_validation:{state_id}")
    return reasons


def _is_exact_executable(correctness: dict) -> bool:
    return (
        correctness.get("correctness_class") == "certified_batch"
        and correctness.get("can_hard_fold") == "true"
        and correctness.get("validation_status") == "all_permutations_same"
    )


def score_batch_candidate(state_dir: Path, batch_row: dict, context: dict) -> dict:
    state_dir = Path(state_dir)
    summary = _first_row(state_dir / "batch_summary.csv")
    per_state = _first_row(state_dir / "per_state_summary.csv")
    batch_id = batch_row.get("batch_id", "")
    correctness = context.get("correctness_by_batch", {}).get(batch_id, {})
    batch_size = _int(batch_row.get("batch_size"))
    active_passes = _int(per_state.get("active_passes")) or _int(summary.get("active_passes"))

    coverage_score = (batch_size / active_passes) if active_passes > 0 else 0.0
    if active_passes <= 1:
        batch_size_score = 1.0 if batch_size > 0 else 0.0
    else:
        batch_size_score = math.log1p(batch_size) / math.log1p(active_passes)

    reduction_estimate = _float(summary.get("batch_reduction_estimate"))
    if reduction_estimate > 0:
        reduction_score = min(1.0, math.log1p(reduction_estimate) / 10.0)
    else:
        reduction_score = batch_size_score

    correctness_class = correctness.get("correctness_class", "")
    validation_status = correctness.get("validation_status") or "not_validated"
    if correctness_class == "certified_batch" or validation_status == "all_permutations_same":
        evidence_score = 1.0
    elif correctness_class == "sampled_batch" or validation_status == "sampled_same":
        evidence_score = 0.5
    else:
        evidence_score = 0.0

    diversity_score = 1.0 if context.get("first_signature_by_batch", {}).get(batch_id, True) else 0.0
    total_components = _int(batch_row.get("num_conflict_components")) or _int(summary.get("conflict_components")) or 1
    unresolved_components = _int(batch_row.get("unresolved_components")) or _int(summary.get("unresolved_components"))
    risk_penalty = unresolved_components / max(1, total_components)
    if correctness_class == "sampled_batch" or validation_status == "sampled_same":
        risk_penalty += 0.25
    if correctness_class not in {"certified_batch", "sampled_batch"} and validation_status not in {"all_permutations_same", "sampled_same"}:
        risk_penalty = 1.0
    risk_penalty = _clamp(risk_penalty)

    final_score = _clamp(
        0.35 * _clamp(coverage_score)
        + 0.25 * _clamp(batch_size_score)
        + 0.20 * _clamp(reduction_score)
        + 0.10 * _clamp(evidence_score)
        + 0.10 * _clamp(diversity_score)
        - 0.20 * risk_penalty
    )
    return {
        "program": batch_row.get("program", summary.get("program", "")),
        "state_id": batch_row.get("state_id", summary.get("state_id", state_dir.name)),
        "state_hash": batch_row.get("state_hash", summary.get("state_hash", "")),
        "batch_id": batch_id,
        "batch_passes": batch_row.get("batch_passes", ""),
        "batch_size": batch_row.get("batch_size", ""),
        "correctness_class": correctness_class,
        "validation_status": validation_status,
        "coverage_score": _format_score(coverage_score),
        "batch_size_score": _format_score(batch_size_score),
        "reduction_score": _format_score(reduction_score),
        "evidence_score": _format_score(evidence_score),
        "diversity_score": _format_score(diversity_score),
        "risk_penalty": _format_score(risk_penalty),
        "final_batch_score": _format_score(final_score),
        "selected_for_execution": "false",
        "selection_reason": "",
    }


def _score_batch_candidates(state_dir: Path, candidates: list[dict], correctness_by_batch: dict[str, dict], context: dict) -> list[dict]:
    first_signature_by_batch: dict[str, bool] = {}
    seen_signatures: set[str] = set()
    for candidate in candidates:
        batch_id = candidate.get("batch_id", "")
        signature = _batch_signature(candidate)
        first_signature_by_batch[batch_id] = signature not in seen_signatures
        seen_signatures.add(signature)

    score_context = {
        **context,
        "correctness_by_batch": correctness_by_batch,
        "first_signature_by_batch": first_signature_by_batch,
    }
    return [score_batch_candidate(state_dir, candidate, score_context) for candidate in candidates]


def _mark_selected_batch_scores(
    state_dir: Path,
    rows: list[dict],
    selected_keys: set[tuple[str, str]],
    selected_reasons: dict[str, str],
) -> None:
    selected_ids = {batch_id for batch_id, _class_name in selected_keys}
    for row in rows:
        batch_id = row.get("batch_id", "")
        if batch_id in selected_ids:
            row["selected_for_execution"] = "true"
            row["selection_reason"] = selected_reasons.get(batch_id, "selected")
        elif row.get("correctness_class") in {"rejected_batch", "failed_batch", "unvalidated_batch", "unknown_batch"}:
            row["selection_reason"] = row.get("correctness_class", "")
        else:
            row["selection_reason"] = "not_selected"
    _write_csv(Path(state_dir) / "batch_candidate_scores.csv", BATCH_CANDIDATE_SCORE_FIELDS, rows)


def _write_unselected_batch_scores(state_dir: Path, context: dict) -> None:
    candidates = _read_csv(Path(state_dir) / "batch_candidates.csv")
    correctness_by_batch = {
        row.get("batch_id", ""): row
        for row in _read_csv(Path(state_dir) / "batch_correctness.csv")
        if row.get("batch_id")
    }
    rows = _score_batch_candidates(Path(state_dir), candidates, correctness_by_batch, context)
    _mark_selected_batch_scores(Path(state_dir), rows, set(), {})


def _analyze_new_state(context: dict, input_ll: Path, state_dir: Path, state_id: str, depth: int, parent_state_id: str, transition_pass: str) -> None:
    _analyze(
        input_ll,
        state_dir,
        context["tools"],
        valid_passes=context["valid_passes"],
        invalid_rows=context["invalid_rows"],
        configured_pass_count=context["configured_pass_count"],
        jobs=context["jobs"],
        timeout=context["timeout"],
        max_pairs=context["max_pairs"],
        program=context["program"],
        state_id=state_id,
        depth=depth,
        parent_state_id=parent_state_id,
        transition_pass=transition_pass,
    )


def _analyze(input_ll: Path, state_dir: Path, tools: dict, **kwargs) -> None:
    try:
        analyze_state(input_ll, state_dir, tools, **kwargs)
    except TypeError as exc:
        if "pass_registry" not in str(exc):
            raise
        fallback = dict(kwargs)
        fallback.pop("pass_registry", None)
        analyze_state(input_ll, state_dir, tools, **fallback)


def _state_row_from_summary(
    state_dir: Path,
    *,
    program: str,
    state_id: str,
    state_hash: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
    ir_path: Path,
    is_duplicate: bool,
    duplicate_of: str,
) -> dict:
    summary = _first_row(state_dir / "per_state_summary.csv")
    return {
        "program": program,
        "state_id": state_id,
        "state_hash": state_hash,
        "depth": str(depth),
        "parent_state_id": parent_state_id,
        "transition_pass": transition_pass,
        "ir_path": str(ir_path),
        "state_dir": str(state_dir),
        "is_duplicate": _bool(is_duplicate),
        "duplicate_of": duplicate_of,
        "active_passes": summary.get("active_passes", ""),
        "pairs_tested": summary.get("pairs_tested", ""),
        "dynamic_commute": summary.get("dynamic_commute", ""),
        "order_sensitive": summary.get("order_sensitive", ""),
        "unknown": summary.get("unknown", ""),
        "max_conflict_component": summary.get("max_conflict_component", ""),
        "total_time_ms": summary.get("total_time_ms", ""),
    }


def _duplicate_state_row(
    canonical: dict,
    *,
    state_id: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
    ir_path: Path,
    duplicate_of: str,
) -> dict:
    row = dict(canonical)
    row.update(
        {
            "state_id": state_id,
            "depth": str(depth),
            "parent_state_id": parent_state_id,
            "transition_pass": transition_pass,
            "ir_path": str(ir_path),
            "state_dir": str(Path(ir_path).parent),
            "is_duplicate": "true",
            "duplicate_of": duplicate_of,
        }
    )
    return row


def _transition_row(
    program: str,
    parent: dict,
    child: dict,
    candidate: dict,
    correctness: dict,
    child_hash: str,
    is_duplicate: bool,
    duplicate_of: str,
) -> dict:
    return {
        "program": program,
        "parent_state_id": parent["state_id"],
        "child_state_id": child["state_id"],
        "batch_id": candidate.get("batch_id", ""),
        "batch_passes": candidate.get("batch_passes", ""),
        "batch_size": candidate.get("batch_size", ""),
        "validation_status": correctness.get("validation_status", ""),
        "correctness_class": correctness.get("correctness_class", ""),
        "can_hard_fold": correctness.get("can_hard_fold", ""),
        "can_execute": correctness.get("can_execute", ""),
        "parent_hash": parent["state_hash"],
        "child_hash": child_hash,
        "is_duplicate": _bool(is_duplicate),
        "duplicate_of": duplicate_of,
    }


def _dag_row(
    program: str,
    parent: dict,
    child: dict,
    candidate: dict,
    correctness: dict,
    child_hash: str,
    is_duplicate: bool,
    duplicate_of: str,
) -> dict:
    return {
        "program": program,
        "source_state_id": parent["state_id"],
        "target_state_id": child["state_id"],
        "source_hash": parent["state_hash"],
        "target_hash": child_hash,
        "transition_kind": "batch",
        "batch_id": candidate.get("batch_id", ""),
        "batch_passes": candidate.get("batch_passes", ""),
        "canonical_order": candidate.get("canonical_order", "") or candidate.get("batch_passes", ""),
        "validation_status": correctness.get("validation_status", ""),
        "correctness_class": correctness.get("correctness_class", ""),
        "is_duplicate": _bool(is_duplicate),
        "duplicate_of": duplicate_of,
    }


def _select_budgeted_batches(
    executable: list[tuple[dict, dict]],
    *,
    policy: str,
    limit: int,
    score_rows: list[dict] | None = None,
    state_id: str = "",
    selection_seed: int = 0,
) -> list[tuple[dict, dict]]:
    if limit <= 0:
        return []
    indexed = list(enumerate(executable))
    if policy == "largest-batch":
        ordered = sorted(indexed, key=lambda item: (-_int(item[1][0].get("batch_size")), item[0]))
    elif policy == "diverse":
        ordered = _order_diverse_batches(indexed)
    elif policy == "score":
        return _select_scored_batches(indexed, limit=limit, score_rows=score_rows or [], state_id=state_id, selection_seed=selection_seed)
    elif policy in {"certified-first", "score", "objective"}:
        ordered = sorted(indexed, key=lambda item: (_validation_rank(item[1][1].get("validation_status")), item[0]))
    else:
        raise ValueError(f"unknown batch frontier policy: {policy}")
    return [pair for _, pair in ordered[:limit]]


def _select_scored_batches(
    indexed: list[tuple[int, tuple[dict, dict]]],
    *,
    limit: int,
    score_rows: list[dict],
    state_id: str,
    selection_seed: int,
) -> list[tuple[dict, dict]]:
    score_by_id = {row.get("batch_id", ""): row for row in score_rows}
    score_slots = max(1, math.floor(0.5 * limit))
    diversity_slots = math.floor(0.3 * limit)
    neutral_slots = max(0, limit - score_slots - diversity_slots)
    selected: list[tuple[int, tuple[dict, dict]]] = []

    def add_from(ordered: list[tuple[int, tuple[dict, dict]]], count: int) -> None:
        if count <= 0:
            return
        selected_ids = {item[1][0].get("batch_id", "") for item in selected}
        added = 0
        for item in ordered:
            batch_id = item[1][0].get("batch_id", "")
            if batch_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(batch_id)
            added += 1
            if len(selected) >= limit or added >= count:
                break

    by_score = sorted(
        indexed,
        key=lambda item: (
            -_float(score_by_id.get(item[1][0].get("batch_id", ""), {}).get("final_batch_score")),
            _validation_rank(item[1][1].get("validation_status")),
            item[0],
        ),
    )
    by_diversity = _order_diverse_batches(indexed)
    by_neutral = sorted(
        indexed,
        key=lambda item: _stable_hash_int(str(selection_seed), state_id, item[1][0].get("batch_id", "")),
    )

    add_from(by_score, score_slots)
    add_from(by_diversity, diversity_slots)
    add_from(by_neutral, neutral_slots)
    add_from(by_score, limit - len(selected))
    return [pair for _, pair in selected[:limit]]


def _order_diverse_batches(indexed: list[tuple[int, tuple[dict, dict]]]) -> list[tuple[int, tuple[dict, dict]]]:
    first_by_signature: list[tuple[int, tuple[dict, dict]]] = []
    repeated: list[tuple[int, tuple[dict, dict]]] = []
    seen: set[str] = set()
    for item in indexed:
        candidate = item[1][0]
        signature = candidate.get("component_choices") or candidate.get("batch_passes") or candidate.get("canonical_order") or candidate.get("batch_id", "")
        if signature not in seen:
            first_by_signature.append(item)
            seen.add(signature)
        else:
            repeated.append(item)
    return first_by_signature + repeated


def _select_budgeted_frontier(
    child_ids: list[str],
    *,
    round_index: int,
    policy: str,
    beam_width: int,
    state_rows_by_id: dict[str, dict],
    objective_by_state: dict[str, int],
    child_info_by_state: dict[str, dict],
    context: dict,
) -> tuple[list[str], list[dict]]:
    score_rows = _frontier_score_rows(
        child_ids,
        round_index=round_index,
        policy=policy,
        state_rows_by_id=state_rows_by_id,
        objective_by_state=objective_by_state,
        child_info_by_state=child_info_by_state,
        context=context,
    )
    pareto_ids = _pareto_kept_state_ids(score_rows)
    for row in score_rows:
        row["pareto_kept"] = _bool(row["state_id"] in pareto_ids)

    selection_buckets: dict[str, str] = {}
    if beam_width <= 0:
        ordered: list[str] = []
    elif policy == "objective":
        ordered = sorted(child_ids, key=lambda state_id: (objective_by_state[state_id], state_id))
        selection_buckets = {state_id: "objective_policy" for state_id in ordered[:beam_width]}
    elif policy == "largest-batch":
        ordered = sorted(
            child_ids,
            key=lambda state_id: (-_int(child_info_by_state.get(state_id, {}).get("last_batch_size")), objective_by_state[state_id], state_id),
        )
        selection_buckets = {state_id: "largest_batch_policy" for state_id in ordered[:beam_width]}
    elif policy == "certified-first":
        ordered = sorted(
            child_ids,
            key=lambda state_id: (
                _validation_rank(child_info_by_state.get(state_id, {}).get("validation_status")),
                objective_by_state[state_id],
                state_id,
            ),
        )
        selection_buckets = {state_id: "certified_first_policy" for state_id in ordered[:beam_width]}
    elif policy == "score":
        ordered_with_buckets = _select_diversity_preserving_beam(score_rows, beam_width=beam_width)
        ordered = [state_id for state_id, _bucket in ordered_with_buckets]
        selection_buckets = {state_id: bucket for state_id, bucket in ordered_with_buckets}
    elif policy == "diverse":
        ordered = _order_diverse_frontier(child_ids, objective_by_state, child_info_by_state)
        selection_buckets = {state_id: "diverse_policy" for state_id in ordered[:beam_width]}
    else:
        raise ValueError(f"unknown batch frontier policy: {policy}")

    selected = set(ordered[: max(0, beam_width)])
    rank_by_state = {state_id: str(rank) for rank, state_id in enumerate(ordered)}
    rows_by_state = {row["state_id"]: row for row in score_rows}
    output_order = ordered + [state_id for state_id in child_ids if state_id not in set(ordered)]
    rows = []
    for fallback_rank, state_id in enumerate(output_order):
        row = rows_by_state[state_id]
        row["rank"] = rank_by_state.get(state_id, str(len(ordered) + fallback_rank))
        row["selection_bucket"] = selection_buckets.get(state_id, "")
        row["selected_for_frontier"] = _bool(state_id in selected)
        row["selection_reason"] = "within_beam" if state_id in selected else "beam_pruned"
        rows.append(row)
    return ordered[: max(0, beam_width)], rows


def score_frontier_state(child_state: dict, parent_state: dict, transition: dict, context: dict) -> dict:
    objective_by_state = context.get("objective_by_state", {})
    root_inst = _int(context.get("root_inst_count")) or _int(objective_by_state.get("S0000"))
    child_id = child_state.get("state_id", "")
    parent_id = parent_state.get("state_id", transition.get("parent_state_id", ""))
    parent_inst = _int(objective_by_state.get(parent_id))
    child_inst = _int(objective_by_state.get(child_id))
    raw_objective = (root_inst - child_inst) / max(1, root_inst)
    objective_score = _clamp(raw_objective)
    parent_gain = (parent_inst - child_inst) / max(1, parent_inst) if parent_inst else 0.0

    child_dir = Path(child_state.get("state_dir", ""))
    summary = _first_row(child_dir / "batch_summary.csv")
    correctness = _read_csv(child_dir / "batch_correctness.csv")
    coverage = _first_row(child_dir / "coverage_summary.csv")
    active_passes = _int(child_state.get("active_passes"))
    configured_passes = max(1, _int(context.get("configured_pass_count")))
    active_score = math.log1p(active_passes) / math.log1p(configured_passes) if configured_passes > 1 else 0.0
    certified = sum(1 for row in correctness if row.get("correctness_class") == "certified_batch")
    candidate_count = _int(summary.get("batch_candidates")) or len(correctness)
    certified_score = certified / max(1, candidate_count) if candidate_count else 0.0
    enable_count = _int(transition.get("enable_count_from_parent"))
    effect_changed_count = _int(transition.get("effect_changed_count_from_parent"))
    enable_effect_score = (
        math.log1p(enable_count + effect_changed_count) / math.log1p(configured_passes)
        if configured_passes > 1
        else 0.0
    )
    enable_effect_score = _clamp(enable_effect_score)
    future_potential_score = _clamp(0.4 * active_score + 0.4 * certified_score + 0.2 * enable_effect_score)

    coverage_active = _int(coverage.get("active_passes"))
    if coverage_active > 0:
        certified_ratio = _int(coverage.get("certified_covered")) / coverage_active
        heuristic_ratio = _int(coverage.get("heuristic_covered")) / coverage_active
        unknown_ratio = (_int(coverage.get("unvalidated_covered")) + _int(coverage.get("failed_or_unknown"))) / coverage_active
        dropped_ratio = _int(coverage.get("dropped_active_passes")) / coverage_active
    else:
        certified_ratio = heuristic_ratio = unknown_ratio = dropped_ratio = 0.0
    evidence_quality_score = _clamp(certified_ratio + 0.5 * heuristic_ratio - 0.5 * unknown_ratio - dropped_ratio)

    novelty_score = _float(context.get("novelty_by_child", {}).get(child_id, 0.0))
    cost_score = _clamp((_float(child_state.get("total_time_ms")) + _validation_time_ms(child_dir)) / 10000.0)
    unresolved_ratio = (_int(summary.get("unresolved_components")) / max(1, _int(summary.get("conflict_components")))) if summary else 0.0
    validation_rows = _read_csv(child_dir / "batch_validation.csv")
    validation_failure_ratio = (
        sum(1 for row in validation_rows if row.get("validation_status") in {"mismatch", "failed"}) / max(1, len(validation_rows))
        if validation_rows
        else 0.0
    )
    risk_penalty = _clamp(unresolved_ratio + validation_failure_ratio + dropped_ratio)
    final_score = _clamp(
        0.45 * objective_score
        + 0.20 * future_potential_score
        + 0.15 * evidence_quality_score
        + 0.15 * novelty_score
        - 0.05 * cost_score
        - 0.20 * risk_penalty
    )
    return {
        "root_inst_count": str(root_inst),
        "parent_inst_count": str(parent_inst),
        "child_inst_count": str(child_inst),
        "enable_count_from_parent": str(enable_count),
        "effect_changed_count_from_parent": str(effect_changed_count),
        "parent_gain": _format_score(parent_gain),
        "objective_score": _format_score(objective_score),
        "future_potential_score": _format_score(future_potential_score),
        "evidence_quality_score": _format_score(evidence_quality_score),
        "novelty_score": _format_score(novelty_score),
        "cost_score": _format_score(cost_score),
        "risk_penalty": _format_score(risk_penalty),
        "final_state_score": _format_score(final_score),
    }


def _frontier_score_rows(
    child_ids: list[str],
    *,
    round_index: int,
    policy: str,
    state_rows_by_id: dict[str, dict],
    objective_by_state: dict[str, int],
    child_info_by_state: dict[str, dict],
    context: dict,
) -> list[dict]:
    novelty_by_child = _frontier_novelty(child_ids, child_info_by_state)
    score_context = {
        **context,
        "objective_by_state": objective_by_state,
        "root_inst_count": objective_by_state.get("S0000", 0),
        "novelty_by_child": novelty_by_child,
    }
    rows = []
    for state_id in child_ids:
        info = child_info_by_state.get(state_id, {})
        state_row = state_rows_by_id[state_id]
        parent_row = state_rows_by_id.get(info.get("parent_state_id", ""), {})
        score = score_frontier_state(state_row, parent_row, info, score_context)
        rows.append(
            {
                "round": str(round_index),
                "state_id": state_id,
                "parent_state_id": info.get("parent_state_id", ""),
                "last_batch_id": info.get("last_batch_id", ""),
                "batch_passes": info.get("batch_passes", ""),
                "component_choices": info.get("component_choices", ""),
                "active_pass_signature": info.get("active_pass_signature", ""),
                "depth": state_row.get("depth", ""),
                "objective_value": str(objective_by_state[state_id]),
                "active_passes": state_row.get("active_passes", ""),
                "batch_candidates": _batch_candidate_count(state_row),
                "last_batch_size": info.get("last_batch_size", ""),
                "validation_status": info.get("validation_status", ""),
                "correctness_class": info.get("correctness_class", ""),
                "enable_count_from_parent": info.get("enable_count_from_parent", "0"),
                "effect_changed_count_from_parent": info.get("effect_changed_count_from_parent", "0"),
                **score,
                "pareto_kept": "false",
                "policy": policy,
                "rank": "",
                "selection_bucket": "",
                "selected_for_frontier": "false",
                "selection_reason": "",
            }
        )
    return rows


def _frontier_novelty(child_ids: list[str], child_info_by_state: dict[str, dict]) -> dict[str, float]:
    novelty: dict[str, float] = {}
    seen_active: set[str] = set()
    seen_batch: set[str] = set()
    for state_id in child_ids:
        info = child_info_by_state.get(state_id, {})
        active_signature = info.get("active_pass_signature", "")
        batch_signature = info.get("component_choices") or info.get("batch_passes") or info.get("last_batch_id", state_id)
        is_novel = (active_signature and active_signature not in seen_active) or (batch_signature and batch_signature not in seen_batch)
        novelty[state_id] = 1.0 if is_novel else 0.0
        if active_signature:
            seen_active.add(active_signature)
        if batch_signature:
            seen_batch.add(batch_signature)
    return novelty


def _pareto_kept_state_ids(rows: list[dict]) -> set[str]:
    kept: set[str] = set()
    for row in rows:
        if not any(_dominates(other, row) for other in rows if other is not row):
            kept.add(row.get("state_id", ""))
    return kept


def _dominates(candidate: dict, other: dict) -> bool:
    high_fields = ["objective_score", "future_potential_score", "evidence_quality_score", "novelty_score"]
    low_fields = ["cost_score", "risk_penalty"]
    no_worse = all(_float(candidate.get(field)) >= _float(other.get(field)) for field in high_fields)
    no_worse = no_worse and all(_float(candidate.get(field)) <= _float(other.get(field)) for field in low_fields)
    strictly_better = any(_float(candidate.get(field)) > _float(other.get(field)) for field in high_fields)
    strictly_better = strictly_better or any(_float(candidate.get(field)) < _float(other.get(field)) for field in low_fields)
    return no_worse and strictly_better


def _select_diversity_preserving_beam(rows: list[dict], *, beam_width: int) -> list[tuple[str, str]]:
    if beam_width <= 0:
        return []
    pareto = [row for row in rows if row.get("pareto_kept") == "true"]
    non_pareto = [row for row in rows if row.get("pareto_kept") != "true"]
    pool = pareto or rows
    selected: list[tuple[dict, str]] = []
    objective_slots = 1 if beam_width >= 1 else 0
    novelty_slots = 1 if beam_width >= 2 else 0
    if beam_width >= 10:
        novelty_slots = max(novelty_slots, math.floor(0.2 * beam_width))
    score_slots = max(0, beam_width - novelty_slots - objective_slots)

    def selected_ids() -> set[str]:
        return {row.get("state_id", "") for row, _bucket in selected}

    def add_rows(ordered: list[dict], count: int, bucket: str, *, distinct_signature: bool = False) -> None:
        if count <= 0:
            return
        added = 0
        used_signatures = {_row_signature(row) for row, _bucket in selected}
        for row in ordered:
            state_id = row.get("state_id", "")
            if state_id in selected_ids():
                continue
            if distinct_signature and _row_signature(row) in used_signatures:
                continue
            selected.append((row, bucket))
            used_signatures.add(_row_signature(row))
            added += 1
            if len(selected) >= beam_width or added >= count:
                break

    by_score = sorted(pool, key=lambda row: (-_float(row.get("final_state_score")), row.get("state_id", "")))
    by_novelty = sorted(pool, key=lambda row: (-_float(row.get("novelty_score")), -_float(row.get("final_state_score")), row.get("state_id", "")))
    by_objective = sorted(pool, key=lambda row: (_int(row.get("objective_value")), row.get("state_id", "")))
    fallback = sorted(non_pareto, key=lambda row: (-_float(row.get("final_state_score")), row.get("state_id", "")))

    add_rows(by_score, score_slots, "score_bucket")
    add_rows(by_novelty, novelty_slots, "novelty_bucket", distinct_signature=True)
    add_rows(by_objective, objective_slots, "objective_bucket")
    add_rows(by_score, beam_width - len(selected), "score_fill")
    add_rows(fallback, beam_width - len(selected), "non_pareto_fill")
    return [(row.get("state_id", ""), bucket) for row, bucket in selected[:beam_width]]


def _row_signature(row: dict) -> str:
    active_signature = row.get("active_pass_signature", "")
    batch_signature = row.get("component_choices") or row.get("batch_passes") or row.get("last_batch_id") or row.get("state_id", "")
    return f"{active_signature}::{batch_signature}"


def _validation_time_ms(state_dir: Path) -> float:
    return sum(_float(row.get("time_ms")) for row in _read_csv(Path(state_dir) / "batch_validation.csv"))


def _order_diverse_frontier(child_ids: list[str], objective_by_state: dict[str, int], child_info_by_state: dict[str, dict]) -> list[str]:
    unique: list[str] = []
    repeated: list[str] = []
    seen: set[str] = set()
    for state_id in child_ids:
        signature = _frontier_signature(state_id, child_info_by_state)
        if signature not in seen:
            unique.append(state_id)
            seen.add(signature)
        else:
            repeated.append(state_id)
    return unique + sorted(repeated, key=lambda state_id: (objective_by_state[state_id], state_id))


def _frontier_signature(state_id: str, child_info_by_state: dict[str, dict]) -> str:
    info = child_info_by_state.get(state_id, {})
    active_signature = info.get("active_pass_signature", "")
    batch_signature = info.get("component_choices") or info.get("batch_passes") or info.get("last_batch_id", state_id)
    return f"{active_signature}::{batch_signature}"


def _batch_candidate_count(state_row: dict) -> str:
    summary = _first_row(Path(state_row.get("state_dir", "")) / "batch_summary.csv")
    return summary.get("batch_candidates", "")


def _active_pass_signature(state_dir: Path) -> str:
    passes = [
        row.get("pass", "")
        for row in _read_csv(Path(state_dir) / "pass_profile.csv")
        if row.get("pass") and _is_true(row.get("success")) and _is_true(row.get("active"))
    ]
    return ";".join(sorted(passes))


def _enable_effect_counts(parent_dir: Path, child_dir: Path, valid_passes: list[str]) -> dict[str, str]:
    parent_profile = {row.get("pass", ""): row for row in _read_csv(Path(parent_dir) / "pass_profile.csv") if row.get("pass")}
    child_profile = {row.get("pass", ""): row for row in _read_csv(Path(child_dir) / "pass_profile.csv") if row.get("pass")}
    enable_count = 0
    effect_changed_count = 0
    for pass_name in valid_passes:
        parent = parent_profile.get(pass_name, {})
        child = child_profile.get(pass_name, {})
        if not parent or not child:
            continue
        if not (_is_true(parent.get("success")) and _is_true(child.get("success"))):
            continue
        parent_active = _is_true(parent.get("active"))
        child_active = _is_true(child.get("active"))
        if not parent_active and child_active:
            enable_count += 1
        elif parent_active and child_active and _pass_effect_signature(parent) != _pass_effect_signature(child):
            effect_changed_count += 1
    return {
        "enable_count_from_parent": str(enable_count),
        "effect_changed_count_from_parent": str(effect_changed_count),
    }


def _pass_effect_signature(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("inst_delta", "")),
        str(row.get("blocks_changed", "")),
        str(row.get("changed_functions", "")),
    )


def _batch_signature(candidate: dict) -> str:
    if candidate.get("component_choices"):
        return candidate.get("component_choices", "")
    passes = _split_order(candidate.get("batch_passes") or candidate.get("canonical_order"))
    if passes:
        return ";".join(sorted(passes))
    return candidate.get("batch_id", "")


def _validation_rank(status: object) -> int:
    return {
        "all_permutations_same": 0,
        "sampled_same": 1,
        "not_validated": 2,
        "": 2,
    }.get(str(status or ""), 3)


def _unique_state_count(state_rows: list[dict]) -> int:
    return sum(1 for row in state_rows if row.get("is_duplicate") != "true")


def _current_incumbent(state_rows: list[dict], objective_by_state: dict[str, int], path_info_by_state: dict[str, dict]) -> str:
    return _select_best_state(state_rows, objective_by_state, path_info_by_state)


def _is_better_state(candidate_id: str, current_id: str, objective_by_state: dict[str, int], path_info_by_state: dict[str, dict]) -> bool:
    return _state_selection_key(candidate_id, objective_by_state, path_info_by_state) < _state_selection_key(current_id, objective_by_state, path_info_by_state)


def _state_selection_key(state_id: str, objective_by_state: dict[str, int], path_info_by_state: dict[str, dict]) -> tuple[int, int, int, float, str]:
    path_info = path_info_by_state[state_id]
    return (
        objective_by_state[state_id],
        path_info["path_length"],
        path_info["pass_invocations"],
        -_certified_ratio(path_info),
        state_id,
    )


def _event(rows: list[dict], round_index: int, state_id: str, event_type: str, message: str) -> None:
    rows.append(
        {
            "event_id": str(len(rows)),
            "round": str(round_index),
            "state_id": state_id,
            "event_type": event_type,
            "message": message,
        }
    )


def _budgeted_leaf_reasons(states: list[dict], has_transitions: bool) -> dict[str, str]:
    reasons = {}
    for state in states:
        state_id = state["state_id"]
        if state_id == "S0000":
            if not has_transitions:
                reasons[state_id] = "no_executable_batches"
        else:
            reasons[state_id] = "max_rounds_reached"
    return reasons


def _leaf_rows(
    program: str,
    states: list[dict],
    objective: str,
    objective_by_state: dict[str, int],
    path_info_by_state: dict[str, dict],
    selected_state_id: str,
    leaf_reasons: dict[str, str],
) -> list[dict]:
    rows = []
    for state in states:
        state_id = state["state_id"]
        path_info = path_info_by_state[state_id]
        leaf_reason = _prioritized_leaf_reason(state, leaf_reasons.get(state_id, "expanded"))
        rows.append(
            {
                "program": program,
                "state_id": state_id,
                "depth": state.get("depth", ""),
                "state_hash": state.get("state_hash", ""),
                "objective_kind": objective,
                "objective_value": str(objective_by_state[state_id]),
                "is_leaf": _bool(leaf_reason != "expanded"),
                "leaf_reason": leaf_reason,
                "path_length": str(path_info["path_length"]),
                "pass_invocations": str(path_info["pass_invocations"]),
                "selected_as_final": _bool(state_id == selected_state_id),
            }
        )
    return rows


def _prioritized_leaf_reason(state: dict, current_reason: str) -> str:
    active_passes = str(state.get("active_passes", "")).strip()
    if active_passes and _int(active_passes) == 0:
        return "no_active_passes"
    return current_reason


def _select_best_state(states: list[dict], objective_by_state: dict[str, int], path_info_by_state: dict[str, dict]) -> str:
    return min(states, key=lambda row: _state_selection_key(row["state_id"], objective_by_state, path_info_by_state))["state_id"]


def _chosen_path_rows(
    selected_state_id: str,
    parent_by_child: dict[str, dict],
    *,
    state_rows_by_id: dict[str, dict],
    state_input_by_id: dict[str, Path],
    objective_by_state: dict[str, int],
) -> list[dict]:
    rows = []
    current = selected_state_id
    while current != "S0000" and current in parent_by_child:
        edge = parent_by_child[current]
        rows.append(edge)
        current = edge["parent_state_id"]
    rows.reverse()
    chosen = []
    for index, edge in enumerate(rows):
        parent_id = edge["parent_state_id"]
        child_id = edge["child_state_id"]
        parent_state = state_rows_by_id.get(parent_id, {})
        child_state = state_rows_by_id.get(child_id, {})
        duplicate_of = edge.get("duplicate_of", "")
        is_duplicate = _is_true(edge.get("is_duplicate"))
        correctness = _correctness_for_edge(parent_state, edge)
        inst_before = _int(edge.get("ir_inst_before") or objective_by_state.get(parent_id))
        inst_after = _int(edge.get("ir_inst_after") or objective_by_state.get(child_id))
        chosen.append(
            {
                "step": str(index),
                "round": parent_state.get("depth", str(index)),
                "parent_state_id": parent_id,
                "parent_depth": parent_state.get("depth", ""),
                "parent_state_hash": parent_state.get("state_hash", edge.get("parent_hash", "")),
                "batch_id": edge["batch_id"],
                "batch_passes": edge["batch_passes"],
                "batch_size": edge.get("batch_size", ""),
                "canonical_order": edge["canonical_order"],
                "validation_status": edge["validation_status"],
                "correctness_class": edge["correctness_class"],
                "can_hard_fold": edge.get("can_hard_fold") or correctness.get("can_hard_fold", ""),
                "can_execute": edge.get("can_execute") or correctness.get("can_execute", ""),
                "child_state_id": child_id,
                "child_depth": child_state.get("depth", ""),
                "child_state_hash": child_state.get("state_hash", edge.get("child_hash", "")),
                "is_duplicate_transition": _bool(is_duplicate),
                "duplicate_of": duplicate_of,
                "parent_ir_path": str(state_input_by_id.get(parent_id, "")),
                "child_ir_path": str(_resolved_child_ir_path(child_id, is_duplicate, duplicate_of, state_input_by_id)),
                "parent_active_passes": parent_state.get("active_passes", ""),
                "child_active_passes": child_state.get("active_passes", ""),
                "parent_tested_pairs": parent_state.get("pairs_tested", ""),
                "child_tested_pairs": child_state.get("pairs_tested", ""),
                "parent_commute_pairs": parent_state.get("dynamic_commute", ""),
                "child_commute_pairs": child_state.get("dynamic_commute", ""),
                "parent_order_sensitive_pairs": parent_state.get("order_sensitive", ""),
                "child_order_sensitive_pairs": child_state.get("order_sensitive", ""),
                "parent_unknown_pairs": parent_state.get("unknown", ""),
                "child_unknown_pairs": child_state.get("unknown", ""),
                "ir_inst_before": str(inst_before),
                "ir_inst_after": str(inst_after),
                "ir_inst_delta": str(inst_after - inst_before),
                "ir_inst_reduction_pct": _format_pct(_reduction_pct(inst_before, inst_after)),
                "selection_reason": "selected_final_path",
            }
        )
    return chosen


def _correctness_for_edge(parent_state: dict, edge: dict) -> dict:
    parent_dir = Path(parent_state.get("state_dir", ""))
    batch_id = edge.get("batch_id", "")
    for row in _read_csv(parent_dir / "batch_correctness.csv"):
        if row.get("batch_id") == batch_id:
            return row
    return {}


def _resolved_child_ir_path(child_id: str, is_duplicate: bool, duplicate_of: str, state_input_by_id: dict[str, Path]) -> Path | str:
    target_id = duplicate_of if is_duplicate and duplicate_of else child_id
    return state_input_by_id.get(target_id, state_input_by_id.get(child_id, ""))


def _chosen_path_summary_row(program: str, selected_state_id: str, chosen_path_rows: list[dict], root_inst: int, final_inst: int) -> dict:
    passes: list[str] = []
    for row in chosen_path_rows:
        passes.extend(_split_order(row.get("canonical_order", "")))
    correctness_classes = {row.get("correctness_class", "") for row in chosen_path_rows}
    return {
        "program": program,
        "selected_final_state": selected_state_id,
        "path_steps": str(len(chosen_path_rows)),
        "total_pass_invocations": str(len(passes)),
        "unique_pass_types": str(len(set(passes))),
        "root_ir_inst_count": str(root_inst),
        "final_ir_inst_count": str(final_inst),
        "total_ir_inst_delta": str(final_inst - root_inst),
        "total_ir_inst_reduction_pct": _format_pct(_reduction_pct(root_inst, final_inst)),
        "all_batches_certified": _bool(all(row.get("correctness_class") == "certified_batch" for row in chosen_path_rows)),
        "any_sampled_batch": _bool("sampled_batch" in correctness_classes),
        "any_rejected_batch": _bool("rejected_batch" in correctness_classes),
        "any_unvalidated_batch": _bool(bool(correctness_classes & {"unvalidated_batch", "unknown_batch"})),
        "replay_verified": "false",
    }


def _write_optimized_pipeline_readable(path: Path, chosen_path_rows: list[dict], optimized_pipeline: str, optimized_pipeline_names: str) -> None:
    lines = ["# Flattened pipeline from selected certified batches"]
    for row in chosen_path_rows:
        lines.append(f"# Step {row['step']}: {row['batch_id']} from {row['parent_state_id']} to {row['child_state_id']}")
    lines.append("# Pass names:")
    lines.append(optimized_pipeline_names)
    lines.append("# Opt pipeline text:")
    lines.append(optimized_pipeline)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_state(path: Path, selected_state: dict, selected_ir: Path, final_objective: int) -> None:
    lines = [
        f"selected_final_state={selected_state.get('state_id', '')}",
        f"final_state_hash={selected_state.get('state_hash', '')}",
        f"final_ir_path={selected_ir}",
        f"final_objective={final_objective}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_path_artifacts(
    path: Path,
    selected_state_id: str,
    chosen_path_rows: list[dict],
    summary: dict,
    optimized_pipeline: str,
    optimized_pipeline_names: str,
) -> None:
    lines = [
        "# Chosen Path",
        "",
        "## Summary",
        "",
        f"- selected final state: {selected_state_id}",
        f"- path steps: {summary.get('path_steps', '')}",
        f"- total pass invocations: {summary.get('total_pass_invocations', '')}",
        f"- root IR instructions: {summary.get('root_ir_inst_count', '')}",
        f"- final IR instructions: {summary.get('final_ir_inst_count', '')}",
        f"- total reduction: {summary.get('total_ir_inst_reduction_pct', '')}%",
        "",
        "## Path Table",
        "",
        "| step | parent | batch | canonical order | validation | correctness | child | inst before | inst after | delta |",
        "|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    if chosen_path_rows:
        for row in chosen_path_rows:
            lines.append(
                "| {step} | {parent_state_id} | {batch_id} | {canonical_order} | {validation_status} | {correctness_class} | {child_state_id} | {ir_inst_before} | {ir_inst_after} | {ir_inst_delta} |".format(
                    **row
                )
            )
    else:
        lines.append("| - | S0000 | - | - | - | - | S0000 | {0} | {0} | 0 |".format(summary.get("root_ir_inst_count", "")))
    lines.extend(
        [
            "",
            "## Flattened Pipeline Names",
            "",
            optimized_pipeline_names or "(root state selected; no batch pipeline)",
            "",
            "## Flattened Opt Pipeline",
            "",
            optimized_pipeline or "(root state selected; no batch pipeline)",
            "",
            "## Correctness Note",
            "",
            "Every hard-folded batch in the chosen path must be supported by batch correctness evidence. Objective values are reported only for path selection and evaluation; they are not used as commutation proof.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flatten_pipeline_names(chosen_path_rows: list[dict]) -> str:
    passes: list[str] = []
    for row in chosen_path_rows:
        passes.extend(_split_order(row.get("canonical_order", "")))
    return ",".join(passes)


def _flatten_pipeline(chosen_path_rows: list[dict], pass_registry: PassRegistry | None = None) -> str:
    passes: list[str] = []
    for row in chosen_path_rows:
        passes.extend(_split_order(row.get("canonical_order", "")))
    return ",".join(resolve_pipeline_sequence(passes, pass_registry))


def _write_optimized_batches(path: Path, chosen_path_rows: list[dict]) -> None:
    if not chosen_path_rows:
        path.write_text("# No batch selected; root state is final.\n", encoding="utf-8")
        return
    lines = []
    for index, row in enumerate(chosen_path_rows):
        lines.extend(
            [
                f"Round {index}:",
                f"  parent_state: {row['parent_state_id']}",
                f"  parent_depth: {row['parent_depth']}",
                f"  batch_id: {row['batch_id']}",
                f"  batch_passes: {row['batch_passes']}",
                f"  canonical_order: {row['canonical_order']}",
                f"  validation_status: {row['validation_status']}",
                f"  correctness_class: {row['correctness_class']}",
                f"  can_hard_fold: {row['can_hard_fold']}",
                f"  can_execute: {row['can_execute']}",
                f"  objective_before: {row['ir_inst_before']}",
                f"  objective_after: {row['ir_inst_after']}",
                f"  objective_delta: {row['ir_inst_delta']}",
                f"  child_state: {row['child_state_id']}",
                f"  child_depth: {row['child_depth']}",
                f"  duplicate: {row['is_duplicate_transition']}",
                f"  duplicate_of: {row['duplicate_of']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    input_path: Path,
    requested_mode: str,
    mode: str,
    auto_reason: str,
    objective: str,
    max_rounds: int,
    beam_width: int,
    max_states: int,
    max_batches_per_state: int,
    batch_frontier_policy: str,
    batch_selection_policy: str,
    frontier_selection_policy: str,
    selection_seed: int,
    states: list[dict],
    transitions: list[dict],
    duplicate_transitions: int,
    leaf_rows: list[dict],
    chosen_path_rows: list[dict],
    selected_state_id: str,
    root_objective: int,
    final_objective: int,
    optimized_pipeline: str,
    exact_status: str,
    exact_reasons: list[str],
    budget_exhausted: bool,
    stop_reason: str,
) -> None:
    duplicates = sum(1 for row in states if row.get("is_duplicate") == "true")
    selected_state = next((row for row in states if row.get("state_id") == selected_state_id), {})
    lines = [
        "# Optimize Batches Summary",
        "",
        f"- input: {input_path}",
        f"- requested_mode: {requested_mode}",
        f"- selected_mode: {mode}",
        f"- mode: {mode}",
        f"- auto_reason: {auto_reason}",
        f"- objective: {objective}",
        f"- exact_status: {exact_status}",
        f"- exact_incomplete_reasons: {';'.join(exact_reasons)}",
        f"- max_rounds: {max_rounds}",
        f"- beam_width: {beam_width}",
        f"- max_states: {max_states}",
        f"- max_batches_per_state: {max_batches_per_state}",
        f"- batch_frontier_policy: {batch_frontier_policy}",
        f"- batch_selection_policy: {batch_selection_policy}",
        f"- frontier_selection_policy: {frontier_selection_policy}",
        f"- selection_seed: {selection_seed}",
        f"- states_reached: {len(states)}",
        f"- states generated: {len(states)}",
        f"- unique states: {len(states) - duplicates}",
        f"- duplicate states: {duplicates}",
        f"- transitions: {len(transitions)}",
        f"- batch transitions: {len(transitions)}",
        f"- duplicate_transitions: {duplicate_transitions}",
        f"- budget_exhausted: {_bool(budget_exhausted)}",
        f"- stop_reason: {stop_reason}",
        f"- leaf_states: {sum(1 for row in leaf_rows if row.get('is_leaf') == 'true')}",
        f"- incumbent_state: {selected_state_id}",
        f"- incumbent_objective: {final_objective}",
        f"- incumbent_depth: {selected_state.get('depth', '')}",
        f"- selected final state: {selected_state_id}",
        f"- final objective value: {final_objective}",
        f"- root objective value: {root_objective}",
        f"- objective delta: {final_objective - root_objective}",
        f"- optimized pipeline: {optimized_pipeline}",
        '- note: "Objective is used only for path selection, not as commutation proof."',
        "",
        "## Mode",
        "",
        f"- requested mode: {requested_mode}",
        f"- selected mode: {mode}",
        f"- mode reason: {auto_reason or 'explicit command-line mode'}",
        "",
        "## Search Bounds",
        "",
        f"- max_rounds: {max_rounds}",
        f"- beam_width: {beam_width}",
        f"- max_states: {max_states}",
        f"- max_batches_per_state: {max_batches_per_state}",
        "",
        "## Batch Selection Policy",
        "",
        "- Certified batches are executable by default when validation proves canonical-IR equality.",
        "- Sampled batches execute only when explicitly allowed; sampled evidence is heuristic, not a hard certificate.",
        "- Rejected, failed, and unvalidated batches are not executed by the optimizer.",
        "- Budgeted score mode combines coverage, batch size, estimated reduction, evidence, diversity, and risk.",
        "- Batch selection and frontier selection are separate policies; batch_frontier_policy is a compatibility alias.",
        "",
        "## Frontier Selection Policy",
        "",
        "- objective_score ranks IR instruction-count improvement from the root state.",
        "- future_potential_score estimates remaining useful search space from active passes and certified child batches.",
        "- evidence_quality_score reflects certified, heuristic, unknown, and dropped pass coverage.",
        "- novelty_score preserves distinct active-pass or batch signatures in the frontier.",
        "- cost_score and risk_penalty discourage expensive or uncertain states.",
        "- Pareto filtering keeps non-dominated states before beam selection.",
        "- Score frontier mode uses explicit score, novelty, and objective buckets, targeting a 70/20/10 style beam composition.",
        "",
        "## Incumbent Path",
        "",
        "| step | parent | batch | child | validation | correctness | delta | duplicate |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    if chosen_path_rows:
        for row in chosen_path_rows:
            lines.append(
                "| {step} | {parent_state_id} | {batch_id} | {child_state_id} | {validation_status} | {correctness_class} | {ir_inst_delta} | {is_duplicate_transition} |".format(
                    **row
                )
            )
    else:
        lines.append("| - | S0000 | - | S0000 | - | - | 0 | false |")
    lines.extend(
        [
            "",
            "## Final Pipeline",
            "",
            optimized_pipeline or "(root state selected; no batch pipeline)",
            "",
            "## Correctness Boundary",
            "",
            "Batch correctness is based on certified canonical-IR equality or explicit validation status. Objective scores are used only for search ranking and final path selection; they are not used as commutation or independence proof.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_exact_status(path: Path, status: str, reasons: list[str]) -> None:
    lines = [status]
    if reasons:
        lines.append("reasons: " + ";".join(reasons))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_state_input(state_dir: Path, source_ir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    input_ll = state_dir / "input.ll"
    if source_ir.resolve() != input_ll.resolve():
        shutil.copyfile(source_ir, input_ll)
    return input_ll


def _successor_artifact(state_dir: Path, batch_id: str) -> Path:
    path = state_dir / "artifacts" / "batch_successors" / f"{batch_id}.ll"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _root_path_info() -> dict:
    return {"path_length": 0, "pass_invocations": 0, "certified_steps": 0, "batch_sequence": []}


def _extend_path_info(parent: dict, candidate: dict, correctness: dict, order: list[str]) -> dict:
    return {
        "path_length": parent["path_length"] + 1,
        "pass_invocations": parent["pass_invocations"] + len(order),
        "certified_steps": parent["certified_steps"] + (1 if correctness.get("correctness_class") == "certified_batch" else 0),
        "batch_sequence": [*parent["batch_sequence"], candidate.get("batch_id", "")],
    }


def _path_is_better(candidate: dict, current: dict) -> bool:
    return (
        candidate["path_length"],
        candidate["pass_invocations"],
        tuple(candidate["batch_sequence"]),
    ) < (
        current["path_length"],
        current["pass_invocations"],
        tuple(current["batch_sequence"]),
    )


def _certified_ratio(path_info: dict) -> float:
    length = path_info["path_length"]
    if length == 0:
        return 1.0
    return path_info["certified_steps"] / length


def _edge_for_path(transition: dict, canonical_order: str, inst_before: int, inst_after: int) -> dict:
    return {
        **transition,
        "canonical_order": canonical_order,
        "ir_inst_before": str(inst_before),
        "ir_inst_after": str(inst_after),
        "ir_inst_delta": str(inst_after - inst_before),
    }


def _exact_status(reasons: list[str], *, continued: bool) -> str:
    if not reasons:
        return "exact_complete"
    return "exact_incomplete_continued" if continued else "exact_incomplete"


def _tool_paths(metadata: dict) -> dict[str, str]:
    return {
        name: details["path"]
        for name, details in metadata.get("tools", {}).items()
        if details.get("path")
    }


def _split_order(value: str | None) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


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


def _add_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _int(value: object) -> int:
    try:
        return int(str(value or "0"))
    except ValueError:
        return 0


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _format_score(value: float) -> str:
    return f"{_clamp(value):.4f}"


def _reduction_pct(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return ((before - after) / before) * 100.0


def _format_pct(value: float) -> str:
    return f"{value:.2f}"


def _stable_hash_int(*parts: str) -> int:
    text = "\0".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"
