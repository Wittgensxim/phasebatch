from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path

from .ir_equivalence import EqualityResult, compare_ir_equivalence, safe_canonical_hash as hash_ir
from .pass_config import PassRegistry, resolve_pipeline_sequence
from .runner import materialize_run_result, run_opt, run_opt_from_result, worker_handles_enabled
from .schema import BATCH_VALIDATION_DAG_SUMMARY_FIELDS, RunResult
from .validation_runtime import ValidationRuntime, ValidationTransition, ValidationTransitionKey


EQUIVALENCE_CACHE_VERSION = "ir_equivalence_v1"


class _DagTransitionFailed(RuntimeError):
    pass

VALIDATION_DAG_NODE_FIELDS = [
    "batch_id",
    "node_id",
    "subset_mask",
    "subset_passes",
    "depth",
    "state_hash",
    "representative_path",
    "is_final",
    "merged_into",
]

VALIDATION_DAG_EDGE_FIELDS = [
    "batch_id",
    "source_node",
    "target_node",
    "source_subset",
    "target_subset",
    "applied_pass",
    "output_hash",
    "transition_cache_hit",
    "equality_tier",
    "equality_reason",
]


@dataclass
class DagNode:
    node_id: str
    subset_mask: int
    ir_path: Path
    canonical_hash: str
    representative_path: tuple[str, ...]
    depth: int
    merged_into: str = ""
    run_result: RunResult | None = None


@dataclass
class DagMetrics:
    nodes: int = 0
    edges: int = 0
    transition_cache_hits: int = 0
    transition_cache_misses: int = 0
    equivalence_cache_hits: int = 0
    equivalence_cache_misses: int = 0
    hash_merges: int = 0
    structural_merges: int = 0
    equality_failed_count: int = 0
    profile_reuse_hits: int = 0
    state_transition_cache_hits: int = 0
    state_equivalence_cache_hits: int = 0
    materializations: int = 0
    materializations_avoided: int = 0


@dataclass(frozen=True)
class _TransitionSpec:
    source: DagNode
    pass_name: str
    pass_pipeline: tuple[str, ...]
    cache_key: ValidationTransitionKey
    new_subset: int


@dataclass(frozen=True)
class _TransitionOutcome:
    spec: _TransitionSpec
    transition: ValidationTransition | None
    cache_hit: bool
    failed: bool


def validate_batch_with_permutation_dag(
    state_ir_path: Path,
    batch_pass_names: list[str],
    canonical_order: list[str],
    pass_registry_or_mapping: PassRegistry | None,
    tools: dict,
    out_dir: Path,
    max_nodes: int,
    max_edges: int,
    dump_dag: bool = False,
    timeout: int = 10,
    batch_id: str = "",
    program: str = "",
    state_id: str = "",
    state_hash: str = "",
    validation_mode: str = "dag",
    runtime: ValidationRuntime | None = None,
    jobs: int = 1,
    keep_ir_artifacts: bool = False,
) -> dict:
    del batch_pass_names
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_ir_artifacts = keep_ir_artifacts or bool(tools.get("_keep_ir_artifacts"))
    if dump_dag or keep_ir_artifacts:
        reason = "validation DAG dump requested" if dump_dag else "validation DAG IR retained"
        (out_dir / ".keep_ir_artifacts").write_text(f"{reason}\n", encoding="utf-8")

    start = _now()
    jobs = max(1, jobs)
    owns_runtime = runtime is None
    runtime = runtime or ValidationRuntime(out_dir, max_workers=jobs)

    def finish(row: dict) -> dict:
        if owns_runtime:
            runtime.close(timeout=timeout)
        return row

    defer_materialization = (
        worker_handles_enabled()
        and not dump_dag
        and not keep_ir_artifacts
    )
    passes = list(canonical_order)
    pass_index = {pass_name: index for index, pass_name in enumerate(passes)}
    factorial_log10 = _factorial_log10(len(passes))
    factorial_text = _factorial_text(len(passes))
    metrics = DagMetrics(nodes=1)
    node_counter = 1
    nodes: list[DagNode] = [
        DagNode(
            node_id="N0000",
            subset_mask=0,
            ir_path=Path(state_ir_path),
            canonical_hash=hash_ir(Path(state_ir_path)),
            representative_path=(),
            depth=0,
        )
    ]
    nodes_by_subset: dict[int, list[DagNode]] = defaultdict(list)
    nodes_by_subset[0].append(nodes[0])
    edge_rows: list[dict] = []
    full_mask = (1 << len(passes)) - 1

    if metrics.nodes > max_nodes:
        return finish(_result_row(
            program,
            state_id,
            state_hash,
            batch_id,
            passes,
            validation_mode,
            "incomplete",
            "permutation_dag_incomplete",
            False,
            False,
            "max_validation_dag_nodes",
            nodes_by_subset,
            full_mask,
            metrics,
            factorial_text,
            factorial_log10,
            start,
            budget_exceeded=True,
        ))

    for depth in range(len(passes)):
        current_nodes = sorted(
            (node for node in nodes if node.depth == depth),
            key=lambda node: node.node_id,
        )
        transition_specs: list[_TransitionSpec] = []
        for node in current_nodes:
            for pass_name in passes:
                bit = 1 << pass_index[pass_name]
                if node.subset_mask & bit:
                    continue
                pass_pipeline = tuple(resolve_pipeline_sequence([pass_name], pass_registry_or_mapping))
                transition_specs.append(
                    _TransitionSpec(
                        source=node,
                        pass_name=pass_name,
                        pass_pipeline=pass_pipeline,
                        cache_key=ValidationTransitionKey(
                            node.canonical_hash,
                            pass_name,
                            ",".join(pass_pipeline),
                        ),
                        new_subset=node.subset_mask | bit,
                    )
                )

        remaining_edges = max(0, max_edges - metrics.edges)
        selected_specs = transition_specs[:remaining_edges]
        edge_budget_exceeded = len(selected_specs) < len(transition_specs)
        metrics.edges += len(selected_specs)
        if jobs == 1 or len(selected_specs) <= 1:
            outcomes = [
                _run_transition_spec(spec, runtime, tools, timeout, defer_materialization)
                for spec in selected_specs
            ]
        else:
            with ThreadPoolExecutor(max_workers=min(jobs, len(selected_specs))) as executor:
                outcomes = list(
                    executor.map(
                        lambda spec: _run_transition_spec(
                            spec,
                            runtime,
                            tools,
                            timeout,
                            defer_materialization,
                        ),
                        selected_specs,
                    )
                )

        for outcome in outcomes:
            if outcome.cache_hit:
                metrics.transition_cache_hits += 1
                metrics.state_transition_cache_hits += 1
                if outcome.transition is not None and outcome.transition.source == "profile":
                    metrics.profile_reuse_hits += 1
            else:
                metrics.transition_cache_misses += 1
                if outcome.transition is not None and outcome.transition.run_result is not None:
                    if outcome.transition.run_result.materialized:
                        metrics.materializations += 1
                    else:
                        metrics.materializations_avoided += 1

        for outcome in outcomes:
            spec = outcome.spec
            if outcome.failed or outcome.transition is None:
                failure_reason = f"validation_dag_opt_failed:{spec.pass_name}"
                row = _result_row(
                    program,
                    state_id,
                    state_hash,
                    batch_id,
                    passes,
                    validation_mode,
                    "failed",
                    "failed",
                    False,
                    False,
                    "validation_dag_opt_failed",
                    nodes_by_subset,
                    full_mask,
                    metrics,
                    factorial_text,
                    factorial_log10,
                    start,
                    equality_tier="failed",
                    equality_reason=failure_reason,
                )
                _write_optional_dump(out_dir, batch_id, passes, nodes, edge_rows, dump_dag, full_mask)
                return finish(row)

            transition = outcome.transition
            out_ir = transition.ir_path
            out_hash = transition.canonical_hash
            target, equality_tier, equality_reason, failed = _find_or_create_node(
                out_ir=out_ir,
                out_hash=out_hash,
                out_result=transition.run_result,
                new_subset=spec.new_subset,
                pass_name=spec.pass_name,
                source=spec.source,
                nodes_by_subset=nodes_by_subset,
                nodes=nodes,
                node_counter=node_counter,
                metrics=metrics,
                runtime=runtime,
                tools=tools,
                timeout=timeout,
            )
            if failed:
                row = _result_row(
                    program,
                    state_id,
                    state_hash,
                    batch_id,
                    passes,
                    validation_mode,
                    "failed",
                    "failed",
                    False,
                    False,
                    equality_reason or "validation_dag_equivalence_failed",
                    nodes_by_subset,
                    full_mask,
                    metrics,
                    factorial_text,
                    factorial_log10,
                    start,
                    equality_tier="failed",
                    equality_reason=equality_reason or "validation_dag_equivalence_failed",
                )
                _write_optional_dump(out_dir, batch_id, passes, nodes, edge_rows, dump_dag, full_mask)
                return finish(row)
            if target.node_id == f"N{node_counter:04d}":
                node_counter += 1
                metrics.nodes = len(nodes)
                if metrics.nodes > max_nodes:
                    edge_rows.append(
                        _edge_row(
                            batch_id,
                            spec.source,
                            target,
                            spec.new_subset,
                            spec.pass_name,
                            out_hash,
                            outcome.cache_hit,
                            equality_tier,
                            equality_reason,
                        )
                    )
                    row = _result_row(
                        program,
                        state_id,
                        state_hash,
                        batch_id,
                        passes,
                        validation_mode,
                        "incomplete",
                        "permutation_dag_incomplete",
                        False,
                        False,
                        "max_validation_dag_nodes",
                        nodes_by_subset,
                        full_mask,
                        metrics,
                        factorial_text,
                        factorial_log10,
                        start,
                        budget_exceeded=True,
                    )
                    _write_optional_dump(out_dir, batch_id, passes, nodes, edge_rows, dump_dag, full_mask)
                    return finish(row)
            edge_rows.append(
                _edge_row(
                    batch_id,
                    spec.source,
                    target,
                    spec.new_subset,
                    spec.pass_name,
                    out_hash,
                    outcome.cache_hit,
                    equality_tier,
                    equality_reason,
                )
            )

        if edge_budget_exceeded:
            metrics.edges += 1
            row = _result_row(
                program,
                state_id,
                state_hash,
                batch_id,
                passes,
                validation_mode,
                "incomplete",
                "permutation_dag_incomplete",
                False,
                False,
                "max_validation_dag_edges",
                nodes_by_subset,
                full_mask,
                metrics,
                factorial_text,
                factorial_log10,
                start,
                budget_exceeded=True,
            )
            _write_optional_dump(out_dir, batch_id, passes, nodes, edge_rows, dump_dag, full_mask)
            return finish(row)

    final_nodes = nodes_by_subset.get(full_mask, [])
    final_classes = len(final_nodes)
    if final_classes == 1:
        equality_tier = "structural_diff" if metrics.structural_merges else "canonical_hash"
        equality_reason = "structural_merges_present" if metrics.structural_merges else "all_full_subset_paths_merged"
        status = "all_permutations_same"
        tier = "permutation_dag_exact"
        hard_certificate = True
    else:
        equality_tier = "different"
        equality_reason = "multiple_final_equivalence_classes"
        status = "mismatch"
        tier = "permutation_dag_mismatch"
        hard_certificate = False

    row = _result_row(
        program,
        state_id,
        state_hash,
        batch_id,
        passes,
        validation_mode,
        status,
        tier,
        True,
        hard_certificate,
        "",
        nodes_by_subset,
        full_mask,
        metrics,
        factorial_text,
        factorial_log10,
        start,
        equality_tier=equality_tier,
        equality_reason=equality_reason,
    )
    _write_optional_dump(out_dir, batch_id, passes, nodes, edge_rows, dump_dag, full_mask)
    return finish(row)


def _run_transition_spec(
    spec: _TransitionSpec,
    runtime: ValidationRuntime,
    tools: dict,
    timeout: int,
    defer_materialization: bool,
) -> _TransitionOutcome:
    computed = False

    def compute_transition(output_path: Path) -> ValidationTransition:
        nonlocal computed
        computed = True
        try:
            if (
                defer_materialization
                and spec.source.run_result is not None
                and spec.source.run_result.backend == "worker"
                and spec.source.run_result.module_handle
            ):
                result = runtime.run_with_opt_slot(
                    lambda: run_opt_from_result(
                        spec.source.run_result,
                        list(spec.pass_pipeline),
                        output_path,
                        timeout,
                        materialize=False,
                    )
                )
            elif defer_materialization:
                result = runtime.run_with_opt_slot(
                    lambda: run_opt(
                        str(tools["opt"]),
                        spec.source.ir_path,
                        list(spec.pass_pipeline),
                        output_path,
                        timeout,
                        materialize=False,
                    )
                )
            else:
                result = runtime.run_with_opt_slot(
                    lambda: run_opt(
                        str(tools["opt"]),
                        spec.source.ir_path,
                        list(spec.pass_pipeline),
                        output_path,
                        timeout,
                    )
                )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _DagTransitionFailed(str(exc)) from exc
        if not result.success or (result.materialized and not output_path.exists()):
            raise _DagTransitionFailed("validation_dag_opt_failed")
        output_hash = result.canonical_hash or hash_ir(output_path)
        return ValidationTransition(output_path, output_hash, "computed", result)

    try:
        transition = runtime.get_or_compute_transition(spec.cache_key, compute_transition)
    except _DagTransitionFailed:
        return _TransitionOutcome(spec, None, not computed, True)
    return _TransitionOutcome(spec, transition, not computed, False)


def write_batch_validation_dag_summary(state_dir: Path, validation_rows: list[dict]) -> Path:
    state_dir = Path(state_dir)
    rows = []
    for row in validation_rows:
        if not _is_dag_row(row):
            continue
        rows.append(
            {
                "program": row.get("program", ""),
                "state_id": row.get("state_id", ""),
                "state_hash": row.get("state_hash", ""),
                "batch_id": row.get("batch_id", ""),
                "batch_size": row.get("batch_size", ""),
                "factorial_permutations_log10": row.get("factorial_permutations_log10", ""),
                "validation_tier": row.get("validation_tier", ""),
                "validation_status": row.get("validation_status", ""),
                "validation_complete": row.get("validation_complete", ""),
                "validation_hard_certificate": row.get("validation_hard_certificate", ""),
                "dag_nodes": row.get("validation_dag_nodes", ""),
                "dag_edges": row.get("validation_dag_edges", ""),
                "final_equivalence_classes": row.get("validation_dag_final_classes", ""),
                "hash_merges": row.get("validation_dag_hash_merges", ""),
                "structural_merges": row.get("validation_dag_structural_merges", ""),
                "transition_cache_hits": row.get("validation_dag_transition_cache_hits", ""),
                "equivalence_cache_hits": row.get("validation_dag_equivalence_cache_hits", ""),
                "materializations": row.get("validation_materializations", ""),
                "materializations_avoided": row.get("validation_materializations_avoided", ""),
                "compression_vs_permutation": row.get("compression_vs_permutation", ""),
                "time_ms": row.get("time_ms", ""),
            }
        )
    path = state_dir / "batch_validation_dag_summary.csv"
    _write_csv(path, BATCH_VALIDATION_DAG_SUMMARY_FIELDS, rows)
    return path


def _find_or_create_node(
    *,
    out_ir: Path,
    out_hash: str,
    out_result: RunResult | None,
    new_subset: int,
    pass_name: str,
    source: DagNode,
    nodes_by_subset: dict[int, list[DagNode]],
    nodes: list[DagNode],
    node_counter: int,
    metrics: DagMetrics,
    runtime: ValidationRuntime,
    tools: dict,
    timeout: int,
) -> tuple[DagNode, str, str, bool]:
    for existing in nodes_by_subset.get(new_subset, []):
        if existing.canonical_hash == out_hash:
            metrics.hash_merges += 1
            return existing, "canonical_hash", "hash_equal", False

    for existing in nodes_by_subset.get(new_subset, []):
        try:
            if out_result is not None and not out_result.materialized:
                materialize_run_result(out_result, out_ir, timeout=timeout)
                metrics.materializations += 1
                metrics.materializations_avoided = max(0, metrics.materializations_avoided - 1)
            if existing.run_result is not None and not existing.run_result.materialized:
                materialize_run_result(existing.run_result, existing.ir_path, timeout=timeout)
                metrics.materializations += 1
                metrics.materializations_avoided = max(0, metrics.materializations_avoided - 1)
        except (OSError, RuntimeError, ValueError) as exc:
            metrics.equality_failed_count += 1
            return existing, "failed", f"materialize_failed:{exc}", True
        key = _equivalence_cache_key(out_hash, existing.canonical_hash)
        computed = False

        def compute_equivalence() -> EqualityResult:
            nonlocal computed
            computed = True
            return compare_ir_equivalence(out_ir, existing.ir_path, tools=tools, timeout=timeout)

        equality = runtime.get_or_compute_equivalence(key, compute_equivalence)
        if computed:
            metrics.equivalence_cache_misses += 1
        else:
            metrics.equivalence_cache_hits += 1
            metrics.state_equivalence_cache_hits += 1
        if equality.tier == "failed":
            metrics.equality_failed_count += 1
            return existing, "failed", equality.reason or "tool_failed", True
        if equality.can_hard_fold:
            metrics.structural_merges += 1
            return existing, equality.tier, equality.reason, False

    new_node = DagNode(
        node_id=f"N{node_counter:04d}",
        subset_mask=new_subset,
        ir_path=out_ir,
        canonical_hash=out_hash,
        representative_path=(*source.representative_path, pass_name),
        depth=source.depth + 1,
        run_result=out_result,
    )
    nodes.append(new_node)
    nodes_by_subset[new_subset].append(new_node)
    return new_node, "new_class", "new_equivalence_class", False


def _result_row(
    program: str,
    state_id: str,
    state_hash: str,
    batch_id: str,
    passes: list[str],
    validation_mode: str,
    validation_status: str,
    validation_tier: str,
    validation_complete: bool,
    hard_certificate: bool,
    incomplete_reason: str,
    nodes_by_subset: dict[int, list[DagNode]],
    full_mask: int,
    metrics: DagMetrics,
    factorial_text: str,
    factorial_log10: float,
    start: float,
    *,
    budget_exceeded: bool = False,
    equality_tier: str = "",
    equality_reason: str = "",
) -> dict:
    final_nodes = sorted(nodes_by_subset.get(full_mask, []), key=lambda node: node.node_id)
    final_classes = [
        {
            "class_id": f"F{index:04d}",
            "representative_node_id": node.node_id,
            "representative_path": list(node.representative_path),
            "final_hash": node.canonical_hash,
        }
        for index, node in enumerate(final_nodes)
    ]
    first_final = final_nodes[0] if final_nodes else None
    first_mismatch = final_nodes[1] if len(final_nodes) > 1 else None
    different_count = max(0, len(final_nodes) - 1) if validation_status == "mismatch" else 0
    hash_equal_count = metrics.hash_merges
    structural_equal_count = metrics.structural_merges
    transition_work = metrics.transition_cache_hits + metrics.transition_cache_misses
    return {
        "program": program,
        "state_id": state_id,
        "state_hash": state_hash,
        "batch_id": batch_id,
        "batch_size": str(len(passes)),
        "canonical_order": ";".join(passes),
        "validation_mode": validation_mode,
        "validation_tier": validation_tier,
        "validation_sequences_tested": str(metrics.transition_cache_misses),
        "validation_sequences_total_estimate": factorial_text,
        "validation_complete": _bool(validation_complete),
        "validation_hard_certificate": _bool(hard_certificate),
        "validation_incomplete_reason": incomplete_reason,
        "tested_orders": str(metrics.transition_cache_misses),
        "same_hash_count": str(hash_equal_count),
        "different_hash_count": str(different_count),
        "hash_equal_count": str(hash_equal_count),
        "structural_equal_count": str(structural_equal_count),
        "different_count": str(different_count),
        "canonical_hash_equal_count": str(hash_equal_count),
        "structural_diff_equal_count": str(structural_equal_count),
        "equality_failed_count": str(metrics.equality_failed_count),
        "validation_equality_tier": equality_tier,
        "validation_equality_reason": equality_reason,
        "validation_status": validation_status,
        "canonical_hash": first_final.canonical_hash if first_final else "",
        "first_mismatch_order": ";".join(first_mismatch.representative_path) if first_mismatch else "",
        "first_mismatch_hash": first_mismatch.canonical_hash if first_mismatch else "",
        "validation_dag_nodes": str(metrics.nodes),
        "validation_dag_edges": str(metrics.edges),
        "validation_dag_final_classes": str(len(final_nodes)),
        "validation_dag_final_classes_json": json.dumps(final_classes, separators=(",", ":"), ensure_ascii=True),
        "validation_dag_transition_cache_hits": str(metrics.transition_cache_hits),
        "validation_dag_transition_cache_misses": str(metrics.transition_cache_misses),
        "validation_dag_equivalence_cache_hits": str(metrics.equivalence_cache_hits),
        "validation_dag_equivalence_cache_misses": str(metrics.equivalence_cache_misses),
        "validation_dag_hash_merges": str(metrics.hash_merges),
        "validation_dag_structural_merges": str(metrics.structural_merges),
        "validation_materializations": str(metrics.materializations),
        "validation_materializations_avoided": str(metrics.materializations_avoided),
        "validation_dag_budget_exceeded": _bool(budget_exceeded),
        "validation_dag_incomplete_reason": incomplete_reason if validation_tier == "permutation_dag_incomplete" else "",
        "factorial_permutations": factorial_text,
        "factorial_permutations_log10": f"{factorial_log10:.6f}",
        "compression_vs_permutation": _compression_vs_permutation(factorial_text, factorial_log10, metrics.edges),
        "validation_opt_invocations": str(metrics.transition_cache_misses),
        "validation_pass_invocations_baseline": str(transition_work),
        "validation_pass_invocations_actual": str(metrics.transition_cache_misses),
        "validation_pass_invocations_saved": str(metrics.transition_cache_hits),
        "validation_profile_reuse_hits": str(metrics.profile_reuse_hits),
        "validation_state_transition_cache_hits": str(metrics.state_transition_cache_hits),
        "validation_state_equivalence_cache_hits": str(metrics.state_equivalence_cache_hits),
        "time_ms": f"{(_now() - start) * 1000:.2f}",
        "final_classes": final_classes,
    }


def _write_optional_dump(
    out_dir: Path,
    batch_id: str,
    passes: list[str],
    nodes: list[DagNode],
    edges: list[dict],
    dump_dag: bool,
    full_mask: int,
) -> None:
    if not dump_dag:
        return
    node_rows = [_node_row(batch_id, node, passes, full_mask) for node in sorted(nodes, key=lambda item: item.node_id)]
    _write_csv(out_dir / f"validation_dag_{_safe_name(batch_id)}_nodes.csv", VALIDATION_DAG_NODE_FIELDS, node_rows)
    _write_csv(out_dir / f"validation_dag_{_safe_name(batch_id)}_edges.csv", VALIDATION_DAG_EDGE_FIELDS, edges)
    _write_dot(out_dir / f"validation_dag_{_safe_name(batch_id)}.dot", nodes, edges, passes, full_mask)


def _node_row(batch_id: str, node: DagNode, passes: list[str], full_mask: int) -> dict:
    return {
        "batch_id": batch_id,
        "node_id": node.node_id,
        "subset_mask": str(node.subset_mask),
        "subset_passes": _subset_passes(node.subset_mask, passes),
        "depth": str(node.depth),
        "state_hash": node.canonical_hash,
        "representative_path": ";".join(node.representative_path),
        "is_final": _bool(node.subset_mask == full_mask),
        "merged_into": node.merged_into,
    }


def _edge_row(
    batch_id: str,
    source: DagNode,
    target: DagNode,
    target_subset: int,
    pass_name: str,
    output_hash: str,
    transition_cache_hit: bool,
    equality_tier: str,
    equality_reason: str,
) -> dict:
    return {
        "batch_id": batch_id,
        "source_node": source.node_id,
        "target_node": target.node_id,
        "source_subset": str(source.subset_mask),
        "target_subset": str(target_subset),
        "applied_pass": pass_name,
        "output_hash": output_hash,
        "transition_cache_hit": _bool(transition_cache_hit),
        "equality_tier": equality_tier,
        "equality_reason": equality_reason,
    }


def _write_dot(path: Path, nodes: list[DagNode], edges: list[dict], passes: list[str], full_mask: int) -> None:
    lines = ["digraph validation_dag {", "  rankdir=LR;"]
    depths = sorted({node.depth for node in nodes})
    for depth in depths:
        lines.append(f"  subgraph depth_{depth} {{")
        lines.append("    rank=same;")
        for node in sorted((item for item in nodes if item.depth == depth), key=lambda item: item.node_id):
            label = f"{node.node_id}\\n{_subset_passes(node.subset_mask, passes)}\\n{node.canonical_hash[:8]}"
            attrs = [f'label="{_dot_escape(label)}"']
            if node.subset_mask == full_mask:
                attrs.extend(["shape=doublecircle", "style=filled", "fillcolor=\"#d9ead3\""])
            lines.append(f"    {node.node_id} [{', '.join(attrs)}];")
        lines.append("  }")
    for edge in edges:
        label = edge.get("applied_pass", "")
        if edge.get("transition_cache_hit") == "true":
            label += " cache"
        lines.append(f"  {edge['source_node']} -> {edge['target_node']} [label=\"{_dot_escape(label)}\"];")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subset_passes(mask: int, passes: list[str]) -> str:
    return ";".join(pass_name for index, pass_name in enumerate(passes) if mask & (1 << index))


def _equivalence_cache_key(hash_a: str, hash_b: str) -> tuple[str, str, str]:
    first, second = sorted([hash_a, hash_b])
    return first, second, EQUIVALENCE_CACHE_VERSION


def _is_dag_row(row: dict) -> bool:
    tier = row.get("validation_tier", "")
    return tier.startswith("permutation_dag_") or _to_int(row.get("validation_dag_nodes")) > 0


def _factorial_text(size: int) -> str:
    if size <= 50:
        return str(math.factorial(size))
    return ""


def _factorial_log10(size: int) -> float:
    return sum(math.log10(value) for value in range(1, size + 1))


def _compression_vs_permutation(factorial_text: str, factorial_log10: float, edges: int) -> str:
    denominator = max(1, edges)
    if factorial_text:
        return f"{int(factorial_text) / denominator:.6g}"
    return f"1e{factorial_log10 - math.log10(denominator):.6f}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in str(value)) or "batch"


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _to_int(value: object) -> int:
    try:
        return int(float(str(value).strip() or "0"))
    except ValueError:
        return 0


def _now() -> float:
    import time

    return time.perf_counter()
