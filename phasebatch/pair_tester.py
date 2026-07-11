from __future__ import annotations

import csv
import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ir_equivalence import EqualityResult, compare_ir_equivalence
from .normalizer import canonical_hash, count_ir_features
from .pair_cache import PairRelationCache
from .pass_config import PassRegistry
from .runner import materialize_run_result, release_run_result, run_opt, worker_handles_enabled
from .schema import PAIR_RELATION_FIELDS, RunResult


def run_pair_tests(
    input_ll: Path,
    active_profiles: list[dict],
    tools: dict,
    out_dir: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    pass_registry: PassRegistry | None = None,
    pair_cache: PairRelationCache | None = None,
    pair_testing_mode: str = "full",
    pair_test_budget_per_state: int = 0,
    pair_priority_policy: str = "mixed",
    write_output: bool = True,
    keep_ir_artifacts: bool = False,
) -> list[dict]:
    out_dir = Path(out_dir)
    pair_dir = out_dir / "artifacts" / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    active_profiles = [row for row in active_profiles if _is_true(row.get("active"))]
    profiles = {row["pass"]: row for row in active_profiles}
    mode = pair_testing_mode if pair_testing_mode in {"full", "lazy"} else "full"
    policy = pair_priority_policy if pair_priority_policy in {"default", "history", "effect-size", "mixed"} else "mixed"
    history = _resolve_pair_history(tools)
    if mode == "lazy":
        scored_pairs = _prioritized_pairs(active_profiles, history, policy)
        limit = len(scored_pairs) if pair_test_budget_per_state <= 0 else max(0, pair_test_budget_per_state)
        tested_scored_pairs = scored_pairs[:limit]
        skipped_scored_pairs = scored_pairs[limit:]
    else:
        scored_pairs = _full_order_scored_pairs(active_profiles)
        tested_scored_pairs = scored_pairs if max_pairs is None else scored_pairs[: max(0, max_pairs)]
        skipped_scored_pairs = [] if max_pairs is None else scored_pairs[max(0, max_pairs) :]
    tested_pairs = [item["pair"] for item in tested_scored_pairs]
    priority_by_pair = {item["pair"]: item for item in scored_pairs}
    cache = _resolve_pair_cache(tools, input_ll, pair_cache)
    defer_materialization = worker_handles_enabled() and not keep_ir_artifacts

    def run_one(pair: tuple[str, str]) -> dict:
        pass_a, pass_b = pair
        profile_a = profiles[pass_a]
        profile_b = profiles[pass_b]
        safe = f"{_safe_name(pass_a)}__{_safe_name(pass_b)}"
        current_dir = pair_dir / safe
        current_dir.mkdir(parents=True, exist_ok=True)
        ab_path = current_dir / "ab.ll"
        ba_path = current_dir / "ba.ll"
        pipeline_a = _pipeline_for(pass_a, pass_registry)
        pipeline_b = _pipeline_for(pass_b, pass_registry)
        row = _base_row(input_ll, profile_a, profile_b, ab_path, ba_path)
        _apply_pair_scheduling(row, priority_by_pair[pair], mode)
        cache_key = cache.cache_key(
            state_hash=str(row.get("state_hash", "")),
            pass_a_name=pass_a,
            pass_b_name=pass_b,
            pass_a_pipeline=pipeline_a,
            pass_b_pipeline=pipeline_b,
        )
        cached = cache.get(cache_key)
        row["cache_key"] = cache_key
        if cached is not None:
            row.update(cached)
            row.update(
                {
                    "cache_hit": "true",
                    "pair_test_opt_runs": "0",
                    "pair_test_time_ms": "0.000",
                    "avoided_opt_runs": "2",
                    "pair_test_pass_invocations_baseline": "4",
                    "pair_test_pass_invocations_actual": "0",
                    "pair_test_pass_invocations_saved": "4",
                    "reused_single_pass_outputs": "false",
                    "full_pipeline_runs_avoided": "2",
                    "second_stage_runs": "0",
                    "time_ms": "0.000",
                    "llvm_diff_time_ms": "",
                    "comparator_time_ms": "0.000",
                }
            )
            return row

        reuse_a = _existing_output(profile_a)
        reuse_b = _existing_output(profile_b)
        reused_single_pass_outputs = reuse_a is not None and reuse_b is not None
        if reused_single_pass_outputs:
            if defer_materialization:
                ab = run_opt(str(tools["opt"]), reuse_a, [pipeline_b], ab_path, timeout, materialize=False)
                ba = run_opt(str(tools["opt"]), reuse_b, [pipeline_a], ba_path, timeout, materialize=False)
            else:
                ab = run_opt(str(tools["opt"]), reuse_a, [pipeline_b], ab_path, timeout)
                ba = run_opt(str(tools["opt"]), reuse_b, [pipeline_a], ba_path, timeout)
        else:
            if defer_materialization:
                ab = run_opt(
                    str(tools["opt"]),
                    input_ll,
                    [pipeline_a, pipeline_b],
                    ab_path,
                    timeout,
                    materialize=False,
                )
                ba = run_opt(
                    str(tools["opt"]),
                    input_ll,
                    [pipeline_b, pipeline_a],
                    ba_path,
                    timeout,
                    materialize=False,
                )
            else:
                ab = run_opt(str(tools["opt"]), input_ll, [pipeline_a, pipeline_b], ab_path, timeout)
                ba = run_opt(str(tools["opt"]), input_ll, [pipeline_b, pipeline_a], ba_path, timeout)
        opt_time_ms = ab.time_ms + ba.time_ms
        retry_opt_runs = 0
        retry_pass_invocations = 0
        row["ab_success"] = _bool(ab.success)
        row["ba_success"] = _bool(ba.success)
        row["time_ms"] = f"{opt_time_ms:.3f}"
        row["cache_hit"] = "false"
        row["pair_test_opt_runs"] = "2"
        row["pair_test_time_ms"] = f"{opt_time_ms:.3f}"
        row["avoided_opt_runs"] = "0"
        row["pair_test_pass_invocations_baseline"] = "4"
        row["pair_test_pass_invocations_actual"] = "2" if reused_single_pass_outputs else "4"
        row["pair_test_pass_invocations_saved"] = "2" if reused_single_pass_outputs else "0"
        row["reused_single_pass_outputs"] = _bool(reused_single_pass_outputs)
        row["full_pipeline_runs_avoided"] = "2" if reused_single_pass_outputs else "0"
        row["second_stage_runs"] = "2" if reused_single_pass_outputs else "0"
        worker_hash_fast_path = (
            ab.success
            and ba.success
            and ab.backend == "worker"
            and ba.backend == "worker"
            and bool(ab.canonical_hash)
            and ab.canonical_hash == ba.canonical_hash
        )
        materializations = 0
        row["worker_hash_fast_path"] = _bool(worker_hash_fast_path)

        if ab.success and ba.success:
            compare_start = time.perf_counter()
            if worker_hash_fast_path:
                equality = EqualityResult(
                    equal=True,
                    tier="canonical_hash",
                    can_hard_fold=True,
                    reason="worker_full_ir_hash_equal",
                    text_hash_equal=True,
                    left_hash=ab.canonical_hash,
                    right_hash=ba.canonical_hash,
                )
            else:
                materialization_retry_reason = ""
                try:
                    if not ab.materialized:
                        materialize_run_result(ab, ab_path, timeout=timeout)
                        materializations += 1
                    if not ba.materialized:
                        materialize_run_result(ba, ba_path, timeout=timeout)
                        materializations += 1
                except (OSError, RuntimeError, ValueError) as exc:
                    materialization_retry_reason = str(exc) or "borrowed_handle_unavailable"
                if not (ab_path.exists() and ba_path.exists()):
                    materialization_retry_reason = materialization_retry_reason or "materialized_output_missing"
                if materialization_retry_reason:
                    if ab.backend == "worker":
                        release_run_result(ab, timeout=timeout)
                    if ba.backend == "worker":
                        release_run_result(ba, timeout=timeout)
                    ab_path.unlink(missing_ok=True)
                    ba_path.unlink(missing_ok=True)
                    ab, ba = _rerun_pair_materialized(
                        tools=tools,
                        input_ll=input_ll,
                        reuse_a=reuse_a,
                        reuse_b=reuse_b,
                        pipeline_a=pipeline_a,
                        pipeline_b=pipeline_b,
                        ab_path=ab_path,
                        ba_path=ba_path,
                        timeout=timeout,
                    )
                    retry_opt_runs = 2
                    retry_pass_invocations = 2 if reused_single_pass_outputs else 4
                    opt_time_ms += ab.time_ms + ba.time_ms
                    row["materialization_retry"] = "true"
                    row["materialization_retry_reason"] = materialization_retry_reason
                    if ab.success and ba.success and ab_path.exists() and ba_path.exists():
                        materializations += 2
                        equality = compare_ir_equivalence(ab_path, ba_path, tools=tools, timeout=timeout)
                    else:
                        equality = EqualityResult(
                            equal=False,
                            tier="failed",
                            can_hard_fold=False,
                            reason="materialization_retry_failed",
                            error_message="direct worker materialization retry failed",
                        )
                else:
                    equality = compare_ir_equivalence(ab_path, ba_path, tools=tools, timeout=timeout)
            row["comparator_time_ms"] = f"{(time.perf_counter() - compare_start) * 1000:.3f}"
            ab_hash = equality.left_hash or (canonical_hash(ab_path) if ab_path.exists() else ab.canonical_hash)
            ba_hash = equality.right_hash or (canonical_hash(ba_path) if ba_path.exists() else ba.canonical_hash)
            same_hash = ab_hash == ba_hash
            if equality.can_hard_fold:
                dynamic_relation = "dynamic_commute"
                failure_kind = ""
            elif equality.tier == "failed":
                dynamic_relation = "dynamic_failed"
                failure_kind = equality.reason or "tool_failed"
            else:
                dynamic_relation = "dynamic_order_sensitive"
                failure_kind = ""
            ab_inst = _instruction_count(ab, ab_path)
            ba_inst = _instruction_count(ba, ba_path)
            row.update(
                {
                    "dynamic_relation": dynamic_relation,
                    "final_relation": _final_relation(dynamic_relation),
                    "ab_hash": ab_hash,
                    "ba_hash": ba_hash,
                    "same_hash": _bool(same_hash),
                    "text_hash_equal": _bool_or_empty(equality.text_hash_equal),
                    "llvm_diff_equal": _bool_or_empty(equality.llvm_diff_equal),
                    "module_fingerprint_equal": _bool_or_empty(equality.module_fingerprint_equal),
                    "equality_tier": equality.tier,
                    "equality_reason": equality.reason,
                    "can_hard_fold": _bool(equality.can_hard_fold),
                    "ab_inst": ab_inst,
                    "ba_inst": ba_inst,
                    "inst_delta_ab_ba": ab_inst - ba_inst,
                    "failure_kind": failure_kind,
                }
            )
        else:
            relation = "dynamic_timeout" if ab.timed_out or ba.timed_out else "dynamic_failed"
            failure = "timeout" if relation == "dynamic_timeout" else "failed"
            row.update(
                {
                    "dynamic_relation": relation,
                    "final_relation": _final_relation(relation),
                    "same_hash": "false",
                    "equality_tier": "failed",
                    "equality_reason": failure,
                    "can_hard_fold": "false",
                    "failure_kind": failure,
                }
            )
        row["ab_materialized"] = _bool(ab.materialized and ab_path.exists())
        row["ba_materialized"] = _bool(ba.materialized and ba_path.exists())
        row["pair_materializations"] = str(materializations if defer_materialization else 2)
        row["pair_materializations_avoided"] = str(max(0, 2 - materializations) if defer_materialization else 0)
        actual_pass_invocations = (2 if reused_single_pass_outputs else 4) + retry_pass_invocations
        row["ab_success"] = _bool(ab.success)
        row["ba_success"] = _bool(ba.success)
        row["time_ms"] = f"{opt_time_ms:.3f}"
        row["pair_test_time_ms"] = f"{opt_time_ms:.3f}"
        row["pair_test_opt_runs"] = str(2 + retry_opt_runs)
        row["pair_test_pass_invocations_actual"] = str(actual_pass_invocations)
        row["pair_test_pass_invocations_saved"] = str(4 - actual_pass_invocations)
        row["second_stage_runs"] = str((2 + retry_opt_runs) if reused_single_pass_outputs else 0)
        row["pair_test_retry_opt_runs"] = str(retry_opt_runs)
        cache.store(cache_key, row)
        if ab.backend == "worker":
            release_run_result(ab, timeout=timeout)
        if ba.backend == "worker":
            release_run_result(ba, timeout=timeout)
        return row

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        rows = list(executor.map(run_one, tested_pairs))

    for item in skipped_scored_pairs:
        pass_a, pass_b = item["pair"]
        row = _base_row(input_ll, profiles[pass_a], profiles[pass_b], Path(""), Path(""))
        _apply_pair_scheduling(row, item, mode)
        if mode == "lazy":
            row.update(
                {
                    "dynamic_relation": "not_tested",
                    "final_relation": "final_unknown",
                    "ab_success": "false",
                    "ba_success": "false",
                    "same_hash": "",
                    "equality_tier": "failed",
                    "equality_reason": "lazy_budget",
                    "can_hard_fold": "false",
                    "failure_kind": "lazy_budget",
                    "cache_hit": "false",
                    "cache_key": "",
                    "pair_test_opt_runs": "0",
                    "pair_test_time_ms": "0.000",
                    "avoided_opt_runs": "0",
                    "pair_test_pass_invocations_baseline": "4",
                    "pair_test_pass_invocations_actual": "0",
                    "pair_test_pass_invocations_saved": "4",
                    "reused_single_pass_outputs": "false",
                    "full_pipeline_runs_avoided": "2",
                    "second_stage_runs": "0",
                    "llvm_diff_time_ms": "",
                    "comparator_time_ms": "",
                    "skipped_by_budget": "true",
                }
            )
            rows.append(row)
            continue
        row.update(
            {
                "dynamic_relation": "not_tested",
                "final_relation": _final_relation("not_tested"),
                "ab_success": "false",
                "ba_success": "false",
                "same_hash": "",
                "equality_tier": "failed",
                "equality_reason": "max_pairs",
                "can_hard_fold": "false",
                "failure_kind": "max_pairs",
                "cache_hit": "false",
                "cache_key": "",
                "pair_test_opt_runs": "0",
                "pair_test_time_ms": "0.000",
                "avoided_opt_runs": "0",
                "pair_test_pass_invocations_baseline": "0",
                "pair_test_pass_invocations_actual": "0",
                "pair_test_pass_invocations_saved": "0",
                "reused_single_pass_outputs": "false",
                "full_pipeline_runs_avoided": "0",
                "second_stage_runs": "0",
                "llvm_diff_time_ms": "",
                "comparator_time_ms": "",
            }
        )
        rows.append(row)

    _update_pair_history(history, rows)
    if write_output:
        _write_csv(out_dir / "pair_relation.csv", PAIR_RELATION_FIELDS, rows)
    return rows


def _ordered_pairs(active_profiles: list[dict]) -> list[tuple[str, str]]:
    sorted_profiles = sorted(active_profiles, key=lambda row: row["pass"])

    def priority(pair: tuple[dict, dict]) -> tuple[int, str, str]:
        a, b = pair
        overlap = bool(_split(a.get("changed_functions")) & _split(b.get("changed_functions")))
        return (0 if overlap else 1, a["pass"], b["pass"])

    ordered = sorted(itertools.combinations(sorted_profiles, 2), key=priority)
    return [(a["pass"], b["pass"]) for a, b in ordered]


def _full_order_scored_pairs(active_profiles: list[dict]) -> list[dict]:
    return [
        {
            "pair": pair,
            "score": 0.0,
            "reason": "full_order",
        }
        for pair in _ordered_pairs(active_profiles)
    ]


def _prioritized_pairs(active_profiles: list[dict], history: dict, policy: str) -> list[dict]:
    profiles = {row["pass"]: row for row in active_profiles}
    pairs = sorted(itertools.combinations(sorted(profiles), 2))
    scored = []
    for pass_a, pass_b in pairs:
        effect_score = _effect_score(profiles[pass_a], profiles[pass_b])
        history_score = _history_score(history.get(_pair_key(pass_a, pass_b), {}))
        if policy == "default":
            score = 0.0
        elif policy == "effect-size":
            score = effect_score
        elif policy == "history":
            score = history_score
        else:
            score = effect_score + history_score
        reason = f"effect={effect_score:.3f};history={history_score:.3f};policy={policy}"
        scored.append({"pair": (pass_a, pass_b), "score": score, "reason": reason})
    return sorted(scored, key=lambda item: (-float(item["score"]), item["pair"][0], item["pair"][1]))


def _effect_score(profile_a: dict, profile_b: dict) -> float:
    inst = abs(_to_float(profile_a.get("inst_delta"))) + abs(_to_float(profile_b.get("inst_delta")))
    blocks = abs(_to_float(profile_a.get("blocks_changed"))) + abs(_to_float(profile_b.get("blocks_changed")))
    funcs = len(_split(profile_a.get("changed_functions"))) + len(_split(profile_b.get("changed_functions")))
    return inst + 0.25 * blocks + 0.10 * funcs


def _history_score(entry: dict) -> float:
    tested = _to_float(entry.get("tested_count"))
    if tested <= 0:
        return 1.0
    commute = _to_float(entry.get("commute_count")) / tested
    sensitive = _to_float(entry.get("order_sensitive_count")) / tested
    unknown = _to_float(entry.get("unknown_count")) / tested
    mixed = 1.0 - max(commute, sensitive, unknown)
    score = unknown + mixed
    if sensitive >= 0.8 and unknown == 0:
        score -= 0.5
    return max(0.0, score)


def _apply_pair_scheduling(row: dict, scored_pair: dict, mode: str) -> None:
    row["pair_testing_mode"] = mode
    row["pair_priority_score"] = f"{float(scored_pair.get('score', 0.0)):.6f}"
    row["pair_priority_reason"] = str(scored_pair.get("reason", ""))
    row["skipped_by_budget"] = "false"


def _resolve_pair_history(tools: dict) -> dict:
    if not isinstance(tools, dict):
        return {}
    history = tools.get("_pair_history")
    if isinstance(history, dict):
        return history
    history = {}
    tools["_pair_history"] = history
    return history


def _update_pair_history(history: dict, rows: list[dict]) -> None:
    for row in rows:
        if row.get("dynamic_relation") == "not_tested":
            continue
        key = _pair_key(row.get("pass_a", ""), row.get("pass_b", ""))
        entry = history.setdefault(
            key,
            {
                "tested_count": 0,
                "commute_count": 0,
                "order_sensitive_count": 0,
                "unknown_count": 0,
            },
        )
        entry["tested_count"] = _to_int(entry.get("tested_count")) + 1
        relation = row.get("dynamic_relation")
        if relation == "dynamic_commute":
            entry["commute_count"] = _to_int(entry.get("commute_count")) + 1
        elif relation == "dynamic_order_sensitive":
            entry["order_sensitive_count"] = _to_int(entry.get("order_sensitive_count")) + 1
        else:
            entry["unknown_count"] = _to_int(entry.get("unknown_count")) + 1


def _rerun_pair_materialized(
    *,
    tools: dict,
    input_ll: Path,
    reuse_a: Path | None,
    reuse_b: Path | None,
    pipeline_a: str,
    pipeline_b: str,
    ab_path: Path,
    ba_path: Path,
    timeout: int,
) -> tuple[RunResult, RunResult]:
    if reuse_a is not None and reuse_b is not None:
        ab = run_opt(
            str(tools["opt"]), reuse_a, [pipeline_b], ab_path, timeout, materialize=True
        )
        ba = run_opt(
            str(tools["opt"]), reuse_b, [pipeline_a], ba_path, timeout, materialize=True
        )
    else:
        ab = run_opt(
            str(tools["opt"]),
            input_ll,
            [pipeline_a, pipeline_b],
            ab_path,
            timeout,
            materialize=True,
        )
        ba = run_opt(
            str(tools["opt"]),
            input_ll,
            [pipeline_b, pipeline_a],
            ba_path,
            timeout,
            materialize=True,
        )
    return ab, ba


def _base_row(input_ll: Path, profile_a: dict, profile_b: dict, ab_path: Path, ba_path: Path) -> dict:
    return {
        "program": profile_a.get("program") or Path(input_ll).parent.name or Path(input_ll).stem,
        "state_id": profile_a.get("state_id", ""),
        "depth": profile_a.get("depth", ""),
        "parent_state_id": profile_a.get("parent_state_id", ""),
        "transition_pass": profile_a.get("transition_pass", ""),
        "state_hash": profile_a.get("state_hash", ""),
        "pass_a": profile_a["pass"],
        "pass_b": profile_b["pass"],
        "a_active": profile_a.get("active", "true"),
        "b_active": profile_b.get("active", "true"),
        "static_relation": "",
        "dynamic_relation": "",
        "final_relation": "",
        "ab_success": "",
        "ba_success": "",
        "ab_hash": "",
        "ba_hash": "",
        "same_hash": "",
        "text_hash_equal": "",
        "llvm_diff_equal": "",
        "module_fingerprint_equal": "",
        "equality_tier": "",
        "equality_reason": "",
        "can_hard_fold": "",
        "pair_testing_mode": "full",
        "pair_priority_score": "",
        "pair_priority_reason": "",
        "skipped_by_budget": "false",
        "ab_inst": "",
        "ba_inst": "",
        "inst_delta_ab_ba": "",
        "changed_funcs_a": profile_a.get("changed_functions", ""),
        "changed_funcs_b": profile_b.get("changed_functions", ""),
        "changed_blocks_a": profile_a.get("changed_blocks", ""),
        "changed_blocks_b": profile_b.get("changed_blocks", ""),
        "overlap_functions": "",
        "overlap_blocks": "",
        "time_ms": "",
        "failure_kind": "",
        "ab_path": str(ab_path) if str(ab_path) != "." else "",
        "ba_path": str(ba_path) if str(ba_path) != "." else "",
        "ab_materialized": "",
        "ba_materialized": "",
        "pair_materializations": "",
        "pair_materializations_avoided": "",
        "worker_hash_fast_path": "false",
        "cache_hit": "false",
        "cache_key": "",
        "pair_test_opt_runs": "",
        "pair_test_time_ms": "",
        "avoided_opt_runs": "",
        "pair_test_pass_invocations_baseline": "",
        "pair_test_pass_invocations_actual": "",
        "pair_test_pass_invocations_saved": "",
        "reused_single_pass_outputs": "false",
        "full_pipeline_runs_avoided": "",
        "second_stage_runs": "",
        "materialization_retry": "false",
        "materialization_retry_reason": "",
        "pair_test_retry_opt_runs": "0",
        "llvm_diff_time_ms": "",
        "comparator_time_ms": "",
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "pass"


def _split(value: object) -> set[str]:
    if not value:
        return set()
    return {item for item in str(value).split(";") if item}


def _pair_key(pass_a: str, pass_b: str) -> str:
    left, right = sorted([pass_a, pass_b])
    return f"{left}--{right}"


def _to_float(value: object) -> float:
    try:
        return float(str(value).strip() or "0")
    except ValueError:
        return 0.0


def _to_int(value: object) -> int:
    try:
        return int(float(str(value).strip() or "0"))
    except ValueError:
        return 0


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _bool_or_empty(value: bool | None) -> str:
    return "" if value is None else _bool(value)


def _instruction_count(result: RunResult, path: Path) -> int:
    if isinstance(result.feature_counts, dict) and "instructions" in result.feature_counts:
        return _to_int(result.feature_counts.get("instructions"))
    if path.exists():
        return count_ir_features(path).get("instructions", 0)
    return 0


def _final_relation(dynamic_relation: str) -> str:
    if dynamic_relation == "dynamic_commute":
        return "final_commute"
    if dynamic_relation == "dynamic_order_sensitive":
        return "final_order_sensitive"
    return "final_unknown"


def _pipeline_for(pass_name: str, registry: PassRegistry | None) -> str:
    return registry.pipeline_for(pass_name) if registry else pass_name


def _resolve_pair_cache(tools: dict, input_ll: Path, pair_cache: PairRelationCache | None) -> PairRelationCache:
    if pair_cache is not None:
        return pair_cache
    existing = tools.get("_pair_cache") if isinstance(tools, dict) else None
    if isinstance(existing, PairRelationCache):
        return existing
    cache = PairRelationCache.from_tools(tools, input_ll)
    if isinstance(tools, dict):
        tools["_pair_cache"] = cache
    return cache


def _existing_output(profile: dict) -> Path | None:
    output_path = profile.get("output_path", "")
    if not output_path:
        return None
    path = Path(output_path)
    return path if path.exists() else None
