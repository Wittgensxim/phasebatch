from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Callable

from .artifact_cleanup import cleanup_ir_artifacts, mark_ir_artifacts_kept
from .ir_equivalence import compare_ir_equivalence
from .optimizer import optimize_batches
from .runtime_rerank import rerank_terminal_states, select_runtime_candidates
from .runner import run_opt
from .staged_config import StageSpec, load_staged_config
from .tools import collect_toolchain, write_metadata


StageRunner = Callable[..., dict]

STAGED_SUMMARY_FIELDS = [
    "stage_index",
    "stage_id",
    "mode",
    "max_rounds",
    "passes_path",
    "states",
    "transitions",
    "selected_final_state",
    "exact_status",
    "pair_matrix_complete",
    "pipeline",
    "selection_source",
    "runtime_reason",
    "runtime_median_ms",
]

STAGED_PIPELINE_FIELDS = ["stage_index", "stage_id", "pipeline", "selection_source"]
STAGED_REPLAY_FIELDS = [
    "replay_status",
    "hashes_match",
    "equality_tier",
    "equality_reason",
    "can_hard_fold",
    "stage_segments",
    "time_ms",
    "error_message",
]


def optimize_staged(
    input_path: Path,
    out_dir: Path,
    manifest_path: Path,
    *,
    jobs: int,
    timeout: int,
    keep_ir_artifacts: bool = False,
    stage_runner: StageRunner = optimize_batches,
) -> dict:
    start = time.perf_counter()
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_staged_config(manifest_path)
    toolchain = collect_toolchain()
    tools = _tool_paths(toolchain)
    metadata = {
        **toolchain,
        "input": str(input_path),
        "out_dir": str(out_dir),
        "manifest": str(Path(manifest_path)),
        "root_ir_mode": config.root_ir_mode,
        "stage_count": len(config.stages),
        "jobs": jobs,
        "timeout": timeout,
    }
    write_metadata(out_dir, metadata)

    stage_rows: list[dict] = []
    pipeline_rows: list[dict] = []
    handoff_dir = out_dir / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    current_input = input_path
    first_root_ir: Path | None = None

    for index, stage in enumerate(config.stages):
        stage_dir = out_dir / "stages" / f"{index:02d}_{stage.stage_id}"
        result = stage_runner(
            current_input,
            stage_dir,
            stage.passes_path,
            **_stage_kwargs(stage, jobs=jobs, timeout=timeout, root_ir_mode=config.root_ir_mode),
        )
        final_ir = stage_dir / "final.ll"
        if not final_ir.exists():
            raise RuntimeError(f"stage {stage.stage_id} did not produce final.ll")
        if first_root_ir is None:
            first_root_ir = stage_dir / "states" / "S0000" / "input.ll"
            if not first_root_ir.exists():
                raise RuntimeError(f"stage {stage.stage_id} did not retain its root IR")
        pipeline = _read_text(stage_dir / "optimized_pipeline.txt")
        stage_metadata = _read_json(stage_dir / "metadata.json")
        selected_final_state = str(result.get("selected_final_state", ""))
        selection_source = "ir_objective"
        runtime_reason = "not_requested"
        runtime_median_ms = ""
        if stage.require_transition and not pipeline:
            required_candidate = _select_required_transition(stage_dir, stage.passes_path)
            if required_candidate is None:
                raise RuntimeError(f"stage {stage.stage_id} requires a reached non-identity transition")
            final_ir = required_candidate.ir_path
            pipeline = ",".join(required_candidate.pipeline)
            selected_final_state = required_candidate.state_id
            selection_source = "required_transition"
        if config.runtime.enabled and stage.runtime_rerank:
            runtime_result = rerank_terminal_states(
                stage_dir,
                stage.passes_path,
                stage_dir / "runtime_rerank",
                config.runtime,
                tools=tools,
            )
            runtime_reason = runtime_result.reason
            winner = runtime_result.winner
            if winner is not None and winner.ir_path.exists() and (not stage.require_transition or winner.pipeline):
                final_ir = winner.ir_path
                pipeline = ",".join(winner.pipeline)
                selected_final_state = winner.state_id
                selection_source = "runtime_median"
                runtime_median_ms = (runtime_result.winner_summary or {}).get("median_ms", "")
            else:
                selection_source = "ir_objective_fallback"
        stage_rows.append(
            {
                "stage_index": str(index),
                "stage_id": stage.stage_id,
                "mode": stage.mode,
                "max_rounds": str(stage.max_rounds),
                "passes_path": str(stage.passes_path),
                "states": str(result.get("states", "")),
                "transitions": str(result.get("batch_transitions", result.get("transitions", ""))),
                "selected_final_state": selected_final_state,
                "exact_status": str(stage_metadata.get("exact_status", "")),
                "pair_matrix_complete": _bool(stage_metadata.get("pair_matrix_complete")),
                "pipeline": pipeline,
                "selection_source": selection_source,
                "runtime_reason": runtime_reason,
                "runtime_median_ms": runtime_median_ms,
            }
        )
        pipeline_rows.append(
            {
                "stage_index": str(index),
                "stage_id": stage.stage_id,
                "pipeline": pipeline,
                "selection_source": selection_source,
            }
        )
        handoff = handoff_dir / f"{index:02d}_{stage.stage_id}.ll"
        shutil.copyfile(final_ir, handoff)
        current_input = handoff

    assert first_root_ir is not None
    final_ir = out_dir / "final.ll"
    shutil.copyfile(current_input, final_ir)
    replay_row = _replay_stages(first_root_ir, final_ir, pipeline_rows, out_dir, tools, timeout)
    _write_csv(out_dir / "staged_summary.csv", STAGED_SUMMARY_FIELDS, stage_rows)
    _write_csv(out_dir / "staged_pipeline.csv", STAGED_PIPELINE_FIELDS, pipeline_rows)
    _write_csv(out_dir / "staged_replay.csv", STAGED_REPLAY_FIELDS, [replay_row])

    exact_complete = all(
        row["mode"] == "exact" and row["exact_status"] == "exact_complete"
        for row in stage_rows
    )
    metadata.update(
        {
            "staged_exact_scope": "all_stages_exact" if exact_complete else "mixed_or_budgeted_stages",
            "pair_matrix_complete": all(row["pair_matrix_complete"] == "true" for row in stage_rows),
            "replay_verified": replay_row["can_hard_fold"],
            "elapsed_ms": f"{(time.perf_counter() - start) * 1000:.3f}",
        }
    )
    write_metadata(out_dir, metadata)
    _write_summary(out_dir / "staged_summary.md", metadata, stage_rows, replay_row)

    cleanup = mark_ir_artifacts_kept() if keep_ir_artifacts else cleanup_ir_artifacts(out_dir)
    return {
        "out_dir": str(out_dir),
        "stages": len(stage_rows),
        "selected_final_state": stage_rows[-1]["selected_final_state"],
        "replay_verified": replay_row["can_hard_fold"] == "true",
        "staged_summary_csv": str(out_dir / "staged_summary.csv"),
        "staged_pipeline_csv": str(out_dir / "staged_pipeline.csv"),
        "staged_replay_csv": str(out_dir / "staged_replay.csv"),
        "staged_summary_md": str(out_dir / "staged_summary.md"),
        **cleanup,
    }


def _stage_kwargs(stage: StageSpec, *, jobs: int, timeout: int, root_ir_mode: str) -> dict:
    return {
        "mode": stage.mode,
        "objective": "ir-inst-count",
        "max_rounds": stage.max_rounds,
        "beam_width": stage.beam_width,
        "max_batches_per_state": stage.max_batches_per_state,
        "budgeted_validation_strategy": stage.budgeted_validation_strategy,
        "max_component_size": stage.max_component_size,
        "max_batch_candidates": stage.max_batch_candidates,
        "batchify_terminal_states": True,
        "validate_batches": True,
        "allow_sampled_batches": False,
        "allow_bounded_validation": False,
        "batch_validation_mode": stage.batch_validation_mode,
        "pair_testing_mode": "full",
        "pair_test_budget_per_state": 0,
        "pair_priority_policy": "mixed",
        "batch_construction_mode": "pairwise",
        "max_states": stage.max_states,
        "jobs": jobs,
        "timeout": timeout,
        "max_pairs": None,
        "verify_final_pipeline": True,
        "keep_ir_artifacts": True,
        "root_ir_mode": root_ir_mode,
    }


def _select_required_transition(stage_dir: Path, passes_path: Path):
    candidates = select_runtime_candidates(stage_dir, passes_path, top_k=1_000_000)
    transitioned = [candidate for candidate in candidates if candidate.pipeline]
    if not transitioned:
        return None
    return min(transitioned, key=lambda candidate: (candidate.objective_value, candidate.state_id))


def _replay_stages(
    root_ir: Path,
    final_ir: Path,
    pipeline_rows: list[dict],
    out_dir: Path,
    tools: dict[str, str],
    timeout: int,
) -> dict:
    start = time.perf_counter()
    current = root_ir
    try:
        for index, row in enumerate(pipeline_rows):
            output = out_dir / "staged_replay" / f"{index:02d}_{row['stage_id']}.ll"
            pipeline = row.get("pipeline", "")
            if pipeline:
                result = run_opt(tools["opt"], current, [pipeline], output, timeout)
                if not result.success or not output.exists():
                    raise RuntimeError(result.stderr or result.failure_kind or "staged replay opt failed")
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(current, output)
            current = output
        equality = compare_ir_equivalence(current, final_ir, tools=tools, timeout=timeout)
        return {
            "replay_status": "success" if equality.can_hard_fold else "mismatch",
            "hashes_match": _bool(equality.left_hash == equality.right_hash),
            "equality_tier": equality.tier,
            "equality_reason": equality.reason,
            "can_hard_fold": _bool(equality.can_hard_fold),
            "stage_segments": str(len(pipeline_rows)),
            "time_ms": f"{(time.perf_counter() - start) * 1000:.3f}",
            "error_message": equality.error_message,
        }
    except Exception as exc:
        return {
            "replay_status": "failed",
            "hashes_match": "false",
            "equality_tier": "failed",
            "equality_reason": "staged_replay_failed",
            "can_hard_fold": "false",
            "stage_segments": str(len(pipeline_rows)),
            "time_ms": f"{(time.perf_counter() - start) * 1000:.3f}",
            "error_message": str(exc),
        }


def _write_summary(path: Path, metadata: dict, stages: list[dict], replay: dict) -> None:
    lines = [
        "# Staged Optimization Summary",
        "",
        f"- root_ir_mode: {metadata.get('root_ir_mode', '')}",
        f"- staged_exact_scope: {metadata.get('staged_exact_scope', '')}",
        f"- pair_matrix_complete: {_bool(metadata.get('pair_matrix_complete'))}",
        f"- replay_verified: {replay.get('can_hard_fold', 'false')}",
        "",
        "| stage | mode | rounds | states | transitions | exact status | selection | runtime median ms | pipeline |",
        "|---|---|---:|---:|---:|---|---|---:|---|",
    ]
    for row in stages:
        lines.append(
            f"| {row['stage_id']} | {row['mode']} | {row['max_rounds']} | {row['states']} | "
            f"{row['transitions']} | {row['exact_status']} | {row['selection_source']} | "
            f"{row['runtime_median_ms']} | `{row['pipeline']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tool_paths(metadata: dict) -> dict[str, str]:
    return {
        name: details["path"]
        for name, details in (metadata.get("tools") or {}).items()
        if isinstance(details, dict) and details.get("path")
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value) -> str:
    if isinstance(value, str):
        return "true" if value.strip().lower() == "true" else "false"
    return "true" if value else "false"
