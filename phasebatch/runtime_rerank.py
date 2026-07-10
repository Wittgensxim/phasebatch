from __future__ import annotations

import csv
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .normalizer import count_ir_features
from .pass_config import load_pass_registry, resolve_pipeline_sequence
from .staged_config import RuntimeConfig


@dataclass(frozen=True)
class RuntimeCandidate:
    state_id: str
    state_hash: str
    ir_path: Path
    objective_value: float
    pipeline: tuple[str, ...]
    selected_as_final: bool
    direct_calls: int
    memory_ops: int
    branches: int


@dataclass(frozen=True)
class CommandOutcome:
    returncode: int
    time_ms: float
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[list[str], int, Path], CommandOutcome]


@dataclass(frozen=True)
class RuntimeRerankResult:
    candidates: tuple[RuntimeCandidate, ...]
    winner: RuntimeCandidate | None
    winner_summary: dict[str, str] | None
    reason: str
    candidates_csv: Path
    trials_csv: Path
    summary_csv: Path
    selection_md: Path


RUNTIME_CANDIDATE_FIELDS = [
    "selection_index",
    "state_id",
    "state_hash",
    "objective_value",
    "selected_as_final",
    "direct_calls",
    "memory_ops",
    "branches",
    "pipeline",
    "ir_path",
    "llc_returncode",
    "link_returncode",
    "compile_eligible",
    "compile_error",
    "executable_path",
]
RUNTIME_TRIAL_FIELDS = [
    "kind",
    "trial",
    "order_position",
    "state_id",
    "returncode",
    "time_ms",
    "stdout",
    "stderr",
]
RUNTIME_SUMMARY_FIELDS = [
    "state_id",
    "eligible",
    "successful_trials",
    "median_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
]


def select_runtime_candidates(
    run_dir: str | Path,
    passes_path: str | Path,
    *,
    top_k: int,
) -> list[RuntimeCandidate]:
    run_dir = Path(run_dir)
    state_rows = _read_csv(run_dir / "states.csv")
    leaf_rows = _read_csv(run_dir / "leaf_states.csv")
    if not state_rows or not leaf_rows:
        return []

    states = {row.get("state_id", ""): row for row in state_rows if row.get("state_id")}
    registry = load_pass_registry(passes_path)
    eligible: list[RuntimeCandidate] = []
    for leaf in leaf_rows:
        if not (_as_bool(leaf.get("is_leaf")) or _as_bool(leaf.get("selected_as_final"))):
            continue
        state_id = leaf.get("state_id", "")
        state = states.get(state_id)
        if state is None:
            continue
        ir_path = _resolve_ir_path(run_dir, state)
        if ir_path is None:
            continue
        logical_pipeline = _reconstruct_pipeline(state_id, states)
        if logical_pipeline is None:
            continue
        features = count_ir_features(ir_path)
        eligible.append(
            RuntimeCandidate(
                state_id=state_id,
                state_hash=state.get("state_hash", "") or state_id,
                ir_path=ir_path,
                objective_value=_as_float(leaf.get("objective_value")),
                pipeline=tuple(resolve_pipeline_sequence(logical_pipeline, registry)),
                selected_as_final=_as_bool(leaf.get("selected_as_final")),
                direct_calls=features["direct_calls"],
                memory_ops=features["loads"] + features["stores"],
                branches=features["branches"],
            )
        )

    representatives: dict[str, RuntimeCandidate] = {}
    for candidate in eligible:
        current = representatives.get(candidate.state_hash)
        if current is None or _representative_key(candidate) < _representative_key(current):
            representatives[candidate.state_hash] = candidate
    return _select_diverse_candidates(representatives.values(), top_k)


def benchmark_executables(
    executables: dict[str, Path],
    config: RuntimeConfig,
    *,
    command_runner: CommandRunner | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    runner = command_runner or _run_command
    ordered_ids = sorted(executables)
    trial_rows: list[dict[str, str]] = []
    eligible = {state_id: True for state_id in ordered_ids}
    times: dict[str, list[float]] = {state_id: [] for state_id in ordered_ids}

    for warmup in range(1, config.warmups + 1):
        for position, state_id in enumerate(ordered_ids, start=1):
            outcome = _execute(executables[state_id], config, runner)
            trial_rows.append(_trial_row("warmup", warmup, position, state_id, outcome))
            if outcome.returncode != config.expected_exit_code:
                eligible[state_id] = False

    for trial in range(1, config.trials + 1):
        order = _rotate(ordered_ids, trial - 1)
        for position, state_id in enumerate(order, start=1):
            if not eligible[state_id]:
                continue
            outcome = _execute(executables[state_id], config, runner)
            trial_rows.append(_trial_row("execute", trial, position, state_id, outcome))
            if outcome.returncode != config.expected_exit_code:
                eligible[state_id] = False
                continue
            times[state_id].append(outcome.time_ms)

    summary_rows = [
        _summary_row(state_id, times[state_id], eligible[state_id], config.trials)
        for state_id in ordered_ids
    ]
    return trial_rows, summary_rows


def choose_runtime_winner(summary_rows: Iterable[dict[str, str]]) -> dict[str, str] | None:
    eligible = [
        row
        for row in summary_rows
        if _as_bool(row.get("eligible")) and int(row.get("successful_trials", "0") or 0) > 0
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (_as_float(row.get("median_ms")), row.get("state_id", "")))


def rerank_terminal_states(
    run_dir: str | Path,
    passes_path: str | Path,
    out_dir: str | Path,
    config: RuntimeConfig,
    *,
    tools: dict[str, str],
    command_runner: CommandRunner | None = None,
) -> RuntimeRerankResult:
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = command_runner or _run_command
    candidates = select_runtime_candidates(run_dir, passes_path, top_k=config.top_k)
    candidate_rows: list[dict[str, str]] = []
    executables: dict[str, Path] = {}

    for index, candidate in enumerate(candidates, start=1):
        row, executable = _compile_candidate(candidate, index, out_dir, config, tools, runner)
        candidate_rows.append(row)
        if executable is not None:
            executables[candidate.state_id] = executable

    trial_rows, summary_rows = benchmark_executables(
        executables,
        config,
        command_runner=runner,
    )
    winner_summary = choose_runtime_winner(summary_rows)
    candidates_by_id = {candidate.state_id: candidate for candidate in candidates}
    winner = candidates_by_id.get(winner_summary["state_id"]) if winner_summary else None
    if winner is not None:
        reason = "runtime_median"
    elif not candidates:
        reason = "no_replayable_terminal_candidates"
    elif not executables:
        reason = "no_compiled_candidates"
    else:
        reason = "no_successful_runtime_candidates"

    candidates_csv = out_dir / "runtime_candidates.csv"
    trials_csv = out_dir / "runtime_trials.csv"
    summary_csv = out_dir / "runtime_summary.csv"
    selection_md = out_dir / "runtime_selection.md"
    _write_csv(candidates_csv, RUNTIME_CANDIDATE_FIELDS, candidate_rows)
    _write_csv(trials_csv, RUNTIME_TRIAL_FIELDS, trial_rows)
    _write_csv(summary_csv, RUNTIME_SUMMARY_FIELDS, summary_rows)
    _write_selection(selection_md, config, candidate_rows, summary_rows, winner, reason)
    return RuntimeRerankResult(
        candidates=tuple(candidates),
        winner=winner,
        winner_summary=winner_summary,
        reason=reason,
        candidates_csv=candidates_csv,
        trials_csv=trials_csv,
        summary_csv=summary_csv,
        selection_md=selection_md,
    )


def _select_diverse_candidates(
    candidates: Iterable[RuntimeCandidate],
    top_k: int,
) -> list[RuntimeCandidate]:
    if top_k < 1:
        return []
    ordered = sorted(candidates, key=_objective_key)
    selected: list[RuntimeCandidate] = []
    buckets = (
        lambda candidate: (not candidate.selected_as_final, *_objective_key(candidate)),
        _objective_key,
        lambda candidate: (candidate.direct_calls, *_objective_key(candidate)),
        lambda candidate: (candidate.memory_ops, *_objective_key(candidate)),
        lambda candidate: (candidate.branches, *_objective_key(candidate)),
    )
    for key in buckets:
        if len(selected) >= top_k or not ordered:
            break
        candidate = min(ordered, key=key)
        if candidate not in selected:
            selected.append(candidate)
    for candidate in ordered:
        if len(selected) >= top_k:
            break
        if candidate not in selected:
            selected.append(candidate)
    return selected


def _representative_key(candidate: RuntimeCandidate) -> tuple:
    return (not candidate.selected_as_final, candidate.objective_value, candidate.state_id)


def _objective_key(candidate: RuntimeCandidate) -> tuple:
    return (candidate.objective_value, candidate.state_id)


def _compile_candidate(
    candidate: RuntimeCandidate,
    selection_index: int,
    out_dir: Path,
    config: RuntimeConfig,
    tools: dict[str, str],
    runner: CommandRunner,
) -> tuple[dict[str, str], Path | None]:
    build_dir = out_dir / "build" / candidate.state_id
    build_dir.mkdir(parents=True, exist_ok=True)
    object_path = build_dir / f"{candidate.state_id}.obj"
    executable_path = build_dir / f"{candidate.state_id}.exe"
    llc_code = ""
    link_code = ""
    error = ""

    llc = tools.get("llc")
    clang = tools.get("clang")
    if not llc or not clang:
        error = "runtime rerank requires llc and clang"
    else:
        opt_level = config.llc_opt_level.lstrip("-")
        llc_result = runner(
            [llc, str(candidate.ir_path), f"-{opt_level}", "-filetype=obj", "-o", str(object_path)],
            config.timeout,
            build_dir,
        )
        llc_code = str(llc_result.returncode)
        if llc_result.returncode != 0 or not object_path.exists():
            error = llc_result.stderr or "llc did not produce an object file"
        else:
            link_result = runner(
                [clang, str(object_path), "-o", str(executable_path)],
                config.timeout,
                build_dir,
            )
            link_code = str(link_result.returncode)
            if link_result.returncode != 0 or not executable_path.exists():
                error = link_result.stderr or "clang did not produce an executable"

    executable = executable_path if not error else None
    row = {
        "selection_index": str(selection_index),
        "state_id": candidate.state_id,
        "state_hash": candidate.state_hash,
        "objective_value": _format_float(candidate.objective_value),
        "selected_as_final": _bool(candidate.selected_as_final),
        "direct_calls": str(candidate.direct_calls),
        "memory_ops": str(candidate.memory_ops),
        "branches": str(candidate.branches),
        "pipeline": ",".join(candidate.pipeline),
        "ir_path": str(candidate.ir_path),
        "llc_returncode": llc_code,
        "link_returncode": link_code,
        "compile_eligible": _bool(executable is not None),
        "compile_error": error,
        "executable_path": str(executable_path) if executable is not None else "",
    }
    return row, executable


def _reconstruct_pipeline(state_id: str, states: dict[str, dict[str, str]]) -> list[str] | None:
    segments: list[list[str]] = []
    seen: set[str] = set()
    current_id = state_id
    while current_id:
        if current_id in seen:
            return None
        seen.add(current_id)
        row = states.get(current_id)
        if row is None:
            return None
        transition = row.get("transition_pass", "")
        if transition:
            segments.append([part for part in transition.split(";") if part])
        current_id = row.get("parent_state_id", "")
    return [pass_name for segment in reversed(segments) for pass_name in segment]


def _resolve_ir_path(run_dir: Path, state: dict[str, str]) -> Path | None:
    raw = state.get("ir_path", "")
    candidates = [Path(raw)] if raw else []
    if raw:
        candidates.append(run_dir / raw)
    state_dir = state.get("state_dir", "")
    if state_dir:
        candidates.extend((Path(state_dir) / "input.ll", run_dir / state_dir / "input.ll"))
    candidates.append(run_dir / "states" / state.get("state_id", "") / "input.ll")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _execute(executable: Path, config: RuntimeConfig, runner: CommandRunner) -> CommandOutcome:
    command = [part.replace("{exe}", str(executable)) for part in config.command]
    return runner(command, config.timeout, executable.parent)


def _run_command(command: list[str], timeout: int, cwd: Path) -> CommandOutcome:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return CommandOutcome(
            returncode=completed.returncode,
            time_ms=(time.perf_counter() - start) * 1000,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandOutcome(
            returncode=-1,
            time_ms=(time.perf_counter() - start) * 1000,
            stdout=_decode_timeout_output(exc.stdout),
            stderr=_decode_timeout_output(exc.stderr) or f"timed out after {timeout} seconds",
        )
    except OSError as exc:
        return CommandOutcome(
            returncode=-1,
            time_ms=(time.perf_counter() - start) * 1000,
            stderr=str(exc),
        )


def _trial_row(
    kind: str,
    trial: int,
    position: int,
    state_id: str,
    outcome: CommandOutcome,
) -> dict[str, str]:
    return {
        "kind": kind,
        "trial": str(trial),
        "order_position": str(position),
        "state_id": state_id,
        "returncode": str(outcome.returncode),
        "time_ms": f"{outcome.time_ms:.3f}",
        "stdout": outcome.stdout,
        "stderr": outcome.stderr,
    }


def _summary_row(state_id: str, values: list[float], eligible: bool, expected_trials: int) -> dict[str, str]:
    complete = eligible and len(values) == expected_trials
    return {
        "state_id": state_id,
        "eligible": _bool(complete),
        "successful_trials": str(len(values)),
        "median_ms": f"{statistics.median(values):.3f}" if values else "",
        "mean_ms": f"{statistics.mean(values):.3f}" if values else "",
        "min_ms": f"{min(values):.3f}" if values else "",
        "max_ms": f"{max(values):.3f}" if values else "",
    }


def _rotate(values: list[str], amount: int) -> list[str]:
    if not values:
        return []
    offset = amount % len(values)
    return values[offset:] + values[:offset]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_selection(
    path: Path,
    config: RuntimeConfig,
    candidate_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    winner: RuntimeCandidate | None,
    reason: str,
) -> None:
    summary_by_id = {row["state_id"]: row for row in summary_rows}
    lines = [
        "# Runtime Rerank Selection",
        "",
        f"- reason: {reason}",
        f"- top_k: {config.top_k}",
        f"- warmups: {config.warmups}",
        f"- trials: {config.trials}",
        f"- winner: {winner.state_id if winner else ''}",
        "",
        "| state | compiled | eligible | median ms | static objective |",
        "|---|---|---|---:|---:|",
    ]
    for row in candidate_rows:
        summary = summary_by_id.get(row["state_id"], {})
        lines.append(
            f"| {row['state_id']} | {row['compile_eligible']} | {summary.get('eligible', 'false')} | "
            f"{summary.get('median_ms', '')} | {row['objective_value']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return float("inf")


def _bool(value: object) -> str:
    return "true" if bool(value) else "false"


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _format_float(value: float) -> str:
    if value == float("inf"):
        return ""
    return f"{value:g}"
