from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .schema import OBJECTIVE_EVAL_FIELDS, OBJECTIVE_SIGNAL_FIELDS


SUPPORTED_OBJECTIVES = {"ir-inst-count"}
OBJECTIVE_SIGNAL_NOTE = "objective signal only; not used as commutation proof"
OBJECTIVE_SUMMARY_REMINDER = (
    "Objective signals are used only for evaluation and ranking. "
    "They are not used as commutation or independence proof."
)


def eval_batch_objectives(run_dir: Path, objective: str = "ir-inst-count", recursive: bool = False) -> dict:
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")

    run_dir = Path(run_dir)
    if recursive:
        program_dirs = _recursive_program_dirs(run_dir)
        all_rows: list[dict] = []
        for program_dir in program_dirs:
            all_rows.extend(_eval_one_program(program_dir, objective))
        aggregate_path = run_dir / "aggregate_objective_signal.csv"
        summary_path = run_dir / "objective_summary.md"
        _write_csv(aggregate_path, OBJECTIVE_SIGNAL_FIELDS, all_rows)
        _write_objective_signal_summary(summary_path, all_rows)
        return {
            "run_dir": str(run_dir),
            "objective": objective,
            "recursive": True,
            "program_dirs": len(program_dirs),
            "rows": len(all_rows),
            "aggregate_objective_signal_csv": str(aggregate_path),
            "objective_summary_md": str(summary_path),
        }

    rows = _eval_one_program(run_dir, objective)
    summary_path = run_dir / "objective_summary.md"
    _write_objective_signal_summary(summary_path, rows)
    return {
        "run_dir": str(run_dir),
        "objective": objective,
        "recursive": False,
        "rows": len(rows),
        "objective_signal_csv": str(run_dir / "objective_signal.csv"),
        "objective_eval_csv": str(run_dir / "objective_eval.csv"),
        "objective_summary_md": str(summary_path),
    }


def count_ir_instructions(ir_path: Path) -> int:
    """Return an approximate IR instruction count for real instructions in function bodies."""
    count = 0
    in_function = False
    for raw_line in Path(ir_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if not in_function:
            if line.startswith("define ") and "{" in line:
                in_function = True
            continue
        if line.startswith("}"):
            in_function = False
            continue
        if _is_non_instruction_function_line(line):
            continue
        if _starts_instruction(line):
            count += 1
    return count


def _eval_one_program(run_dir: Path, objective: str) -> list[dict]:
    states = _read_csv(run_dir / "states.csv")
    transitions = _read_csv(run_dir / "batch_state_transitions.csv")
    if not states:
        raise RuntimeError(f"missing or empty states.csv under {run_dir}")
    if not transitions:
        raise RuntimeError(f"missing or empty batch_state_transitions.csv under {run_dir}")

    states_by_id = {row.get("state_id", ""): row for row in states if row.get("state_id")}
    rows = []
    for transition in transitions:
        parent_state_id = transition.get("parent_state_id", "")
        child_state_id = transition.get("child_state_id", "")
        parent_state = states_by_id.get(parent_state_id, {})
        parent_ir = _parent_input_path(parent_state_id, parent_state, run_dir)
        child_ir = _child_input_path_for_transition(transition, run_dir)
        correctness = _correctness_for_transition(parent_state, run_dir, transition)
        missing_notes = []
        if not parent_ir.exists():
            missing_notes.append("missing_parent_ir")
        if not child_ir.exists():
            missing_notes.append("missing_child_ir")
        if missing_notes:
            parent_count = child_count = inst_delta = None
            reduction_pct = None
            objective_note = f"{OBJECTIVE_SIGNAL_NOTE}; {';'.join(missing_notes)}"
        else:
            parent_count = count_ir_instructions(parent_ir)
            child_count = count_ir_instructions(child_ir)
            inst_delta = child_count - parent_count
            reduction_pct = _reduction_pct(parent_count, child_count)
            objective_note = OBJECTIVE_SIGNAL_NOTE
        rows.append(
            {
                "program": transition.get("program") or run_dir.name,
                "parent_state_id": parent_state_id,
                "child_state_id": child_state_id,
                "transition_kind": "batch",
                "batch_id": transition.get("batch_id", ""),
                "batch_passes": transition.get("batch_passes", ""),
                "batch_size": transition.get("batch_size", ""),
                "validation_status": transition.get("validation_status", ""),
                "correctness_class": correctness.get("correctness_class", ""),
                "parent_ir_path": str(parent_ir),
                "child_ir_path": str(child_ir),
                "ir_inst_before": "" if parent_count is None else str(parent_count),
                "ir_inst_after": "" if child_count is None else str(child_count),
                "ir_inst_delta": "" if inst_delta is None else str(inst_delta),
                "ir_inst_reduction_pct": "" if reduction_pct is None else _format_float(reduction_pct),
                "objective_kind": objective,
                "objective_note": objective_note,
            }
        )

    _write_csv(run_dir / "objective_signal.csv", OBJECTIVE_SIGNAL_FIELDS, rows)
    _write_legacy_objective_eval(run_dir / "objective_eval.csv", rows, transitions)
    return rows


def _recursive_program_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "batch_state_transitions.csv").exists():
        return [run_dir]
    return sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and (path / "batch_state_transitions.csv").exists()
    )


def _correctness_for_transition(parent_state: dict, run_dir: Path, transition: dict) -> dict:
    parent_dir = _state_dir(parent_state, run_dir)
    batch_id = transition.get("batch_id", "")
    for row in _read_csv(parent_dir / "batch_correctness.csv"):
        if row.get("batch_id") == batch_id:
            return row
    return {}


def _parent_input_path(state_id: str, state: dict, run_dir: Path) -> Path:
    direct = run_dir / "states" / state_id / "input.ll"
    if direct.exists():
        return direct
    try:
        return _state_input_path(state, run_dir)
    except FileNotFoundError:
        return direct


def _child_input_path_for_transition(transition: dict, run_dir: Path) -> Path:
    child_state_id = transition.get("child_state_id", "")
    duplicate_of = transition.get("duplicate_of", "")
    target_state_id = duplicate_of if _is_true(transition.get("is_duplicate")) and duplicate_of else child_state_id
    return run_dir / "states" / target_state_id / "input.ll"


def _state_input_path(state: dict, run_dir: Path) -> Path:
    state_id = state.get("state_id", "")
    state_dir = _state_dir(state, run_dir)
    candidate = state_dir / "input.ll"
    if candidate.exists():
        return candidate

    ir_path = _resolve_path(state.get("ir_path", ""), run_dir)
    if ir_path and ir_path.exists():
        return ir_path

    fallback = run_dir / "states" / state_id / "input.ll"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"could not find input.ll for state {state_id or '<unknown>'}")


def _state_dir(state: dict, run_dir: Path) -> Path:
    state_id = state.get("state_id", "")
    value = state.get("state_dir", "")
    candidates = []
    if value:
        path = Path(value)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(run_dir.parent / path)
            candidates.append(run_dir / path)
    candidates.append(run_dir / "states" / state_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return run_dir / "states" / state_id


def _resolve_path(value: str, run_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = run_dir / path
    if candidate.exists():
        return candidate
    parent_candidate = run_dir.parent / path
    if parent_candidate.exists():
        return parent_candidate
    return path


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _is_non_instruction_function_line(line: str) -> bool:
    if line.endswith(":"):
        return True
    if line.startswith((";", "!", "attributes ")):
        return True
    return False


_INSTRUCTION_OPCODES = {
    "add",
    "addrspacecast",
    "alloca",
    "and",
    "ashr",
    "atomicrmw",
    "bitcast",
    "br",
    "call",
    "callbr",
    "catchret",
    "catchswitch",
    "cleanupret",
    "cmpxchg",
    "extractelement",
    "extractvalue",
    "fadd",
    "fcmp",
    "fdiv",
    "fence",
    "fmul",
    "fneg",
    "fpext",
    "fptosi",
    "fptoui",
    "fptrunc",
    "freeze",
    "frem",
    "fsub",
    "getelementptr",
    "icmp",
    "indirectbr",
    "insertelement",
    "insertvalue",
    "inttoptr",
    "invoke",
    "landingpad",
    "load",
    "lshr",
    "mul",
    "musttail",
    "or",
    "phi",
    "ptrtoint",
    "resume",
    "ret",
    "sdiv",
    "select",
    "sext",
    "shl",
    "shufflevector",
    "sitofp",
    "srem",
    "store",
    "sub",
    "switch",
    "tail",
    "trunc",
    "udiv",
    "uitofp",
    "unreachable",
    "urem",
    "va_arg",
    "xor",
    "zext",
}


def _starts_instruction(line: str) -> bool:
    if " = " in line:
        return True
    first = line.split(None, 1)[0].rstrip(",")
    return first in _INSTRUCTION_OPCODES


def _reduction_pct(parent_count: int, child_count: int) -> float:
    if parent_count <= 0:
        return 0.0
    return ((parent_count - child_count) / parent_count) * 100.0


def _write_objective_signal_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Objective Signal Summary",
        "",
        "## Reminder",
        "",
        f'"{OBJECTIVE_SUMMARY_REMINDER}"',
        "",
        "## Aggregate Table",
        "",
    ]
    lines.extend(_aggregate_table(rows))
    lines.extend(["", "## Top Improvements", ""])
    lines.extend(_transition_table(_top_improvements(rows)))
    lines.extend(["", "## Worsened Transitions", ""])
    lines.extend(_transition_table(_top_worsened(rows)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_table(rows: list[dict]) -> list[str]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("program", "")].append(row)

    table_rows = []
    for program in sorted(buckets):
        program_rows = buckets[program]
        deltas = [_to_int(row.get("ir_inst_delta")) for row in program_rows]
        reductions = [_to_float(row.get("ir_inst_reduction_pct")) for row in program_rows]
        table_rows.append(
            [
                program,
                str(len(program_rows)),
                _format_float(sum(deltas) / len(deltas) if deltas else 0.0),
                _format_float(max(reductions, default=0.0)),
                str(sum(1 for delta in deltas if delta > 0)),
                str(sum(1 for delta in deltas if delta < 0)),
                str(sum(1 for delta in deltas if delta == 0)),
            ]
        )
    return _markdown_table(
        [
            "program",
            "transitions",
            "avg inst delta",
            "best reduction pct",
            "worsened transitions",
            "improved transitions",
            "unchanged transitions",
        ],
        table_rows,
    )


def _top_improvements(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (_to_float(row.get("ir_inst_reduction_pct")), -_to_int(row.get("ir_inst_delta"))), reverse=True)[:10]


def _top_worsened(rows: list[dict]) -> list[dict]:
    worsened = [row for row in rows if _to_float(row.get("ir_inst_reduction_pct")) < 0]
    return sorted(worsened, key=lambda row: _to_float(row.get("ir_inst_reduction_pct")))[:10]


def _transition_table(rows: list[dict]) -> list[str]:
    return _markdown_table(
        [
            "program",
            "parent",
            "child",
            "batch",
            "inst delta",
            "reduction pct",
            "validation",
            "batch passes",
        ],
        [
            [
                row.get("program", ""),
                row.get("parent_state_id", ""),
                row.get("child_state_id", ""),
                row.get("batch_id", ""),
                row.get("ir_inst_delta", ""),
                row.get("ir_inst_reduction_pct", ""),
                row.get("validation_status", ""),
                row.get("batch_passes", ""),
            ]
            for row in rows
        ],
    )


def _write_legacy_objective_eval(path: Path, signal_rows: list[dict], transitions: list[dict]) -> None:
    by_key = {
        (row.get("parent_state_id"), row.get("child_state_id"), row.get("batch_id")): transition
        for row, transition in zip(signal_rows, transitions)
    }
    rows = []
    for row in signal_rows:
        transition = by_key.get((row.get("parent_state_id"), row.get("child_state_id"), row.get("batch_id")), {})
        rows.append(
            {
                "program": row.get("program", ""),
                "objective": row.get("objective_kind", ""),
                "parent_state_id": row.get("parent_state_id", ""),
                "child_state_id": row.get("child_state_id", ""),
                "batch_id": row.get("batch_id", ""),
                "batch_passes": row.get("batch_passes", ""),
                "batch_size": row.get("batch_size", ""),
                "validation_status": row.get("validation_status", ""),
                "parent_hash": transition.get("parent_hash", ""),
                "child_hash": transition.get("child_hash", ""),
                "parent_inst_count": row.get("ir_inst_before", ""),
                "child_inst_count": row.get("ir_inst_after", ""),
                "inst_delta": row.get("ir_inst_delta", ""),
                "inst_reduction_pct": row.get("ir_inst_reduction_pct", ""),
                "is_duplicate": transition.get("is_duplicate", ""),
                "duplicate_of": transition.get("duplicate_of", ""),
            }
        )
    _write_csv(path, OBJECTIVE_EVAL_FIELDS, rows)


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
    if not rows:
        return ["none"]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(str(cell) for cell in row)} |" for row in rows)
    return lines


def _to_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _format_float(value: float) -> str:
    return f"{value:.2f}"
