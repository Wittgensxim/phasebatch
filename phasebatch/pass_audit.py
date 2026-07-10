from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .batch_objective import count_ir_instructions
from .normalizer import canonical_hash
from .pass_config import PassSpec, load_pass_config
from .runner import prepare_input_ir, run_opt
from .schema import RunResult
from .tools import collect_toolchain, write_metadata


PASS_AUDIT_FIELDS = [
    "pass",
    "category",
    "stage",
    "enabled",
    "candidate_index",
    "candidate_pipeline",
    "resolved_pipeline",
    "recognized_by_opt",
    "valid_on_input",
    "active_on_input",
    "input_hash",
    "output_hash",
    "ir_inst_before",
    "ir_inst_after",
    "inst_delta",
    "time_ms",
    "failure_kind",
    "stderr_summary",
    "recommended_action",
]

AUDIT_INVALID_PASS_FIELDS = [
    "pass",
    "category",
    "stage",
    "attempted_candidates",
    "failure_kind",
    "stderr_summary",
]

OptRunner = Callable[[str, Path, str, Path, int], RunResult]


def audit_passes(
    input_path: Path,
    passes_path: Path,
    out_dir: Path,
    *,
    timeout: int = 10,
    jobs: int = 1,
    tools: dict[str, str] | None = None,
    opt_runner: OptRunner | None = None,
) -> dict:
    input_path = Path(input_path)
    passes_path = Path(passes_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tools is None:
        metadata = collect_toolchain()
        tools = _tool_paths(metadata)
    else:
        metadata = {"tools": {name: {"path": path, "version": None} for name, path in tools.items()}}
    write_metadata(out_dir, metadata)

    input_ll = _prepare_audit_input(input_path, out_dir, tools, timeout)
    specs = load_pass_config(passes_path)
    opt = tools.get("opt")
    if not opt:
        raise RuntimeError("opt tool path is required for pass audit")
    opt_runner = opt_runner or run_opt_pipeline

    input_hash = canonical_hash(input_ll)
    inst_before = count_ir_instructions(input_ll)

    def run_one(item: tuple[int, PassSpec]) -> dict:
        index, spec = item
        return _audit_one_pass(index, spec, input_ll, out_dir, opt, timeout, opt_runner, input_hash, inst_before)

    indexed_specs = list(enumerate(specs))
    if jobs > 1 and len(indexed_specs) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            audit_rows = list(executor.map(run_one, indexed_specs))
    else:
        audit_rows = [run_one(item) for item in indexed_specs]

    invalid_rows = [
        {
            "pass": row["pass"],
            "category": row["category"],
            "stage": row["stage"],
            "attempted_candidates": row.get("_attempted_candidates", ""),
            "failure_kind": row.get("failure_kind", ""),
            "stderr_summary": row.get("stderr_summary", ""),
        }
        for row in audit_rows
        if row.get("valid_on_input") != "true"
    ]

    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in audit_rows]
    _write_csv(out_dir / "pass_audit.csv", PASS_AUDIT_FIELDS, public_rows)
    _write_csv(out_dir / "invalid_passes.csv", AUDIT_INVALID_PASS_FIELDS, invalid_rows)
    valid_rows = [row for row in public_rows if row.get("valid_on_input") == "true"]
    if valid_rows:
        _write_resolved_config(out_dir / "resolved_passes.yaml", valid_rows)
    _write_summary(out_dir / "pass_audit_summary.md", input_path, passes_path, public_rows, invalid_rows)

    return {
        "out_dir": str(out_dir),
        "pass_audit_csv": str(out_dir / "pass_audit.csv"),
        "invalid_passes_csv": str(out_dir / "invalid_passes.csv"),
        "resolved_passes_yaml": str(out_dir / "resolved_passes.yaml") if valid_rows else "",
        "summary_md": str(out_dir / "pass_audit_summary.md"),
        "total_passes": len(public_rows),
        "valid_passes": len(valid_rows),
        "invalid_passes": len(invalid_rows),
        "active_on_input": sum(1 for row in valid_rows if row.get("active_on_input") == "true"),
        "dormant_on_input": sum(1 for row in valid_rows if row.get("active_on_input") == "false"),
    }


def run_opt_pipeline(opt: str, input_ll: Path, pipeline: str, output_ll: Path, timeout: int) -> RunResult:
    return run_opt(opt, input_ll, [pipeline], output_ll, timeout)


def _audit_one_pass(
    index: int,
    spec: PassSpec,
    input_ll: Path,
    out_dir: Path,
    opt: str,
    timeout: int,
    opt_runner: OptRunner,
    input_hash: str,
    inst_before: int,
) -> dict:
    attempted: list[str] = []
    errors: list[str] = []
    failure_kind = ""
    total_time_ms = 0.0

    for candidate_index, candidate in enumerate(spec.pipeline_candidates):
        attempted.append(candidate)
        output_ll = out_dir / "attempts" / _safe_name(f"{index:04d}_{spec.name}") / f"candidate_{candidate_index}.ll"
        result = opt_runner(opt, input_ll, candidate, output_ll, timeout)
        total_time_ms += result.time_ms

        if result.success and output_ll.exists():
            output_hash = canonical_hash(output_ll)
            inst_after = count_ir_instructions(output_ll)
            active = output_hash != input_hash
            return {
                "pass": spec.name,
                "category": spec.category,
                "stage": spec.stage,
                "enabled": _bool(spec.enabled),
                "candidate_index": str(candidate_index),
                "candidate_pipeline": candidate,
                "resolved_pipeline": candidate,
                "recognized_by_opt": "true",
                "valid_on_input": "true",
                "active_on_input": _bool(active),
                "input_hash": input_hash,
                "output_hash": output_hash,
                "ir_inst_before": str(inst_before),
                "ir_inst_after": str(inst_after),
                "inst_delta": str(inst_after - inst_before),
                "time_ms": _fmt_ms(total_time_ms),
                "failure_kind": "",
                "stderr_summary": "",
                "recommended_action": _recommended_action(spec, candidate_index, active),
                "_attempted_candidates": ";".join(attempted),
            }

        failure_kind = "output_missing" if result.success else (result.failure_kind or "failed")
        errors.append(_stderr_summary(result.stderr or result.stdout or failure_kind))

    return {
        "pass": spec.name,
        "category": spec.category,
        "stage": spec.stage,
        "enabled": _bool(spec.enabled),
        "candidate_index": "",
        "candidate_pipeline": "",
        "resolved_pipeline": "",
        "recognized_by_opt": "false",
        "valid_on_input": "false",
        "active_on_input": "false",
        "input_hash": input_hash,
        "output_hash": "",
        "ir_inst_before": str(inst_before),
        "ir_inst_after": "",
        "inst_delta": "",
        "time_ms": _fmt_ms(total_time_ms),
        "failure_kind": failure_kind or "failed",
        "stderr_summary": " | ".join(error for error in errors if error)[:500],
        "recommended_action": _invalid_action(spec),
        "_attempted_candidates": ";".join(attempted),
    }


def _prepare_audit_input(input_path: Path, out_dir: Path, tools: dict[str, str], timeout: int) -> Path:
    suffix = input_path.suffix.lower()
    if suffix == ".ll":
        return input_path
    if suffix == ".c":
        return prepare_input_ir(input_path, out_dir, tools, timeout)
    raise RuntimeError(f"unsupported input type '{input_path.suffix}': {input_path}")


def _recommended_action(spec: PassSpec, candidate_index: int, active: bool) -> str:
    if spec.category in {"ipo", "backend", "module"}:
        return "move_to_later_stage"
    first = spec.pipeline_candidates[0] if spec.pipeline_candidates else ""
    if candidate_index > 0 and _looks_nested(spec.pipeline_candidates[candidate_index]) and first == spec.name:
        return "needs_nested_pipeline"
    if not active:
        return "keep_dormant"
    return "keep"


def _invalid_action(spec: PassSpec) -> str:
    if spec.category in {"ipo", "backend", "module"}:
        return "move_to_later_stage"
    return "drop_invalid"


def _looks_nested(pipeline: str) -> bool:
    return "(" in pipeline and ")" in pipeline


def _write_resolved_config(path: Path, rows: list[dict]) -> None:
    lines = ["passes:"]
    for row in rows:
        lines.extend(
            [
                f"  - name: {_yaml_scalar(row.get('pass', ''))}",
                f"    pipeline: {_yaml_scalar(row.get('resolved_pipeline', ''))}",
                f"    category: {_yaml_scalar(row.get('category', 'unknown') or 'unknown')}",
                f"    stage: {_yaml_scalar(row.get('stage', '') or '')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, input_path: Path, passes_path: Path, rows: list[dict], invalid_rows: list[dict]) -> None:
    valid_rows = [row for row in rows if row.get("valid_on_input") == "true"]
    active_rows = [row for row in valid_rows if row.get("active_on_input") == "true"]
    dormant_rows = [row for row in valid_rows if row.get("active_on_input") == "false"]
    lines = [
        "# Pass Audit Summary",
        "",
        f"- input: {input_path}",
        f"- pass config: {passes_path}",
        f"- total passes: {len(rows)}",
        f"- valid passes: {len(valid_rows)}",
        f"- invalid passes: {len(invalid_rows)}",
        f"- active on input: {len(active_rows)}",
        f"- dormant on input: {len(dormant_rows)}",
        "",
        "## Valid Passes",
        "",
        "| pass | category | stage | resolved pipeline | active | inst delta |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in valid_rows:
        lines.append(
            "| {pass_name} | {category} | {stage} | {pipeline} | {active} | {delta} |".format(
                pass_name=_md(row.get("pass", "")),
                category=_md(row.get("category", "")),
                stage=_md(row.get("stage", "")),
                pipeline=_md(row.get("resolved_pipeline", "")),
                active=_md(row.get("active_on_input", "")),
                delta=_md(row.get("inst_delta", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Invalid Passes",
            "",
            "| pass | attempted candidates | failure kind | stderr summary |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in invalid_rows:
        lines.append(
            "| {pass_name} | {attempted} | {failure} | {stderr} |".format(
                pass_name=_md(row.get("pass", "")),
                attempted=_md(row.get("attempted_candidates", "")),
                failure=_md(row.get("failure_kind", "")),
                stderr=_md(row.get("stderr_summary", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Pass validity means the local opt accepted the pass pipeline on this input. Dormant passes are still retained because they may become active in later states.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _tool_paths(metadata: dict) -> dict[str, str]:
    return {
        name: details["path"]
        for name, details in metadata.get("tools", {}).items()
        if isinstance(details, dict) and details.get("path")
    }


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _stderr_summary(text: str) -> str:
    return " ".join(str(text or "").split())[:240]


def _fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _md(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _yaml_scalar(value: object) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char in text for char in ":#[]{}&,*?|-<>=!%@`'\" \t"):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
