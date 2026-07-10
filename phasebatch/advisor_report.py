from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .advisor_benchmarks import discover_advisor_benchmarks
from .advisor_figures import generate_advisor_figures
from .advisor_markdown import generate_advisor_markdown
from .advisor_metrics import summarize_advisor_metrics
from .artifact_cleanup import cleanup_ir_artifacts
from .batch_validation_ladder import write_batch_validation_ladder_summary
from .component_summary import summarize_components
from .dag_visualizer import visualize_dag
from .equality_summary import write_equality_tier_summary
from .opt_backend import opt_backend_metadata
from .opt_worker import WorkerError
from .optimizer import optimize_batches
from .pair_cost import write_pair_cost_summary
from .pair_scheduling import write_pair_scheduling_summary
from .reduction_summary import summarize_reduction
from .tools import collect_toolchain, find_graphviz_dot


STUDY_RUN_FIELDS = [
    "program", "input_path", "output_dir", "status", "states", "transitions", "max_depth",
    "total_time_ms", "error_stage", "error_message",
]


def run_advisor_report_zh(
    *,
    test_suite_root: Path,
    out_dir: Path,
    passes_path: Path,
    benchmark_manifest: Path | None = None,
    num_programs: int = 15,
    max_source_bytes: int = 200_000,
    selection_seed: int = 0,
    resume: bool = False,
    overwrite: bool = False,
    continue_on_error: bool = False,
    mode: str = "budgeted",
    max_rounds: int = 2,
    beam_width: int = 4,
    max_states: int = 150,
    max_batches_per_state: int = 10,
    budgeted_validation_strategy: str = "all",
    pair_testing_mode: str = "full",
    batch_construction_mode: str = "pairwise",
    batch_validation_mode: str = "auto",
    validate_batches: bool = True,
    jobs: int = 8,
    timeout: int = 15,
    max_pairs: int | None = 300,
    command: list[str] | None = None,
) -> dict:
    _validate_stable_mainline(pair_testing_mode, batch_construction_mode, batch_validation_mode, validate_batches, budgeted_validation_strategy)
    out_dir = Path(out_dir)
    passes_path = Path(passes_path).resolve()
    if not passes_path.is_file():
        raise FileNotFoundError(f"pass config not found: {passes_path}")
    _prepare_output(out_dir, resume=resume, overwrite=overwrite)
    started_at = datetime.now(timezone.utc)
    tools = collect_toolchain()
    clang = str(tools.get("tools", {}).get("clang", {}).get("path") or "clang")
    discovery = discover_advisor_benchmarks(
        test_suite_root=Path(test_suite_root),
        out_dir=out_dir,
        clang=clang,
        num_programs=num_programs,
        max_source_bytes=max_source_bytes,
        selection_seed=selection_seed,
        timeout=timeout,
        benchmark_manifest=benchmark_manifest,
    )

    previous = {row.get("program", ""): row for row in _read_csv(out_dir / "study_runs.csv")}
    run_rows: list[dict] = []
    for item in discovery["selected"]:
        program = item["name"]
        input_path = Path(item["path"])
        optimize_dir = out_dir / "programs" / program / "optimize"
        if resume and _is_successful_run(optimize_dir):
            counts = _run_counts(optimize_dir)
            old = previous.get(program, {})
            run_rows.append(
                {
                    "program": program, "input_path": str(input_path), "output_dir": str(optimize_dir), "status": "success",
                    "states": counts["states"], "transitions": counts["transitions"], "max_depth": counts["max_depth"],
                    "total_time_ms": old.get("total_time_ms", ""), "error_stage": "", "error_message": "",
                }
            )
            _write_csv(out_dir / "study_runs.csv", STUDY_RUN_FIELDS, run_rows)
            continue

        if optimize_dir.exists():
            _remove_partial_run(optimize_dir, out_dir)

        program_started = time.perf_counter()
        row = {
            "program": program, "input_path": str(input_path), "output_dir": str(optimize_dir), "status": "failed",
            "states": "", "transitions": "", "max_depth": "", "total_time_ms": "", "error_stage": "optimize", "error_message": "",
        }
        try:
            optimize_batches(
                input_path,
                optimize_dir,
                passes_path,
                mode=mode,
                objective="ir-inst-count",
                max_rounds=max_rounds,
                beam_width=beam_width,
                max_states=max_states,
                max_batches_per_state=max_batches_per_state,
                budgeted_validation_strategy="all",
                validate_batches=True,
                allow_sampled_batches=False,
                allow_bounded_validation=False,
                batch_validation_mode="auto",
                jobs=jobs,
                timeout=timeout,
                max_pairs=max_pairs,
                pair_testing_mode="full",
                pair_test_budget_per_state=0,
                batch_construction_mode="pairwise",
                exact_fail_on_incomplete=True,
                verify_final_pipeline=True,
                keep_ir_artifacts=False,
            )
            _generate_per_run_summaries(optimize_dir)
            cleanup_ir_artifacts(optimize_dir)
            counts = _run_counts(optimize_dir)
            row.update(status="success", error_stage="", states=counts["states"], transitions=counts["transitions"], max_depth=counts["max_depth"])
        except WorkerError as exc:
            row["error_message"] = _one_line(exc)
            row["total_time_ms"] = f"{(time.perf_counter() - program_started) * 1000:.3f}"
            run_rows.append(row)
            _write_csv(out_dir / "study_runs.csv", STUDY_RUN_FIELDS, run_rows)
            raise
        except Exception as exc:
            row["error_message"] = _one_line(exc)
            if not continue_on_error:
                row["total_time_ms"] = f"{(time.perf_counter() - program_started) * 1000:.3f}"
                run_rows.append(row)
                _write_csv(out_dir / "study_runs.csv", STUDY_RUN_FIELDS, run_rows)
                raise
        finally:
            if not row["total_time_ms"]:
                row["total_time_ms"] = f"{(time.perf_counter() - program_started) * 1000:.3f}"
        if not run_rows or run_rows[-1].get("program") != program:
            run_rows.append(row)
        _write_csv(out_dir / "study_runs.csv", STUDY_RUN_FIELDS, run_rows)

    ended_at = datetime.now(timezone.utc)
    metadata = _study_metadata(
        out_dir=out_dir,
        test_suite_root=Path(test_suite_root),
        passes_path=passes_path,
        tools=tools,
        command=command or sys.argv,
        started_at=started_at,
        ended_at=ended_at,
        benchmark_count=len(discovery["selected"]),
        config={
            "mode": mode, "max_rounds": max_rounds, "beam_width": beam_width, "max_states": max_states,
            "max_batches_per_state": max_batches_per_state, "budgeted_validation_strategy": "all",
            "pair_testing_mode": "full", "batch_construction_mode": "pairwise", "batch_validation_mode": "auto",
            "validate_batches": True, "jobs": jobs, "timeout": timeout, "max_pairs": max_pairs,
        },
    )
    (out_dir / "study_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_result = summarize_advisor_report_zh(out_dir)
    successes = sum(1 for row in run_rows if row["status"] == "success")
    return {
        "study_dir": str(out_dir), "programs": len(run_rows), "successes": successes,
        "failures": len(run_rows) - successes, "study_runs_csv": str(out_dir / "study_runs.csv"),
        "report": report_result,
    }


def summarize_advisor_report_zh(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    if not study_dir.is_dir():
        raise FileNotFoundError(f"advisor study directory not found: {study_dir}")
    metrics = summarize_advisor_metrics(study_dir)
    figures = generate_advisor_figures(study_dir)
    dags = generate_advisor_dags(study_dir)
    metadata = _summary_metadata(study_dir)
    markdown = generate_advisor_markdown(study_dir, metadata=metadata)
    return {"study_dir": str(study_dir), "metrics": metrics, "figures": figures, "dags": dags, **markdown}


def generate_advisor_dags(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    selected = _representative_programs(study_dir)
    dot = find_graphviz_dot()
    manifest = []
    for program, criterion in selected:
        run_dir = study_dir / "programs" / program / "optimize"
        dag_dir = study_dir / "dags" / program
        result = visualize_dag(
            run_dir,
            dag_dir,
            view="selected-neighborhood",
            formats=["svg"] if dot else ["dot"],
            max_full_nodes=150,
            include_selected_path=True,
            include_depth_overview=True,
        )
        for path in dag_dir.glob("*.dot"):
            _localize_dot(path)
            if dot and (path.name != "state_dag_full.dot" or result["unique_states"] <= 150):
                subprocess.run([dot, "-Tsvg", str(path), "-o", str(path.with_suffix(".svg"))], capture_output=True, text=True, check=False)
        warning = "；".join(result.get("warnings", []))
        if not dot:
            warning = (warning + "；" if warning else "") + "Graphviz 不存在，仅输出 DOT。"
        (dag_dir / "dag_summary.md").write_text(
            "\n".join(
                [
                    f"# {program} 状态 DAG 摘要", "", f"- 代表性选择依据：{criterion}",
                    f"- canonical IR states：{result['unique_states']}", f"- batch transitions：{result['transitions']}",
                    f"- duplicate transitions：{result['duplicate_transitions']}", f"- 警告：{warning or '无'}", "",
                    "节点是 canonical IR state，边是 correctness classifier 允许的 batch transition。DAG 可视化不产生新的正确性证据。", "",
                ]
            ),
            encoding="utf-8",
        )
        manifest.append({"program": program, "criterion": criterion, "unique_states": str(result["unique_states"]), "graphviz_available": _bool(bool(dot)), "warning": warning})
    _write_csv(study_dir / "dag_manifest.csv", ["program", "criterion", "unique_states", "graphviz_available", "warning"], manifest)
    return {"programs": len(selected), "graphviz_available": bool(dot), "dag_manifest_csv": str(study_dir / "dag_manifest.csv")}


def _generate_per_run_summaries(run_dir: Path) -> None:
    operations = [
        lambda: summarize_reduction(run_dir),
        lambda: summarize_components(run_dir=run_dir),
        lambda: write_pair_cost_summary(run_dir),
        lambda: write_pair_scheduling_summary(run_dir),
        lambda: write_batch_validation_ladder_summary(run_dir),
        lambda: write_equality_tier_summary(run_dir),
    ]
    for operation in operations:
        try:
            operation()
        except (FileNotFoundError, RuntimeError, ValueError):
            continue


def _validate_stable_mainline(pair_mode: str, construction: str, validation: str, validate: bool, strategy: str) -> None:
    if pair_mode != "full":
        raise ValueError("pair_testing_mode must be full for Advisor Data Report v1")
    if construction != "pairwise":
        raise ValueError("batch_construction_mode must be pairwise for Advisor Data Report v1")
    if validation != "auto":
        raise ValueError("batch_validation_mode must be auto for Advisor Data Report v1")
    if not validate:
        raise ValueError("validate_batches must be enabled for Advisor Data Report v1")
    if strategy != "all":
        raise ValueError("budgeted_validation_strategy must be all for Advisor Data Report v1")


def _prepare_output(out_dir: Path, *, resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("use either --resume or --overwrite, not both")
    if out_dir.exists() and overwrite:
        resolved = out_dir.resolve()
        cwd = Path.cwd().resolve()
        if resolved == cwd or resolved.parent == resolved:
            raise RuntimeError(f"refusing to remove unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
    elif out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise RuntimeError(f"output directory already exists: {out_dir}; use --resume or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


def _is_successful_run(run_dir: Path) -> bool:
    states = _read_csv(run_dir / "states.csv")
    return bool((run_dir / "optimize_summary.md").is_file() and states and all((run_dir / "states" / row.get("state_id", "")).is_dir() for row in states if row.get("state_id")))


def _remove_partial_run(run_dir: Path, study_dir: Path) -> None:
    resolved = run_dir.resolve()
    study = study_dir.resolve()
    if study not in resolved.parents or resolved.name != "optimize":
        raise RuntimeError(f"refusing to remove unsafe partial run directory: {resolved}")
    shutil.rmtree(resolved)


def _run_counts(run_dir: Path) -> dict:
    states = [row for row in _read_csv(run_dir / "states.csv") if not _is_true(row.get("is_duplicate"))]
    transitions = _read_csv(run_dir / "state_dag.csv")
    return {"states": str(len(states)), "transitions": str(len(transitions)), "max_depth": str(max((_int(row.get("depth")) for row in states), default=0))}


def _study_metadata(*, out_dir: Path, test_suite_root: Path, passes_path: Path, tools: dict, command: list[str], started_at: datetime, ended_at: datetime, benchmark_count: int, config: dict) -> dict:
    tool_rows = tools.get("tools", {}) if isinstance(tools, dict) else {}
    clang_version = str(tool_rows.get("clang", {}).get("version") or "")
    target = next((line.split(":", 1)[1].strip() for line in clang_version.splitlines() if line.startswith("Target:")), "")
    git_commit, git_dirty = _git_state()
    return {
        **config,
        "study_dir": str(out_dir.resolve()), "test_suite_root": str(test_suite_root.resolve()),
        "benchmark_count": benchmark_count, "pass_config": str(passes_path), "pass_config_sha256": _sha256(passes_path),
        "llvm_opt_version": tool_rows.get("opt", {}).get("version"), "clang_version": tool_rows.get("clang", {}).get("version"),
        "llvm_diff_version": tool_rows.get("llvm-diff", {}).get("version"), "target_triple": target,
        "git_commit": git_commit, "git_dirty": git_dirty, "normalizer_version": "normalizer-v1",
        "ir_equivalence_version": "ir-equality-v1", "command": subprocess.list2cmdline([str(item) for item in command]),
        "started_at": started_at.isoformat(), "ended_at": ended_at.isoformat(),
        "total_wall_time_ms": f"{(ended_at - started_at).total_seconds() * 1000:.3f}",
        "opt_backend": opt_backend_metadata(), "tools": tool_rows,
    }


def _representative_programs(study_dir: Path) -> list[tuple[str, str]]:
    plans = []
    criteria = [
        (_read_csv(study_dir / "batch_reduction_program_summary.csv"), "max_local_reduction_log10", "最大局部 reduction"),
        (_read_csv(study_dir / "conflict_component_program_summary.csv"), "max_component_size", "最大 conflict component"),
        (_read_csv(study_dir / "program_summary.csv"), "states", "reached states 最多"),
    ]
    seen = set()
    for rows, field, label in criteria:
        if not rows:
            continue
        winner = max(rows, key=lambda row: (_float(row.get(field)), row.get("program", "")))
        program = winner.get("program", "")
        if program and program not in seen and (study_dir / "programs" / program / "optimize").is_dir():
            seen.add(program)
            plans.append((program, label))
    return plans[:3]


def _summary_metadata(study_dir: Path) -> dict:
    metadata = _read_json(study_dir / "study_metadata.json")
    if metadata:
        return metadata
    run_metadata = []
    for path in sorted((study_dir / "programs").glob("*/optimize/metadata.json")):
        value = _read_json(path)
        if value:
            run_metadata.append(value)
    if not run_metadata:
        return {}
    first = run_metadata[0]
    tools = first.get("tools", {}) if isinstance(first.get("tools"), dict) else {}
    backend = first.get("opt_backend", {}) if isinstance(first.get("opt_backend"), dict) else {}
    return {
        "benchmark_count": len(run_metadata),
        "pass_config": first.get("pass_config", ""),
        "mode": first.get("mode", ""),
        "max_rounds": first.get("max_rounds", ""),
        "pair_testing_mode": first.get("pair_testing_mode", ""),
        "batch_construction_mode": first.get("batch_construction_mode", ""),
        "batch_validation_mode": first.get("batch_validation_mode", ""),
        "opt_backend": backend.get("backend", "worker"),
        "llvm_opt_version": tools.get("opt", {}).get("version", "") if isinstance(tools.get("opt"), dict) else "",
        "clang_version": tools.get("clang", {}).get("version", "") if isinstance(tools.get("clang"), dict) else "",
        "llvm_diff_version": tools.get("llvm-diff", {}).get("version", "") if isinstance(tools.get("llvm-diff"), dict) else "",
        "normalizer_version": "normalizer-v1",
        "ir_equivalence_version": "ir-equality-v1",
        "metadata_source": "per-program optimize metadata",
    }


def _localize_dot(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "Compressed Batch-State DAG Overview": "压缩后的 Batch-State DAG 深度概览",
        "Compressed Batch-State DAG": "压缩后的 Batch-State DAG",
        "Selected Batch-State DAG": "选中路径邻域",
        "transitions=": "转移=", "states=": "状态=", "dup=": "重复=",
        "active=": "active=", "sensitive=": "顺序敏感=", "unknown=": "unknown=",
        "duplicate ->": "重复到达 ->",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    path.write_text(text, encoding="utf-8")


def _git_state() -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False, timeout=10).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "", None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())[:4000]


def _float(value: object) -> float:
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


def _int(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _bool(value: bool) -> str:
    return "true" if value else "false"
