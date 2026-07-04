from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from .batcher import build_batch_family, validate_batch_candidates
from .config import load_passes
from .graph import cluster_distribution_rows, write_cluster_distribution
from .normalizer import canonical_hash
from .pair_tester import run_pair_tests
from .profiler import profile_passes, validate_passes
from .relation import annotate_pair_relations, write_pair_relations
from .report import write_aggregate_report, write_per_state_summary, write_summary
from .runner import prepare_input_ir
from .tools import collect_toolchain, write_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phasebatch",
        description="LLVM phase-ordering data MVP command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one C or LLVM IR input.")
    _add_common_args(analyze)
    analyze.add_argument("--input", required=True, help="Input .c or .ll file.")
    analyze.set_defaults(func=_run_analyze)

    batch = subparsers.add_parser("batch", help="Analyze multiple C or LLVM IR inputs.")
    _add_common_args(batch)
    batch.add_argument("--inputs", required=True, nargs="+", help="Input .c or .ll files.")
    batch.set_defaults(func=_run_batch)

    explore = subparsers.add_parser("explore", help="Explore multiple IR states.")
    _add_common_args(explore)
    explore.add_argument("--input", required=True, help="Input .c or .ll file.")
    explore.add_argument("--max-depth", type=int, default=1, help="Maximum exploration depth.")
    explore.add_argument(
        "--frontier-policy",
        choices=["all-active", "top-k-change", "sensitive-first"],
        default="all-active",
        help="Policy for choosing successor states.",
    )
    explore.add_argument("--top-k", type=int, default=5, help="Frontier cap for top-k policies.")
    explore.set_defaults(func=_run_explore)

    explore_batches_parser = subparsers.add_parser("explore-batches", help="Explore states using batch candidates.")
    _add_common_args(explore_batches_parser)
    explore_batches_parser.add_argument("--input", required=True, help="Input .c or .ll file.")
    explore_batches_parser.add_argument("--max-depth", type=int, default=1, help="Maximum batch exploration depth.")
    explore_batches_parser.add_argument("--max-component-size", type=int, default=10, help="Maximum exact conflict component size.")
    explore_batches_parser.add_argument("--max-batch-candidates", type=int, default=50, help="Maximum batch candidates per state.")
    explore_batches_parser.add_argument("--max-batches-per-state", type=int, default=20, help="Maximum batch candidates to apply per state.")
    explore_batches_parser.add_argument("--max-frontier-states", type=int, default=20, help="Maximum non-duplicate frontier states to keep after each depth.")
    explore_batches_parser.add_argument(
        "--batch-frontier-policy",
        choices=["all", "largest-batch", "certified-first", "diverse-hash"],
        default="all",
        help="Policy for selecting batches and frontier states.",
    )
    explore_batches_parser.add_argument("--validate-batches", action="store_true", help="Validate batch candidates before applying them.")
    explore_batches_parser.add_argument(
        "--allow-sampled-batches",
        action="store_true",
        help="When validating, also apply sampled_same batch candidates.",
    )
    explore_batches_parser.set_defaults(func=_run_explore_batches)

    batchify = subparsers.add_parser("batchify", help="Build batch candidates for one analyzed state.")
    batchify.add_argument("--state-dir", required=True, help="State directory containing pass_profile.csv and pair_relation.csv.")
    batchify.add_argument("--max-component-size", type=int, default=10, help="Maximum exact conflict component size.")
    batchify.add_argument("--max-batch-candidates", type=int, default=200, help="Maximum global batch candidates to emit.")
    batchify.add_argument("--validate-batches", action="store_true", help="Run opt to validate candidate order hashes.")
    batchify.set_defaults(func=_run_batchify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--passes", required=True, help="Path to pass config YAML.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")


def _run_analyze(args: argparse.Namespace) -> int:
    result = run_analysis(
        Path(args.input),
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
    )
    print(
        "analyzed {program}: valid={valid_passes} active={active_passes} "
        "pairs={pair_rows} summary={summary_path}".format(**result)
    )
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    result = run_batch(
        args.inputs,
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
    )
    print(f"batch analyzed {len(result['program_dirs'])} programs: {result['aggregate_summary']}")
    return 0


def _run_explore(args: argparse.Namespace) -> int:
    from .explorer import explore_states

    result = explore_states(
        Path(args.input),
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        max_depth=args.max_depth,
        frontier_policy=args.frontier_policy,
        top_k=args.top_k,
    )
    print(
        "explored {program}: states={states} transitions={transitions} "
        "states_csv={states_csv}".format(**result)
    )
    return 0


def _run_explore_batches(args: argparse.Namespace) -> int:
    result = run_explore_batches(
        Path(args.input),
        Path(args.out),
        Path(args.passes),
        jobs=args.jobs,
        timeout=args.timeout,
        max_pairs=args.max_pairs,
        max_depth=args.max_depth,
        max_component_size=args.max_component_size,
        max_batch_candidates=args.max_batch_candidates,
        max_batches_per_state=args.max_batches_per_state,
        max_frontier_states=args.max_frontier_states,
        batch_frontier_policy=args.batch_frontier_policy,
        validate_batches=args.validate_batches,
        allow_sampled_batches=args.allow_sampled_batches,
    )
    print(
        "batch-explored {program}: states={states} batch_transitions={batch_transitions} "
        "states_csv={states_csv}".format(**result)
    )
    return 0


def _run_batchify(args: argparse.Namespace) -> int:
    result = run_batchify(
        Path(args.state_dir),
        max_component_size=args.max_component_size,
        max_batch_candidates=args.max_batch_candidates,
        validate_batches=args.validate_batches,
    )
    print(
        "batchified {state_id}: candidates={batch_candidates} "
        "summary={batch_summary_md}".format(**result)
    )
    return 0


def run_batchify(
    state_dir: Path,
    max_component_size: int = 10,
    max_batch_candidates: int = 200,
    validate_batches: bool = False,
) -> dict:
    state_dir = Path(state_dir)
    result = build_batch_family(
        state_dir,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
    )
    if not validate_batches:
        return result

    tools = _tool_paths(collect_toolchain())
    validation = validate_batch_candidates(state_dir, tools, timeout=10, jobs=1)
    result.update(validation)
    return result


def run_explore_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_batches_per_state: int = 20,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
) -> dict:
    return explore_batches(
        input_path,
        out_dir,
        passes_path,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        max_depth=max_depth,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
        max_batches_per_state=max_batches_per_state,
        max_frontier_states=max_frontier_states,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        allow_sampled_batches=allow_sampled_batches,
    )


def explore_batches(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    *,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    max_depth: int,
    max_component_size: int,
    max_batch_candidates: int,
    max_batches_per_state: int = 20,
    max_frontier_states: int = 20,
    batch_frontier_policy: str = "all",
    validate_batches: bool = False,
    allow_sampled_batches: bool = False,
) -> dict:
    from .batch_explorer import explore_batches as explore_batches_impl

    return explore_batches_impl(
        input_path,
        out_dir,
        passes_path,
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        max_depth=max_depth,
        max_component_size=max_component_size,
        max_batch_candidates=max_batch_candidates,
        max_batches_per_state=max_batches_per_state,
        max_frontier_states=max_frontier_states,
        batch_frontier_policy=batch_frontier_policy,
        validate_batches=validate_batches,
        allow_sampled_batches=allow_sampled_batches,
    )


def run_analysis(
    input_path: Path,
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configured_passes = load_passes(passes_path)

    metadata = collect_toolchain()
    metadata.update(
        {
            "input": str(input_path),
            "out_dir": str(out_dir),
            "pass_config": str(passes_path),
            "configured_pass_count": len(configured_passes),
            "jobs": jobs,
            "timeout": timeout,
            "max_pairs": max_pairs,
        }
    )
    write_metadata(out_dir, metadata)
    tools = _tool_paths(metadata)

    input_ll = prepare_input_ir(Path(input_path), out_dir, tools, timeout)
    state_hash = canonical_hash(input_ll)
    program = out_dir.name
    metadata["state_hash"] = state_hash
    write_metadata(out_dir, metadata)

    valid_passes, invalid_rows = validate_passes(input_ll, configured_passes, tools, out_dir, timeout)

    result = analyze_state(
        input_ll,
        out_dir,
        tools,
        valid_passes=valid_passes,
        invalid_rows=invalid_rows,
        configured_pass_count=len(configured_passes),
        jobs=jobs,
        timeout=timeout,
        max_pairs=max_pairs,
        program=program,
        state_id="S0000",
        depth=0,
        parent_state_id="",
        transition_pass="",
    )

    metadata = _read_metadata(out_dir)
    metadata.update(
        {
            "valid_passes": result.get("valid_passes"),
            "invalid_passes": result.get("invalid_passes"),
            "active_passes": result.get("active_passes"),
            "pair_rows": result.get("pair_rows"),
            "summary": result.get("summary_path"),
            "total_time_ms": result.get("total_time_ms"),
        }
    )
    write_metadata(out_dir, metadata)
    return result


def analyze_state(
    input_ll: Path,
    out_dir: Path,
    tools: dict,
    *,
    valid_passes: list[str],
    invalid_rows: list[dict],
    configured_pass_count: int,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
    program: str,
    state_id: str,
    depth: int,
    parent_state_id: str,
    transition_pass: str,
) -> dict:
    start = time.perf_counter()
    input_ll = Path(input_ll)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_hash = canonical_hash(input_ll)

    metadata = _read_metadata(out_dir)
    metadata.update(
        {
            "input": str(input_ll),
            "state_hash": state_hash,
            "state_id": state_id,
            "depth": depth,
            "parent_state_id": parent_state_id,
            "transition_pass": transition_pass,
        }
    )
    write_metadata(out_dir, metadata)

    profile_start = time.perf_counter()
    profile_rows = profile_passes(
        input_ll,
        valid_passes,
        tools,
        out_dir,
        jobs,
        timeout,
        program=program,
        state_id=state_id,
        depth=depth,
        parent_state_id=parent_state_id,
        transition_pass=transition_pass,
    )
    profile_time_ms = (time.perf_counter() - profile_start) * 1000
    active_profiles = [row for row in profile_rows if row.get("success") == "true" and row.get("active") == "true"]

    pair_start = time.perf_counter()
    pair_rows = run_pair_tests(input_ll, active_profiles, tools, out_dir, jobs, timeout, max_pairs)
    profile_map = {row["pass"]: row for row in profile_rows}
    pair_rows = annotate_pair_relations(pair_rows, profile_map)
    write_pair_relations(out_dir / "pair_relation.csv", pair_rows)
    pair_time_ms = (time.perf_counter() - pair_start) * 1000

    cluster_rows = cluster_distribution_rows(pair_rows, program, state_hash)
    write_cluster_distribution(out_dir / "cluster_distribution.csv", cluster_rows)

    total_time_ms = (time.perf_counter() - start) * 1000
    write_per_state_summary(
        out_dir,
        program,
        state_hash,
        state_id=state_id,
        depth=depth,
        parent_state_id=parent_state_id,
        transition_pass=transition_pass,
        pass_set_size=configured_pass_count,
        valid_passes=len(valid_passes),
        invalid_passes=len(invalid_rows),
        profile_time_ms=profile_time_ms,
        pair_time_ms=pair_time_ms,
        total_time_ms=total_time_ms,
    )
    summary = write_summary(out_dir)

    metadata.update(
        {
            "valid_passes": len(valid_passes),
            "invalid_passes": len(invalid_rows),
            "active_passes": len(active_profiles),
            "pair_rows": len(pair_rows),
            "summary": str(summary),
            "total_time_ms": total_time_ms,
        }
    )
    write_metadata(out_dir, metadata)
    return {
        "program": program,
        "out_dir": str(out_dir),
        "state_id": state_id,
        "depth": depth,
        "parent_state_id": parent_state_id,
        "transition_pass": transition_pass,
        "valid_passes": len(valid_passes),
        "active_passes": len(active_profiles),
        "pair_rows": len(pair_rows),
        "summary_path": str(summary),
        "total_time_ms": total_time_ms,
    }


def run_batch(
    inputs: list[str],
    out_dir: Path,
    passes_path: Path,
    jobs: int,
    timeout: int,
    max_pairs: int | None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expanded = _expand_inputs(inputs)
    program_dirs: list[Path] = []

    for index, input_path in enumerate(expanded):
        program_name = _unique_program_name(input_path, program_dirs, index)
        program_out = out_dir / program_name
        run_analysis(input_path, program_out, passes_path, jobs, timeout, max_pairs)
        program_dirs.append(program_out)

    aggregate = write_aggregate_report(out_dir, program_dirs)
    return {
        "out_dir": str(out_dir),
        "program_dirs": [str(path) for path in program_dirs],
        "aggregate_summary": str(aggregate),
    }


def _tool_paths(metadata: dict) -> dict[str, str]:
    return {
        name: details["path"]
        for name, details in metadata.get("tools", {}).items()
        if details.get("path")
    }


def _read_metadata(out_dir: Path) -> dict:
    path = Path(out_dir) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = sorted(Path(match) for match in glob.glob(item))
        if matches:
            paths.extend(matches)
        elif any(char in item for char in "*?[]"):
            raise RuntimeError(f"input pattern matched no files: {item}")
        else:
            paths.append(Path(item))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _unique_program_name(input_path: Path, existing_dirs: list[Path], index: int) -> str:
    stem = input_path.stem
    existing = {path.name for path in existing_dirs}
    if stem not in existing:
        return stem
    return f"{stem}_{index}"
