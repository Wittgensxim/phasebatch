from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
REPORT = OUTPUTS / "verify_step1_report.md"
BRANCH_INPUT = "benchmarks/tiny/branch.c"
PASSES = "configs/core_passes.yaml"
PYTHON = sys.executable
BEAM_WIDTH = 4
MAX_BATCHES_PER_STATE = 10


@dataclass
class CommandResult:
    label: str
    command: list[str]
    returncode: int
    elapsed_ms: int
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class Section:
    name: str
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def check(self, condition: bool, failure: str, note: str | None = None) -> None:
        if condition:
            if note:
                self.notes.append(note)
        else:
            self.failures.append(failure)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    llvm_bin = Path(r"E:\llvm\build\bin")
    if "PHASEBATCH_LLVM_BIN" not in env and llvm_bin.exists():
        env["PHASEBATCH_LLVM_BIN"] = str(llvm_bin)

    commands: list[CommandResult] = []
    sections: dict[str, Section] = {
        "Test Suite": Section("Test Suite"),
        "CLI Checks": Section("CLI Checks"),
        "Commit 1 Skeleton": Section("Commit 1 Skeleton"),
        "Commit 2 Exact Mode": Section("Commit 2 Exact Mode"),
        "Commit 2 Exact Cap": Section("Commit 2 Exact Cap"),
        "Commit 3 Budgeted Beam": Section("Commit 3 Budgeted Beam"),
        "Budgeted Policy Variants": Section("Budgeted Policy Variants"),
        "Commit 4 Scoring / Pareto": Section("Commit 4 Scoring / Pareto"),
        "Auto Mode": Section("Auto Mode"),
        "State DAG Invariants": Section("State DAG Invariants"),
        "Path Reconstruction Invariants": Section("Path Reconstruction Invariants"),
        "Unsafe Batch Execution Check": Section("Unsafe Batch Execution Check"),
        "Summary Wording": Section("Summary Wording"),
    }

    pytest_result = run_command("pytest", [PYTHON, "-m", "pytest", "-q"], env, commands)
    sections["Test Suite"].check(pytest_result.returncode == 0, "python -m pytest -q failed")
    sections["Test Suite"].note(tail(pytest_result.stdout_tail or pytest_result.stderr_tail, 300))

    phase_help = run_command("phasebatch help", [PYTHON, "-m", "phasebatch", "--help"], env, commands)
    opt_help = run_command("optimize-batches help", [PYTHON, "-m", "phasebatch", "optimize-batches", "--help"], env, commands)
    verify_cli(sections["CLI Checks"], phase_help, opt_help)

    run_dirs: list[Path] = []
    commit1 = OUTPUTS / "verify_step1_commit1_branch_budgeted"
    run_optimize_case(
        "commit1 budgeted skeleton",
        commit1,
        [
            "--mode", "budgeted",
            "--objective", "ir-inst-count",
            "--max-rounds", "1",
            "--max-batches-per-state", "20",
            "--validate-batches",
            "--jobs", "8",
            "--timeout", "10",
            "--max-pairs", "300",
        ],
        env,
        commands,
    )
    run_dirs.append(commit1)
    verify_commit1(sections["Commit 1 Skeleton"], commit1)

    exact = OUTPUTS / "verify_step1_commit2_branch_exact"
    run_optimize_case(
        "commit2 exact",
        exact,
        [
            "--mode", "exact",
            "--objective", "ir-inst-count",
            "--max-rounds", "2",
            "--max-states", "5000",
            "--validate-batches",
            "--jobs", "8",
            "--timeout", "10",
            "--max-pairs", "300",
        ],
        env,
        commands,
    )
    run_dirs.append(exact)
    verify_exact(sections["Commit 2 Exact Mode"], exact)

    exact_cap = OUTPUTS / "verify_step1_commit2_branch_exact_cap"
    run_optimize_case(
        "commit2 exact cap",
        exact_cap,
        [
            "--mode", "exact",
            "--objective", "ir-inst-count",
            "--max-rounds", "5",
            "--max-states", "2",
            "--validate-batches",
            "--jobs", "8",
            "--timeout", "10",
            "--max-pairs", "300",
        ],
        env,
        commands,
    )
    run_dirs.append(exact_cap)
    verify_exact_cap(sections["Commit 2 Exact Cap"], exact_cap)

    budgeted = OUTPUTS / "verify_step1_commit3_branch_budgeted"
    run_optimize_case(
        "commit3 budgeted beam",
        budgeted,
        [
            "--mode", "budgeted",
            "--objective", "ir-inst-count",
            "--max-rounds", "3",
            "--beam-width", str(BEAM_WIDTH),
            "--max-states", "200",
            "--max-batches-per-state", str(MAX_BATCHES_PER_STATE),
            "--batch-frontier-policy", "certified-first",
            "--validate-batches",
            "--jobs", "8",
            "--timeout", "10",
            "--max-pairs", "300",
        ],
        env,
        commands,
    )
    run_dirs.append(budgeted)
    verify_budgeted(sections["Commit 3 Budgeted Beam"], budgeted, beam_width=BEAM_WIDTH, max_states=200)

    policy_dirs: dict[str, Path] = {}
    for policy in ["certified-first", "largest-batch", "objective", "diverse", "score"]:
        out_dir = OUTPUTS / f"verify_step1_policy_{policy}"
        policy_dirs[policy] = out_dir
        run_optimize_case(
            f"policy {policy}",
            out_dir,
            [
                "--mode", "budgeted",
                "--objective", "ir-inst-count",
                "--max-rounds", "2",
                "--beam-width", str(BEAM_WIDTH),
                "--max-states", "200",
                "--max-batches-per-state", str(MAX_BATCHES_PER_STATE),
                "--batch-frontier-policy", policy,
                "--validate-batches",
                "--jobs", "8",
                "--timeout", "10",
                "--max-pairs", "300",
            ],
            env,
            commands,
        )
        run_dirs.append(out_dir)
        verify_policy(sections["Budgeted Policy Variants"], out_dir, policy)

    verify_scoring(sections["Commit 4 Scoring / Pareto"], policy_dirs["score"], beam_width=BEAM_WIDTH)

    auto_dir = OUTPUTS / "verify_step1_auto_branch"
    run_optimize_case(
        "auto mode",
        auto_dir,
        [
            "--mode", "auto",
            "--objective", "ir-inst-count",
            "--max-rounds", "2",
            "--beam-width", str(BEAM_WIDTH),
            "--max-states", "200",
            "--max-batches-per-state", str(MAX_BATCHES_PER_STATE),
            "--validate-batches",
            "--jobs", "8",
            "--timeout", "10",
            "--max-pairs", "300",
        ],
        env,
        commands,
    )
    run_dirs.append(auto_dir)
    verify_auto(sections["Auto Mode"], auto_dir)

    for run_dir in run_dirs:
        verify_state_dag(sections["State DAG Invariants"], run_dir)
        verify_path_reconstruction(sections["Path Reconstruction Invariants"], run_dir)
        allow_sampled = False
        exact_mode = run_dir.name in {"verify_step1_commit2_branch_exact", "verify_step1_commit2_branch_exact_cap"}
        verify_unsafe_batches(sections["Unsafe Batch Execution Check"], run_dir, exact_mode=exact_mode, allow_sampled=allow_sampled)
        verify_summary_wording(sections["Summary Wording"], run_dir)

    write_report(sections, commands, run_dirs)
    failed = [section for section in sections.values() if not section.passed]
    print(f"wrote {REPORT}")
    print("overall:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


def run_optimize_case(label: str, out_dir: Path, extra_args: list[str], env: dict[str, str], commands: list[CommandResult]) -> CommandResult:
    safe_remove_output(out_dir)
    command = [
        PYTHON,
        "-m",
        "phasebatch",
        "optimize-batches",
        "--input",
        BRANCH_INPUT,
        "--out",
        str(out_dir.relative_to(REPO)),
        "--passes",
        PASSES,
        *extra_args,
    ]
    return run_command(label, command, env, commands)


def run_command(label: str, command: list[str], env: dict[str, str], commands: list[CommandResult]) -> CommandResult:
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True, check=False)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result = CommandResult(
        label=label,
        command=command,
        returncode=completed.returncode,
        elapsed_ms=elapsed_ms,
        stdout_tail=tail(completed.stdout, 20000),
        stderr_tail=tail(completed.stderr, 20000),
    )
    commands.append(result)
    return result


def safe_remove_output(path: Path) -> None:
    path = path.resolve()
    outputs = OUTPUTS.resolve()
    if path.exists() and path.is_relative_to(outputs):
        shutil.rmtree(path)


def verify_cli(section: Section, phase_help: CommandResult, opt_help: CommandResult) -> None:
    section.check(phase_help.returncode == 0, "python -m phasebatch --help failed")
    section.check(opt_help.returncode == 0, "python -m phasebatch optimize-batches --help failed")
    text = opt_help.stdout_tail
    for flag in [
        "--input",
        "--out",
        "--passes",
        "--mode",
        "--objective",
        "--max-rounds",
        "--max-states",
        "--beam-width",
        "--max-batches-per-state",
        "--batch-frontier-policy",
        "--validate-batches",
        "--allow-sampled-batches",
        "--jobs",
        "--timeout",
        "--max-pairs",
    ]:
        section.check(flag in text, f"optimize-batches help missing {flag}")
    for mode in ["exact", "budgeted", "auto"]:
        section.check(mode in text, f"optimize-batches --mode help missing choice {mode}")


def verify_commit1(section: Section, out_dir: Path) -> None:
    verify_required_files(
        section,
        out_dir,
        [
            "metadata.json",
            "states.csv",
            "state_dag.csv",
            "batch_state_transitions.csv",
            "leaf_states.csv",
            "chosen_path.csv",
            "optimized_batches.txt",
            "optimized_pipeline.txt",
            "final.ll",
            "optimize_summary.md",
        ],
    )
    summary = read_text(out_dir / "optimize_summary.md")
    states = read_csv(out_dir / "states.csv")
    transitions = read_csv(out_dir / "batch_state_transitions.csv")
    chosen = read_csv(out_dir / "chosen_path.csv")
    pipeline = read_text(out_dir / "optimized_pipeline.txt")
    section.check("selected_mode: budgeted" in summary or "mode: budgeted" in summary, "commit1 summary does not record budgeted mode")
    section.check("max_rounds: 1" in summary, "commit1 summary does not record max_rounds: 1")
    section.check(any(row.get("state_id") == "S0000" for row in states), "commit1 states.csv missing S0000")
    if transitions:
        section.check(len(states) > 1, "commit1 executable transitions exist but no child state was recorded")
    section.check((out_dir / "chosen_path.csv").exists(), "commit1 chosen_path.csv missing")
    section.check(not any(line.strip().startswith("#") for line in pipeline.splitlines()), "optimized_pipeline.txt contains comment lines")
    section.check((out_dir / "final.ll").exists(), "commit1 final.ll missing")
    section.check(summary_has_objective_boundary(summary), "commit1 summary missing objective correctness boundary")
    if chosen:
        verify_chosen_rows(section, out_dir, chosen)


def verify_exact(section: Section, out_dir: Path) -> None:
    verify_required_files(section, out_dir, ["exact_status.txt", "optimize_summary.md", "state_dag.csv", "leaf_states.csv", "chosen_path.csv", "optimized_pipeline.txt", "optimized_batches.txt", "final.ll"])
    summary = read_text(out_dir / "optimize_summary.md")
    exact_status = read_text(out_dir / "exact_status.txt")
    dag = read_csv(out_dir / "state_dag.csv")
    leaves = read_csv(out_dir / "leaf_states.csv")
    chosen = read_csv(out_dir / "chosen_path.csv")
    selected = selected_leaf(leaves)
    section.check("selected_mode: exact" in summary or "mode: exact" in summary, "exact summary does not record exact mode")
    section.check("exact_status" in summary, "exact summary does not record exact_status")
    if "exact_complete" in exact_status:
        section.check("exact_incomplete_reasons: " in summary, "exact complete summary missing empty exact_incomplete_reasons field")
    section.check(bool(dag), "exact state_dag.csv has no batch transitions")
    if any(row.get("source_state_id") != "S0000" for row in dag):
        section.note("exact mode generated at least one non-root DAG edge")
    if selected and selected.get("depth") == "2":
        section.check(len(chosen) == 2, "selected exact final state has depth 2 but chosen_path.csv does not have 2 rows")
    verify_pipeline(section, out_dir, chosen)
    verify_final_matches_selected(section, out_dir)


def verify_exact_cap(section: Section, out_dir: Path) -> None:
    verify_required_files(section, out_dir, ["exact_status.txt", "optimize_summary.md"])
    exact_status = read_text(out_dir / "exact_status.txt")
    summary = read_text(out_dir / "optimize_summary.md")
    status_text = exact_status + "\n" + summary
    section.check(
        "exact_incomplete" in status_text or "state_cap_exceeded" in status_text or "state_cap_reached" in status_text,
        "exact cap run did not report exact_incomplete or state cap reason",
    )
    section.check("exact_complete" not in exact_status or "state_cap_exceeded" not in status_text, "exact cap run silently appears exact_complete despite state cap")


def verify_budgeted(section: Section, out_dir: Path, *, beam_width: int, max_states: int) -> None:
    verify_required_files(section, out_dir, ["frontier_scores.csv", "optimizer_events.csv", "state_dag.csv", "leaf_states.csv", "chosen_path.csv", "optimized_pipeline.txt", "final.ll", "optimize_summary.md"])
    summary = read_text(out_dir / "optimize_summary.md")
    events = read_csv(out_dir / "optimizer_events.csv")
    scores = read_csv(out_dir / "frontier_scores.csv")
    states = read_csv(out_dir / "states.csv")
    section.check("selected_mode: budgeted" in summary or "mode: budgeted" in summary, "budgeted summary does not record budgeted mode")
    for expected in ["max_rounds: 3", f"beam_width: {beam_width}", f"max_states: {max_states}", f"max_batches_per_state: {MAX_BATCHES_PER_STATE}"]:
        section.check(expected in summary, f"budgeted summary missing {expected}")
    event_types = {row.get("event_type", "") for row in events}
    for event in ["build_batches", "apply_batch", "select_frontier"]:
        section.check(event in event_types, f"optimizer_events.csv missing {event}")
    section.check(bool(scores), "frontier_scores.csv has no rows")
    verify_frontier_beam(section, scores, beam_width)
    unique_states = [row for row in states if row.get("is_duplicate") != "true"]
    section.check(len(unique_states) <= max_states or "budget_exhausted: true" in summary, "budgeted unique state count exceeds max_states without budget_exhausted")
    verify_pipeline(section, out_dir, read_csv(out_dir / "chosen_path.csv"))
    verify_final_matches_selected(section, out_dir)


def verify_policy(section: Section, out_dir: Path, policy: str) -> None:
    for name in ["optimize_summary.md", "chosen_path.csv", "optimized_pipeline.txt", "final.ll", "frontier_scores.csv"]:
        section.check((out_dir / name).exists(), f"policy {policy} missing {name}")


def verify_scoring(section: Section, out_dir: Path, *, beam_width: int) -> None:
    scores = read_csv(out_dir / "frontier_scores.csv")
    required = [
        "objective_score",
        "future_potential_score",
        "evidence_quality_score",
        "novelty_score",
        "cost_score",
        "risk_penalty",
        "final_state_score",
        "pareto_kept",
        "selected_for_frontier",
        "selection_reason",
    ]
    for column in required:
        section.check(bool(scores) and column in scores[0], f"frontier_scores.csv missing {column}")
    for row in scores:
        for column in ["objective_score", "future_potential_score", "evidence_quality_score", "novelty_score", "cost_score", "risk_penalty", "final_state_score"]:
            value = parse_float(row.get(column, ""))
            section.check(value is not None and math.isfinite(value), f"frontier score {column} is not finite for state {row.get('state_id')}")
        section.check(row.get("pareto_kept") in {"true", "false"}, f"pareto_kept is not true/false for state {row.get('state_id')}")
    verify_frontier_beam(section, scores, beam_width)
    for round_id, rows in group_by(scores, "round").items():
        if rows:
            section.check(any(row.get("selected_for_frontier") == "true" for row in rows), f"round {round_id} has candidates but no selected frontier state")

    root_scores = read_csv(out_dir / "states" / "S0000" / "batch_candidate_scores.csv")
    required_batch = [
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
    section.check(bool(root_scores), "root batch_candidate_scores.csv missing or empty")
    for column in required_batch:
        section.check(bool(root_scores) and column in root_scores[0], f"batch_candidate_scores.csv missing {column}")
    selected = [row for row in root_scores if row.get("selected_for_execution") == "true"]
    section.check(len(selected) <= MAX_BATCHES_PER_STATE, "selected_for_execution count exceeds max_batches_per_state")
    for row in root_scores:
        evidence = parse_float(row.get("evidence_score"))
        if row.get("correctness_class") == "certified_batch":
            section.check(evidence is not None and abs(evidence - 1.0) <= 0.001, f"certified batch {row.get('batch_id')} does not have evidence_score close to 1.0")
        if row.get("correctness_class") in {"rejected_batch", "failed_batch", "unvalidated_batch", "unknown_batch"}:
            section.check(row.get("selected_for_execution") != "true", f"unsafe batch {row.get('batch_id')} selected for execution")


def verify_auto(section: Section, out_dir: Path) -> None:
    verify_required_files(section, out_dir, ["optimize_summary.md", "chosen_path.csv", "optimized_pipeline.txt", "final.ll"])
    summary = read_text(out_dir / "optimize_summary.md")
    selected_mode = summary_value(summary, "selected_mode")
    section.check(bool(selected_mode), "auto summary missing selected_mode")
    section.check(bool(summary_value(summary, "auto_reason")), "auto summary missing auto_reason")
    section.check(selected_mode in {"exact", "budgeted"}, f"auto selected invalid mode {selected_mode}")
    if selected_mode == "exact":
        section.check((out_dir / "exact_status.txt").exists(), "auto selected exact but exact_status.txt is missing")
    if selected_mode == "budgeted":
        section.check((out_dir / "frontier_scores.csv").exists(), "auto selected budgeted but frontier_scores.csv is missing")


def verify_state_dag(section: Section, out_dir: Path) -> None:
    states = read_csv(out_dir / "states.csv")
    dag = read_csv(out_dir / "state_dag.csv")
    states_by_id = {row.get("state_id", ""): row for row in states}
    section.check("S0000" in states_by_id, f"{out_dir.name}: states.csv missing S0000")
    if "S0000" in states_by_id:
        section.check(states_by_id["S0000"].get("depth") == "0", f"{out_dir.name}: S0000 depth is not 0")
    for row in states:
        if row.get("state_id") != "S0000":
            section.check(parse_int(row.get("depth")) >= 1, f"{out_dir.name}: non-root state {row.get('state_id')} has depth < 1")
    seen_hashes: dict[str, str] = {}
    for row in states:
        state_hash = row.get("state_hash", "")
        if row.get("is_duplicate") != "true" and state_hash:
            section.check(state_hash not in seen_hashes, f"{out_dir.name}: non-duplicate hash reused by {row.get('state_id')} and {seen_hashes.get(state_hash)}")
            seen_hashes[state_hash] = row.get("state_id", "")
        if row.get("is_duplicate") == "true":
            duplicate_of = row.get("duplicate_of", "")
            section.check(duplicate_of in states_by_id, f"{out_dir.name}: duplicate state {row.get('state_id')} points to missing {duplicate_of}")
    for edge in dag:
        source = edge.get("source_state_id", "")
        target = edge.get("target_state_id", "")
        section.check(source in states_by_id, f"{out_dir.name}: DAG source {source} missing from states.csv")
        section.check(target in states_by_id, f"{out_dir.name}: DAG target {target} missing from states.csv")
        section.check(bool(edge.get("source_hash")), f"{out_dir.name}: DAG edge {edge.get('batch_id')} has empty source_hash")
        section.check(bool(edge.get("target_hash")), f"{out_dir.name}: DAG edge {edge.get('batch_id')} has empty target_hash")
        section.check(edge.get("transition_kind") == "batch", f"{out_dir.name}: transition_kind is not batch")
        section.check(bool(edge.get("batch_id")), f"{out_dir.name}: batch transition missing batch_id")
        section.check(bool(edge.get("canonical_order")), f"{out_dir.name}: batch transition missing canonical_order")
        if edge.get("is_duplicate") == "true":
            duplicate_of = edge.get("duplicate_of", "")
            section.check(duplicate_of in states_by_id, f"{out_dir.name}: duplicate edge points to missing {duplicate_of}")
            if duplicate_of in states_by_id:
                section.check(edge.get("target_hash") == states_by_id[duplicate_of].get("state_hash"), f"{out_dir.name}: duplicate edge target_hash does not match duplicate_of")


def verify_path_reconstruction(section: Section, out_dir: Path) -> None:
    chosen = read_csv(out_dir / "chosen_path.csv")
    states_by_id = {row.get("state_id", ""): row for row in read_csv(out_dir / "states.csv")}
    for index, row in enumerate(chosen):
        section.check(row.get("step") == str(index), f"{out_dir.name}: chosen_path step {index} is {row.get('step')}")
        if index == 0:
            section.check(row.get("parent_state_id") == "S0000", f"{out_dir.name}: first chosen parent is not S0000")
        if index + 1 < len(chosen):
            child = row.get("child_state_id")
            next_parent = chosen[index + 1].get("parent_state_id")
            child_state = states_by_id.get(child, {})
            acceptable = child == next_parent or child_state.get("duplicate_of") == next_parent
            section.check(acceptable, f"{out_dir.name}: chosen path row {index} child does not connect to next parent")
        verify_chosen_batch_exists(section, out_dir, row)
        section.check(bool(row.get("canonical_order")), f"{out_dir.name}: chosen path row {index} has empty canonical_order")
        if row.get("ir_inst_before"):
            section.check(parse_int(row.get("ir_inst_before")) is not None, f"{out_dir.name}: ir_inst_before is not numeric")
        if row.get("ir_inst_after"):
            section.check(parse_int(row.get("ir_inst_after")) is not None, f"{out_dir.name}: ir_inst_after is not numeric")
    verify_pipeline(section, out_dir, chosen)
    verify_final_matches_selected(section, out_dir)


def verify_unsafe_batches(section: Section, out_dir: Path, *, exact_mode: bool, allow_sampled: bool) -> None:
    for edge in read_csv(out_dir / "batch_state_transitions.csv"):
        parent = edge.get("parent_state_id", "")
        batch_id = edge.get("batch_id", "")
        correctness = correctness_for(out_dir, parent, batch_id)
        class_name = correctness.get("correctness_class") or edge.get("correctness_class")
        status = correctness.get("validation_status") or edge.get("validation_status")
        section.check(class_name not in {"rejected_batch", "failed_batch", "unvalidated_batch", "unknown_batch"}, f"{out_dir.name}: unsafe batch {batch_id} executed as {class_name}")
        if not allow_sampled:
            section.check(class_name != "sampled_batch", f"{out_dir.name}: sampled batch {batch_id} executed without --allow-sampled-batches")
        if exact_mode:
            section.check(class_name == "certified_batch", f"{out_dir.name}: exact mode executed non-certified batch {batch_id} ({class_name})")
            section.check(status == "all_permutations_same", f"{out_dir.name}: exact mode batch {batch_id} has validation_status {status}")
            if correctness:
                section.check(correctness.get("can_hard_fold") == "true", f"{out_dir.name}: exact mode batch {batch_id} can_hard_fold is not true")


def verify_summary_wording(section: Section, out_dir: Path) -> None:
    summary = read_text(out_dir / "optimize_summary.md")
    for needle in ["selected_mode", "objective", "max_rounds", "root objective value", "final objective value", "objective delta", "optimized pipeline"]:
        section.check(needle in summary, f"{out_dir.name}: optimize_summary.md missing {needle}")
    section.check(summary_has_objective_boundary(summary), f"{out_dir.name}: summary missing objective-not-proof wording")


def verify_required_files(section: Section, out_dir: Path, names: list[str]) -> None:
    for name in names:
        section.check((out_dir / name).exists(), f"{out_dir.name}: missing {name}")


def verify_chosen_rows(section: Section, out_dir: Path, chosen: list[dict]) -> None:
    for row in chosen:
        section.check(bool(row.get("validation_status")), f"{out_dir.name}: chosen batch {row.get('batch_id')} missing validation_status")
        section.check(bool(row.get("correctness_class")), f"{out_dir.name}: chosen batch {row.get('batch_id')} missing correctness_class")
        verify_chosen_batch_exists(section, out_dir, row)
        correctness = correctness_for(out_dir, row.get("parent_state_id", ""), row.get("batch_id", ""))
        if correctness:
            section.check(correctness.get("can_execute") == "true" or correctness.get("can_hard_fold") == "true", f"{out_dir.name}: chosen batch {row.get('batch_id')} is not executable")


def verify_chosen_batch_exists(section: Section, out_dir: Path, row: dict) -> None:
    parent = row.get("parent_state_id", "")
    batch_id = row.get("batch_id", "")
    candidates = read_csv(out_dir / "states" / parent / "batch_candidates.csv")
    correctness = read_csv(out_dir / "states" / parent / "batch_correctness.csv")
    section.check(any(item.get("batch_id") == batch_id for item in candidates), f"{out_dir.name}: chosen batch {batch_id} missing from {parent}/batch_candidates.csv")
    section.check(any(item.get("batch_id") == batch_id for item in correctness), f"{out_dir.name}: chosen batch {batch_id} missing from {parent}/batch_correctness.csv")


def verify_pipeline(section: Section, out_dir: Path, chosen: list[dict]) -> None:
    expected_parts: list[str] = []
    for row in chosen:
        expected_parts.extend(split_order(row.get("canonical_order", "")))
    expected = ",".join(expected_parts)
    actual = read_text(out_dir / "optimized_pipeline.txt").strip()
    section.check(actual == expected, f"{out_dir.name}: optimized_pipeline.txt mismatch; expected {expected!r}, got {actual!r}")
    section.check(not any(line.strip().startswith("#") for line in actual.splitlines()), f"{out_dir.name}: optimized_pipeline.txt contains markdown/comment lines")
    section.check(" " not in actual, f"{out_dir.name}: optimized_pipeline.txt contains spaces")


def verify_final_matches_selected(section: Section, out_dir: Path) -> None:
    selected = selected_leaf(read_csv(out_dir / "leaf_states.csv"))
    if not selected:
        section.fail(f"{out_dir.name}: no selected final state in leaf_states.csv")
        return
    selected_input = out_dir / "states" / selected.get("state_id", "") / "input.ll"
    final_ll = out_dir / "final.ll"
    section.check(final_ll.exists(), f"{out_dir.name}: final.ll missing")
    section.check(selected_input.exists(), f"{out_dir.name}: selected state input.ll missing for {selected.get('state_id')}")
    if final_ll.exists() and selected_input.exists():
        section.check(final_ll.read_bytes() == selected_input.read_bytes(), f"{out_dir.name}: final.ll does not match selected state input.ll byte-for-byte")


def verify_frontier_beam(section: Section, scores: list[dict], beam_width: int) -> None:
    for round_id, rows in group_by(scores, "round").items():
        selected = [row for row in rows if row.get("selected_for_frontier") == "true"]
        section.check(len(selected) <= beam_width, f"round {round_id} selected {len(selected)} frontier states, above beam_width {beam_width}")


def correctness_for(out_dir: Path, parent_state_id: str, batch_id: str) -> dict:
    for row in read_csv(out_dir / "states" / parent_state_id / "batch_correctness.csv"):
        if row.get("batch_id") == batch_id:
            return row
    return {}


def selected_leaf(rows: list[dict]) -> dict:
    for row in rows:
        if row.get("selected_as_final") == "true":
            return row
    return {}


def summary_value(summary: str, key: str) -> str:
    prefix = f"- {key}:"
    for line in summary.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def summary_has_objective_boundary(summary: str) -> bool:
    return (
        "Objective is used only for path selection, not as commutation proof." in summary
        or "Objective scores are used only for search ranking and final path selection" in summary
    )


def split_order(value: str) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, ""), []).append(row)
    return grouped


def parse_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def parse_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.strip()
    return text[-limit:].strip()


def write_report(sections: dict[str, Section], commands: list[CommandResult], run_dirs: list[Path]) -> None:
    failed_sections = [section for section in sections.values() if not section.passed]
    lines = [
        "# Step 1 Optimize-Batches Verification Report",
        "",
        "## Overall Result",
        "",
        "FAIL" if failed_sections else "PASS",
        "",
    ]
    for heading in [
        "Test Suite",
        "CLI Checks",
        "Commit 1 Skeleton",
        "Commit 2 Exact Mode",
        "Commit 2 Exact Cap",
        "Commit 3 Budgeted Beam",
        "Commit 4 Scoring / Pareto",
        "Auto Mode",
        "State DAG Invariants",
        "Path Reconstruction Invariants",
        "Unsafe Batch Execution Check",
        "Summary Wording",
    ]:
        section = sections[heading]
        lines.extend(section_lines(section))

    lines.extend(section_lines(sections["Budgeted Policy Variants"]))
    lines.extend(
        [
            "## Important Output Directories",
            "",
            *[f"- {path.relative_to(REPO)}" for path in run_dirs],
            "",
            "## Command Results",
            "",
        ]
    )
    for result in commands:
        command_text = " ".join(result.command)
        lines.extend(
            [
                f"- {result.label}: returncode={result.returncode}, elapsed_ms={result.elapsed_ms}",
                f"  - command: `{command_text}`",
            ]
        )
        if result.returncode != 0:
            if result.stdout_tail:
                lines.append(f"  - stdout_tail: `{one_line(result.stdout_tail)}`")
            if result.stderr_tail:
                lines.append(f"  - stderr_tail: `{one_line(result.stderr_tail)}`")
    lines.extend(["", "## Remaining Issues", ""])
    if failed_sections:
        for section in failed_sections:
            for failure in section.failures:
                lines.append(f"- {section.name}: {failure}")
    else:
        lines.append("- None.")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def section_lines(section: Section) -> list[str]:
    lines = [f"## {section.name}", "", "PASS" if section.passed else "FAIL", ""]
    if section.notes:
        lines.append("Notes:")
        for note in section.notes[:20]:
            lines.append(f"- {one_line(note)}")
        lines.append("")
    if section.failures:
        lines.append("Failures:")
        for failure in section.failures:
            lines.append(f"- {failure}")
        lines.append("")
    return lines


def one_line(text: str) -> str:
    return " ".join(str(text).replace("`", "'").split())


if __name__ == "__main__":
    raise SystemExit(main())
