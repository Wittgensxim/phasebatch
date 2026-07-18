from __future__ import annotations

import csv
from dataclasses import replace
import json
from collections import Counter
from pathlib import Path

from .deterministic_io import sha256_file
from .models import ArtifactRef, FrozenDataset, PairRow, SourceAttempt


EXPECTED_AUTHORITATIVE_SHA256 = (
    "a05749071d8c108bcb6a35ea63e85a16ab9016188520a878b585040afb463cb8"
)
EXPECTED_RELATIONS = {
    "dynamic_commute": 833,
    "dynamic_order_sensitive": 569,
    "failed": 9,
}
EXPECTED_LEGACY = {
    "H_func": {"selected": 30, "commute": 28, "order_sensitive": 2},
    "H_block": {"selected": 46, "commute": 44, "order_sensitive": 2},
    "H_effect": {"selected": 47, "commute": 45, "order_sensitive": 2},
}


def load_frozen_dataset(experiment_root: Path) -> FrozenDataset:
    experiment_root = Path(experiment_root).resolve()
    workspace = experiment_root.parents[1]
    old_root = workspace / "experiments" / "observed_effect_granularity_20260717"
    authoritative_csv = (
        workspace
        / "outputs"
        / "ei_idem_root_only_50programs_v2_20260716"
        / "pair_rule_mining"
        / "pair_observations.csv"
    )
    authoritative_hash = sha256_file(authoritative_csv)
    if authoritative_hash != EXPECTED_AUTHORITATIVE_SHA256:
        raise ValueError(
            "authoritative pair table hash mismatch: "
            f"expected={EXPECTED_AUTHORITATIVE_SHA256}:actual={authoritative_hash}"
        )

    authoritative_rows = _read_csv(authoritative_csv)
    ground_truth_rows = _read_csv(old_root / "aggregate" / "pair_ground_truth.csv")
    if len(authoritative_rows) != 1411 or len(ground_truth_rows) != 1411:
        raise ValueError("authoritative row count must be exactly 1411")
    auth_by_id = _unique_by(authoritative_rows, "observation_id")
    truth_by_id = _unique_by(ground_truth_rows, "observation_id")
    if set(auth_by_id) != set(truth_by_id):
        raise ValueError("authoritative and old ground-truth observation IDs differ")

    pairs: list[PairRow] = []
    base_by_program: dict[str, tuple[Path, str]] = {}
    for truth in ground_truth_rows:
        observation_id = truth["observation_id"]
        auth = auth_by_id[observation_id]
        authoritative_relation = _normalize_relation(auth["dynamic_relation"])
        truth_relation = _normalize_relation(truth["authoritative_relation"])
        dynamic_relation = _normalize_relation(truth["dynamic_all_relation"])
        if authoritative_relation != truth_relation or dynamic_relation != truth_relation:
            raise ValueError(f"dynamic relation mismatch: {observation_id}")
        if _as_bool(truth["dynamic_all_stable"]) is not True:
            raise ValueError(f"unstable old DYNAMIC_ALL label: {observation_id}")
        if _as_bool(truth["dynamic_disagreement"]):
            raise ValueError(f"old DYNAMIC_ALL disagreement: {observation_id}")
        state_path = Path(auth["state_ir_path"]).resolve()
        program = auth["program"]
        prior = base_by_program.setdefault(
            program, (state_path, auth["hard_state_hash"])
        )
        if prior != (state_path, auth["hard_state_hash"]):
            raise ValueError(f"multiple frozen S identities for program: {program}")
        pairs.append(
            PairRow(
                observation_id=observation_id,
                program=program,
                action_a_id=auth["action_a_id"],
                action_b_id=auth["action_b_id"],
                action_a_name=auth["action_a_name"],
                action_b_name=auth["action_b_name"],
                action_a_pipeline=auth["action_a_pipeline"],
                action_b_pipeline=auth["action_b_pipeline"],
                dynamic_relation=truth_relation,
                state_ir_path=state_path,
                state_hard_hash=auth["hard_state_hash"],
                h_func_selected=_as_bool(truth["h_func_selected"]),
                h_block_selected=_as_bool(truth["h_block_selected"]),
                h_effect_selected=_as_bool(truth["h_effect_selected"]),
            )
        )

    relation_counts = Counter(pair.dynamic_relation for pair in pairs)
    if dict(relation_counts) != EXPECTED_RELATIONS:
        raise ValueError(
            f"relation count mismatch: expected={EXPECTED_RELATIONS}:actual={dict(relation_counts)}"
        )
    programs = sorted({pair.program for pair in pairs})
    if len(programs) != 49:
        raise ValueError(f"program count mismatch: {len(programs)}")

    base_artifacts: dict[str, ArtifactRef] = {}
    for program, (path, hard_hash) in sorted(base_by_program.items()):
        errors: list[str] = []
        try:
            stat = path.stat()
            raw_hash = sha256_file(path)
        except OSError as exc:
            stat = None
            raw_hash = ""
            errors.append(f"base_artifact_unreadable:{type(exc).__name__}")
        base_artifacts[program] = ArtifactRef(
            path=path,
            expected_size=stat.st_size if stat else -1,
            expected_sha256=raw_hash,
            expected_hard_hash=hard_hash,
            consistency_errors=tuple(errors),
        )

    attempts = tuple(
        _load_attempt(old_root, repetition, authoritative_hash, truth_by_id)
        for repetition in (1, 2, 3)
    )
    transition_keys = frozenset(attempts[0].outputs)
    if len(transition_keys) != 686:
        raise ValueError(f"transition count mismatch: {len(transition_keys)}")
    if any(frozenset(attempt.outputs) != transition_keys for attempt in attempts[1:]):
        raise ValueError("DYNAMIC_ALL transition keys differ across repetitions")
    if {program for program, _ in transition_keys} != set(programs):
        raise ValueError("single-pass programs differ from pair programs")

    attempts = _mark_cross_repetition_inconsistency(attempts, transition_keys)
    action_ids = tuple(sorted({action_id for _, action_id in transition_keys}))
    if len(action_ids) != 14:
        raise ValueError(f"action count mismatch: {len(action_ids)}")

    legacy_counts = _legacy_counts(pairs)
    if legacy_counts != EXPECTED_LEGACY:
        raise ValueError(
            f"legacy count mismatch: expected={EXPECTED_LEGACY}:actual={legacy_counts}"
        )
    if len(programs) * len(action_ids) != len(transition_keys):
        raise ValueError("expected complete 49 x 14 transition grid")

    return FrozenDataset(
        experiment_root=experiment_root,
        old_experiment_root=old_root,
        authoritative_csv=authoritative_csv,
        authoritative_csv_sha256=authoritative_hash,
        pairs=tuple(pairs),
        attempts=attempts,
        base_artifacts=base_artifacts,
        action_ids=action_ids,
        transition_keys=transition_keys,
        relation_counts=dict(relation_counts),
        legacy_counts=legacy_counts,
    )


def _load_attempt(
    old_root: Path,
    repetition: int,
    authoritative_hash: str,
    truth_by_id: dict[str, dict[str, str]],
) -> SourceAttempt:
    config_dir = (
        old_root
        / "raw"
        / "repetitions"
        / f"r{repetition:02d}"
        / "DYNAMIC_ALL"
    )
    candidates: list[tuple[Path, dict]] = []
    for path in sorted(config_dir.glob("attempt-*")):
        completion_path = path / "completion.json"
        if not completion_path.is_file():
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "complete":
            candidates.append((path, completion))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one complete DYNAMIC_ALL attempt for repetition {repetition}, "
            f"got {len(candidates)}"
        )
    root, completion = candidates[0]
    _require_equal(completion, "configuration", "DYNAMIC_ALL")
    _require_equal(completion, "repetition", repetition)
    _require_equal(completion, "actual_program_count", 49)
    _require_equal(completion, "actual_action_count", 14)
    _require_equal(completion, "actual_row_count", 1411)
    _require_equal(completion, "authoritative_csv_sha256", authoritative_hash)
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"completion artifact manifest missing: {root}")

    single_path = root / "single_pass_runs.csv"
    pair_path = root / "pair_runs.csv"
    _validate_manifested_file(root, single_path, artifacts)
    _validate_manifested_file(root, pair_path, artifacts)
    single_rows = _read_csv(single_path)
    pair_rows = _read_csv(pair_path)
    if len(single_rows) != 686 or len(pair_rows) != 1411:
        raise ValueError(f"attempt CSV counts mismatch: {root}")

    outputs: dict[tuple[str, str], ArtifactRef] = {}
    for row in single_rows:
        key = (row["program"], row["action_id"])
        if key in outputs:
            raise ValueError(f"duplicate single-pass key: {key}")
        output_path = Path(row["output_path"]).resolve()
        errors: list[str] = []
        try:
            relative = output_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = ""
            errors.append("output_path_outside_attempt")
        artifact = artifacts.get(relative) if relative else None
        if not isinstance(artifact, dict):
            errors.append("output_missing_from_completion_manifest")
            size = -1
            raw_hash = ""
        else:
            size = int(artifact.get("size", -1))
            raw_hash = str(artifact.get("sha256", ""))
        if row["profile_status"] != "success":
            errors.append("single_pass_profile_not_success")
        outputs[key] = ArtifactRef(
            path=output_path,
            expected_size=size,
            expected_sha256=raw_hash,
            expected_hard_hash=row["hard_state_hash"],
            manifest_relative_path=relative,
            consistency_errors=tuple(sorted(errors)),
        )

    pair_costs: dict[str, float] = {}
    pair_relations: dict[str, str] = {}
    for row in pair_rows:
        observation_id = row["observation_id"]
        if observation_id in pair_costs:
            raise ValueError(f"duplicate pair run: {observation_id}")
        if observation_id not in truth_by_id:
            raise ValueError(f"unknown pair run: {observation_id}")
        relation = _normalize_relation(row["relation"])
        expected_relation = _normalize_relation(
            truth_by_id[observation_id]["authoritative_relation"]
        )
        if relation != expected_relation:
            raise ValueError(
                f"DYNAMIC_ALL pair relation mismatch: repetition={repetition}:"
                f"{observation_id}:{relation}:{expected_relation}"
            )
        pair_costs[observation_id] = float(row["elapsed_time_ms"])
        pair_relations[observation_id] = relation
    if set(pair_costs) != set(truth_by_id):
        raise ValueError(f"attempt pair IDs incomplete: {root}")

    return SourceAttempt(
        repetition=repetition,
        configuration="DYNAMIC_ALL",
        status="complete",
        root=root,
        completion_path=root / "completion.json",
        completion_sha256=sha256_file(root / "completion.json"),
        outputs=outputs,
        pair_cost_ms=pair_costs,
        pair_relations=pair_relations,
        single_pass_csv_sha256=sha256_file(single_path),
        pair_runs_csv_sha256=sha256_file(pair_path),
    )


def _mark_cross_repetition_inconsistency(
    attempts: tuple[SourceAttempt, ...],
    keys: frozenset[tuple[str, str]],
) -> tuple[SourceAttempt, ...]:
    updated: list[SourceAttempt] = []
    for attempt in attempts:
        outputs = dict(attempt.outputs)
        for key in keys:
            hard_hashes = {source.outputs[key].expected_hard_hash for source in attempts}
            errors = list(outputs[key].consistency_errors)
            if len(hard_hashes) != 1 or "" in hard_hashes:
                errors.append("hard_hash_unstable_across_sources")
            outputs[key] = replace(
                outputs[key], consistency_errors=tuple(sorted(set(errors)))
            )
        updated.append(replace(attempt, outputs=outputs))
    return tuple(updated)


def _legacy_counts(pairs: list[PairRow]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, attribute in (
        ("H_func", "h_func_selected"),
        ("H_block", "h_block_selected"),
        ("H_effect", "h_effect_selected"),
    ):
        selected = [pair for pair in pairs if getattr(pair, attribute)]
        result[name] = {
            "selected": len(selected),
            "commute": sum(
                pair.dynamic_relation == "dynamic_commute" for pair in selected
            ),
            "order_sensitive": sum(
                pair.dynamic_relation == "dynamic_order_sensitive"
                for pair in selected
            ),
        }
    return result


def _validate_manifested_file(root: Path, path: Path, artifacts: dict) -> None:
    relative = path.relative_to(root).as_posix()
    record = artifacts.get(relative)
    if not isinstance(record, dict):
        raise ValueError(f"file missing from completion artifact manifest: {relative}")
    stat = path.stat()
    if stat.st_size != int(record["size"]) or sha256_file(path) != record["sha256"]:
        raise ValueError(f"completion artifact hash mismatch: {path}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _unique_by(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f"duplicate {field}: {key}")
        result[key] = row
    return result


def _normalize_relation(value: str) -> str:
    return "failed" if value in {"dynamic_failed", "failed"} else value


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _require_equal(payload: dict, key: str, expected: object) -> None:
    if payload.get(key) != expected:
        raise ValueError(
            f"completion field mismatch: {key}:expected={expected!r}:"
            f"actual={payload.get(key)!r}"
        )

