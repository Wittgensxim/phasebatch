"""Frozen pass discovery, identity, preflight, and nested group selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import yaml


FROZEN_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "llvm_commit",
        "ir_unit",
        "u14_config",
        "u30_seed",
        "preflight_programs",
        "candidate_pipelines",
        "forbidden_prefixes",
        "forbidden_exact",
        "preflight_repeats",
        "require_success",
        "require_verifier",
    }
)
FROZEN_POLICY_SHA256 = (
    "2bb02795b808be7b6c211bd61c1e15339adbc503a4a7fc86bf219d892a39f213"
)
FROZEN_PREFLIGHT_PROGRAMS = (
    "20021219-1",
    "crc8.be",
    "fannkuch",
    "ffbench",
    "queens",
)


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _adaptor_path(pipeline: str) -> tuple[str, ...]:
    remaining = pipeline.strip()
    path: list[str] = []
    while True:
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_.-]*)\((.*)\)$",
            remaining,
            flags=re.DOTALL,
        )
        if not match or not _outer_parentheses_enclose_all(
            remaining, match.start(2) - 1
        ):
            break
        path.append(match.group(1))
        remaining = match.group(2).strip()
    return tuple(path)


def _outer_parentheses_enclose_all(text: str, open_index: int) -> bool:
    depth = 0
    for index, char in enumerate(text[open_index:], start=open_index):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index == len(text) - 1
    return False


def _pipeline_parameters(pipeline: str) -> tuple[str, ...]:
    parameters: list[str] = []
    for match in re.finditer(r"<([^<>]*)>", pipeline):
        parameters.extend(
            part.strip() for part in match.group(1).split(",") if part.strip()
        )
    return tuple(parameters)


@dataclass(frozen=True)
class ActionRecord:
    """Full manifest-compatible, content-addressed action identity."""

    config_index: int
    name: str
    pipeline: str
    category: str
    stage: str
    ir_unit: str
    adaptor_path: tuple[str, ...]
    parameters: tuple[str, ...]
    name_occurrence_index: int
    canonical_json: str = field(init=False)
    action_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.config_index) is not int or self.config_index < 0:
            raise ValueError("action config_index must be a non-negative integer")
        if (
            type(self.name_occurrence_index) is not int
            or self.name_occurrence_index < 0
        ):
            raise ValueError(
                "action name_occurrence_index must be a non-negative integer"
            )
        name = self.name.strip()
        pipeline = self.pipeline.strip()
        category = self.category.strip()
        stage = self.stage.strip()
        ir_unit = self.ir_unit.strip()
        adaptor_path = tuple(str(item).strip() for item in self.adaptor_path)
        parameters = tuple(str(item).strip() for item in self.parameters)
        if not name or not pipeline or not category or not ir_unit:
            raise ValueError(
                "action name, pipeline, category, and ir_unit must be non-empty"
            )
        if any(not item for item in (*adaptor_path, *parameters)):
            raise ValueError("action adaptor_path and parameters cannot contain blanks")
        canonical = _canonical_json(
            {
                "config_index": self.config_index,
                "ir_unit": ir_unit,
                "name": name,
                "pipeline": pipeline,
                "category": category,
                "stage": stage,
                "adaptor_path": list(adaptor_path),
                "parameters": list(parameters),
                "name_occurrence_index": self.name_occurrence_index,
            }
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pipeline", pipeline)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "ir_unit", ir_unit)
        object.__setattr__(self, "adaptor_path", adaptor_path)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "canonical_json", canonical)
        object.__setattr__(
            self,
            "action_id",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_manifest_record(cls, row: Mapping[str, object]) -> "ActionRecord":
        required = {
            "config_index",
            "name",
            "pipeline",
            "category",
            "stage",
            "ir_unit",
            "adaptor_path",
            "parameters",
            "name_occurrence_index",
            "action_id",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"manifest action missing required fields: {missing}")
        unexpected = sorted(set(row) - required)
        if unexpected:
            raise ValueError(f"manifest action has unexpected fields: {unexpected}")
        for field_name in ("config_index", "name_occurrence_index"):
            if type(row[field_name]) is not int:
                raise ValueError(f"manifest action {field_name} must be an integer")
        for field_name in (
            "name",
            "pipeline",
            "category",
            "stage",
            "ir_unit",
        ):
            if type(row[field_name]) is not str:
                raise ValueError(f"manifest action {field_name} must be a string")
        adaptor_path = row["adaptor_path"]
        parameters = row["parameters"]
        if type(adaptor_path) is not list or not all(
            type(item) is str for item in adaptor_path
        ):
            raise ValueError("manifest action adaptor_path must be a string list")
        if type(parameters) is not list or not all(
            type(item) is str for item in parameters
        ):
            raise ValueError("manifest action parameters must be a string list")
        recorded_id = row["action_id"]
        if type(recorded_id) is not str or not recorded_id.strip():
            raise ValueError("manifest action requires a non-empty action_id")
        action = cls(
            config_index=row["config_index"],
            name=row["name"],
            pipeline=row["pipeline"],
            category=row["category"],
            stage=row["stage"],
            ir_unit=row["ir_unit"],
            adaptor_path=tuple(adaptor_path),
            parameters=tuple(parameters),
            name_occurrence_index=row["name_occurrence_index"],
        )
        if recorded_id != action.action_id:
            raise ValueError("manifest action_id does not match canonical record")
        return action

    @classmethod
    def for_function_candidate(
        cls,
        *,
        name: str,
        pipeline: str,
        config_index: int,
        name_occurrence_index: int = 0,
    ) -> "ActionRecord":
        """Freeze a new registry candidate with explicit effective semantics."""

        return cls(
            config_index=config_index,
            name=name,
            pipeline=pipeline,
            category="function_transform",
            stage="advisor_pair_scale_2n_v1",
            ir_unit="function",
            adaptor_path=("module-to-function",),
            parameters=_pipeline_parameters(pipeline),
            name_occurrence_index=name_occurrence_index,
        )

    def as_manifest_record(self) -> dict[str, object]:
        return {"action_id": self.action_id, **json.loads(self.canonical_json)}


@dataclass(frozen=True)
class PassInventoryRow:
    name: str
    pipeline: str
    registry_section: str
    policy_candidate: bool
    policy_reason: str


@dataclass(frozen=True)
class PreflightDecision:
    action_id: str
    eligible: bool
    rejection_reasons: tuple[str, ...]


def validate_frozen_policy(policy: Mapping[str, object]) -> None:
    """Fail closed unless every runtime policy field matches the approved v1."""

    if not isinstance(policy, Mapping):
        raise ValueError("frozen policy must be a mapping")
    if set(policy) != FROZEN_POLICY_FIELDS:
        missing = sorted(FROZEN_POLICY_FIELDS - set(policy))
        unexpected = sorted(set(policy) - FROZEN_POLICY_FIELDS)
        raise ValueError(
            "frozen policy fields mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_scalars: dict[str, object] = {
        "schema_version": "phasebatch-advisor-pass-policy-v1",
        "llvm_commit": "aac212f0bc9acbc40a8a2e9638f4b7496c25d0b2",
        "ir_unit": "function",
        "u14_config": "configs/core_passes_v1.yaml",
        "u30_seed": "advisor-2n-scale-v1",
        "preflight_repeats": 2,
        "require_success": True,
        "require_verifier": True,
    }
    for field_name, expected in expected_scalars.items():
        actual = policy[field_name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"frozen policy {field_name} mismatch: expected {expected!r}"
            )
    for field_name in (
        "preflight_programs",
        "candidate_pipelines",
        "forbidden_prefixes",
        "forbidden_exact",
    ):
        entries = policy[field_name]
        if type(entries) is not list or not all(type(item) is str for item in entries):
            raise ValueError(f"frozen policy {field_name} must be a string list")
        normalized = [item.strip() for item in entries]
        if not normalized or any(not item for item in normalized):
            raise ValueError(f"frozen policy {field_name} cannot be empty or blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"frozen policy {field_name} contains duplicates")
    if tuple(policy["preflight_programs"]) != FROZEN_PREFLIGHT_PROGRAMS:
        raise ValueError("frozen policy preflight_programs mismatch")
    candidates = set(policy["candidate_pipelines"])
    forbidden_exact = set(policy["forbidden_exact"])
    forbidden_prefixes = tuple(policy["forbidden_prefixes"])
    forbidden_candidates = sorted(
        candidate
        for candidate in candidates
        if candidate in forbidden_exact
        or any(candidate.startswith(prefix) for prefix in forbidden_prefixes)
    )
    if forbidden_candidates:
        raise ValueError(
            f"frozen policy candidates conflict with forbidden rules: {forbidden_candidates}"
        )
    digest = hashlib.sha256(_canonical_json(policy).encode("utf-8")).hexdigest()
    if digest != FROZEN_POLICY_SHA256:
        raise ValueError("frozen policy content digest mismatch")


def load_frozen_policy(path: str | Path) -> dict[str, object]:
    """Read and validate the prepare-facing pass policy input."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen policy JSON root must be an object")
    validate_frozen_policy(payload)
    return payload


def _string_sequence(policy: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = policy.get(key, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"policy {key} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"policy {key} contains an empty entry")
    if len(result) != len(set(result)):
        raise ValueError(f"policy {key} contains duplicate entries")
    return result


def _section_lines(text: str, heading: str) -> tuple[str, ...]:
    lines = text.splitlines()
    collecting = False
    rows: list[str] = []
    for raw in lines:
        if raw.strip() == heading:
            collecting = True
            continue
        if not collecting:
            continue
        if raw and not raw[0].isspace() and raw.strip().endswith(":"):
            break
        item = raw.strip()
        if item:
            rows.append(item)
    return tuple(rows)


def _forbidden_reason(
    name: str,
    forbidden_exact: set[str],
    forbidden_prefixes: Sequence[str],
) -> str | None:
    if name in forbidden_exact:
        return f"forbidden_exact:{name}"
    for prefix in forbidden_prefixes:
        if name.startswith(prefix):
            return f"forbidden_prefix:{prefix}"
    return None


def parse_function_pass_inventory(
    text: str,
    policy: Mapping[str, object],
) -> list[PassInventoryRow]:
    """Parse only Function pass registry sections and bind the frozen policy."""

    validate_frozen_policy(policy)
    candidates = set(_string_sequence(policy, "candidate_pipelines"))
    forbidden_prefixes = _string_sequence(policy, "forbidden_prefixes")
    forbidden_exact = set(_string_sequence(policy, "forbidden_exact"))
    raw_bindings = policy.get("parameter_bindings", {})
    if not isinstance(raw_bindings, Mapping):
        raise ValueError("policy parameter_bindings must be a mapping")
    bindings = {str(key).strip(): str(value).strip() for key, value in raw_bindings.items()}
    unclassified = sorted(set(bindings) - candidates)
    if unclassified:
        raise ValueError(f"unclassified policy entries: {unclassified}")

    registered: dict[str, str] = {}
    for declaration in _section_lines(text, "Function passes:"):
        registered[declaration] = declaration
    for declaration in _section_lines(text, "Function passes with params:"):
        name = declaration.split("<", 1)[0].strip()
        registered[name] = bindings.get(name, name)

    missing = sorted(candidates - set(registered))
    if missing:
        raise ValueError(f"candidate Function passes not registered: {missing}")
    forbidden_candidates = sorted(
        name
        for name in candidates
        if _forbidden_reason(name, forbidden_exact, forbidden_prefixes) is not None
    )
    if forbidden_candidates:
        raise ValueError(f"forbidden entries cannot be candidates: {forbidden_candidates}")

    rows: list[PassInventoryRow] = []
    for name in sorted(registered):
        reason = _forbidden_reason(name, forbidden_exact, forbidden_prefixes)
        candidate = name in candidates
        rows.append(
            PassInventoryRow(
                name=name,
                pipeline=registered[name],
                registry_section="function",
                policy_candidate=candidate,
                policy_reason="candidate" if candidate else reason or "policy_not_candidate",
            )
        )
    return rows


def load_u14_actions(path: str | Path) -> tuple[ActionRecord, ...]:
    """Load the exact ordered U14 actions from the existing read-only YAML."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("passes"), list):
        raise ValueError("U14 config must contain a passes list")
    actions: list[ActionRecord] = []
    occurrences: dict[str, int] = {}
    for config_index, row in enumerate(payload["passes"]):
        if not isinstance(row, Mapping):
            raise ValueError("U14 pass entry must be a mapping")
        name = str(row.get("name", "")).strip()
        pipeline = str(row.get("pipeline", "")).strip()
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        adaptor_path = _adaptor_path(pipeline)
        actions.append(
            ActionRecord(
                config_index=config_index,
                name=name,
                pipeline=pipeline,
                category=str(row.get("category") or "unknown"),
                stage=str(row.get("stage") or ""),
                ir_unit=adaptor_path[-1] if adaptor_path else "unknown",
                adaptor_path=adaptor_path,
                parameters=_pipeline_parameters(pipeline),
                name_occurrence_index=occurrence,
            )
        )
    if len(actions) != 14:
        raise ValueError(f"U14 requires exactly 14 actions, got {len(actions)}")
    if len({action.action_id for action in actions}) != len(actions):
        raise ValueError("duplicate action IDs in U14")
    return tuple(actions)


def validate_u14_binding(
    core_actions: Sequence[ActionRecord],
    inventory: Sequence[PassInventoryRow],
    expected_manifest_actions: Sequence[Mapping[str, object]],
) -> None:
    if len(core_actions) != 14:
        raise ValueError(f"U14 requires exactly 14 actions, got {len(core_actions)}")
    if len(expected_manifest_actions) != 14:
        raise ValueError(
            "U14 reference manifest must contain exactly 14 action records"
        )
    expected = tuple(
        ActionRecord.from_manifest_record(row) for row in expected_manifest_actions
    )
    if tuple(action.action_id for action in core_actions) != tuple(
        action.action_id for action in expected
    ) or tuple(action.canonical_json for action in core_actions) != tuple(
        action.canonical_json for action in expected
    ):
        raise ValueError("U14 action identity drift from root-only manifest")
    registered = {row.pipeline for row in inventory}
    missing = sorted(action.pipeline for action in core_actions if action.pipeline not in registered)
    if missing:
        raise ValueError(f"U14 pipelines absent from Function registry: {missing}")
    candidates = {row.pipeline for row in inventory if row.policy_candidate}
    unbound = sorted(action.pipeline for action in core_actions if action.pipeline not in candidates)
    if unbound:
        raise ValueError(f"U14 pipelines not policy candidates: {unbound}")
    if len({action.action_id for action in core_actions}) != len(core_actions):
        raise ValueError("duplicate action IDs in U14")


def build_nested_groups(
    core_actions: Sequence[ActionRecord],
    eligible_additions: Sequence[ActionRecord],
    *,
    seed: str,
) -> dict[str, tuple[str, ...]]:
    """Build exact nested groups without using pair or 2N outcomes."""

    if len(core_actions) != 14:
        raise ValueError(f"U14 requires exactly 14 actions, got {len(core_actions)}")
    all_actions = [*core_actions, *eligible_additions]
    all_ids = [action.action_id for action in all_actions]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate action IDs across U14 and eligible additions")
    ranked = sorted(
        eligible_additions,
        key=lambda action: (
            hashlib.sha256((seed + "\0" + action.canonical_json).encode("utf-8")).hexdigest(),
            action.action_id,
        ),
    )
    if len(ranked) < 16:
        raise ValueError("U30 requires at least 16 eligible non-U14 actions")
    core_ids = tuple(action.action_id for action in core_actions)
    u30 = core_ids + tuple(action.action_id for action in ranked[:16])
    uall = u30 + tuple(action.action_id for action in ranked[16:])
    return {"U14": core_ids, "U30": u30, "Uall": uall}


def join_preflight_results(
    actions: Sequence[ActionRecord],
    observations: Sequence[Mapping[str, object]],
    preflight_programs: Sequence[str],
    *,
    repeats: int,
) -> tuple[PreflightDecision, ...]:
    """Retain all actions and explain every failed two-repeat eligibility gate."""

    if repeats != 2:
        raise ValueError("frozen pass preflight requires exactly 2 repeats")
    action_ids = [action.action_id for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("duplicate action IDs in preflight actions")
    programs = tuple(str(program).strip() for program in preflight_programs)
    if not programs:
        raise ValueError("preflight program list cannot be empty")
    if any(not program for program in programs):
        raise ValueError("preflight program IDs cannot be blank")
    if len(programs) != len(set(programs)):
        raise ValueError("duplicate preflight program IDs")

    indexed: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    known_actions = set(action_ids)
    known_programs = set(programs)
    for row in observations:
        action_id = str(row.get("action_id", ""))
        program_id = str(row.get("program_id", "")).strip()
        if action_id not in known_actions or program_id not in known_programs:
            raise ValueError("preflight observation is outside the frozen action/program universe")
        try:
            repetition = int(row.get("repetition", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("preflight repetition must be an integer") from error
        if repetition not in range(1, repeats + 1):
            raise ValueError("preflight repetition is outside the frozen repeat range")
        indexed.setdefault((action_id, program_id, repetition), []).append(row)

    decisions: list[PreflightDecision] = []
    for action in actions:
        reasons: list[str] = []
        for program in programs:
            valid_hashes: list[str] = []
            complete = True
            for repetition in range(1, repeats + 1):
                matches = indexed.get((action.action_id, program, repetition), [])
                if not matches:
                    reasons.append(f"missing_preflight_run:{program}:{repetition}")
                    complete = False
                    continue
                if len(matches) > 1:
                    reasons.append(f"duplicate_preflight_run:{program}:{repetition}")
                    complete = False
                    continue
                row = matches[0]
                execution = str(row.get("execution_status", ""))
                verifier = str(row.get("verifier_status", ""))
                hard_hash = str(row.get("output_hard_state_id", row.get("hard_hash", ""))).strip()
                if execution != "success":
                    reasons.append(f"execution_{execution or 'unknown'}:{program}:{repetition}")
                    complete = False
                if verifier != "success":
                    reasons.append(f"verifier_{verifier or 'unknown'}:{program}:{repetition}")
                    complete = False
                if not hard_hash:
                    reasons.append(f"missing_hard_hash:{program}:{repetition}")
                    complete = False
                else:
                    valid_hashes.append(hard_hash)
            if complete and len(set(valid_hashes)) != 1:
                reasons.append(f"unstable_hard_hash:{program}")
        decisions.append(
            PreflightDecision(
                action_id=action.action_id,
                eligible=not reasons,
                rejection_reasons=tuple(reasons),
            )
        )
    return tuple(decisions)
