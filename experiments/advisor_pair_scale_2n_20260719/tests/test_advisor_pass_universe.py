from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from advisor_study.pass_universe import (
    ActionRecord,
    build_nested_groups,
    join_preflight_results,
    load_frozen_policy,
    load_u14_actions,
    parse_function_pass_inventory,
    validate_frozen_policy,
    validate_u14_binding,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


ROOT_ONLY_MANIFEST = (
    REPO
    / "outputs"
    / "ei_idem_root_only_50programs_v2_20260716"
    / "programs"
    / "20021219-1"
    / "root_only"
    / "experiment_manifest.json"
)

EXPECTED_CANDIDATE_PIPELINES = [
    "adce",
    "aggressive-instcombine",
    "alignment-from-assumptions",
    "assume-builder",
    "assume-simplify",
    "bdce",
    "break-crit-edges",
    "callsite-splitting",
    "chr",
    "complex-deinterleaving",
    "consthoist",
    "constraint-elimination",
    "correlated-propagation",
    "dce",
    "dfa-jump-threading",
    "div-rem-pairs",
    "dse",
    "early-cse",
    "expand-memcmp",
    "expand-reductions",
    "fix-irreducible",
    "flatten-cfg",
    "float2int",
    "guard-widening",
    "gvn",
    "gvn-hoist",
    "gvn-sink",
    "infer-address-spaces",
    "infer-alignment",
    "instcombine",
    "instnamer",
    "instsimplify",
    "interleaved-access",
    "interleaved-load-combine",
    "irce",
    "jump-table-to-switch",
    "jump-threading",
    "lcssa",
    "libcalls-shrinkwrap",
    "load-store-vectorizer",
    "loop-data-prefetch",
    "loop-distribute",
    "loop-fusion",
    "loop-load-elim",
    "loop-simplify",
    "loop-sink",
    "loop-versioning",
    "lower-constant-intrinsics",
    "lower-expect",
    "lower-guard-intrinsic",
    "lower-switch",
    "lower-widenable-condition",
    "make-guards-explicit",
    "mem2reg",
    "memcpyopt",
    "mergeicmps",
    "mergereturn",
    "move-auto-init",
    "nary-reassociate",
    "newgvn",
    "partially-inline-libcalls",
    "reassociate",
    "redundant-dbg-inst-elim",
    "replace-with-veclib",
    "sccp",
    "select-optimize",
    "separate-const-offset-from-gep",
    "simplifycfg",
    "sink",
    "slp-vectorizer",
    "slsr",
    "sroa",
    "tailcallelim",
    "typepromotion",
    "unify-loop-exits",
    "unreachableblockelim",
    "vector-combine",
]


def _action(name: str, config_index: int = 0) -> ActionRecord:
    return ActionRecord.for_function_candidate(
        name=name,
        pipeline=name,
        config_index=config_index,
    )


def _frozen_policy() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs" / "advisor_pair_scale_pass_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _registry_text(
    *,
    extra: tuple[str, ...] = (),
    omit: tuple[str, ...] = (),
    parameterized: tuple[str, ...] = (),
) -> str:
    omitted = set(omit)
    parameterized_names = set(parameterized)
    plain = [
        name
        for name in EXPECTED_CANDIDATE_PIPELINES
        if name not in omitted and name not in parameterized_names
    ]
    plain.extend(extra)
    lines = ["Module passes:", "  globalopt", "Function passes:"]
    lines.extend(f"  {name}" for name in plain)
    lines.append("Function passes with params:")
    lines.extend(
        f"  {name}<no-verify-fixpoint;verify-fixpoint;max-iterations=N>"
        for name in parameterized
        if name not in omitted
    )
    lines.extend(["Function analyses:", "  domtree"])
    return "\n".join(lines) + "\n"


def test_frozen_policy_is_fully_locked() -> None:
    policy = json.loads(
        (ROOT / "configs" / "advisor_pair_scale_pass_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert policy == {
        "schema_version": "phasebatch-advisor-pass-policy-v1",
        "llvm_commit": "aac212f0bc9acbc40a8a2e9638f4b7496c25d0b2",
        "ir_unit": "function",
        "u14_config": "configs/core_passes_v1.yaml",
        "u30_seed": "advisor-2n-scale-v1",
        "preflight_programs": [
            "20021219-1",
            "crc8.be",
            "fannkuch",
            "ffbench",
            "queens",
        ],
        "candidate_pipelines": EXPECTED_CANDIDATE_PIPELINES,
        "forbidden_prefixes": ["dot-", "print", "trigger-", "verify", "view-"],
        "forbidden_exact": [
            "aa-eval",
            "annotation-remarks",
            "count-visits",
            "helloworld",
            "invalidate<all>",
            "no-op-function",
            "pa-eval",
        ],
        "preflight_repeats": 2,
        "require_success": True,
        "require_verifier": True,
    }


def test_action_record_has_canonical_content_addressed_identity() -> None:
    first = ActionRecord(
        config_index=2,
        name="instcombine",
        pipeline="instcombine",
        category="scalar",
        stage="v1",
        ir_unit="unknown",
        adaptor_path=(),
        parameters=(),
        name_occurrence_index=0,
    )
    again = replace(first)
    parameterized = replace(
        first,
        pipeline="function(instcombine<max-iterations=4>)",
        ir_unit="function",
        adaptor_path=("function",),
        parameters=("max-iterations=4",),
    )

    assert first == again
    assert first.action_id == again.action_id
    assert json.loads(first.canonical_json) == {
        "adaptor_path": [],
        "category": "scalar",
        "config_index": 2,
        "ir_unit": "unknown",
        "name": "instcombine",
        "name_occurrence_index": 0,
        "parameters": [],
        "pipeline": "instcombine",
        "stage": "v1",
    }
    assert first.action_id == "3a5e77ee2ee6737ab281c7b8d3753ffcb39217ad7b086b27ff010ee7ecdb32e4"
    assert first.action_id != parameterized.action_id


def test_new_candidate_action_freezes_effective_function_adaptor_semantics() -> None:
    action = ActionRecord.for_function_candidate(
        name="instcombine",
        pipeline="instcombine<max-iterations=4>",
        config_index=14,
    )
    assert action.ir_unit == "function"
    assert action.adaptor_path == ("module-to-function",)
    assert action.parameters == ("max-iterations=4",)
    assert action.category == "function_transform"
    assert action.stage == "advisor_pair_scale_2n_v1"


def test_manifest_action_record_requires_all_identity_fields_and_action_id() -> None:
    valid = load_u14_actions(REPO / "configs" / "core_passes_v1.yaml")[0]
    row = valid.as_manifest_record()
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
    assert set(row) == required
    for field in sorted(required):
        incomplete = dict(row)
        del incomplete[field]
        with pytest.raises(ValueError, match="missing required fields"):
            ActionRecord.from_manifest_record(incomplete)
    for invalid_id in ("", " ", None, 123):
        invalid = dict(row)
        invalid["action_id"] = invalid_id
        with pytest.raises(ValueError, match="non-empty action_id"):
            ActionRecord.from_manifest_record(invalid)


@pytest.mark.parametrize("field", ["config_index", "name_occurrence_index"])
@pytest.mark.parametrize("invalid", [True, 1.0, "1"])
def test_manifest_action_record_rejects_non_integer_identity_index(
    field: str, invalid: object
) -> None:
    row = load_u14_actions(REPO / "configs" / "core_passes_v1.yaml")[0].as_manifest_record()
    row[field] = invalid
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        ActionRecord.from_manifest_record(row)


def test_manifest_action_record_rejects_hash_mismatch() -> None:
    row = load_u14_actions(REPO / "configs" / "core_passes_v1.yaml")[0].as_manifest_record()
    row["action_id"] = "0" * 64
    with pytest.raises(ValueError, match="does not match canonical record"):
        ActionRecord.from_manifest_record(row)


def test_runtime_policy_loader_and_validator_accept_only_frozen_policy() -> None:
    path = ROOT / "configs" / "advisor_pair_scale_pass_policy_v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    validate_frozen_policy(policy)
    assert load_frozen_policy(path) == policy

    for field in sorted(policy):
        drifted = dict(policy)
        del drifted[field]
        with pytest.raises(ValueError, match="frozen policy fields"):
            validate_frozen_policy(drifted)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema_version", "v2"),
        ("llvm_commit", "0" * 40),
        ("ir_unit", "module"),
        ("u14_config", "other.yaml"),
        ("u30_seed", "other-seed"),
        ("preflight_programs", ["queens"]),
        ("preflight_repeats", 3),
        ("require_success", False),
        ("require_verifier", False),
        ("candidate_pipelines", EXPECTED_CANDIDATE_PIPELINES[:-1]),
        ("forbidden_prefixes", ["print"]),
        ("forbidden_exact", ["no-op-function"]),
    ],
)
def test_runtime_policy_validator_rejects_every_frozen_field_drift(
    field: str, invalid: object
) -> None:
    policy = json.loads(
        (ROOT / "configs" / "advisor_pair_scale_pass_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy[field] = invalid
    with pytest.raises(ValueError, match="frozen policy"):
        validate_frozen_policy(policy)


def test_inventory_accepts_only_policy_bound_function_transforms() -> None:
    text = _registry_text(
        extra=("print", "trigger-crash-function"),
        parameterized=("instcombine",),
    )
    policy = _frozen_policy()

    rows = parse_function_pass_inventory(text, policy)

    candidates = {
        row.name: row.pipeline for row in rows if row.policy_candidate
    }
    assert len(candidates) == 77
    assert candidates["adce"] == "adce"
    assert candidates["instcombine"] == "instcombine"
    rejected = {row.name: row.policy_reason for row in rows if not row.policy_candidate}
    assert rejected == {
        "print": "forbidden_prefix:print",
        "trigger-crash-function": "forbidden_prefix:trigger-",
    }
    assert all(row.registry_section == "function" for row in rows)


def test_inventory_retains_unselected_pass_with_machine_reason() -> None:
    rows = parse_function_pass_inventory(
        _registry_text(extra=("extra-transform",)),
        _frozen_policy(),
    )
    extra = next(row for row in rows if row.name == "extra-transform")
    assert (extra.policy_candidate, extra.policy_reason) == (
        False,
        "policy_not_candidate",
    )


def test_inventory_rejects_unregistered_or_forbidden_candidate() -> None:
    with pytest.raises(ValueError, match="not registered"):
        parse_function_pass_inventory(
            _registry_text(omit=("vector-combine",)),
            _frozen_policy(),
        )
    drifted = _frozen_policy()
    drifted["candidate_pipelines"] = [*EXPECTED_CANDIDATE_PIPELINES, "print"]
    with pytest.raises(ValueError, match="frozen policy"):
        parse_function_pass_inventory(_registry_text(extra=("print",)), drifted)


def test_inventory_rejects_unclassified_parameter_binding() -> None:
    drifted = _frozen_policy()
    drifted["parameter_bindings"] = {"bdce": "bdce"}
    with pytest.raises(ValueError, match="frozen policy fields"):
        parse_function_pass_inventory(_registry_text(), drifted)


def test_u14_is_bound_to_exact_existing_config_and_registry() -> None:
    actions = load_u14_actions(REPO / "configs" / "core_passes_v1.yaml")
    manifest = json.loads(ROOT_ONLY_MANIFEST.read_text(encoding="utf-8"))
    manifest_actions = manifest["pass_config"]["actions"]
    assert [action.name for action in actions] == [
        "mem2reg",
        "sroa",
        "instcombine",
        "aggressive-instcombine",
        "instsimplify",
        "simplifycfg",
        "early-cse",
        "dce",
        "adce",
        "bdce",
        "reassociate",
        "gvn",
        "jump-threading",
        "correlated-propagation",
    ]
    assert [action.as_manifest_record() for action in actions] == manifest_actions
    inventory = parse_function_pass_inventory(_registry_text(), _frozen_policy())
    validate_u14_binding(actions, inventory, manifest_actions)
    with pytest.raises(ValueError, match="U14 pipelines absent"):
        validate_u14_binding(
            actions,
            [row for row in inventory if row.pipeline != actions[-1].pipeline],
            manifest_actions,
        )
    drifted_policy_inventory = [
        replace(row, policy_candidate=False)
        if row.pipeline == actions[-1].pipeline
        else row
        for row in inventory
    ]
    with pytest.raises(ValueError, match="U14 pipelines not policy candidates"):
        validate_u14_binding(actions, drifted_policy_inventory, manifest_actions)


@pytest.mark.parametrize(
    "drifted",
    [
        lambda rows: [replace(rows[0], category="drift"), *rows[1:]],
        lambda rows: [replace(rows[0], stage="drift"), *rows[1:]],
        lambda rows: [rows[1], rows[0], *rows[2:]],
        lambda rows: [replace(rows[0], adaptor_path=("function",)), *rows[1:]],
        lambda rows: [replace(rows[0], parameters=("x=1",)), *rows[1:]],
    ],
)
def test_u14_binding_rejects_any_canonical_action_drift(drifted) -> None:
    actions = list(load_u14_actions(REPO / "configs" / "core_passes_v1.yaml"))
    manifest = json.loads(ROOT_ONLY_MANIFEST.read_text(encoding="utf-8"))
    manifest_actions = manifest["pass_config"]["actions"]
    inventory = parse_function_pass_inventory(_registry_text(), _frozen_policy())

    with pytest.raises(ValueError, match="U14 action identity drift"):
        validate_u14_binding(drifted(actions), inventory, manifest_actions)


def test_nested_groups_are_result_independent_and_unbounded_uall() -> None:
    core = [_action(f"core-{index}", index) for index in range(14)]
    additions = [_action(f"extra-{index}", 14 + index) for index in range(80)]

    groups = build_nested_groups(core, additions, seed="advisor-2n-scale-v1")

    assert len(groups["U14"]) == 14
    assert len(groups["U30"]) == 30
    assert len(groups["Uall"]) == 94
    assert set(groups["U14"]) < set(groups["U30"]) < set(groups["Uall"])
    assert build_nested_groups(
        core, list(reversed(additions)), seed="advisor-2n-scale-v1"
    ) == groups


def test_nested_groups_reject_duplicate_ids_and_short_addition_pool() -> None:
    core = [_action(f"core-{index}", index) for index in range(14)]
    with pytest.raises(ValueError, match="duplicate action IDs"):
        build_nested_groups(core, [core[0], *[_action(f"x-{i}", 14 + i) for i in range(16)]], seed="s")
    with pytest.raises(ValueError, match="at least 16"):
        build_nested_groups(core, [_action(f"x-{i}", 14 + i) for i in range(15)], seed="s")


def test_preflight_join_requires_two_successful_verified_stable_runs() -> None:
    actions = [_action("adce"), _action("bdce"), _action("dce")]
    programs = ("p1", "p2")
    rows: list[dict[str, object]] = []
    for action in actions:
        for program in programs:
            for repetition in (1, 2):
                rows.append(
                    {
                        "action_id": action.action_id,
                        "program_id": program,
                        "repetition": repetition,
                        "execution_status": "success",
                        "verifier_status": "success",
                        "output_hard_state_id": f"{action.name}-{program}",
                    }
                )
    rows[4]["output_hard_state_id"] = "unstable-first"
    rows[9]["verifier_status"] = "invalid"

    decisions = join_preflight_results(actions, rows, programs, repeats=2)

    assert [decision.action_id for decision in decisions] == [a.action_id for a in actions]
    assert decisions[0].eligible is True
    assert decisions[0].rejection_reasons == ()
    assert decisions[1].eligible is False
    assert decisions[1].rejection_reasons == ("unstable_hard_hash:p1",)
    assert decisions[2].eligible is False
    assert "verifier_invalid:p1:2" in decisions[2].rejection_reasons


def test_preflight_join_retains_missing_and_failed_rejection_reasons() -> None:
    action = _action("adce")
    rows = [
        {
            "action_id": action.action_id,
            "program_id": "p1",
            "repetition": 1,
            "execution_status": "error",
            "verifier_status": "not_run",
            "output_hard_state_id": "",
        }
    ]
    decision = join_preflight_results([action], rows, ["p1"], repeats=2)[0]
    assert decision.eligible is False
    assert decision.rejection_reasons == (
        "execution_error:p1:1",
        "verifier_not_run:p1:1",
        "missing_hard_hash:p1:1",
        "missing_preflight_run:p1:2",
    )


@pytest.mark.parametrize(
    "programs",
    [[], [""], ["   "], ["p1", "p1"]],
)
def test_preflight_join_rejects_empty_blank_or_duplicate_programs(
    programs: list[str],
) -> None:
    with pytest.raises(ValueError, match="preflight program"):
        join_preflight_results([_action("adce")], [], programs, repeats=2)
