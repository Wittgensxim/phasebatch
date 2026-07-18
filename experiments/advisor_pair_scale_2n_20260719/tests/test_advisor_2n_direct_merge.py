"""Protocol tests for the isolated direct-merge Python client.

The fake helper is deliberately a JSON-lines subprocess, so these tests cover
the same persistent-process boundary used by the experiment without requiring
the LLVM helper binary to be built.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from advisor_study.direct_merge import (
    DirectMergeClient,
    DirectMergeProtocolError,
    DirectMergeUnavailable,
    EffectRecord,
    MergeRecord,
    PatchRecord,
    evaluate_group_2n,
)


def _digest(letter: str) -> str:
    return letter * 64


def _patch_hash(functions: list[tuple[str, str, str]]) -> str:
    canonical = "advisor_2n_patch_record_v1\n" + "".join(
        f"{name}\n{base}\n{output}\n" for name, base, output in sorted(functions)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _patch_reply(
    *,
    function: str,
    base_module_hash: str = _digest("a"),
    output_module_hash: str = _digest("b"),
) -> dict[str, object]:
    changed = [(function, _digest("c"), _digest("d"))]
    patch_hash = _patch_hash(changed)
    entries = [
        {
            "name": name,
            "base_isolated_hash": base,
            "output_isolated_hash": output,
        }
        for name, base, output in changed
    ]
    return {
        "status": "ok",
        "base_module_hash": base_module_hash,
        "output_module_hash": output_module_hash,
        "base_skeleton_hash": _digest("e"),
        "output_skeleton_hash": _digest("e"),
        "base_symbol_inventory_hash": _digest("f"),
        "output_symbol_inventory_hash": _digest("f"),
        "changed_functions": [function],
        "changed_function_hashes": entries,
        "patch_record": {"schema_version": 1, "changed_functions": entries},
        "patch_hash": patch_hash,
    }


def _fake_helper(
    tmp_path: Path,
    responses: dict[str, list[dict[str, object]]],
    mode: str = "ok",
    request_log: Path | None = None,
) -> list[str]:
    script = tmp_path / "fake_merge_helper.py"
    script.write_text(
        """
import json
import shutil
import sys
import time

responses = json.loads(sys.argv[1])
mode = sys.argv[2]
request_log = sys.argv[3]
for raw in sys.stdin:
    request = json.loads(raw)
    if request_log:
        with open(request_log, 'a', encoding='utf-8') as stream:
            stream.write(request['op'] + '\\n')
    if mode == 'malformed':
        print('{not json', flush=True)
        continue
    if mode == 'timeout':
        time.sleep(2)
        continue
    if mode == 'crash':
        sys.exit(17)
    operation = request['op']
    reply = responses[operation].pop(0)
    if mode == 'mismatch_id':
        reply['request_id'] = request['request_id'] + 1
    else:
        reply.setdefault('request_id', request['request_id'])
    if operation == 'merge' and mode != 'missing_output' and reply.get('status') == 'ok':
        shutil.copyfile(request['base_path'], request['merged_path'])
    print(json.dumps(reply, sort_keys=True), flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    return [
        sys.executable,
        "-u",
        str(script),
        json.dumps(responses),
        mode,
        str(request_log) if request_log is not None else "",
    ]


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    base = tmp_path / "base.ll"
    first = tmp_path / "first.ll"
    second = tmp_path / "second.ll"
    second_round = tmp_path / "second-round.ll"
    merged = tmp_path / "merged.ll"
    for path, text in (
        (base, "; base\n"),
        (first, "; first\n"),
        (second, "; second\n"),
        (second_round, "; second-round\n"),
    ):
        path.write_text(text, encoding="utf-8")
    return base, first, second, second_round, merged


def _normal_responses() -> dict[str, list[dict[str, object]]]:
    first = _patch_reply(function="f", output_module_hash=_digest("b"))
    second = _patch_reply(function="g", output_module_hash=_digest("c"))
    return {
        "ping": [
            {
                "status": "ok",
                "protocol_version": 1,
                "llvm_version": "test-llvm",
                "operations": ["ping", "inspect_patch", "merge", "compare_effect"],
            }
        ],
        "inspect_patch": [first, second],
        "merge": [
            {
                "status": "ok",
                "base_module_hash": _digest("a"),
                "base_skeleton_hash": _digest("e"),
                "output_module_hash": _digest("d"),
                "output_skeleton_hash": _digest("e"),
                "merged_functions": ["f", "g"],
                "contributed_functions": ["f", "g"],
                "input_patch_hashes": sorted([first["patch_hash"], second["patch_hash"]]),
                "input_output_module_hashes": sorted([_digest("b"), _digest("c")]),
                "merge_input_count": 2,
                "merge_wall_time_ns": 12,
            }
        ],
        "compare_effect": [
            {
                "status": "ok",
                "same_effect": True,
                "first_changed_functions": ["f"],
                "second_changed_functions": ["f"],
                "first_patch_hash": first["patch_hash"],
                "second_patch_hash": first["patch_hash"],
                "protected_functions": ["g"],
                "expected_protected_functions": ["g"],
                "protected_functions_preserved": True,
                "skeletons_unchanged": True,
                "symbol_inventories_unchanged": True,
            }
        ],
    }


def test_client_accepts_the_full_typed_helper_protocol_and_uses_canonical_record_ids(
    tmp_path: Path,
) -> None:
    base, first_output, second_output, second_round, merged = _paths(tmp_path)
    with DirectMergeClient(_fake_helper(tmp_path, _normal_responses()), timeout_s=0.5) as client:
        ping = client.ping()
        first = client.inspect_patch(base, first_output)
        second = client.inspect_patch(base, second_output)
        merged_record = client.merge(base, [second, first], merged)
        effect = client.compare_effect(
            first_base=base,
            first_output=first_output,
            second_base=merged,
            second_output=second_round,
            protected_functions=["g"],
            expected_first_patch=first,
        )

    assert ping["protocol_version"] == 1
    assert first.changed_functions == ("f",)
    assert second.changed_functions == ("g",)
    assert merged_record.contributed_functions == ("f", "g")
    assert merged_record.input_patch_hashes == tuple(sorted((first.patch_hash, second.patch_hash)))
    assert effect.same_effect is True
    assert effect.first_patch_hash == first.patch_hash
    assert first.operation_record_id.startswith("inspect_patch-")
    assert merged_record.operation_record_id.startswith("merge-")
    assert effect.operation_record_id.startswith("compare_effect-")


def test_client_rejects_helper_reported_unavailable_without_text_or_sequential_fallback(
    tmp_path: Path,
) -> None:
    responses = _normal_responses()
    responses["inspect_patch"] = [
        {
            "status": "error",
            "error_kind": "module_skeleton_changed",
            "error_message": "module-level change",
        }
    ]
    base, first_output, *_ = _paths(tmp_path)
    with DirectMergeClient(_fake_helper(tmp_path, responses), timeout_s=0.5) as client:
        with pytest.raises(DirectMergeUnavailable, match="module_skeleton_changed"):
            client.inspect_patch(base, first_output)
        assert not client.closed


@pytest.mark.parametrize(
    "mode, expected, error_kind",
    [
        ("malformed", "malformed JSON", "protocol_error"),
        ("mismatch_id", "request_id", "protocol_error"),
        ("timeout", "timed out", "timeout"),
        ("crash", "exited", "transport_error"),
    ],
)
def test_client_fails_closed_and_terminates_on_transport_or_protocol_failure(
    tmp_path: Path, mode: str, expected: str, error_kind: str
) -> None:
    base, first_output, *_ = _paths(tmp_path)
    # Process creation on the Windows experiment host can exceed 50ms even
    # when the protocol response is immediate; the timeout fixture sleeps 2s.
    client = DirectMergeClient(_fake_helper(tmp_path, _normal_responses(), mode), timeout_s=0.5)
    with pytest.raises(DirectMergeProtocolError, match=expected) as raised:
        client.inspect_patch(base, first_output)
    assert raised.value.error_kind == error_kind
    assert client.closed


def test_client_rejects_unknown_status_and_incomplete_success_reply(tmp_path: Path) -> None:
    base, first_output, *_ = _paths(tmp_path)
    unknown = _normal_responses()
    unknown["inspect_patch"] = [{"status": "mystery"}]
    with DirectMergeClient(_fake_helper(tmp_path, unknown), timeout_s=0.5) as client:
        with pytest.raises(DirectMergeProtocolError, match="unknown status"):
            client.inspect_patch(base, first_output)
        assert client.closed

    incomplete = _normal_responses()
    incomplete["inspect_patch"] = [{"status": "ok", "patch_hash": _digest("a")}]
    with DirectMergeClient(_fake_helper(tmp_path, incomplete), timeout_s=0.5) as client:
        with pytest.raises(DirectMergeProtocolError, match="missing|required"):
            client.inspect_patch(base, first_output)
        assert client.closed


def test_client_rejects_patch_hash_mismatch_and_missing_merged_artifact(tmp_path: Path) -> None:
    base, first_output, second_output, _, merged = _paths(tmp_path)
    bad_hash = _normal_responses()
    bad_hash["inspect_patch"] = [_patch_reply(function="f")]
    bad_hash["inspect_patch"][0]["patch_hash"] = _digest("0")
    with DirectMergeClient(_fake_helper(tmp_path, bad_hash), timeout_s=0.5) as client:
        with pytest.raises(DirectMergeProtocolError, match="patch_hash mismatch"):
            client.inspect_patch(base, first_output)
        assert client.closed

    missing = _normal_responses()
    with DirectMergeClient(
        _fake_helper(tmp_path, missing, "missing_output"), timeout_s=0.5
    ) as client:
        first = client.inspect_patch(base, first_output)
        second = client.inspect_patch(base, second_output)
        with pytest.raises(DirectMergeProtocolError, match="missing merged artifact"):
            client.merge(base, [first, second], merged)
        assert client.closed


def test_client_rejects_merge_input_hash_and_effect_patch_hash_mismatches(tmp_path: Path) -> None:
    base, first_output, second_output, second_round, merged = _paths(tmp_path)
    responses = _normal_responses()
    responses["merge"][0]["input_patch_hashes"] = [_digest("9")]
    with DirectMergeClient(_fake_helper(tmp_path, responses), timeout_s=0.5) as client:
        first = client.inspect_patch(base, first_output)
        second = client.inspect_patch(base, second_output)
        with pytest.raises(DirectMergeProtocolError, match="input_patch_hashes mismatch"):
            client.merge(base, [first, second], merged)
        assert client.closed

    effect_responses = _normal_responses()
    effect_responses["compare_effect"][0]["first_patch_hash"] = _digest("9")
    with DirectMergeClient(_fake_helper(tmp_path, effect_responses), timeout_s=0.5) as client:
        first = client.inspect_patch(base, first_output)
        client.inspect_patch(base, second_output)
        with pytest.raises(DirectMergeProtocolError, match="first_patch_hash mismatch"):
            client.compare_effect(
                first_base=base,
                first_output=first_output,
                second_base=merged,
                second_output=second_round,
                protected_functions=["g"],
                expected_first_patch=first,
            )
        assert client.closed


@pytest.mark.parametrize("mutated_artifact", ["base", "patch"])
def test_client_rehashes_patch_family_before_merge_and_never_sends_stale_inputs(
    tmp_path: Path, mutated_artifact: str
) -> None:
    base, first_output, second_output, _, merged = _paths(tmp_path)
    request_log = tmp_path / "requests.log"
    command = _fake_helper(
        tmp_path,
        _normal_responses(),
        request_log=request_log,
    )
    with DirectMergeClient(command, timeout_s=0.5) as client:
        first = client.inspect_patch(base, first_output)
        second = client.inspect_patch(base, second_output)
        before_merge = request_log.read_text(encoding="utf-8")
        target = base if mutated_artifact == "base" else first_output
        target.write_text("; mutated after inspection\n", encoding="utf-8")

        with pytest.raises(DirectMergeProtocolError, match="artifact hash mismatch"):
            client.merge(base, [first, second], merged)

        assert request_log.read_text(encoding="utf-8") == before_merge


def test_client_operation_record_id_binds_current_input_bytes_at_the_same_path(
    tmp_path: Path,
) -> None:
    base, first_output, *_ = _paths(tmp_path)
    responses = _normal_responses()
    responses["inspect_patch"].append(_patch_reply(function="f"))
    with DirectMergeClient(_fake_helper(tmp_path, responses), timeout_s=0.5) as client:
        original = client.inspect_patch(base, first_output)
        first_output.write_text("; replaced at the same path\n", encoding="utf-8")
        replaced = client.inspect_patch(base, first_output)

    assert original.operation_record_id != replaced.operation_record_id


@pytest.mark.parametrize(
    "mutated_artifact",
    ["first_base", "first_output", "second_base", "second_output"],
)
def test_compare_effect_rebinds_all_expected_patch_artifacts_before_helper_request(
    tmp_path: Path, mutated_artifact: str
) -> None:
    base, first_output, second_base, second_output, _ = _paths(tmp_path)
    request_log = tmp_path / "compare-requests.log"
    responses = _normal_responses()
    responses["inspect_patch"].append(_patch_reply(function="h"))
    with DirectMergeClient(
        _fake_helper(tmp_path, responses, request_log=request_log), timeout_s=0.5
    ) as client:
        expected_first = client.inspect_patch(base, first_output)
        expected_second = client.inspect_patch(second_base, second_output)
        before_compare = request_log.read_text(encoding="utf-8")
        {
            "first_base": base,
            "first_output": first_output,
            "second_base": second_base,
            "second_output": second_output,
        }[mutated_artifact].write_text("; mutated after inspection\n", encoding="utf-8")

        with pytest.raises(DirectMergeProtocolError, match="artifact hash mismatch"):
            client.compare_effect(
                first_base=base,
                first_output=first_output,
                second_base=second_base,
                second_output=second_output,
                protected_functions=["g"],
                expected_first_patch=expected_first,
                expected_second_patch=expected_second,
            )

        assert request_log.read_text(encoding="utf-8") == before_compare

def test_client_rejects_invalid_paths_before_sending_a_helper_request(tmp_path: Path) -> None:
    client = DirectMergeClient(_fake_helper(tmp_path, _normal_responses()), timeout_s=0.5)
    with pytest.raises(DirectMergeProtocolError, match="missing input artifact"):
        client.inspect_patch(tmp_path / "missing.ll", tmp_path / "also-missing.ll")
    assert client.closed is False
    client.close()


def _artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluator_patch(
    base: Path,
    output: Path,
    functions: tuple[str, ...],
) -> PatchRecord:
    function_hashes = tuple(
        (function, _digest("c"), _digest("d")) for function in functions
    )
    return PatchRecord(
        operation_record_id=f"inspect-{output.stem}",
        base_path=base,
        output_path=output,
        base_artifact_sha256=_artifact_sha256(base),
        output_artifact_sha256=_artifact_sha256(output),
        base_module_hash=_digest("a"),
        output_module_hash=_digest("b"),
        base_skeleton_hash=_digest("e"),
        output_skeleton_hash=_digest("e"),
        base_symbol_inventory_hash=_digest("f"),
        output_symbol_inventory_hash=_digest("f"),
        changed_functions=functions,
        changed_function_hashes=function_hashes,
        patch_hash=_patch_hash(list(function_hashes)),
    )


class _EvaluatorMergeClient:
    """Typed fake that makes evaluator semantics visible without LLVM."""

    def __init__(
        self,
        patches: dict[Path, PatchRecord],
        *,
        effect_equal: dict[str, bool] | None = None,
        unavailable_merge_for: str | None = None,
    ) -> None:
        self.patches = patches
        self.effect_equal = effect_equal or {}
        self.unavailable_merge_for = unavailable_merge_for
        self.inspections: list[Path] = []
        self.merges: list[tuple[Path, tuple[Path, ...], Path]] = []
        self.comparisons: list[tuple[Path, tuple[str, ...]]] = []

    def inspect_patch(self, base: Path, output: Path) -> PatchRecord:
        self.inspections.append(output)
        return self.patches[output]

    def merge(
        self,
        base: Path,
        patches: list[PatchRecord],
        merged_path: Path,
    ) -> MergeRecord:
        omitted = next(
            action for action, patch in _EVALUATOR_PATCHES.items() if patch.output_path not in {
                item.output_path for item in patches
            }
        )
        if omitted == self.unavailable_merge_for:
            raise DirectMergeUnavailable(
                "structured direct merge undefined",
                operation="merge",
                operation_record_id="merge-unavailable",
                error_kind="module_skeleton_changed",
            )
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text("; merged\n", encoding="utf-8")
        canonical = tuple(sorted(patches, key=lambda item: str(item.output_path)))
        contributed = tuple(sorted(
            function for patch in canonical for function in patch.changed_functions
        ))
        self.merges.append((base, tuple(item.output_path for item in patches), merged_path))
        return MergeRecord(
            operation_record_id=f"merge-{omitted}",
            base_path=base,
            output_paths=tuple(item.output_path for item in canonical),
            merged_path=merged_path,
            base_artifact_sha256=_artifact_sha256(base),
            merged_artifact_sha256=_artifact_sha256(merged_path),
            base_module_hash=_digest("a"),
            base_skeleton_hash=_digest("e"),
            output_module_hash=_digest("b"),
            output_skeleton_hash=_digest("e"),
            merged_functions=contributed,
            contributed_functions=contributed,
            input_patch_hashes=tuple(sorted(item.patch_hash for item in canonical)),
            input_output_module_hashes=tuple(sorted(item.output_module_hash for item in canonical)),
            merge_input_count=len(canonical),
            merge_wall_time_ns=1,
        )

    def compare_effect(
        self,
        *,
        first_base: Path,
        first_output: Path,
        second_base: Path,
        second_output: Path,
        protected_functions: list[str] | tuple[str, ...],
        expected_first_patch: PatchRecord,
    ) -> EffectRecord:
        action = first_output.stem
        same = self.effect_equal.get(action, True)
        first = expected_first_patch.changed_functions
        second = first if same else tuple(sorted((*first, "enabled_after_merge")))
        self.comparisons.append((first_output, tuple(protected_functions)))
        return EffectRecord(
            operation_record_id=f"effect-{action}",
            first_base_path=first_base,
            first_output_path=first_output,
            second_base_path=second_base,
            second_output_path=second_output,
            first_base_artifact_sha256=_artifact_sha256(first_base),
            first_output_artifact_sha256=_artifact_sha256(first_output),
            second_base_artifact_sha256=_artifact_sha256(second_base),
            second_output_artifact_sha256=_artifact_sha256(second_output),
            same_effect=same,
            first_changed_functions=first,
            second_changed_functions=second,
            first_patch_hash=expected_first_patch.patch_hash,
            second_patch_hash=(
                expected_first_patch.patch_hash if same else _digest("9")
            ),
            protected_functions=tuple(protected_functions),
            expected_protected_functions=tuple(protected_functions),
            protected_functions_preserved=True,
            skeletons_unchanged=True,
            symbol_inventories_unchanged=True,
        )


_EVALUATOR_PATCHES: dict[str, PatchRecord] = {}


def _evaluator_fixture(
    tmp_path: Path,
    function_sets: dict[str, tuple[str, ...]],
) -> tuple[Path, list[dict[str, object]], dict[str, str], _EvaluatorMergeClient]:
    base = tmp_path / "root.ll"
    base.write_text("; root\n", encoding="utf-8")
    profiles: list[dict[str, object]] = []
    action_map: dict[str, str] = {}
    patches: dict[Path, PatchRecord] = {}
    global _EVALUATOR_PATCHES
    _EVALUATOR_PATCHES = {}
    for action, functions in sorted(function_sets.items()):
        output = tmp_path / f"{action}.ll"
        output.write_text(f"; {action}\n", encoding="utf-8")
        patch = _evaluator_patch(base, output, functions)
        patches[output] = patch
        _EVALUATOR_PATCHES[action] = patch
        action_map[action] = action
        profiles.append(
            {
                "action_id": action,
                "execution_status": "success",
                "verifier_status": "success",
                "activity_status": "active" if functions else "no_op",
                "output_path": str(output),
                "output_hard_state_id": _digest(action.lower()[0]),
                "physical_pass_invocations": 1,
            }
        )
    return base, profiles, action_map, _EvaluatorMergeClient(patches)


def _second_runner(
    calls: list[tuple[Path, str, Path]], *, verifier_status: str = "success"
):
    def run(base: Path, action: str, output: Path) -> dict[str, object]:
        calls.append((base, action, output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"; second {action}\n", encoding="utf-8")
        return {
            "success": True,
            "output_path": output,
            "hard_state_id": _digest(action.lower()[0]),
            "verifier_status": verifier_status,
            "stderr": "",
            "command": ("fake-worker", action),
        }

    return run


def test_evaluator_runs_exactly_one_all_other_direct_merge_and_second_round_per_action(
    tmp_path: Path,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",), "C": ("h",)}
    )
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls),
    )

    assert len(client.inspections) == 3
    assert len(client.merges) == len(calls) == len(client.comparisons) == 3
    assert all(len(inputs) == 2 for _, inputs, _ in client.merges)
    assert {row["directional_status"] for row in result.directional_rows} == {
        "authorized_all_others"
    }
    from phasebatch.ir_equivalence import DEFAULT_HARD_STATE_POLICY, hard_state_hash

    assert all(
        row["merged_input_hard_state_id"]
        == hard_state_hash(Path(row["merged_input_path"]), DEFAULT_HARD_STATE_POLICY)
        for row in result.directional_rows
    )
    assert result.group_row["group_authorization_status"] == "authorized"
    assert result.group_row["authority_granted"] == "false"
    assert result.group_row["proved_commute"] == "false"
    assert {row["two_n_pair_status"] for row in result.pair_rows} == {
        "both_directions_authorized"
    }


def test_evaluator_stops_before_any_second_round_when_complete_patch_family_overlaps(
    tmp_path: Path,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("f",), "C": ("h",)}
    )
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U30",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls),
    )

    assert len(client.inspections) == 3
    assert not client.merges and not calls and not client.comparisons
    assert result.group_row["first_round_disjoint_status"] == "overlap"
    assert result.group_row["group_authorization_status"] == "group_precondition_unavailable"
    assert {row["directional_status"] for row in result.directional_rows} == {
        "direct_merge_not_defined"
    }
    assert {row["two_n_pair_status"] for row in result.pair_rows} == {
        "group_precondition_unavailable"
    }


def test_evaluator_treats_a_noop_becoming_active_after_direct_merge_as_effect_changed(
    tmp_path: Path,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": (), "B": ("g",)}
    )
    client.effect_equal["A"] = False
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls),
    )

    action_a = next(row for row in result.directional_rows if row["action_id"] == "A")
    assert action_a["directional_status"] == "rejected_effect_changed"
    assert action_a["first_round_effect_sha256"] != action_a["second_round_effect_sha256"]
    assert result.group_row["group_authorization_status"] == "rejected"


def test_evaluator_requires_explicit_second_round_verifier_success_for_authorization(
    tmp_path: Path,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",)}
    )
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls, verifier_status=""),
    )

    assert len(calls) == 2
    assert not client.comparisons
    assert {row["directional_status"] for row in result.directional_rows} == {
        "second_round_failed"
    }
    assert result.group_row["all_n_second_round_status"] == "second_round_failed"
    assert result.group_row["group_authorization_status"] == "group_precondition_unavailable"


@pytest.mark.parametrize(
    ("first_status", "expected_group_status", "expected_directional_status"),
    [
        ("error", "round1_precondition_failed", "round1_precondition_failed"),
        ("timeout", "timeout", "timeout"),
    ],
)
def test_evaluator_never_merges_or_runs_second_round_after_any_first_round_failure(
    tmp_path: Path,
    first_status: str,
    expected_group_status: str,
    expected_directional_status: str,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",)}
    )
    profiles[0]["execution_status"] = first_status
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls),
    )

    assert not client.inspections and not client.merges and not calls and not client.comparisons
    assert result.group_row["round1_status"] == expected_group_status
    assert {row["directional_status"] for row in result.directional_rows} == {
        expected_directional_status
    }


def test_evaluator_records_direct_merge_nonapplicability_without_sequential_fallback(
    tmp_path: Path,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",)}
    )
    client.unavailable_merge_for = "A"
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls),
    )

    action_a = next(row for row in result.directional_rows if row["action_id"] == "A")
    assert action_a["directional_status"] == "direct_merge_not_defined"
    assert action_a["second_round_status"] == "not_run"
    assert len(calls) == 1
    assert result.group_row["all_n_merge_status"] == "direct_merge_not_defined"
    assert result.group_row["group_authorization_status"] == "group_precondition_unavailable"


def test_evaluator_retains_one_sided_or_endpoint_false_authorization_against_ab_ba(
    tmp_path: Path,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",)}
    )
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls),
        pair_observations=[
            {
                "study_manifest_id": "manifest",
                "group_id": "U14",
                "program_id": "p1",
                "action_a_id": "A",
                "action_b_id": "B",
                "row_id": "pair-row",
                "dynamic_result": "order_sensitive",
            }
        ],
    )

    assert result.pair_rows[0]["two_n_pair_status"] == "both_directions_authorized"
    assert result.pair_rows[0]["validation_status"] == "false_authorization"
    assert result.pair_rows[0]["false_authorization"] == "true"
    assert result.pair_rows[0]["stable_false_authorization"] == "false"
    assert result.pair_rows[0]["worker_replay_status"] == "unavailable"
    assert result.pair_rows[0]["external_opt_replay_status"] == "unavailable"
    assert result.pair_rows[0]["two_n_replay_status"] == "unavailable"


def test_evaluator_canonicalizes_mapping_action_order_but_runs_mapping_values(
    tmp_path: Path,
) -> None:
    base, profiles, _, first_client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",)}
    )
    action_a = object()
    action_b = object()
    received_first: list[object] = []
    received_second: list[object] = []

    def runner(received: list[object]):
        def run(_base: Path, action: object, output: Path) -> dict[str, object]:
            received.append(action)
            assert action in {action_a, action_b}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("; second\n", encoding="utf-8")
            return {
                "success": True,
                "output_path": output,
                "hard_state_id": _digest("a"),
                "verifier_status": "success",
                "command": ("fake-worker",),
            }

        return run

    unordered = {"B": action_b, "A": action_a}
    ordered = {"A": action_a, "B": action_b}
    first = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=unordered,
        profiles=profiles,
        merge_client=first_client,
        out_dir=tmp_path / "canonical",
        run_second=runner(received_first),
        clock_ns=iter((1_000_000, 3_100_000)).__next__,
    )
    _, _, _, second_client = _evaluator_fixture(tmp_path, {"A": ("f",), "B": ("g",)})
    second = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=ordered,
        profiles=profiles,
        merge_client=second_client,
        out_dir=tmp_path / "canonical",
        run_second=runner(received_second),
        clock_ns=iter((1_000_000, 3_100_000)).__next__,
    )

    assert received_first == received_second == [action_a, action_b]
    assert first.group_row == second.group_row
    assert first.directional_rows == second.directional_rows
    assert first.pair_rows == second.pair_rows


def test_evaluator_records_measured_nonzero_group_wall_time_from_runner_clock(
    tmp_path: Path,
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",)}
    )
    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "measured-wall",
        run_second=_second_runner([]),
        clock_ns=iter((5_000_000, 7_000_001)).__next__,
    )

    assert result.group_row["wall_time_ms"] == 3


@pytest.mark.parametrize("pair_manifest_id", ["other-manifest", ""])
def test_evaluator_treats_ab_ba_rows_from_another_or_missing_manifest_as_unavailable(
    tmp_path: Path, pair_manifest_id: str
) -> None:
    base, profiles, actions, client = _evaluator_fixture(
        tmp_path, {"A": ("f",), "B": ("g",)}
    )
    calls: list[tuple[Path, str, Path]] = []

    result = evaluate_group_2n(
        root_ir=base,
        group_id="U14",
        program_id="p1",
        study_manifest_id="manifest",
        actions=actions,
        profiles=profiles,
        merge_client=client,
        out_dir=tmp_path / "two-n",
        run_second=_second_runner(calls),
        pair_observations=[
            {
                "study_manifest_id": pair_manifest_id,
                "group_id": "U14",
                "program_id": "p1",
                "action_a_id": "A",
                "action_b_id": "B",
                "row_id": "wrong-or-missing-manifest",
                "dynamic_result": "order_sensitive",
            }
        ],
    )

    assert result.pair_rows[0]["dynamic_result"] == "unknown"
    assert result.pair_rows[0]["validation_status"] == "unavailable"
    assert result.pair_rows[0]["false_authorization"] == "false"
