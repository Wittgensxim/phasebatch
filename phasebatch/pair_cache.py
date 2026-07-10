from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock


NORMALIZER_VERSION = "phasebatch.normalizer.v1"
IR_EQUIVALENCE_VERSION = "phasebatch.ir_equivalence.v1"

PAIR_CACHE_VALUE_FIELDS = [
    "dynamic_relation",
    "final_relation",
    "ab_success",
    "ba_success",
    "ab_hash",
    "ba_hash",
    "same_hash",
    "text_hash_equal",
    "llvm_diff_equal",
    "module_fingerprint_equal",
    "equality_tier",
    "equality_reason",
    "can_hard_fold",
    "ab_inst",
    "ba_inst",
    "inst_delta_ab_ba",
    "failure_kind",
    "ab_path",
    "ba_path",
    "pair_test_pass_invocations_baseline",
    "pair_test_pass_invocations_actual",
    "pair_test_pass_invocations_saved",
]


@dataclass
class PairRelationCache:
    llvm_version: str = ""
    target_triple: str = ""
    normalizer_version: str = NORMALIZER_VERSION
    ir_equivalence_version: str = IR_EQUIVALENCE_VERSION
    _entries: dict[str, dict] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @classmethod
    def from_tools(cls, tools: dict | None, input_ll: Path | None = None) -> "PairRelationCache":
        tools = tools or {}
        metadata = tools.get("_toolchain_metadata") if isinstance(tools.get("_toolchain_metadata"), dict) else {}
        opt_meta = metadata.get("tools", {}).get("opt", {}) if isinstance(metadata.get("tools"), dict) else {}
        llvm_version = str(tools.get("_llvm_version") or opt_meta.get("version") or tools.get("opt") or "")
        target_triple = str(tools.get("_target_triple") or _read_target_triple(input_ll) or "")
        return cls(llvm_version=llvm_version, target_triple=target_triple)

    def cache_key(
        self,
        *,
        state_hash: str,
        pass_a_name: str,
        pass_b_name: str,
        pass_a_pipeline: str,
        pass_b_pipeline: str,
    ) -> str:
        pass_pair = sorted(
            [
                {"name": pass_a_name, "pipeline": pass_a_pipeline},
                {"name": pass_b_name, "pipeline": pass_b_pipeline},
            ],
            key=lambda item: (item["name"], item["pipeline"]),
        )
        payload = {
            "state_hash": state_hash,
            "pass_pair": pass_pair,
            "llvm_version": self.llvm_version,
            "target_triple": self.target_triple,
            "normalizer_version": self.normalizer_version,
            "ir_equivalence_version": self.ir_equivalence_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict | None:
        with self._lock:
            value = self._entries.get(cache_key)
            return dict(value) if value is not None else None

    def store(self, cache_key: str, row: dict) -> None:
        with self._lock:
            self._entries[cache_key] = {field: str(row.get(field, "")) for field in PAIR_CACHE_VALUE_FIELDS}


def _read_target_triple(input_ll: Path | None) -> str:
    if input_ll is None:
        return ""
    try:
        text = Path(input_ll).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    match = re.search(r'^target\s+triple\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return match.group(1) if match else ""
