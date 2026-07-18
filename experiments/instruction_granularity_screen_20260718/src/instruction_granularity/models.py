from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping


class ExtractionLevel(str, Enum):
    FUNC_ONLY = "FUNC_ONLY"
    BLOCK_ONLY = "BLOCK_ONLY"
    EFFECT_ONLY = "EFFECT_ONLY"
    INSTRUCTION_ONLY = "INSTRUCTION_ONLY"


LEVEL_RANK = {
    ExtractionLevel.FUNC_ONLY: 0,
    ExtractionLevel.BLOCK_ONLY: 1,
    ExtractionLevel.EFFECT_ONLY: 2,
    ExtractionLevel.INSTRUCTION_ONLY: 3,
}


@dataclass(slots=True)
class ExtractionTrace:
    function_builds: int = 0
    block_builds: int = 0
    effect_builds: int = 0
    instruction_builds: int = 0


EffectKey = tuple[str, str, str]
InstructionToken = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class FingerprintCollision:
    fingerprint: str
    canonical_forms: tuple[str, ...]
    function: str = ""
    block: str = ""
    effect_class: str = ""


@dataclass(slots=True)
class ParsedModule:
    level: ExtractionLevel
    text_sha256: str
    functions: dict[str, str]
    blocks: dict[str, str] = field(default_factory=dict)
    effect_slices: dict[EffectKey, str] = field(default_factory=dict)
    instruction_counters: dict[EffectKey, Counter[str]] = field(default_factory=dict)
    canonical_instructions: dict[EffectKey, tuple[str, ...]] = field(default_factory=dict)
    function_headers: dict[str, str] = field(default_factory=dict)
    attribute_groups: dict[str, str] = field(default_factory=dict)
    function_attribute_references: dict[str, frozenset[str]] = field(default_factory=dict)
    module_structure: str = ""
    cfg_signatures: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    wildcard_reasons: tuple[str, ...] = ()
    opcodes: frozenset[str] = frozenset()
    collisions: tuple[FingerprintCollision, ...] = ()
    logical_instruction_count: int = 0


@dataclass(frozen=True, slots=True)
class TransitionFeature:
    functions: frozenset[str]
    blocks: frozenset[str]
    effect_tokens: frozenset[EffectKey]
    instruction_tokens: frozenset[InstructionToken]
    wildcard_reasons: tuple[str, ...] = ()
    observed_opcodes: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"selected", "not_selected", "unknown"}:
            raise ValueError(f"invalid selection status: {self.status}")


@dataclass(frozen=True, slots=True)
class PairRow:
    observation_id: str
    program: str
    action_a_id: str
    action_b_id: str
    action_a_name: str
    action_b_name: str
    action_a_pipeline: str
    action_b_pipeline: str
    dynamic_relation: str
    state_ir_path: Path
    state_hard_hash: str
    h_func_selected: bool
    h_block_selected: bool
    h_effect_selected: bool


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: Path
    expected_size: int
    expected_sha256: str
    expected_hard_hash: str = ""
    manifest_relative_path: str = ""
    consistency_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceAttempt:
    repetition: int
    configuration: str
    status: str
    root: Path
    completion_path: Path
    completion_sha256: str
    outputs: Mapping[tuple[str, str], ArtifactRef]
    pair_cost_ms: Mapping[str, float]
    pair_relations: Mapping[str, str]
    single_pass_csv_sha256: str
    pair_runs_csv_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenDataset:
    experiment_root: Path
    old_experiment_root: Path
    authoritative_csv: Path
    authoritative_csv_sha256: str
    pairs: tuple[PairRow, ...]
    attempts: tuple[SourceAttempt, ...]
    base_artifacts: Mapping[str, ArtifactRef]
    action_ids: tuple[str, ...]
    transition_keys: frozenset[tuple[str, str]]
    relation_counts: Mapping[str, int]
    legacy_counts: Mapping[str, Mapping[str, int]]

    @property
    def pair_by_id(self) -> dict[str, PairRow]:
        return {pair.observation_id: pair for pair in self.pairs}


@dataclass(frozen=True, slots=True)
class ArtifactRead:
    text: str | None
    error: str = ""
    actual_size: int = 0
    actual_sha256: str = ""


@dataclass(slots=True)
class ExtractionRun:
    level: ExtractionLevel
    source_repetition: int
    features: dict[tuple[str, str], TransitionFeature]
    decisions: dict[str, SelectionDecision]
    trace: ExtractionTrace
    artifact_errors: tuple[dict[str, str], ...]
    collisions: tuple[FingerprintCollision, ...]
    artifact_read_ms: float
    parse_ms: float
    feature_build_ms: float
    pair_selection_ms: float
    total_extraction_ms: float


@dataclass(frozen=True, slots=True)
class TimingTask:
    phase: str
    cycle: int
    measured_repetition: int
    source_repetition: int
    level: ExtractionLevel
    sequence_index: int


@dataclass(frozen=True, slots=True)
class RuntimeRecord:
    phase: str
    cycle: int
    measured_repetition: int
    source_repetition: int
    level: ExtractionLevel
    artifact_read_ms: float
    parse_ms: float
    feature_build_ms: float
    pair_selection_ms: float
    total_extraction_ms: float
    programs: int
    transitions: int
    pairs: int
    sequence_index: int = 0
    selected_count: int = 0
    unknown_count: int = 0
    function_builds: int = 0
    block_builds: int = 0
    effect_builds: int = 0
    instruction_builds: int = 0


@dataclass(frozen=True, slots=True)
class PairedIncrementalRow:
    comparison: str
    measured_repetition: int
    source_repetition: int
    lower_level: str
    upper_level: str
    lower_total_ms: float
    upper_total_ms: float
    delta_ms: float


Hasher = Callable[[str], str]

