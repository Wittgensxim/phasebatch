from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runner import ROOT_IR_MODES


@dataclass(frozen=True)
class RuntimeConfig:
    enabled: bool = False
    top_k: int = 5
    warmups: int = 1
    trials: int = 5
    timeout: int = 30
    expected_exit_code: int = 0
    command: tuple[str, ...] = ("{exe}",)
    llc_opt_level: str = "O2"


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    passes_path: Path
    mode: str = "exact"
    max_rounds: int = 1
    beam_width: int = 8
    max_states: int = 500
    max_batches_per_state: int = 20
    max_component_size: int = 10
    max_batch_candidates: int = 200
    budgeted_validation_strategy: str = "all"
    batch_validation_mode: str = "auto"
    runtime_rerank: bool = False
    require_transition: bool = False


@dataclass(frozen=True)
class StagedConfig:
    root_ir_mode: str
    stages: tuple[StageSpec, ...]
    runtime: RuntimeConfig


def load_staged_config(path: str | Path) -> StagedConfig:
    manifest_path = Path(path).resolve()
    raw = _load_mapping(manifest_path)
    root_ir_mode = str(raw.get("root_ir_mode") or "legacy-o0")
    if root_ir_mode not in ROOT_IR_MODES:
        raise ValueError(f"unknown root IR mode: {root_ir_mode}")

    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("staged manifest must contain a non-empty stages list")

    stages: list[StageSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_stages):
        if not isinstance(item, dict):
            raise ValueError(f"stage {index} must be a mapping")
        stage_id = str(item.get("id") or "").strip()
        if not stage_id:
            raise ValueError(f"stage {index} is missing id")
        if stage_id in seen_ids:
            raise ValueError(f"duplicate stage id: {stage_id}")
        seen_ids.add(stage_id)

        passes_value = str(item.get("passes") or "").strip()
        if not passes_value:
            raise ValueError(f"stage {stage_id} is missing passes")
        passes_path = _resolve_path(passes_value, manifest_path.parent)
        if not passes_path.exists():
            raise FileNotFoundError(f"stage pass config not found: {passes_path}")

        mode = str(item.get("mode") or "exact")
        if mode not in {"exact", "budgeted", "auto"}:
            raise ValueError(f"stage {stage_id} has unknown mode: {mode}")
        stages.append(
            StageSpec(
                stage_id=stage_id,
                passes_path=passes_path,
                mode=mode,
                max_rounds=_positive_int(item, "max_rounds", 1, stage_id),
                beam_width=_positive_int(item, "beam_width", 8, stage_id),
                max_states=_positive_int(item, "max_states", 500, stage_id),
                max_batches_per_state=_positive_int(item, "max_batches_per_state", 20, stage_id),
                max_component_size=_positive_int(item, "max_component_size", 10, stage_id),
                max_batch_candidates=_positive_int(item, "max_batch_candidates", 200, stage_id),
                budgeted_validation_strategy=str(item.get("budgeted_validation_strategy") or "all"),
                batch_validation_mode=str(item.get("batch_validation_mode") or "auto"),
                runtime_rerank=_as_bool(item.get("runtime_rerank", False)),
                require_transition=_as_bool(item.get("require_transition", False)),
            )
        )

    return StagedConfig(
        root_ir_mode=root_ir_mode,
        stages=tuple(stages),
        runtime=_runtime_config(raw.get("runtime")),
    )


def _runtime_config(value: Any) -> RuntimeConfig:
    if value in (None, False):
        return RuntimeConfig()
    if not isinstance(value, dict):
        raise ValueError("runtime must be a mapping")
    command = value.get("command", ["{exe}"])
    if not isinstance(command, list) or not command:
        raise ValueError("runtime command must be a non-empty list")
    command_parts = tuple(str(part) for part in command)
    enabled = _as_bool(value.get("enabled", True))
    if enabled and not any("{exe}" in part for part in command_parts):
        raise ValueError("runtime command must contain the {exe} placeholder")
    return RuntimeConfig(
        enabled=enabled,
        top_k=_positive_int(value, "top_k", 5, "runtime"),
        warmups=_nonnegative_int(value, "warmups", 1, "runtime"),
        trials=_positive_int(value, "trials", 5, "runtime"),
        timeout=_positive_int(value, "timeout", 30, "runtime"),
        expected_exit_code=int(value.get("expected_exit_code", 0)),
        command=command_parts,
        llc_opt_level=str(value.get("llc_opt_level") or "O2"),
    )


def _load_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"staged manifest must be a mapping: {path}")
    return loaded


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (base / path).resolve()


def _positive_int(item: dict, key: str, default: int, owner: str) -> int:
    value = int(item.get(key, default))
    if value < 1:
        raise ValueError(f"{owner} {key} must be positive")
    return value


def _nonnegative_int(item: dict, key: str, default: int, owner: str) -> int:
    value = int(item.get(key, default))
    if value < 0:
        raise ValueError(f"{owner} {key} must be non-negative")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
