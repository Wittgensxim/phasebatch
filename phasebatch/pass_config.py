from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PassSpec:
    name: str
    pipeline: str | None
    pipeline_candidates: list[str]
    category: str
    stage: str
    enabled: bool


@dataclass(frozen=True)
class PassRegistry:
    specs_by_name: dict[str, PassSpec]
    order: list[str]

    @classmethod
    def from_specs(cls, specs: list[PassSpec]) -> "PassRegistry":
        specs_by_name: dict[str, PassSpec] = {}
        order: list[str] = []
        for spec in specs:
            if spec.name in specs_by_name:
                raise ValueError(f"duplicate pass name in config: {spec.name}")
            specs_by_name[spec.name] = spec
            order.append(spec.name)
        return cls(specs_by_name=specs_by_name, order=order)

    def names(self) -> list[str]:
        return list(self.order)

    def pipeline_for(self, name: str) -> str:
        spec = self.specs_by_name.get(name)
        if spec is None:
            return name
        if spec.pipeline:
            return spec.pipeline
        if spec.pipeline_candidates:
            return spec.pipeline_candidates[0]
        return name

    def category_for(self, name: str) -> str:
        spec = self.specs_by_name.get(name)
        return spec.category if spec else "unknown"

    def stage_for(self, name: str) -> str:
        spec = self.specs_by_name.get(name)
        return spec.stage if spec else ""


def load_pass_config(path: str | Path) -> list[PassSpec]:
    """Load pass specs from the legacy or rich pass configuration format."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"passes config not found: {config_path}")

    raw_items = _load_raw_pass_items(config_path)
    specs: list[PassSpec] = []
    for index, item in enumerate(raw_items):
        spec = _parse_pass_item(item, config_path, index)
        if spec.enabled:
            specs.append(spec)

    if not specs:
        raise ValueError(f"no enabled passes found in {config_path}")
    return specs


def load_pass_registry(path: str | Path) -> PassRegistry:
    return PassRegistry.from_specs(load_pass_config(path))


def resolve_pipeline_sequence(pass_names: list[str], registry: PassRegistry | None = None) -> list[str]:
    if registry is None:
        return list(pass_names)
    return [registry.pipeline_for(pass_name) for pass_name in pass_names]


def _load_raw_pass_items(config_path: Path) -> list[Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    if yaml is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("passes"), list):
            raise ValueError(f"passes config must contain a passes list: {config_path}")
        return loaded["passes"]

    return _parse_small_yaml_subset(config_path)


def _parse_pass_item(item: Any, config_path: Path, index: int) -> PassSpec:
    if isinstance(item, str):
        name = item.strip()
        if not name:
            raise ValueError(f"empty pass entry in {config_path}")
        return PassSpec(
            name=name,
            pipeline=name,
            pipeline_candidates=[name],
            category="unknown",
            stage="",
            enabled=True,
        )

    if not isinstance(item, dict):
        raise ValueError(f"pass entry {index} in {config_path} must be a string or mapping")

    enabled = _as_bool(item.get("enabled", True))
    name = str(item.get("name") or "").strip()
    pipeline_value = item.get("pipeline")
    pipeline = str(pipeline_value).strip() if pipeline_value not in (None, "") else None
    raw_candidates = item.get("pipeline_candidates")

    candidates: list[str]
    if raw_candidates is None:
        candidates = [pipeline or name]
    elif isinstance(raw_candidates, list):
        candidates = [str(candidate).strip() for candidate in raw_candidates if str(candidate).strip()]
    else:
        raise ValueError(f"pipeline_candidates for pass entry {index} in {config_path} must be a list")

    if pipeline:
        candidates = [pipeline] + [candidate for candidate in candidates if candidate != pipeline]

    if not name:
        name = pipeline or (candidates[0] if candidates else "")
    if not name:
        raise ValueError(f"pass entry {index} in {config_path} is missing name/pipeline")
    if not candidates:
        candidates = [name]

    return PassSpec(
        name=name,
        pipeline=pipeline,
        pipeline_candidates=candidates,
        category=str(item.get("category") or "unknown").strip() or "unknown",
        stage=str(item.get("stage") or "").strip(),
        enabled=enabled,
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return bool(value)


def _parse_small_yaml_subset(config_path: Path) -> list[Any]:
    items: list[Any] = []
    in_passes = False
    current: dict[str, Any] | None = None
    current_list_key: str | None = None
    current_list_indent = -1

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        without_comment = _strip_comment(raw_line).rstrip()
        if not without_comment.strip():
            continue

        indent = len(without_comment) - len(without_comment.lstrip(" "))
        stripped = without_comment.strip()
        if stripped == "passes:":
            in_passes = True
            continue
        if not in_passes:
            continue
        if indent == 0 and not stripped.startswith("- "):
            break

        if stripped.startswith("- "):
            value = stripped[2:].strip()
            if current is not None and current_list_key and indent > current_list_indent:
                current[current_list_key].append(_parse_scalar(value))
                continue

            if current is not None:
                items.append(current)
                current = None
            current_list_key = None
            current_list_indent = -1

            if not value:
                raise ValueError(f"empty pass entry in {config_path}")
            if ":" in value:
                key, field_value = _split_field(value, config_path)
                current = {key: _parse_scalar(field_value)}
            else:
                items.append(_parse_scalar(value))
            continue

        if current is None:
            raise ValueError(f"unexpected line in passes list in {config_path}: {raw_line}")

        key, field_value = _split_field(stripped, config_path)
        if field_value == "":
            current[key] = []
            current_list_key = key
            current_list_indent = indent
        else:
            current[key] = _parse_scalar(field_value)
            current_list_key = None
            current_list_indent = -1

    if current is not None:
        items.append(current)
    if not items:
        raise ValueError(f"no passes found in {config_path}")
    return items


def _split_field(value: str, config_path: Path) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"expected key/value field in {config_path}: {value}")
    key, field_value = value.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"empty field name in {config_path}: {value}")
    return key, field_value.strip()


def _parse_scalar(value: str) -> str | bool:
    unquoted = _unquote(value.strip())
    lowered = unquoted.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return unquoted


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
