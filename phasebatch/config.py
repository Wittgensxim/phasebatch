from __future__ import annotations

from pathlib import Path


def load_passes(path: str | Path) -> list[str]:
    """Load the MVP pass list from the small YAML subset used by configs."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"passes config not found: {config_path}")

    passes: list[str] = []
    in_passes = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped == "passes:":
            in_passes = True
            continue

        if in_passes and stripped.startswith("- "):
            value = stripped[2:].strip()
            if not value:
                raise ValueError(f"empty pass entry in {config_path}")
            passes.append(_unquote(value))
            continue

        if in_passes and not raw_line.startswith((" ", "\t")):
            in_passes = False

    if not passes:
        raise ValueError(f"no passes found in {config_path}")
    return passes


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
