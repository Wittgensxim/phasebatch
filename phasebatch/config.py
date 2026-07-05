from __future__ import annotations

from pathlib import Path

from .pass_config import load_pass_config


def load_passes(path: str | Path) -> list[str]:
    """Load enabled logical pass names from legacy or rich pass configs."""
    return [spec.name for spec in load_pass_config(path)]
