from __future__ import annotations

import os
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    for name in ("TEMP", "TMP", "MPLCONFIGDIR"):
        value = os.environ.get(name)
        assert value, f"{name} must be set"
        resolved = Path(value).resolve()
        assert resolved == EXPERIMENT_ROOT or EXPERIMENT_ROOT in resolved.parents

