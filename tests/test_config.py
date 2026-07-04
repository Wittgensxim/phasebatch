import tempfile
import unittest
from pathlib import Path

from phasebatch.config import load_passes


class ConfigLoaderTests(unittest.TestCase):
    def test_loads_passes_from_core_yaml_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passes.yaml"
            path.write_text("passes:\n  - mem2reg\n  - instcombine\n", encoding="utf-8")

            self.assertEqual(load_passes(path), ["mem2reg", "instcombine"])


if __name__ == "__main__":
    unittest.main()
