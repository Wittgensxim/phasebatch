import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import phasebatch.tools as tools_module
from phasebatch.tools import collect_toolchain, find_tool, write_metadata


class ToolchainTests(unittest.TestCase):
    def test_find_graphviz_dot_uses_python_environment_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            dot = prefix / "Library" / "bin" / "graphviz" / "dot.exe"
            dot.parent.mkdir(parents=True)
            dot.write_text("", encoding="utf-8")

            with mock.patch("phasebatch.tools.shutil.which", return_value=None):
                resolved = tools_module.find_graphviz_dot(prefixes=[prefix])

        self.assertEqual(Path(resolved), dot)

    def test_find_tool_prefers_explicit_llvm_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = Path(tmp) / "opt.exe"
            tool.write_text("", encoding="utf-8")

            with mock.patch.dict("os.environ", {"PHASEBATCH_LLVM_BIN": tmp}):
                self.assertEqual(Path(find_tool("opt")), tool)

    def test_collect_toolchain_requires_clang_and_opt(self) -> None:
        with mock.patch("phasebatch.tools.find_tool") as fake_find:
            fake_find.side_effect = lambda name, required=True: {
                "clang": "C:/llvm/clang.exe",
                "opt": "C:/llvm/opt.exe",
                "llc": None,
                "llvm-size": None,
                "llvm-diff": None,
            }[name]
            with mock.patch("phasebatch.tools.run_version", return_value="LLVM version 23.0.0git"):
                metadata = collect_toolchain()

        self.assertEqual(metadata["tools"]["clang"]["path"], "C:/llvm/clang.exe")
        self.assertIn("LLVM version 23.0.0git", metadata["tools"]["opt"]["version"])
        self.assertIsNone(metadata["tools"]["llvm-diff"]["path"])

    def test_write_metadata_creates_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_metadata(Path(tmp), {"hello": "world"})
            data = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(data, {"hello": "world"})

    def test_write_metadata_records_active_opt_backend_snapshot(self) -> None:
        snapshot = {
            "requested_mode": "worker",
            "backend": "worker",
            "worker_path": "C:/phasebatch-worker.exe",
            "workers": 2,
            "stats": {"requests": 7, "module_loads": 1},
        }
        with tempfile.TemporaryDirectory() as tmp, \
            mock.patch("phasebatch.opt_backend.opt_backend_metadata", return_value=snapshot):
            write_metadata(Path(tmp), {"hello": "world"})
            data = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(data["opt_backend"], snapshot)
