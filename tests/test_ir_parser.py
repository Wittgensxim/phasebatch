import tempfile
import unittest
from pathlib import Path

from phasebatch.ir_parser import changed_regions, parse_ir_snapshot


class IRParserTests(unittest.TestCase):
    def test_detects_changed_function_and_block(self) -> None:
        before = """define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 0
  ret i32 %a
}
"""
        after = """define i32 @f(i32 %x) {
entry:
  ret i32 %x
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before_path = root / "before.ll"
            after_path = root / "after.ll"
            before_path.write_text(before, encoding="utf-8")
            after_path.write_text(after, encoding="utf-8")

            diff = changed_regions(parse_ir_snapshot(before_path), parse_ir_snapshot(after_path))

        self.assertEqual(diff["changed_functions"], ["f"])
        self.assertEqual(diff["changed_blocks"], ["f::entry"])
        self.assertEqual(diff["funcs_changed"], 1)
        self.assertEqual(diff["blocks_changed"], 1)

    def test_uses_entry_for_instructions_before_first_label(self) -> None:
        text = """define i32 @f() {
  ret i32 0
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.ll"
            path.write_text(text, encoding="utf-8")
            snapshot = parse_ir_snapshot(path)

        self.assertIn("f::entry", snapshot.blocks)
