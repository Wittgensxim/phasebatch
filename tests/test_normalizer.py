import tempfile
import unittest
from pathlib import Path

from phasebatch.normalizer import canonical_hash, count_ir_features, normalize_ir_text


class NormalizerTests(unittest.TestCase):
    def test_strips_debug_metadata_and_module_headers(self) -> None:
        raw = """; ModuleID = 'x'
source_filename = "x.c"
define i32 @f(i32 %x) !dbg !1 {
entry:
  %a = add i32 %x, 0, !dbg !2 ; trailing comment
  ret i32 %a
}
!1 = !{}
!2 = !{}
"""
        normalized = normalize_ir_text(raw)

        self.assertNotIn("ModuleID", normalized)
        self.assertNotIn("source_filename", normalized)
        self.assertNotIn("!dbg", normalized)
        self.assertNotIn("!1 = ", normalized)
        self.assertIn("%a = add i32 %x, 0", normalized)

    def test_hash_is_stable_after_ignored_metadata_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.ll"
            b = Path(tmp) / "b.ll"
            a.write_text("define i32 @f() {\n  ret i32 0, !dbg !1\n}\n!1 = !{}\n", encoding="utf-8")
            b.write_text("; ModuleID = 'b'\ndefine i32 @f() {\n  ret i32 0, !dbg !2\n}\n!2 = !{}\n", encoding="utf-8")

            self.assertEqual(canonical_hash(a), canonical_hash(b))

    def test_counts_basic_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.ll"
            path.write_text(
                """define i32 @f(ptr %p, i1 %c) {
entry:
  %a = alloca i32
  %v = load i32, ptr %p
  store i32 %v, ptr %a
  br i1 %c, label %then, label %else
then:
  %x = call i32 @g()
  ret i32 %x
else:
  %s = select i1 %c, i32 1, i32 0
  ret i32 %s
}
""",
                encoding="utf-8",
            )

            features = count_ir_features(path)

        self.assertEqual(features["functions"], 1)
        self.assertEqual(features["basic_blocks"], 3)
        self.assertGreaterEqual(features["instructions"], 8)
        self.assertEqual(features["branches"], 1)
        self.assertEqual(features["loads"], 1)
        self.assertEqual(features["stores"], 1)
        self.assertEqual(features["calls"], 1)
        self.assertEqual(features["selects"], 1)
        self.assertEqual(features["allocas"], 1)
