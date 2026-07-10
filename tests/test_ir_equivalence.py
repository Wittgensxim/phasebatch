import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phasebatch.ir_equivalence import (
    compare_ir_equivalence,
    module_safety_fingerprint,
    safe_canonical_text,
)
from phasebatch.opt_worker import WorkerError
from phasebatch.tools import find_tool


class IrEquivalenceTests(unittest.TestCase):
    def test_strict_worker_comparator_error_is_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            left.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            right.write_text("define i32 @f() {\n  ret i32 1\n}\n", encoding="utf-8")
            backend = mock.Mock()
            backend.fallback_external = False
            backend.compare_paths.side_effect = WorkerError("worker comparator failed")

            with mock.patch("phasebatch.opt_backend.active_opt_backend", return_value=backend):
                with self.assertRaisesRegex(WorkerError, "worker comparator failed"):
                    compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=3)

    def test_semicolon_inside_inline_asm_string_is_not_treated_as_comment(self) -> None:
        left_text = (
            "define void @f() {\n"
            "  call void asm sideeffect \"nop ; int3\", \"\"()\n"
            "  ret void\n"
            "}\n"
        )
        right_text = left_text.replace("nop ; int3", "nop ; nop ")
        self.assertNotEqual(safe_canonical_text(left_text), safe_canonical_text(right_text))

        with tempfile.TemporaryDirectory() as tmp:
            left, right = _write_pair(Path(tmp), left_text, right_text)
            completed = subprocess.CompletedProcess(["llvm-diff"], 1, "different", "")
            with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed):
                result = compare_ir_equivalence(
                    left,
                    right,
                    tools={"llvm-diff": "llvm-diff"},
                    timeout=1,
                )

        self.assertFalse(result.equal)
        self.assertFalse(result.can_hard_fold)
        self.assertFalse(result.text_hash_equal)

    def test_debug_like_text_inside_string_is_preserved(self) -> None:
        left = '  call void asm sideeffect "nop, !dbg !1 ; keep", ""() ; comment\n'
        right = '  call void asm sideeffect "nop, !dbg !2 ; keep", ""() ; comment\n'

        self.assertIn('"nop, !dbg !1 ; keep"', safe_canonical_text(left))
        self.assertNotEqual(safe_canonical_text(left), safe_canonical_text(right))

    def test_active_worker_runs_structural_diff_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            left.write_text(
                "define i32 @f(i32 %x) {\n  %left = add i32 %x, 1\n  ret i32 %left\n}\n",
                encoding="utf-8",
            )
            right.write_text(
                "define i32 @f(i32 %x) {\n  %right = add i32 %x, 1\n  ret i32 %right\n}\n",
                encoding="utf-8",
            )
            backend = mock.Mock()
            backend.compare_paths.return_value = True
            with mock.patch("phasebatch.opt_backend.active_opt_backend", return_value=backend), \
                mock.patch("phasebatch.ir_equivalence.subprocess.run") as subprocess_run:
                result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=3)

        self.assertTrue(result.equal)
        self.assertEqual(result.tier, "structural_diff")
        backend.compare_paths.assert_called_once_with(left, right, timeout=3)
        subprocess_run.assert_not_called()

    def test_hash_equal_certifies_canonical_hash_without_llvm_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            ir = "define i32 @f() {\n  ret i32 0\n}\n"
            left.write_text(ir, encoding="utf-8")
            right.write_text(ir, encoding="utf-8")

            with mock.patch("phasebatch.ir_equivalence.subprocess.run") as fake_run:
                result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=1)

        self.assertTrue(result.equal)
        self.assertTrue(result.can_hard_fold)
        self.assertEqual(result.tier, "canonical_hash")
        self.assertEqual(result.reason, "hash_equal")
        self.assertTrue(result.text_hash_equal)
        fake_run.assert_not_called()

    def test_hash_mismatch_uses_llvm_diff_and_module_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            left.write_text("define i32 @f(i32 %x) {\nentry:\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n", encoding="utf-8")
            right.write_text("define i32 @f(i32 %x) {\nentry:\n  %b = add i32 %x, 0\n  ret i32 %b\n}\n", encoding="utf-8")

            completed = subprocess.CompletedProcess(["llvm-diff"], 0, "", "")
            with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed):
                result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=1)

        self.assertTrue(result.equal)
        self.assertTrue(result.can_hard_fold)
        self.assertEqual(result.tier, "structural_diff")
        self.assertEqual(result.reason, "llvm_diff_equal_and_module_fingerprint_equal")
        self.assertFalse(result.text_hash_equal)
        self.assertTrue(result.llvm_diff_equal)
        self.assertTrue(result.module_fingerprint_equal)

    def test_module_fingerprint_blocks_llvm_diff_false_commute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            body = "define i32 @f() {\n  ret i32 0\n}\n"
            left.write_text("@g = global i32 0\n" + body, encoding="utf-8")
            right.write_text("@g = global i32 1\n" + body, encoding="utf-8")

            completed = subprocess.CompletedProcess(["llvm-diff"], 0, "", "")
            with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed):
                result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=1)

        self.assertFalse(result.equal)
        self.assertFalse(result.can_hard_fold)
        self.assertEqual(result.tier, "different")
        self.assertEqual(result.reason, "module_fingerprint_difference")
        self.assertTrue(result.llvm_diff_equal)
        self.assertFalse(result.module_fingerprint_equal)

    def test_llvm_diff_difference_is_conservative_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            left.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            right.write_text("define i32 @f() {\n  ret i32 1\n}\n", encoding="utf-8")

            completed = subprocess.CompletedProcess(["llvm-diff"], 1, "different", "")
            with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed):
                result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=1)

        self.assertFalse(result.equal)
        self.assertFalse(result.can_hard_fold)
        self.assertEqual(result.tier, "different")
        self.assertEqual(result.reason, "llvm_diff_difference")
        self.assertFalse(result.llvm_diff_equal)

    def test_missing_llvm_diff_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            left.write_text("define i32 @f() {\n  ret i32 0\n}\n", encoding="utf-8")
            right.write_text("define i32 @f() {\n  ret i32 1\n}\n", encoding="utf-8")

            with mock.patch("phasebatch.ir_equivalence.find_tool", return_value=None):
                result = compare_ir_equivalence(left, right, tools={}, timeout=1)

        self.assertFalse(result.equal)
        self.assertFalse(result.can_hard_fold)
        self.assertEqual(result.tier, "failed")
        self.assertEqual(result.reason, "tool_failed")

    def test_env_llvm_diff_path_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.ll"
            right = root / "right.ll"
            llvm_diff = root / "custom-llvm-diff"
            left.write_text("define i32 @f(i32 %x) {\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n", encoding="utf-8")
            right.write_text("define i32 @f(i32 %x) {\n  %b = add i32 %x, 0\n  ret i32 %b\n}\n", encoding="utf-8")
            llvm_diff.write_text("", encoding="utf-8")

            completed = subprocess.CompletedProcess([str(llvm_diff)], 0, "", "")
            with mock.patch.dict("os.environ", {"PHASEBATCH_LLVM_DIFF": str(llvm_diff)}), \
                mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed) as fake_run:
                result = compare_ir_equivalence(left, right, tools={}, timeout=1)

        self.assertTrue(result.can_hard_fold)
        self.assertEqual(fake_run.call_args.args[0][0], str(llvm_diff))

    def test_llvm_diff_defaults_to_opt_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            opt = bin_dir / "opt.exe"
            llvm_diff = bin_dir / "llvm-diff.exe"
            opt.write_text("", encoding="utf-8")
            llvm_diff.write_text("", encoding="utf-8")
            left = root / "left.ll"
            right = root / "right.ll"
            left.write_text("define i32 @f(i32 %x) {\n  %a = add i32 %x, 0\n  ret i32 %a\n}\n", encoding="utf-8")
            right.write_text("define i32 @f(i32 %x) {\n  %b = add i32 %x, 0\n  ret i32 %b\n}\n", encoding="utf-8")

            completed = subprocess.CompletedProcess([str(llvm_diff)], 0, "", "")
            with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed) as fake_run:
                result = compare_ir_equivalence(left, right, tools={"opt": str(opt)}, timeout=1)

        self.assertTrue(result.can_hard_fold)
        self.assertEqual(fake_run.call_args.args[0][0], str(llvm_diff))

    def test_module_safety_fingerprint_preserves_non_debug_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            left.write_text("define i32 @f() {\n  ret i32 0, !dbg !1\n}\n!1 = distinct !DISubprogram()\n!tbaa = !{!2}\n", encoding="utf-8")
            right.write_text("define i32 @f() {\n  ret i32 0, !dbg !9\n}\n!9 = distinct !DISubprogram()\n!tbaa = !{!3}\n", encoding="utf-8")

            self.assertNotEqual(module_safety_fingerprint(left), module_safety_fingerprint(right))

    def test_basic_block_label_noise_is_structurally_foldable(self) -> None:
        llvm_diff = find_tool("llvm-diff", required=False)
        if not llvm_diff:
            self.skipTest("llvm-diff is not available")
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.ll"
            right = Path(tmp) / "right.ll"
            left.write_text(
                "define i32 @f() {\n"
                "entry:\n"
                "  br label %then\n"
                "then:\n"
                "  ret i32 0\n"
                "}\n",
                encoding="utf-8",
            )
            right.write_text(
                "define i32 @f() {\n"
                "bb0:\n"
                "  br label %bb1\n"
                "bb1:\n"
                "  ret i32 0\n"
                "}\n",
                encoding="utf-8",
            )

            result = compare_ir_equivalence(left, right, tools={"llvm-diff": llvm_diff}, timeout=5)

        self.assertTrue(result.equal)
        self.assertTrue(result.can_hard_fold)
        self.assertEqual(result.tier, "structural_diff")
        self.assertFalse(result.text_hash_equal)
        self.assertTrue(result.llvm_diff_equal)
        self.assertTrue(result.module_fingerprint_equal)

    def test_optimization_metadata_differences_do_not_hard_fold_when_llvm_diff_is_equal(self) -> None:
        cases = {
            "tbaa": (
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, !tbaa !0\n  ret i32 %v\n}\n!0 = !{!1}\n!1 = !{!\"int\", !2, i64 0}\n!2 = !{!\"root\"}\n",
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, !tbaa !0\n  ret i32 %v\n}\n!0 = !{!1}\n!1 = !{!\"float\", !2, i64 0}\n!2 = !{!\"root\"}\n",
            ),
            "alias_scope": (
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, !alias.scope !0\n  ret i32 %v\n}\n!0 = !{!1}\n!1 = distinct !{!1, !2, !\"scope0\"}\n!2 = distinct !{!2, !\"domain\"}\n",
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, !alias.scope !0\n  ret i32 %v\n}\n!0 = !{!1}\n!1 = distinct !{!1, !2, !\"scope1\"}\n!2 = distinct !{!2, !\"domain\"}\n",
            ),
            "noalias": (
                "define void @f(ptr %p, ptr %q) {\n  store i32 0, ptr %p, !noalias !0\n  store i32 1, ptr %q\n  ret void\n}\n!0 = !{!1}\n!1 = distinct !{!1, !2, !\"scope0\"}\n!2 = distinct !{!2, !\"domain\"}\n",
                "define void @f(ptr %p, ptr %q) {\n  store i32 0, ptr %p, !noalias !0\n  store i32 1, ptr %q\n  ret void\n}\n!0 = !{!1}\n!1 = distinct !{!1, !2, !\"scope1\"}\n!2 = distinct !{!2, !\"domain\"}\n",
            ),
            "range": (
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, !range !0\n  ret i32 %v\n}\n!0 = !{i32 0, i32 10}\n",
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, !range !0\n  ret i32 %v\n}\n!0 = !{i32 0, i32 20}\n",
            ),
            "prof_branch_weights": (
                "define void @f(i1 %c) {\nentry:\n  br i1 %c, label %a, label %b, !prof !0\na:\n  ret void\nb:\n  ret void\n}\n!0 = !{!\"branch_weights\", i32 90, i32 10}\n",
                "define void @f(i1 %c) {\nentry:\n  br i1 %c, label %a, label %b, !prof !0\na:\n  ret void\nb:\n  ret void\n}\n!0 = !{!\"branch_weights\", i32 10, i32 90}\n",
            ),
        }
        completed = subprocess.CompletedProcess(["llvm-diff"], 0, "", "")

        for name, (left_text, right_text) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                left, right = _write_pair(Path(tmp), left_text, right_text)

                with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed):
                    result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=1)

                self.assertFalse(result.equal)
                self.assertFalse(result.can_hard_fold)
                self.assertEqual(result.tier, "different")
                self.assertEqual(result.reason, "module_fingerprint_difference")
                self.assertTrue(result.llvm_diff_equal)
                self.assertFalse(result.module_fingerprint_equal)

    def test_attribute_differences_do_not_hard_fold(self) -> None:
        cases = {
            "load_align": (
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, align 4\n  ret i32 %v\n}\n",
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p, align 8\n  ret i32 %v\n}\n",
            ),
            "function_readonly": (
                "define i32 @f(ptr %p) readonly {\n  %v = load i32, ptr %p\n  ret i32 %v\n}\n",
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p\n  ret i32 %v\n}\n",
            ),
            "function_nounwind": (
                "define void @f() nounwind {\n  ret void\n}\n",
                "define void @f() {\n  ret void\n}\n",
            ),
            "function_alwaysinline": (
                "define void @f() alwaysinline {\n  ret void\n}\n",
                "define void @f() {\n  ret void\n}\n",
            ),
            "parameter_nonnull": (
                "define i32 @f(ptr nonnull %p) {\n  %v = load i32, ptr %p\n  ret i32 %v\n}\n",
                "define i32 @f(ptr %p) {\n  %v = load i32, ptr %p\n  ret i32 %v\n}\n",
            ),
            "parameter_noundef": (
                "define i32 @f(i32 noundef %x) {\n  ret i32 %x\n}\n",
                "define i32 @f(i32 %x) {\n  ret i32 %x\n}\n",
            ),
        }

        for name, (left_text, right_text) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                left, right = _write_pair(Path(tmp), left_text, right_text)

                completed = subprocess.CompletedProcess(["llvm-diff"], 0, "", "")
                with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed):
                    result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=1)

                self.assertFalse(result.can_hard_fold)
                self.assertEqual(result.tier, "different")

    def test_module_level_differences_do_not_hard_fold_when_llvm_diff_is_equal(self) -> None:
        body = "define i32 @f() {\n  ret i32 0\n}\n"
        cases = {
            "global_initializer": ("@g = global i32 0\n" + body, "@g = global i32 1\n" + body),
            "linkage": ("define internal i32 @f() {\n  ret i32 0\n}\n", "define i32 @f() {\n  ret i32 0\n}\n"),
            "visibility": ("define hidden i32 @f() {\n  ret i32 0\n}\n", "define protected i32 @f() {\n  ret i32 0\n}\n"),
            "comdat": ("$c = comdat any\n@x = global i32 0, comdat($c)\n" + body, "$c = comdat noduplicates\n@x = global i32 0, comdat($c)\n" + body),
            "alias": ("@g = global i32 0\n@a = alias i32, ptr @g\n" + body, "@g = global i32 0\n" + body),
            "target_triple": ("target triple = \"x86_64-pc-linux-gnu\"\n" + body, "target triple = \"aarch64-unknown-linux-gnu\"\n" + body),
            "datalayout": ("target datalayout = \"e-m:e-p:64:64\"\n" + body, "target datalayout = \"e-m:e-p:32:32\"\n" + body),
            "declaration_attributes": ("declare void @g() nounwind\n" + body, "declare void @g()\n" + body),
        }
        completed = subprocess.CompletedProcess(["llvm-diff"], 0, "", "")

        for name, (left_text, right_text) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                left, right = _write_pair(Path(tmp), left_text, right_text)

                with mock.patch("phasebatch.ir_equivalence.subprocess.run", return_value=completed):
                    result = compare_ir_equivalence(left, right, tools={"llvm-diff": "llvm-diff"}, timeout=1)

                self.assertFalse(result.equal)
                self.assertFalse(result.can_hard_fold)
                self.assertEqual(result.tier, "different")
                self.assertEqual(result.reason, "module_fingerprint_difference")
                self.assertTrue(result.llvm_diff_equal)
                self.assertFalse(result.module_fingerprint_equal)


def _write_pair(root: Path, left_text: str, right_text: str) -> tuple[Path, Path]:
    left = root / "left.ll"
    right = root / "right.ll"
    left.write_text(left_text, encoding="utf-8")
    right.write_text(right_text, encoding="utf-8")
    return left, right
