; This parses successfully but the entry block has a predecessor, which the
; LLVM verifier rejects.
source_filename = "advisor_2n_merge_base.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @f(i32 %x) {
entry:
  br label %entry
}
