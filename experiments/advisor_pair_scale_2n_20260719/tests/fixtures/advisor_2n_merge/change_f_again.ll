; A second, conflicting whole-body edit to @f.
source_filename = "advisor_2n_merge_base.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@shared = global i32 7, align 4

declare i32 @external(i32)

define i32 @f(i32 %x) {
entry:
  %sum = sub nsw i32 %x, 1
  ret i32 %sum
}

define i32 @g(i32 %x) {
entry:
  %product = mul nsw i32 %x, 2
  ret i32 %product
}
