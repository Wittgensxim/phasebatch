# Versioned Pass Sets

Phasebatch pass-set files are intentionally versioned so experiments can expand
coverage without changing the meaning of older results.

## Format

The legacy format still works:

```yaml
passes:
  - mem2reg
  - sroa
  - instcombine
```

Each string entry is interpreted as:

```yaml
name: mem2reg
pipeline: mem2reg
category: unknown
stage: ""
enabled: true
```

Rich entries can record pass metadata and New Pass Manager pipeline candidates:

```yaml
passes:
  - name: licm
    pipeline_candidates:
      - licm
      - loop(licm)
      - function(loop(licm))
    category: loop
    stage: v3
    enabled: true
```

Disabled entries are ignored by the loaded active pass list.

## v1: Core Regression Set

`configs/core_passes_v1.yaml` is the stable regression/core set. It contains the
current scalar, CFG, memory, and cleanup passes used by the existing MVP runs.
Use this file when comparing against earlier results or when debugging the
batching and optimizer control flow.

## v2: Scalar/Memory/CFG Expansion

`configs/scalar_passes_v2.yaml` includes all v1 passes and adds:

- `sccp`
- `dse`
- `memcpyopt`
- `sink`
- `tailcallelim`

This set expands middle-end coverage while staying within scalar, memory, and
CFG transformations.

Recommended smoke workflow:

```bash
python -m phasebatch run-v2-extension-study \
  --inputs benchmarks/tiny/*.c \
  --out outputs/v2_extension_tiny \
  --v1-passes configs/core_passes.yaml \
  --v2-passes configs/scalar_passes_v2.yaml \
  --objective ir-inst-count \
  --max-rounds 4 \
  --beam-width 4 \
  --max-states 500 \
  --max-batches-per-state 20 \
  --batch-frontier-policy score \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 600 \
  --overwrite \
  --continue-on-error
```

Interpret v2 as a scalability and coverage extension. A larger pass set can
increase active pass count, pair tests, conflict components, and validation
cost. It is not expected to improve every objective result, and it does not
replace the Core-v1 case-study baseline.

## v3: Loop Pass Candidates

`configs/middleend_passes_v3.yaml` includes all v2 passes and adds loop-related
passes:

- `loop-simplify`
- `lcssa`
- `loop-rotate`
- `licm`
- `indvars`
- `loop-deletion`

Some loop passes require nested New Pass Manager pipeline syntax that can vary
by LLVM build. For that reason, v3 entries use `pipeline_candidates`. The first
candidate is the default pipeline string used by existing loaders; an
`audit-passes` or pass-resolution step should validate which candidate works for
the local LLVM before larger experiments.

Recommended loop-heavy smoke workflow:

```bash
python -m phasebatch run-v3-loop-smoke \
  --inputs benchmarks/tiny/loop.c E:/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/n-body.c \
  --out outputs/v3_loop_smoke \
  --passes configs/middleend_passes_v3.yaml \
  --optimizer-mode budgeted \
  --objective ir-inst-count \
  --max-rounds 3 \
  --beam-width 4 \
  --max-states 800 \
  --max-batches-per-state 12 \
  --batch-frontier-policy score \
  --validate-batches \
  --jobs 8 \
  --timeout 10 \
  --max-pairs 1000 \
  --overwrite \
  --continue-on-error
```

Use v3 for loop-heavy case studies rather than the default main experiment
unless audit results show that the local LLVM accepts the loop pipelines and
coverage/cost remain manageable.

## Current Boundaries

These pass sets do not include backend/codegen passes, IPO/module passes, or
inlining passes yet. The project main line remains state-local pass relation
discovery and certified batch-state optimization, not brute-force search over
all LLVM passes.

Objective values from `ir-inst-count` are evaluation signals only. They are not
used as commutation proof, independence proof, or a reason to execute unsafe
batches.
