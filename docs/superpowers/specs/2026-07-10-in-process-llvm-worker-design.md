# In-Process LLVM Worker Design

## Goal

Replace thousands of short-lived `opt.exe` subprocesses with a long-lived C++
worker pool while preserving Phasebatch's current pass pipelines, verifier
behavior, equality ladder, deterministic artifacts, fallback behavior, and IR
cleanup policy.

This work does not change lazy pair testing, beam scoring, batch
construction, or correctness classification.

## Measured Motivation

On the local LLVM 23 build and the Salsa20 inlinable root, 100 repetitions gave:

| operation | average |
|---|---:|
| start `opt --version` | 31.559 ms |
| no-op pipeline with parse, verify, and print | 35.759 ms |
| `instcombine` with parse, verify, and print | 38.561 ms |

The E5 scalar stage used 14,515 `opt` invocations and 507.651 seconds. Process
startup is therefore the first performance target.

## Architecture

`phasebatch-worker.exe` is a standalone process linked against the same local
LLVM build as `opt.exe`. It reads one JSON object per line from stdin and writes
one JSON response per line to stdout. LLVM diagnostics go to stderr so protocol
output remains parseable.

Python owns a pool of long-lived workers. Each worker has one `LLVMContext`, a
content-addressed module store, and a path-to-handle cache. A worker processes
one request at a time. Parallelism comes from multiple worker processes, which
also preserves crash and timeout isolation.

The existing `run_opt` function remains the compatibility boundary. External
and worker backends both return `RunResult`. Worker-specific metadata is added
without changing existing success, stderr, timeout, or output-path semantics.

## Protocol

Every request contains `request_id` and `op`. Every response echoes
`request_id`, has `status=ok|error`, and includes elapsed timings.

Supported operations:

```text
ping
load(path) -> module_handle, canonical_hash, features
apply(parent_handle, pipeline, verify_each, materialize_path?)
materialize(module_handle, path)
release(module_handle)
clear
shutdown
```

`apply` clones the parent module, constructs fresh New PM analysis managers,
registers analyses and proxies through `PassBuilder`, installs
`StandardInstrumentations` with verify-each enabled, parses the textual module
pipeline, runs it, verifies the result, computes metadata, interns the child,
and materializes only when requested.

Handles are process-local. Python records `(worker_id, handle)` and never sends
a handle to a different worker. A worker that does not own a parent loads the
materialized path once and then caches it.

## Backend Selection

The CLI and environment support:

```text
--opt-backend external|worker|auto
--opt-worker <path>
--opt-workers <count>
PHASEBATCH_OPT_BACKEND
PHASEBATCH_OPT_WORKER
PHASEBATCH_OPT_WORKERS
```

`external` preserves current behavior. `worker` requires a usable worker.
`auto` prefers the worker and falls back to external when startup fails.

## File-Compatible Phase

The first production phase uses handles internally but materializes every
requested output path. No profiler, pair, validation, equality, replay, or
cleanup caller changes its file expectations. This phase proves semantic
equivalence and captures the process-startup gain before deeper changes.

## Lazy Materialization Phase

`RunResult` gains optional worker metadata:

```text
worker_id
module_handle
canonical_hash
feature_counts
materialized
```

Profiling and newly reached optimizer states remain materialized because the
current IR parser needs their text. Pair AB/BA and validation DAG transitions
may request `materialize=False`.

When two worker results have equal safe canonical hashes, the existing equality
ladder can certify the canonical-hash tier without files. When hashes differ,
Python materializes only those two handles and continues with the unchanged
`llvm-diff + module_fingerprint` ladder. Dump and keep flags always force
materialization.

## Failure Handling

Python enforces request timeouts. A timed-out worker is terminated, killed if
necessary, removed from the pool, and restarted. Handles owned by that process
are invalidated. The failed request may be retried once through external
`opt.exe`; the retry and fallback are recorded.

A worker crash affects only one worker process. Malformed responses, request-ID
mismatches, protocol EOF, and output materialization failures are treated as
backend failures, never successful optimization results.

## Determinism and Correctness

The worker uses the same resolved pipeline strings and canonical order as the
external backend. Fresh analysis managers are constructed for each `apply` so
analyses cannot leak across cloned modules. Module handles are cache identities,
not correctness evidence.

Differential verification must compare external and worker results for valid,
dormant, invalid, nested loop, pair AB/BA, validation DAG, child-state, and
replay pipelines. Required invariants are return status, safe canonical hash,
module fingerprint, active/dormant classification, pair relation, validation
classification, selected pipeline, and replay result.

## Metrics

Worker runs record:

```text
worker_process_starts
worker_restarts
worker_requests
worker_failures
worker_fallbacks
module_loads
module_load_cache_hits
module_clones
handle_cache_hits
materializations
avoided_materializations
parse_ms
clone_ms
pipeline_parse_ms
pass_ms
verify_ms
print_ms
round_trip_ms
```

## Acceptance

The file-compatible backend must produce zero semantic differential failures
and at least 3x throughput on the 100-call Salsa20 microbenchmark before it is
used for full experiments. The handle phase must preserve the same invariants,
reduce materialized pair/DAG IR files, and keep the default cleanup result at
zero `.ll` files and zero empty directories.
