from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import build_screen_snapshot, generate_aggregate_outputs
from .dataset import load_frozen_dataset
from .reporting import generate_reports_and_figures
from .timing import load_runtime_records, run_timing_experiment


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Isolated observed-change instruction granularity experiment"
    )
    parser.add_argument(
        "command",
        choices=("preflight", "screen", "time", "aggregate", "report", "all"),
    )
    parser.add_argument("--root", type=Path, default=experiment_root())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    dataset = load_frozen_dataset(root)
    if args.command == "preflight":
        print("preflight_complete pairs=1411 programs=49 actions=14 transitions=686")
        return 0

    snapshot_path = root / "raw" / "instruction_screen_snapshot.json"
    raw_runtime = root / "aggregate" / "extraction_runtime_raw.csv"
    if args.command in {"screen", "all"}:
        snapshot = build_screen_snapshot(
            dataset,
            snapshot_path,
            progress=lambda repetition, run: print(
                f"screen source={repetition}/3 total_ms={run.total_extraction_ms:.3f}",
                flush=True,
            ),
        )
        print(f"screen_complete rows={snapshot['pair_count']}", flush=True)
        if args.command == "screen":
            return 0
    if args.command in {"time", "all"}:
        records = run_timing_experiment(
            dataset,
            raw_runtime,
            progress=lambda record, done, total: print(
                f"timing {done}/{total} {record.phase} {record.level.value} "
                f"total_ms={record.total_extraction_ms:.3f}",
                flush=True,
            ),
        )
        print(f"timing_complete rows={len(records)}", flush=True)
        if args.command == "time":
            return 0

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    records = load_runtime_records(raw_runtime)
    if args.command in {"aggregate", "all"}:
        metrics = generate_aggregate_outputs(dataset, snapshot, records, root / "aggregate")
        print("aggregate_complete", flush=True)
        if args.command == "aggregate":
            return 0
    else:
        metrics = json.loads((root / "raw" / "aggregate_metrics.json").read_text(encoding="utf-8"))
    if args.command in {"report", "all"}:
        artifacts = generate_reports_and_figures(root, metrics)
        print(f"report_complete artifacts={len(artifacts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

