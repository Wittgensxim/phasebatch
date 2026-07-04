from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_passes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phasebatch",
        description="LLVM phase-ordering data MVP command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one C or LLVM IR input.")
    _add_common_args(analyze)
    analyze.add_argument("--input", required=True, help="Input .c or .ll file.")
    analyze.set_defaults(func=_run_analyze)

    batch = subparsers.add_parser("batch", help="Analyze multiple C or LLVM IR inputs.")
    _add_common_args(batch)
    batch.add_argument("--inputs", required=True, nargs="+", help="Input .c or .ll files.")
    batch.set_defaults(func=_run_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--passes", required=True, help="Path to pass config YAML.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count.")
    parser.add_argument("--timeout", type=int, default=10, help="Per-command timeout in seconds.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Maximum active pass pairs to test.")


def _run_analyze(args: argparse.Namespace) -> int:
    passes = load_passes(args.passes)
    payload = _common_payload(args, passes)
    payload["input"] = args.input
    _print_stub(payload)
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    passes = load_passes(args.passes)
    payload = _common_payload(args, passes)
    payload["inputs"] = args.inputs
    _print_stub(payload)
    return 0


def _common_payload(args: argparse.Namespace, passes: list[str]) -> dict[str, Any]:
    return {
        "command": args.command,
        "out": str(Path(args.out)),
        "passes": str(Path(args.passes)),
        "jobs": args.jobs,
        "timeout": args.timeout,
        "max_pairs": args.max_pairs,
        "pass_count": len(passes),
        "loaded_passes": passes,
        "status": "stub",
    }


def _print_stub(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
