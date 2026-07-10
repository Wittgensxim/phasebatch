from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .schema import EQUALITY_TIER_SUMMARY_FIELDS


EQUALITY_TIERS = ["canonical_hash", "structural_diff", "different", "failed"]
INTERPRETATION_TEXT = (
    "Structural fallback is used only to avoid false conflicts from local naming or harmless structural noise. "
    "Module safety fingerprint guards against false commutativity from optimization-relevant attributes, metadata, "
    "globals, target information, or datalayout differences."
)


def equality_tier_summary_from_rows(rows: list[dict]) -> list[dict]:
    counts: Counter[str] = Counter()
    hard_fold_counts: Counter[str] = Counter()
    for row in rows:
        tier = str(row.get("equality_tier", "")).strip()
        if not tier:
            continue
        counts[tier] += 1
        if _is_true(row.get("can_hard_fold")):
            hard_fold_counts[tier] += 1

    tiers = [*EQUALITY_TIERS, *sorted(tier for tier in counts if tier not in EQUALITY_TIERS)]
    return [
        {
            "tier": tier,
            "count": counts.get(tier, 0),
            "hard_fold": hard_fold_counts.get(tier, 0),
        }
        for tier in tiers
    ]


def equality_tier_summary_for_run(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in _pair_relation_paths(Path(run_dir)):
        rows.extend(_read_csv(path))
    return equality_tier_summary_from_rows(rows)


def equality_tier_summary_for_runs(run_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for run_dir in run_dirs:
        for path in _pair_relation_paths(Path(run_dir)):
            rows.extend(_read_csv(path))
    return equality_tier_summary_from_rows(rows)


def equality_tier_summary_for_run_rows(rows: list[dict], *path_fields: str, base_dir: Path | None = None) -> list[dict]:
    fields = path_fields or ("output_dir", "run_dir", "optimize_dir")
    run_dirs: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("status") and row.get("status") != "success":
            continue
        for field in fields:
            value = row.get(field, "")
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            if not path.exists():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            run_dirs.append(path)
            break
    return equality_tier_summary_for_runs(run_dirs)


def equality_tier_markdown(rows: list[dict], *, heading: str = "## Equality Tier Summary") -> list[str]:
    table_rows = [
        [str(row.get("tier", "")), str(row.get("count", 0)), str(row.get("hard_fold", 0))]
        for row in rows
    ]
    return [
        heading,
        "",
        *_markdown_table(["tier", "count", "hard_fold"], table_rows),
    ]


def write_equality_tier_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    rows = _source_summary_rows(run_dir)
    csv_path = run_dir / "equality_tier_summary.csv"
    md_path = run_dir / "equality_tier_summary.md"
    _write_csv(csv_path, EQUALITY_TIER_SUMMARY_FIELDS, rows)
    _write_run_markdown(md_path, rows)
    return {
        "equality_tier_summary_csv": str(csv_path),
        "equality_tier_summary_md": str(md_path),
    }


def _source_summary_rows(run_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for path in _pair_relation_paths(run_dir):
        for row in _read_csv(path):
            tier = str(row.get("equality_tier", "")).strip()
            if tier:
                entries.append(
                    {
                        "source": "pair_relation",
                        "equality_tier": tier,
                        "equality_reason": str(row.get("equality_reason", "")).strip(),
                        "hard_fold": _is_true(row.get("can_hard_fold")),
                    }
                )

    for path in _batch_validation_paths(run_dir):
        for row in _read_csv(path):
            tier = str(row.get("validation_equality_tier", "")).strip()
            if tier:
                entries.append(
                    {
                        "source": "batch_validation",
                        "equality_tier": tier,
                        "equality_reason": str(row.get("validation_equality_reason", "")).strip(),
                        "hard_fold": (
                            row.get("validation_status") == "all_permutations_same"
                            and tier in {"canonical_hash", "structural_diff"}
                        ),
                    }
                )

    for row in _read_csv(run_dir / "pipeline_replay.csv"):
        tier = str(row.get("equality_tier", "")).strip()
        if tier:
            entries.append(
                {
                    "source": "pipeline_replay",
                    "equality_tier": tier,
                    "equality_reason": str(row.get("equality_reason", "")).strip(),
                    "hard_fold": _is_true(row.get("can_hard_fold")),
                }
            )

    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    hard_folds: dict[tuple[str, str, str], int] = defaultdict(int)
    for entry in entries:
        key = (entry["source"], entry["equality_tier"], entry["equality_reason"])
        counts[key] += 1
        if entry["hard_fold"]:
            hard_folds[key] += 1

    return [
        {
            "source": source,
            "equality_tier": tier,
            "equality_reason": reason,
            "count": str(counts[(source, tier, reason)]),
            "hard_fold_count": str(hard_folds[(source, tier, reason)]),
        }
        for source, tier, reason in sorted(counts, key=_source_sort_key)
    ]


def _batch_validation_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for state_dir in _state_dirs(run_dir):
        path = state_dir / "batch_validation.csv"
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    direct = run_dir / "batch_validation.csv"
    if direct.exists() and str(direct.resolve()) not in seen:
        paths.append(direct)
    return paths


def _write_run_markdown(path: Path, rows: list[dict]) -> None:
    sections = [
        ("## Pair Relations", "pair_relation"),
        ("## Batch Validation", "batch_validation"),
        ("## Replay", "pipeline_replay"),
    ]
    lines = ["# Equality Tier Summary", ""]
    for heading, source in sections:
        source_rows = [row for row in rows if row.get("source") == source]
        lines.extend([heading, ""])
        lines.extend(
            _markdown_table(
                ["tier", "reason", "count", "hard fold count"],
                [
                    [
                        row.get("equality_tier", ""),
                        row.get("equality_reason", ""),
                        row.get("count", "0"),
                        row.get("hard_fold_count", "0"),
                    ]
                    for row in source_rows
                ],
            )
        )
        lines.append("")
    lines.extend(["## Interpretation", "", INTERPRETATION_TEXT])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _pair_relation_paths(run_dir: Path) -> list[Path]:
    from_states = _pair_relation_paths_from_states_csv(run_dir)
    if from_states:
        return from_states

    states_dir = run_dir / "states"
    if states_dir.exists():
        paths = sorted(path for path in states_dir.glob("*/pair_relation.csv") if path.exists())
        if paths:
            return paths

    direct = run_dir / "pair_relation.csv"
    return [direct] if direct.exists() else []


def _pair_relation_paths_from_states_csv(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for row in _read_csv(run_dir / "states.csv"):
        if _is_true(row.get("is_duplicate")):
            continue
        state_dir_value = row.get("state_dir", "")
        state_id = row.get("state_id", "")
        if state_dir_value:
            state_dir = Path(state_dir_value)
            if not state_dir.is_absolute():
                state_dir = run_dir / state_dir
        elif state_id:
            state_dir = run_dir / "states" / state_id
        else:
            continue

        path = state_dir / "pair_relation.csv"
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _state_dirs(run_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for row in _read_csv(run_dir / "states.csv"):
        if _is_true(row.get("is_duplicate")):
            continue
        state_dir_value = row.get("state_dir", "")
        state_id = row.get("state_id", "")
        if state_dir_value:
            state_dir = Path(state_dir_value)
            if not state_dir.is_absolute():
                state_dir = run_dir / state_dir
        elif state_id:
            state_dir = run_dir / "states" / state_id
        else:
            continue
        key = str(state_dir.resolve()) if state_dir.exists() else str(state_dir)
        if key not in seen:
            seen.add(key)
            dirs.append(state_dir)
    states_dir = run_dir / "states"
    if states_dir.exists():
        for state_dir in sorted(path for path in states_dir.iterdir() if path.is_dir()):
            key = str(state_dir.resolve())
            if key not in seen:
                seen.add(key)
                dirs.append(state_dir)
    return dirs


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(_cell(value) for value in row)} |" for row in rows)
    return lines


def _cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _source_sort_key(key: tuple[str, str, str]) -> tuple[int, int, str]:
    source, tier, reason = key
    source_order = {"pair_relation": 0, "batch_validation": 1, "pipeline_replay": 2}
    tier_order = {tier_name: index for index, tier_name in enumerate(EQUALITY_TIERS)}
    return (source_order.get(source, 99), tier_order.get(tier, 99), reason)
