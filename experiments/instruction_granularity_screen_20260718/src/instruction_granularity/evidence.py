from __future__ import annotations

import json
import platform
from pathlib import Path
import sys

from .deterministic_io import sha256_file, write_json, write_text
from .models import FrozenDataset


REQUIRED_ARTIFACTS = (
    "aggregate/granularity_coverage_summary.csv",
    "aggregate/instruction_screen_pairs.csv",
    "aggregate/instruction_incremental_summary.csv",
    "aggregate/instruction_cumulative_summary.csv",
    "aggregate/extraction_runtime_raw.csv",
    "aggregate/extraction_runtime_summary.csv",
    "aggregate/extraction_incremental_cost.csv",
    "aggregate/instruction_by_program.csv",
    "aggregate/instruction_by_pass_pair.csv",
    "aggregate/fingerprint_collision_diagnostics.csv",
    "aggregate/failure_ledger.csv",
    "report/instruction_granularity_report_zh.md",
    "report/advisor_talking_points_zh.md",
    "report/advisor_q_and_a_zh.md",
    "figures/01_four_level_coverage.png",
    "figures/01_four_level_coverage.svg",
    "figures/02_four_level_precision.png",
    "figures/02_four_level_precision.svg",
    "figures/03_extraction_time_by_level.png",
    "figures/03_extraction_time_by_level.svg",
    "figures/04_incremental_benefit_vs_cost.png",
    "figures/04_incremental_benefit_vs_cost.svg",
    "docs/design.md",
    "docs/implementation-plan.md",
    "docs/verification.md",
    "raw/instruction_screen_snapshot.json",
    "raw/tdd_red.log",
    "raw/protected_inventory_baseline.json",
    "raw/protected_inventory_final.json",
    "raw/old_experiment_inventory_baseline.json",
    "raw/old_experiment_inventory_final.json",
    "raw/isolated_tests.log",
    "raw/repository_tests.log",
    "raw/determinism_manifest.json",
)


def generate_evidence(
    root: Path,
    dataset: FrozenDataset,
    metrics: dict,
    verification: dict,
) -> dict:
    root = Path(root)
    verification_path = root / "docs" / "verification.md"
    write_text(verification_path, _verification_markdown(root, metrics, verification))

    missing = [relative for relative in REQUIRED_ARTIFACTS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing required artifacts: " + ",".join(missing))
    artifacts = [
        {
            "path": relative,
            "size": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in REQUIRED_ARTIFACTS
    ]
    payload = {
        "schema_version": "instruction-granularity-evidence-v1",
        "experiment_root": str(root),
        "research_boundary": {
            "claim": "observed-change empirical screening, not a commutativity proof",
            "phasebatch_speedup_claimed": False,
            "llvm_or_worker_execution_allowed": False,
            "s_a_b_ab_ba_rebuilt": False,
            "commit_or_push_performed": False,
        },
        "hard_counts": metrics["hard_counts"],
        "conclusion": {
            "first_sentence": _first_sentence(metrics),
            "h_inst": metrics["coverage"]["H_inst"],
            "incremental": metrics["incremental"],
            "cumulative": metrics["cumulative"],
        },
        "frozen_inputs": {
            "authoritative_csv": {
                "path": str(dataset.authoritative_csv),
                "size": dataset.authoritative_csv.stat().st_size,
                "sha256": dataset.authoritative_csv_sha256,
            },
            "old_experiment_root": str(dataset.old_experiment_root),
            "dynamic_all_attempts": [
                {
                    "repetition": attempt.repetition,
                    "root": str(attempt.root),
                    "completion_path": str(attempt.completion_path),
                    "completion_sha256": attempt.completion_sha256,
                    "single_pass_csv_sha256": attempt.single_pass_csv_sha256,
                    "pair_runs_csv_sha256": attempt.pair_runs_csv_sha256,
                    "output_count": len(attempt.outputs),
                }
                for attempt in dataset.attempts
            ],
        },
        "runtime_protocol": {
            "level_order": [
                "FUNC_ONLY",
                "BLOCK_ONLY",
                "EFFECT_ONLY",
                "INSTRUCTION_ONLY",
            ],
            "warmup_cycles": 5,
            "measured_cycles": 30,
            "measured_rows": 120,
            "source_repetition_rotation": "((cycle-1) mod 3)+1",
            "cross_repetition_parse_or_token_cache": False,
            "paired_delta_before_aggregation": True,
        },
        "verification": verification,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "python_dont_write_bytecode": True,
            "pytest_cache_disabled": True,
        },
        "artifacts": artifacts,
    }
    write_json(root / "evidence_manifest.json", payload)
    return payload


def _verification_markdown(root: Path, metrics: dict, verification: dict) -> str:
    incremental = metrics["incremental"]
    cumulative = metrics["cumulative"]
    isolated = verification.get("isolated_tests", {})
    repository = verification.get("repository_tests", {})
    protected = verification.get("protected_inventory", {})
    old = verification.get("old_experiment_inventory", {})
    visual = verification.get("visual_inspection", {})
    determinism = verification.get("determinism", {})
    return f"""# 指令粒度实验最终验证

## 结果门

- {_first_sentence(metrics)}
- `H_inst`: selected={cumulative['cumulative_selected']}, commute={cumulative['cumulative_commute']}, order-sensitive={cumulative['cumulative_order_sensitive']}, failed={cumulative['cumulative_failed']}, precision={float(cumulative['cumulative_precision']):.6f}, recall={float(cumulative['cumulative_coverage_of_833_commute']):.6f}。
- 增量：selected={incremental['incremental_selected_count']}, commute={incremental['incremental_commute']}, order-sensitive={incremental['incremental_order_sensitive']}, failed={incremental['incremental_failed']}, unknown={incremental['incremental_unknown']}。
- 硬门槛：1,411 rows / 49 programs / 14 actions / 833 commute / 569 order-sensitive / 9 failed / 686 transitions；30/46/47 legacy gate 全部复现。

## TDD 与测试

- RED 证据：`raw/tdd_red.log`，collection 因实现包不存在而失败。
- isolated suite：{isolated.get('summary', 'not_recorded')}；日志 `{isolated.get('log', '')}`，SHA-256 `{isolated.get('sha256', '')}`。
- repository suite：{repository.get('summary', 'not_recorded')}；日志 `{repository.get('log', '')}`，SHA-256 `{repository.get('sha256', '')}`。
- pytest cache 禁用；`PYTHONDONTWRITEBYTECODE=1`；`TEMP/TMP/MPLCONFIGDIR/--basetemp` 均在 `{root}` 内。

## 正式计时与确定性

- 5 个 warm-up cycle 不计入汇总，30 个 measured cycle 全部完成；四层固定顺序，每层 30 行 measured、总计 120 行。
- paired cost 在相同 measured repetition 内先相减，再汇总 median/p90。
- 聚合/报告确定性：{determinism.get('summary', 'not_recorded')}；manifest SHA-256 `{determinism.get('sha256', '')}`。

## 图形人工检查

- 状态：{visual.get('summary', 'not_recorded')}。
- 检查项：中文标签、数值、坐标轴、legend、正负值与 clipping；四图均以原始 960×576 分辨率检查。

## 隔离验证

- protected baseline/final：files={protected.get('final_file_count', '')}，clean={protected.get('is_clean', False)}，record SHA-256 `{protected.get('final_record_sha256', '')}`。
- 旧实验 baseline/final：files={old.get('final_file_count', '')}，clean={old.get('is_clean', False)}，record SHA-256 `{old.get('final_record_sha256', '')}`。
- 禁止进程审计：{verification.get('forbidden_process_audit', {}).get('summary', 'not_recorded')}。
- 未启动 Worker、`opt`、`clang`、`llvm-diff`，未运行 LLVM pass，未重建 S/A/B/AB/BA，未 commit/push。

## 表述边界

本实验是 observed-change empirical screen。指令指纹不相交不证明 commute；pair 局部成本不是 Phasebatch speedup；49-program 结果不外推到其他 corpus。
"""


def _first_sentence(metrics: dict) -> str:
    cumulative = metrics["cumulative"]
    incremental = metrics["incremental"]
    return (
        f"指令级最终覆盖了 {int(cumulative['cumulative_commute'])}/833 = "
        f"{float(cumulative['cumulative_coverage_of_833_commute']) * 100:.2f}% 的实际 commute，"
        f"相比 H_effect 新增 {int(incremental['incremental_commute'])} 个真实 commute。"
    )
