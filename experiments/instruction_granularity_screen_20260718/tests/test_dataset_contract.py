from __future__ import annotations

from pathlib import Path

from instruction_granularity.dataset import load_frozen_dataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_dataset_hard_counts_and_source_attempts() -> None:
    dataset = load_frozen_dataset(EXPERIMENT_ROOT)

    assert len(dataset.pairs) == 1411
    assert len({pair.program for pair in dataset.pairs}) == 49
    assert len(dataset.action_ids) == 14
    assert dataset.relation_counts == {
        "dynamic_commute": 833,
        "dynamic_order_sensitive": 569,
        "failed": 9,
    }
    assert len(dataset.transition_keys) == 686
    assert tuple(attempt.repetition for attempt in dataset.attempts) == (1, 2, 3)
    assert all(attempt.configuration == "DYNAMIC_ALL" for attempt in dataset.attempts)
    assert all(attempt.status == "complete" for attempt in dataset.attempts)
    assert all(len(attempt.outputs) == 686 for attempt in dataset.attempts)


def test_frozen_legacy_labels_match_required_counts() -> None:
    dataset = load_frozen_dataset(EXPERIMENT_ROOT)

    assert dataset.legacy_counts == {
        "H_func": {"selected": 30, "commute": 28, "order_sensitive": 2},
        "H_block": {"selected": 46, "commute": 44, "order_sensitive": 2},
        "H_effect": {"selected": 47, "commute": 45, "order_sensitive": 2},
    }


def test_every_pair_has_two_actions_and_all_rows_are_preserved() -> None:
    dataset = load_frozen_dataset(EXPERIMENT_ROOT)

    assert len({pair.observation_id for pair in dataset.pairs}) == 1411
    assert all(pair.action_a_id != pair.action_b_id for pair in dataset.pairs)
    assert all((pair.program, pair.action_a_id) in dataset.transition_keys for pair in dataset.pairs)
    assert all((pair.program, pair.action_b_id) in dataset.transition_keys for pair in dataset.pairs)

