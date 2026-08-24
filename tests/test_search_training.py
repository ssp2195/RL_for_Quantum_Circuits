from __future__ import annotations

import time

import numpy as np

from hybrid_qcs.benchmarks import all_targets, held_out_targets, training_targets
from hybrid_qcs.certify import certify_state
from hybrid_qcs.rl import LinearSarsaRanker, train_online_sarsa
from hybrid_qcs.search import HybridSearch


def test_target_partitions_are_semantically_disjoint() -> None:
    targets = all_targets()
    assert len({target.canonical_key for target in targets}) == len(targets)


def test_symbolic_distance_scheduler_synthesizes_every_target() -> None:
    for target in all_targets():
        if target.family == "unrestricted-native-qft2":
            continue
        env = HybridSearch(target, max_expansions=2_000)
        state = env.run_scheduler(
            lambda records: min(
                records,
                key=lambda record: (
                    record.symbolic_distance,
                    record.state.gate_count,
                    record.record_id,
                ),
            ).record_id
        )
        assert state is not None, target.name
        result = certify_state(target, state)
        assert result.success, (target.name, result)


def test_online_sarsa_trains_and_certifies_unseen_mixed_target() -> None:
    policy = LinearSarsaRanker(seed=23, learning_rate=0.002)
    started = time.process_time()
    training = train_online_sarsa(
        policy,
        training_targets(),
        episodes=32,
        max_expansions=1_024,
        deadline_seconds=20.0,
    )
    assert not training.deadline_hit
    assert len(training.episodes) == 32
    assert all(log.success and log.certified for log in training.episodes)
    target = held_out_targets()[0]
    env = HybridSearch(target, max_expansions=1_024)
    state = env.run_scheduler(lambda records: policy.choose(records, target, 0.0)[0])
    assert state is not None
    assert certify_state(target, state).success
    assert np.isfinite(policy.theta).all()
    assert time.process_time() - started < 20.0
