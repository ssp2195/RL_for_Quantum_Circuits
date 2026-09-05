from __future__ import annotations

import numpy as np

from hybrid_qcs.certify import certify_state
from hybrid_qcs.cnot_crossover import (
    DeferredCnotSearch,
    EagerCnotSearch,
    LinearCnotLinUCB,
    crossover_evaluation_targets,
    crossover_training_targets,
    inner_context,
    train_deferred_outer,
    train_inner_bandit,
)
from hybrid_qcs.model import HybridState


def test_hybrid_state_supports_controlled_width_extension() -> None:
    target = crossover_evaluation_targets()[-1]
    state = HybridState.identity(target.num_qubits, target.budget)
    assert state.num_qubits == 6
    state.validate()


def test_crossover_targets_are_disjoint_from_training() -> None:
    training = crossover_training_targets()
    evaluation = crossover_evaluation_targets()
    training_keys = {target.canonical_key for target in training}
    assert all(target.canonical_key not in training_keys for target in evaluation)
    assert [target.num_qubits for target in evaluation] == [4, 5, 6]


def test_deferred_record_retains_unprocessed_continuations() -> None:
    target = crossover_training_targets()[0]
    environment = DeferredCnotSearch(target, batch_size=1)
    root = environment.open_records()[0]
    before = root.pending_mask.bit_count()
    token = environment.pending_tokens(root)[0]
    step = environment.process_batch(root.record_id, (token,))
    assert step.attempted_edges == 1
    assert root.pending_mask.bit_count() == before - 1
    assert root.record_id in environment.frontier


def test_inner_context_is_action_conditioned_and_finite() -> None:
    target = crossover_training_targets()[0]
    environment = DeferredCnotSearch(target)
    root = environment.open_records()[0]
    contexts = [inner_context(root, gate, target) for gate in environment.actions]
    assert all(np.isfinite(context).all() for context in contexts)
    assert len({tuple(context) for context in contexts}) > 1


def test_small_hierarchical_training_certifies_heldout_swap() -> None:
    training = crossover_training_targets()
    bandit = train_inner_bandit(training, episodes=80, alpha=0.5)
    outer = train_deferred_outer(
        training,
        bandit,
        episodes=100,
        seed=17,
        batch_size=4,
    )
    target = crossover_evaluation_targets()[0]
    environment = DeferredCnotSearch(
        target,
        max_allocations=512,
        max_edges=4_096,
        batch_size=4,
        fairness_start_k=10_000,
    )
    while environment.frontier and environment.solution_record_id is None:
        record_id, _, _ = outer.choose(
            environment.open_records(), target, 0.0, deferred=True
        )
        record = environment.frontier[record_id]
        remaining = list(environment.pending_tokens(record))
        ordered = []
        while remaining and len(ordered) < 4:
            token, _, _ = bandit.choose(
                record,
                remaining,
                environment.actions,
                target,
                explore=False,
            )
            ordered.append(token)
            remaining.remove(token)
        step = environment.process_batch(
            record_id, ordered, allow_fairness_override=False
        )
        if step.terminated or step.truncated:
            break
    state = environment.solution_state()
    assert state is not None
    assert certify_state(target, state).success


def test_eager_search_uses_complete_directed_cnot_grammar() -> None:
    target = crossover_evaluation_targets()[0]
    environment = EagerCnotSearch(target, max_expansions=1)
    assert len(environment.actions) == target.num_qubits * (target.num_qubits - 1)
