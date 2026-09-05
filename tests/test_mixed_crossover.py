from __future__ import annotations

from functools import lru_cache
import random

import numpy as np

from hybrid_qcs.mixed_crossover import (
    DeferredMixedSearch,
    EagerMixedSearch,
    GATE_FAMILIES,
    _cheap_legal_continuation,
    _projected_tableau_mismatch,
    evaluate_mixed_eager_sarsa,
    evaluate_mixed_hierarchy,
    mixed_evaluation_targets,
    mixed_gate_library,
    mixed_inner_context,
    mixed_training_targets,
    train_mixed_inner_bandit,
    train_mixed_outer_sarsa,
)
from hybrid_qcs.model import Budget, Gate, HybridState


def test_mixed_crossover_uses_complete_native_grammar() -> None:
    for target in mixed_evaluation_targets():
        actions = mixed_gate_library(target.num_qubits)
        assert {gate.name for gate in actions} == set(GATE_FAMILIES)
        assert len(actions) == 5 * target.num_qubits + target.num_qubits * (
            target.num_qubits - 1
        )
        search = EagerMixedSearch(target, max_expansions=1)
        assert search.actions == actions
        assert target.rotation_payloads


def test_mixed_training_and_evaluation_targets_are_disjoint() -> None:
    training = mixed_training_targets()
    evaluation = mixed_evaluation_targets()
    training_keys = {target.canonical_key for target in training}
    assert len(training_keys) == len(training)
    assert all(target.canonical_key not in training_keys for target in evaluation)
    assert [target.num_qubits for target in evaluation] == [4, 5, 6]


def test_cheap_legality_matches_exact_apply_on_random_prefixes() -> None:
    rng = random.Random(20260904)
    budget = Budget(max_t_count=3, max_cnot_count=3, max_gates=7, max_depth=7)
    for n in (4, 5, 6):
        actions = mixed_gate_library(n)
        state = HybridState.identity(n, budget)
        for _ in range(30):
            for gate in actions:
                predicted = _cheap_legal_continuation(state, gate)
                actual = state.apply(gate, partial_order_reduction=True) is not None
                assert predicted == actual, (n, state.reconstruct_gates(), gate)
            legal = [gate for gate in actions if _cheap_legal_continuation(state, gate)]
            if not legal:
                break
            child = state.apply(rng.choice(legal), partial_order_reduction=True)
            assert child is not None
            state = child


def test_projected_tableau_mismatch_matches_exact_clifford_child() -> None:
    target = mixed_evaluation_targets()[0]
    state = HybridState.identity(target.num_qubits, target.budget)
    for gate in (Gate("H", (1,)), Gate("CNOT", (0, 3))):
        child = state.apply(gate, partial_order_reduction=True)
        assert child is not None
        state = child

    for gate in mixed_gate_library(target.num_qubits):
        if not gate.is_clifford or not _cheap_legal_continuation(state, gate):
            continue
        child = state.apply(gate, partial_order_reduction=True)
        assert child is not None
        exact = sum(
            left != right
            for left, right in zip(
                child.tableau.canonical_payload(),
                target.tableau_payload,
                strict=True,
            )
        )
        assert _projected_tableau_mismatch(state, gate, target) == exact


def test_deferred_mixed_record_retains_pending_continuations() -> None:
    target = mixed_evaluation_targets()[0]
    environment = DeferredMixedSearch(target, batch_size=1)
    root = environment.open_records()[0]
    before = root.pending_mask.bit_count()
    token = environment.pending_tokens(root)[0]
    step = environment.process_batch(
        root.record_id, (token,), allow_fairness_override=False
    )
    assert step.attempted_edges == 1
    assert root.pending_mask.bit_count() == before - 1
    assert root.record_id in environment.frontier


def test_mixed_inner_context_is_action_conditioned_and_finite() -> None:
    target = mixed_evaluation_targets()[0]
    environment = DeferredMixedSearch(target)
    root = environment.open_records()[0]
    contexts = [
        mixed_inner_context(root, environment.actions[token], target)
        for token in environment.pending_tokens(root)
    ]
    assert contexts
    assert all(np.isfinite(context).all() for context in contexts)
    assert len({tuple(context) for context in contexts}) > len(GATE_FAMILIES)


@lru_cache(maxsize=1)
def _small_trained_policies():
    training = mixed_training_targets()
    outer = train_mixed_outer_sarsa(
        training, episodes=24, seed=11, max_expansions=96
    )
    inner = train_mixed_inner_bandit(training, episodes=32, alpha=0.5)
    return outer, inner


def test_small_mixed_hierarchy_certifies_all_heldout_targets() -> None:
    outer, inner = _small_trained_policies()
    for target in mixed_evaluation_targets():
        result = evaluate_mixed_hierarchy(
            outer,
            inner,
            target,
            batch_size=4,
            max_allocations=512,
            wall_limit=3.0,
        )
        assert result.success, target.name
        assert result.certified, target.name
        assert result.maximum_matrix_error is not None
        assert result.maximum_matrix_error <= 1e-9
        gate_names = {label.split("(", 1)[0] for label in result.witness}
        assert {"H", "T", "TDG", "CNOT"}.issubset(gate_names)
        assert gate_names.intersection({"S", "SDG"})


def test_mixed_hierarchy_reduces_exact_edges_on_five_qubit_case() -> None:
    outer, inner = _small_trained_policies()
    target = mixed_evaluation_targets()[1]
    eager = evaluate_mixed_eager_sarsa(
        outer, target, max_expansions=512, wall_limit=3.0
    )
    deferred = evaluate_mixed_hierarchy(
        outer,
        inner,
        target,
        batch_size=4,
        max_allocations=512,
        wall_limit=3.0,
    )
    assert eager.success and deferred.success
    assert deferred.attempted_edges < eager.attempted_edges
    assert deferred.frontier_peak < eager.frontier_peak


def test_mixed_runner_aggregation_preserves_false_csv_booleans() -> None:
    from hybrid_qcs.mixed_crossover_runner import _aggregate

    row = {
        "method": "mixed_eager_target_potential",
        "target": "synthetic-failure",
        "num_qubits": "4",
        "generator_length": "6",
        "success": "False",
        "wall_seconds": "1.0",
        "cpu_seconds": "1.0",
        "attempted_edges": "32",
        "outer_decisions": "1",
        "frontier_peak": "1",
        "policy_rows": "1",
        "stop_reason": "wall_limit",
    }
    aggregate = _aggregate([row])
    assert len(aggregate) == 1
    assert aggregate[0]["success_rate"] == 0.0
    assert aggregate[0]["median_success_wall_seconds"] == ""
    assert aggregate[0]["median_observed_attempted_edges"] == 32
