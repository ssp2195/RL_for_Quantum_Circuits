from __future__ import annotations

from dataclasses import replace

from hybrid_qcs.benchmarks import held_out_targets, structured_toffoli_target, training_targets
from hybrid_qcs.certify import certify_state
from hybrid_qcs.model import Budget, Gate, HybridState, generate_gates
from hybrid_qcs.rl import LinearSarsaRanker, train_online_sarsa
from hybrid_qcs.search import HybridSearch
from hybrid_qcs.structured_toffoli import StructuredToffoliSearch


def test_target_records_expose_semantics_but_not_generator_witness() -> None:
    target = training_targets()[0]
    assert not hasattr(target, "hidden_witness")
    assert not hasattr(target, "target_state")
    assert target.canonical_key
    assert target.tableau_payload
    assert target.unitary.flags.writeable is False


def test_gate_expansion_is_target_independent() -> None:
    target = training_targets()[2]
    altered_target = replace(target, canonical_key=("unreachable-target",))
    original = HybridSearch(target, max_expansions=1)
    altered = HybridSearch(altered_target, max_expansions=1)
    assert original.actions == altered.actions == generate_gates(target.num_qubits)

    original_root = original.open_records()[0].state
    altered_root = altered.open_records()[0].state
    original_children = {
        gate.label()
        for gate in original.actions
        if original_root.apply(gate, partial_order_reduction=True) is not None
    }
    altered_children = {
        gate.label()
        for gate in altered.actions
        if altered_root.apply(gate, partial_order_reduction=True) is not None
    }
    assert original_children == altered_children


def test_inverse_pair_is_removed_across_only_disjoint_work() -> None:
    state = HybridState.identity(2, Budget(0, 0, 3, 3))
    first = state.apply(Gate("H", (0,)), partial_order_reduction=True)
    assert first is not None
    independent = first.apply(Gate("S", (1,)), partial_order_reduction=True)
    assert independent is not None
    assert independent.apply(Gate("H", (0,)), partial_order_reduction=True) is None


def test_online_sarsa_frozen_policy_certifies_all_held_out_targets() -> None:
    policy = LinearSarsaRanker(seed=23, learning_rate=0.002)
    training = train_online_sarsa(
        policy,
        training_targets(),
        episodes=160,
        max_expansions=2_048,
        deadline_seconds=20.0,
    )
    assert not training.deadline_hit
    assert len(training.episodes) == 160
    assert all(log.success and log.certified for log in training.episodes)

    for target in held_out_targets():
        environment = HybridSearch(target, max_expansions=8_192)
        state = environment.run_scheduler(
            lambda records: policy.choose(records, target, 0.0)[0]
        )
        assert state is not None, target.name
        assert certify_state(target, state).success, target.name

    toffoli = structured_toffoli_target()
    structured = StructuredToffoliSearch(toffoli, max_expansions=8_192)
    state = structured.run_scheduler(
        lambda records: policy.choose(records, toffoli, 0.0)[0]
    )
    assert state is not None
    assert structured.solution_progress() is not None
    assert structured.solution_progress().stage.value == "DONE"
    assert certify_state(toffoli, state).success
