from __future__ import annotations

import numpy as np

from hybrid_qcs.ancilla_benchmarks import (
    ancilla_training_targets,
    clean_contract,
    parity_phase_witness,
    qft3_clean_ancilla_target,
    qft3_clean_ancilla_witness,
    qft3_matrix,
)
from hybrid_qcs.ancilla_certify import certify_ancilla_state
from hybrid_qcs.ancilla_contract import (
    AncillaContract,
    PhaseMode,
    contract_metrics,
    target_from_hidden_ancilla_gates,
)
from hybrid_qcs.ancilla_search import (
    AncillaDeferredSearch,
    DisjointAncillaLinUCB,
    LinearAncillaOuterSarsa,
    ancilla_inner_context,
    ancilla_outer_features,
    apply_gate_to_isometry,
    evaluate_ancilla_hierarchy,
    train_ancilla_inner_bandit,
    train_ancilla_outer_sarsa,
)
from hybrid_qcs.certify import gate_matrix
from hybrid_qcs.model import Budget, Gate, HybridState


def _state(target, gates):
    state = HybridState.identity(target.num_qubits, target.budget)
    for gate in gates:
        child = state.apply(gate, partial_order_reduction=False)
        assert child is not None
        state = child
    return state


def test_clean_embedding_and_gate_column_update_match_dense_matrix() -> None:
    contract = clean_contract(2, 1)
    isometry = np.array(contract.input_embedding, copy=True)
    for gate in (
        Gate("H", (0,)),
        Gate("S", (2,)),
        Gate("T", (1,)),
        Gate("CNOT", (1, 2)),
    ):
        projected = apply_gate_to_isometry(isometry, gate)
        expected = gate_matrix(contract.total_qubits, gate) @ isometry
        assert np.allclose(projected, expected, atol=1e-12)
        isometry = projected


def test_full_unitary_difference_can_be_clean_ancilla_equivalent() -> None:
    target = target_from_hidden_ancilla_gates(
        "identity-clean-contract",
        "test",
        clean_contract(1, 1),
        (),
        logical_unitary=np.eye(2, dtype=np.complex128),
        gate_slack=2,
        depth_slack=2,
    )
    root = HybridState.identity(2, target.budget)
    candidate = root.apply(Gate("S", (1,)), partial_order_reduction=False)
    assert candidate is not None
    candidate = candidate.apply(Gate("S", (1,)), partial_order_reduction=False)
    assert candidate is not None
    assert candidate.canonical_key != root.canonical_key
    result = certify_ancilla_state(target, candidate)
    assert result.success
    assert result.clean_return


def test_dirty_clean_ancilla_is_rejected() -> None:
    target = target_from_hidden_ancilla_gates(
        "identity-clean-reject-dirty",
        "test",
        clean_contract(1, 1),
        (),
        logical_unitary=np.eye(2, dtype=np.complex128),
        gate_slack=1,
        depth_slack=1,
    )
    candidate = HybridState.identity(2, target.budget)
    child = candidate.apply(Gate("H", (1,)), partial_order_reduction=False)
    assert child is not None
    result = certify_ancilla_state(target, child)
    assert not result.success
    assert not result.clean_return
    assert result.ancilla_leakage > 0.1


def test_borrowed_ancilla_must_be_identity_on_arbitrary_input() -> None:
    contract = AncillaContract(
        total_qubits=2,
        logical_qubits=(0,),
        borrowed_ancillas=(1,),
    )
    target = target_from_hidden_ancilla_gates(
        "borrowed-identity",
        "test",
        contract,
        (),
        logical_unitary=np.eye(2, dtype=np.complex128),
        gate_slack=2,
        depth_slack=2,
        cnot_slack=2,
    )
    restored = _state(
        target,
        (Gate("CNOT", (1, 0)), Gate("CNOT", (1, 0))),
    )
    assert certify_ancilla_state(target, restored).success

    changed = _state(target, (Gate("CNOT", (1, 0)),))
    result = certify_ancilla_state(target, changed)
    assert not result.success
    assert not result.borrowed_identity


def test_projective_and_exact_phase_modes_are_distinguished() -> None:
    projective_target = target_from_hidden_ancilla_gates(
        "projective-phase-test",
        "test",
        clean_contract(1, 1),
        (),
        logical_unitary=np.eye(2, dtype=np.complex128),
    )
    exact_contract = AncillaContract(
        total_qubits=2,
        logical_qubits=(0,),
        clean_ancillas=(1,),
        phase_mode=PhaseMode.EXACT,
    )
    exact_target = target_from_hidden_ancilla_gates(
        "exact-phase-test",
        "test",
        exact_contract,
        (),
        logical_unitary=np.eye(2, dtype=np.complex128),
    )
    shifted = 1j * projective_target.target_isometry
    assert contract_metrics(shifted, projective_target).success
    assert not contract_metrics(shifted, exact_target).success


def test_compute_phase_uncompute_witness_is_certified() -> None:
    gates = parity_phase_witness((0, 1), 2, "T")
    target = target_from_hidden_ancilla_gates(
        "parity-phase",
        "test",
        clean_contract(2, 1),
        gates,
    )
    result = certify_ancilla_state(target, _state(target, gates))
    assert result.success
    assert result.ancilla_leakage < 1e-12


def test_qft3_clean_ancilla_witness_matches_analytical_target() -> None:
    target = qft3_clean_ancilla_target()
    assert target.logical_unitary.shape == (8, 8)
    assert np.allclose(target.logical_unitary, qft3_matrix(), atol=1e-12)
    state = _state(target, qft3_clean_ancilla_witness())
    result = certify_ancilla_state(target, state)
    assert result.success
    assert result.gate_count == 47
    assert result.t_count == 21
    assert result.cnot_count == 19
    assert result.ancilla_leakage < 1e-12


def test_deferred_search_and_role_aware_linear_features_are_finite() -> None:
    target = ancilla_training_targets()[0]
    environment = AncillaDeferredSearch(
        target,
        max_allocations=16,
        max_edges=64,
        batch_size=2,
        fairness_start_k=10_000,
    )
    root = environment.open_records()[0]
    outer = ancilla_outer_features(root, target)
    assert np.isfinite(outer).all()
    token = environment.pending_tokens(root)[0]
    inner = ancilla_inner_context(root, environment.actions[token], target)
    assert np.isfinite(inner).all()
    step = environment.process_batch(
        root.record_id,
        environment.pending_tokens(root)[:2],
        allow_fairness_override=False,
    )
    assert step.attempted_edges == 2
    assert root.pending_mask != 0



def test_symbolic_state_is_not_artificially_limited_to_six_qubits() -> None:
    state = HybridState.identity(7, Budget(0, 0, 0, 0))
    assert state.num_qubits == 7
    assert len(state.wire_depths) == 7


def test_small_staged_training_smoke_and_frozen_evaluation() -> None:
    targets = ancilla_training_targets()[:3]
    outer = train_ancilla_outer_sarsa(
        targets,
        episodes=3,
        max_allocations=24,
        batch_size=2,
    )
    bandit = train_ancilla_inner_bandit(
        outer,
        targets,
        episodes=3,
        max_allocations=24,
    )
    assert isinstance(outer, LinearAncillaOuterSarsa)
    assert isinstance(bandit, DisjointAncillaLinUCB)
    result = evaluate_ancilla_hierarchy(
        outer,
        bandit,
        targets[0],
        max_allocations=64,
        batch_size=2,
        wall_limit=2.0,
    )
    assert result.success
