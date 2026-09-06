"""Regression tests for hierarchical clean-ancilla Boolean-oracle synthesis."""
from __future__ import annotations

import numpy as np

from hybrid_qcs.certify import unitary_from_gates

from hybrid_qcs.oracle_benchmarks import (
    bnn_verification_oracle_spec,
    bnn_verification_oracle_target,
    oracle_training_targets,
)
from hybrid_qcs.oracle_synthesis import (
    DisjointOracleLinUCB,
    LinearOracleOuterSarsa,
    apply_macro_to_mapping,
    assemble_phase_oracle,
    certify_evaluator_witness,
    evaluate_oracle_hierarchy,
    oracle_macro_library,
    train_oracle_inner_bandit,
    train_oracle_outer_sarsa,
)


EXPECTED_MACROS = (
    "TOFFOLI(0,1,4)",
    "CNOT(0,4)",
    "CNOT(4,3)",
    "TOFFOLI(2,4,3)",
    "CNOT(0,4)",
    "TOFFOLI(0,1,4)",
)


def _tokens_for_names(target, names: tuple[str, ...]) -> tuple[int, ...]:
    by_name = {macro.name: token for token, macro in enumerate(oracle_macro_library(target.layout))}
    return tuple(by_name[name] for name in names)


def test_bnn_oracle_spec_marks_exactly_100() -> None:
    spec = bnn_verification_oracle_spec()
    assert spec.marked_bitstrings == ("100",)
    assert spec.truth_table == (0, 1, 0, 0, 0, 0, 0, 0)
    assert spec.marked_inputs == (1,)
    for basis in range(8):
        x1, x2, x3 = ((basis >> index) & 1 for index in range(3))
        assert spec.value(basis) == int(x1 and not x2 and not x3)


def test_macro_grammar_depends_on_layout_not_target_truth_table() -> None:
    targets = [target for target in oracle_training_targets() if target.spec.num_inputs == 3]
    bnn = bnn_verification_oracle_target()
    assert targets
    reference = tuple(macro.name for macro in oracle_macro_library(bnn.layout))
    for target in targets:
        if target.layout == bnn.layout:
            assert tuple(macro.name for macro in oracle_macro_library(target.layout)) == reference


def test_discovered_evaluator_mapping_is_exact_and_workspace_clean() -> None:
    target = bnn_verification_oracle_target()
    actions = oracle_macro_library(target.layout)
    tokens = _tokens_for_names(target, EXPECTED_MACROS)
    mapping = target.root_mapping
    for token in tokens:
        mapping = apply_macro_to_mapping(mapping, actions[token])
    assert mapping == target.target_mapping
    work_mask = sum(1 << qubit for qubit in target.layout.work_qubits)
    assert all((basis & work_mask) == 0 for basis in mapping)


def test_macro_witness_lowers_to_native_clifford_t_and_certifies() -> None:
    target = bnn_verification_oracle_target()
    tokens = _tokens_for_names(target, EXPECTED_MACROS)
    evaluator_state, evaluator_cert = certify_evaluator_witness(target, tokens)
    assert evaluator_cert.success
    assert evaluator_cert.mapping_match
    assert evaluator_cert.native_replay_match
    assert evaluator_cert.workspace_leakage < 1e-12
    assert {gate.name for gate in evaluator_state.reconstruct_gates()} <= {
        "H", "S", "SDG", "T", "TDG", "CNOT"
    }
    assert evaluator_state.gate_count == 48
    assert evaluator_state.t_count == 21
    assert evaluator_state.cnot_count == 21


def test_exact_phase_wrapper_marks_only_100_and_restores_ancillas() -> None:
    target = bnn_verification_oracle_target()
    tokens = _tokens_for_names(target, EXPECTED_MACROS)
    evaluator_state, _ = certify_evaluator_witness(target, tokens)
    phase_state, phase_target, cert = assemble_phase_oracle(target, evaluator_state)
    assert cert.success
    assert cert.exact_isometry_error < 1e-12
    assert cert.projective_isometry_error < 1e-12
    assert cert.ancilla_leakage < 1e-12
    assert phase_state.gate_count == 98
    assert phase_state.t_count == 42
    assert phase_state.cnot_count == 42
    actual = unitary_from_gates(phase_state.num_qubits, phase_state.reconstruct_gates()) @ phase_target.contract.input_embedding
    expected = phase_target.target_isometry
    assert np.max(np.abs(actual - expected)) < 1e-12


def test_hierarchical_outer_sarsa_inner_linucb_generates_oracle() -> None:
    curriculum = oracle_training_targets()[:4]
    outer = LinearOracleOuterSarsa(seed=11, learning_rate=1e-4)
    outer = train_oracle_outer_sarsa(
        curriculum,
        episodes=2,
        seed=11,
        max_allocations=32,
        batch_size=2,
        policy=outer,
    )
    bandit = DisjointOracleLinUCB(alpha=0.05, regularization=1_000.0)
    bandit = train_oracle_inner_bandit(
        outer,
        curriculum,
        episodes=4,
        alpha=0.05,
        max_allocations=32,
        bandit=bandit,
    )
    result = evaluate_oracle_hierarchy(
        outer,
        bandit,
        bnn_verification_oracle_target(),
        max_allocations=1_000,
        max_edges=4_000,
        batch_size=4,
        wall_limit=12.0,
    )
    assert result.success
    assert result.stop_reason == "certified"
    assert result.macro_witness == EXPECTED_MACROS
    assert result.phase_oracle_exact_error is not None
    assert result.phase_oracle_exact_error < 1e-12
    assert result.phase_oracle_leakage is not None
    assert result.phase_oracle_leakage < 1e-12


def test_truth_table_mapping_key_preserves_common_suffix_semantics() -> None:
    target = bnn_verification_oracle_target()
    actions = oracle_macro_library(target.layout)
    tokens = _tokens_for_names(target, EXPECTED_MACROS)
    left = target.root_mapping
    for token in tokens:
        left = apply_macro_to_mapping(left, actions[token])
    right = tuple(left)
    suffix = next(macro for macro in actions if macro.name == "CNOT(4,3)")
    assert apply_macro_to_mapping(left, suffix) == apply_macro_to_mapping(right, suffix)
