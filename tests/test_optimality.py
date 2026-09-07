from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hybrid_qcs.optimality import (
    AdditiveResources,
    certify_marked_state_oracle,
    certify_toffoli_normal_form,
    four_qubit_linear_audits,
    marked_state_target,
    native_unitary,
    oracle_macro_grammar,
    phase_exponents,
    prove_ccz_parity_network_optimum,
    prove_ccz_phase_polynomial_t_optimum,
    prove_marked_evaluator_lexicographic_optimum,
)
from hybrid_qcs.optimality_runner import run


def test_ccz_phase_polynomial_exhaustion_proves_seven_t() -> None:
    proof = prove_ccz_phase_polynomial_t_optimum()
    assert proof.assignments_examined == 8**7
    assert proof.minimum_t_count == 7
    assert proof.exact
    assert sum(value & 1 for value in proof.coefficients_mod8) == 7
    assert phase_exponents(proof.coefficients_mod8) == proof.target_exponents_mod8


def test_ccz_parity_network_proves_six_cnot_and_depth() -> None:
    proof = prove_ccz_parity_network_optimum()
    assert proof.minimum_cnot_count == 6
    assert proof.minimum_cnot_depth == 6
    assert len(proof.cnot_sequence) == 6
    assert {mask for _, mask in proof.parity_host_events} == set(range(1, 8))


def test_all_three_bit_single_violation_phase_oracles_are_exact() -> None:
    for value in range(8):
        bits = "".join("1" if value & (1 << q) else "0" for q in range(3))
        cert = certify_marked_state_oracle(bits)
        assert cert.minimum_t_count == 7
        assert cert.minimum_cnot_count == 6
        assert cert.exact_matrix_error < 1e-12
        actual = native_unitary(3, tuple(
            (text.split("(", 1)[0], tuple(int(q) for q in text.rstrip(")").split("(", 1)[1].split(",") if q))
            for text in cert.native_witness
        ))
        assert np.allclose(actual, marked_state_target(bits), atol=1e-12)


def test_direct_bnn_phase_oracle_strictly_improves_compute_uncompute_resources() -> None:
    cert = certify_marked_state_oracle("100")
    assert cert.minimum_t_count == 7 < 42
    assert cert.minimum_cnot_count == 6 < 42
    assert cert.native_gate_count_upper_bound < 98


def test_oracle_macro_grammar_is_target_independent_and_finite() -> None:
    grammar = oracle_macro_grammar()
    assert len(grammar) == 23
    assert {macro.kind for macro in grammar} == {"X", "CNOT", "TOFFOLI"}
    assert len({(macro.kind, macro.qubits) for macro in grammar}) == len(grammar)


def test_learned_bnn_evaluator_is_exact_lexicographic_macro_optimum() -> None:
    proof = prove_marked_evaluator_lexicographic_optimum("100")
    assert proof.status.value == "exact"
    assert proof.optimum == AdditiveResources(6, 21, 21, 48)
    assert proof.incumbent_matches_optimum
    assert len(proof.witness) == 6
    assert proof.states_settled > 0
    assert proof.edges_generated >= proof.states_settled


def test_three_qubit_toffoli_normal_form_is_exact() -> None:
    audit = certify_toffoli_normal_form()
    assert audit["t_count"] == 7
    assert audit["cnot_count"] == 6
    assert audit["cnot_depth"] == 6
    assert audit["exact_matrix_error"] < 1e-12


def test_four_qubit_linear_targets_have_exact_count_and_depth_proofs() -> None:
    audits = {audit.name: audit for audit in four_qubit_linear_audits()}
    assert set(audits) == {"register-half-swap", "wire-reversal"}
    for audit in audits.values():
        assert audit.minimum_cnot_count == 6
        assert audit.minimum_cnot_depth == 3
        assert sum(len(layer) for layer in audit.depth_layers) == 6


def test_runner_writes_machine_readable_claim_boundaries(tmp_path: Path) -> None:
    results = run(tmp_path)
    assert results["schema"] == "proof-producing-oracle-optimality-v1"
    assert results["single_violation_phase_oracles"]["100"]["minimum_t_count"] == 7
    assert results["bnn_evaluator_optimality"]["optimum"]["macro_count"] == 6
    assert results["qft3_audit"]["optimality"] in {"upper bound only", "no claim"}
    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "summary.csv").is_file()
    loaded = json.loads((tmp_path / "results.json").read_text())
    assert loaded["claims"]["qft3"] == "certified upper bound only"
