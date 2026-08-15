import numpy as np

from benchmarks.native_corpus import (
    DEFAULT_SPLIT_SEEDS,
    NATIVE_GATE_NAMES,
    SPLIT_ORDER,
    build_native_target_corpus,
    ccz_reference_benchmark,
    semantic_target_digest,
)
from benchmarks.toffoli import toffoli_reference_unitary
from certification.base import CertStatus
from certification.simulator import SimulatorCertificationEngine, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType


def _replay_case(case):
    state = CircuitState(
        CircuitDAG(case.num_qubits),
        ResourceBudget(
            max_t_count=len(case.witness),
            max_two_qubit_count=len(case.witness),
            max_depth=len(case.witness),
            max_gates=len(case.witness),
        ),
    )
    for gate in case.witness:
        assert state.apply_gate(gate)
    return state


def test_native_corpus_is_seeded_semantically_split_and_replayable():
    first = build_native_target_corpus()
    second = build_native_target_corpus()

    assert first.manifest() == second.manifest()
    assert first.split_seeds == tuple(DEFAULT_SPLIT_SEEDS.items())
    assert [case.metadata() for case in first.all_cases] == [
        case.metadata() for case in second.all_cases
    ]
    assert {case.num_qubits for case in first.semantic_cases} == {1, 2, 3}
    assert {case.num_qubits for case in first.bounded_synthesis_cases} == {1, 2, 3}

    target_ids = [case.target_id for case in first.all_cases]
    assert len(target_ids) == len(set(target_ids))
    for split in SPLIT_ORDER:
        assert len(first.cases(suite="semantic", split=split)) == 6
        assert len(first.cases(suite="bounded_synthesis", split=split)) == 3
        assert all(
            any(gate.gate_type is GateType.CNOT for gate in case.witness)
            for case in first.cases(suite="bounded_synthesis", split=split)
            if case.num_qubits > 1
        )

    # Every target is independently dense-certified from its authoritative
    # replay DAG.  The canonical digest is diagnostics, not this target oracle.
    for case in first.all_cases:
        assert semantic_target_digest(case.unitary) == case.target_id
        assert not case.unitary.flags.writeable
        assert all(gate.gate_type.name in NATIVE_GATE_NAMES for gate in case.witness)
        state = _replay_case(case)
        certificate = SimulatorCertificationEngine(case.synthesis_target()).certify(state)
        assert certificate.status is CertStatus.SUCCESS
        assert np.allclose(state.symbolic_unitary(), case.unitary)
        assert case.metadata()["target_specific_reachability_oracle"] is False
        assert case.metadata()["witness_use"] == "replay-and-audit-only"


def test_semantic_target_digest_quotients_only_a_global_phase_here():
    target = unitary_from_gates(1, (Gate(GateType.H, (0,)), Gate(GateType.T, (0,))))
    assert semantic_target_digest(target) == semantic_target_digest(
        np.exp(0.375j) * target
    )
    assert semantic_target_digest(target) != semantic_target_digest(
        unitary_from_gates(1, (Gate(GateType.H, (0,)),))
    )
    # Distinct native syntax for one exact target cannot cross a split.
    assert semantic_target_digest(
        unitary_from_gates(
            1,
            (Gate(GateType.T, (0,)), Gate(GateType.T, (0,))),
        )
    ) == semantic_target_digest(
        unitary_from_gates(1, (Gate(GateType.S, (0,)),))
    )


def test_ccz_is_a_separate_native_diagonal_reference():
    benchmark = ccz_reference_benchmark()
    target = benchmark.unitary
    witness = benchmark.witness
    candidate = unitary_from_gates(3, witness)

    assert np.allclose(candidate, target)
    assert np.allclose(target, np.diag([1, 1, 1, 1, 1, 1, 1, -1]))
    assert all(gate.gate_type.name in NATIVE_GATE_NAMES for gate in witness)
    assert benchmark.metadata()["scope"] == "known-native-witness-certification-reference"
    assert benchmark.metadata()["target_specific_reachability_oracle"] is False
    assert benchmark.metadata()["witness_gate_count"] == 13

    # Analytical relationship CCZ = H(target) CCX H(target), established
    # without deriving the CCZ target from the fixed phase-network witness.
    h_target = unitary_from_gates(3, (Gate(GateType.H, (2,)),))
    assert np.allclose(
        target,
        h_target @ toffoli_reference_unitary() @ h_target,
    )
