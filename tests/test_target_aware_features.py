"""Focused dense target-context coverage for labelled GHZ-3 progress.

Feature-vector layout belongs to ``rl.features``.  These tests deliberately
exercise only the target-relative metric provider so it remains independently
usable by feature extraction and potential-based reward shaping.
"""

from __future__ import annotations

import numpy as np
import pytest

from certification.simulator import (
    SimulatorCertificationEngine,
    SynthesisTarget,
    unitary_from_gates,
)
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.target_context import DenseTargetContext


GHZ3_GATES = (
    Gate(GateType.H, (0,)),
    Gate(GateType.CNOT, (0, 1)),
    Gate(GateType.CNOT, (0, 2)),
)
GHZ3_BUDGET = ResourceBudget(
    max_t_count=0,
    max_depth=3,
    max_gates=3,
    max_two_qubit_count=2,
)


def _state(*gates: Gate) -> CircuitState:
    state = CircuitState(CircuitDAG(3), GHZ3_BUDGET)
    for gate in gates:
        assert state.apply_gate(gate)
    state.dag.validate()
    return state


def _context(*, quotient_global_phase: bool = True) -> DenseTargetContext:
    target = SynthesisTarget(
        unitary_from_gates(3, GHZ3_GATES),
        quotient_global_phase=quotient_global_phase,
    )
    return DenseTargetContext.from_synthesis_target(target)


def test_context_derives_from_certifier_and_exposes_immutable_target_data():
    target = SynthesisTarget(unitary_from_gates(3, GHZ3_GATES))
    context = DenseTargetContext.from_certification_engine(
        SimulatorCertificationEngine(target)
    )

    assert context.num_qubits == 3
    assert context.target is not target
    assert context.fingerprint.startswith("sha256:")
    assert context.schema_version == context.target_metrics_schema_version
    assert context.phase_mode == "quotient_global_phase"
    assert not context.target_unitary.flags.writeable
    assert not context.probe_state.flags.writeable
    assert not context.target_probe_state.flags.writeable
    with pytest.raises(ValueError):
        context.target_unitary[0, 0] = 0.0


def test_ghz_target_metrics_are_bounded_and_exact_for_the_full_witness():
    context = _context()
    metrics = context.metrics(_state(*GHZ3_GATES))

    assert metrics.target_fingerprint == context.fingerprint
    assert metrics.phase_mode == "quotient_global_phase"
    assert metrics.process_fidelity == pytest.approx(1.0, abs=1e-12)
    assert metrics.phase_aligned_frobenius_distance <= 1e-6
    assert metrics.probe_state_fidelity == pytest.approx(1.0, abs=1e-12)
    assert metrics.effective_support_size == pytest.approx(2.0, abs=1e-12)
    assert metrics.support_match == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(metrics.one_qubit_linear_entropies, (1.0, 1.0, 1.0))
    assert metrics.entanglement_match == pytest.approx(1.0, abs=1e-12)
    assert metrics.potential == pytest.approx(1.0, abs=1e-12)


def test_canonical_ghz_prefixes_have_strict_target_progress_ordering():
    context = _context()
    root = context.metrics(_state())
    hadamard = context.metrics(_state(GHZ3_GATES[0]))
    wrong_h1 = context.metrics(_state(Gate(GateType.H, (1,))))
    wrong_h2 = context.metrics(_state(Gate(GateType.H, (2,))))
    bell = context.metrics(_state(*GHZ3_GATES[:2]))
    ghz = context.metrics(_state(*GHZ3_GATES))

    assert root.potential < hadamard.potential < bell.potential < ghz.potential
    assert hadamard.process_fidelity > wrong_h1.process_fidelity
    assert hadamard.process_fidelity > wrong_h2.process_fidelity
    assert hadamard.potential > wrong_h1.potential
    assert hadamard.potential > wrong_h2.potential
    # The configured potential is a convex combination of all-one terminal
    # matches, so exact target synthesis reaches its maximum value.
    assert ghz.potential == pytest.approx(1.0, abs=1e-12)


def test_directed_cnot_prefixes_are_distinguished_by_target_progress():
    context = _context()
    h0 = GHZ3_GATES[0]
    correct_01 = context.metrics(_state(h0, Gate(GateType.CNOT, (0, 1))))
    correct_02 = context.metrics(_state(h0, Gate(GateType.CNOT, (0, 2))))
    reversed_10 = context.metrics(_state(h0, Gate(GateType.CNOT, (1, 0))))
    reversed_20 = context.metrics(_state(h0, Gate(GateType.CNOT, (2, 0))))

    assert correct_01.potential == pytest.approx(correct_02.potential, abs=1e-12)
    assert correct_01.potential > reversed_10.potential
    assert correct_02.potential > reversed_20.potential
    assert correct_01.process_fidelity > reversed_10.process_fidelity
    assert correct_02.process_fidelity > reversed_20.process_fidelity


def test_metrics_cache_is_per_context_and_includes_phase_mode_in_the_key():
    quotient_context = _context(quotient_global_phase=True)
    literal_context = _context(quotient_global_phase=False)
    root = _state()

    first = quotient_context.metrics(root)
    second = quotient_context.metrics(_state())
    literal = literal_context.metrics(root)

    # Identical DAG witnesses share an entry only inside the same context.
    assert first is second
    assert quotient_context.cache_size == 1
    assert literal_context.cache_size == 1
    quotient_key = quotient_context.cache_key(root)
    literal_key = literal_context.cache_key(root)
    assert quotient_key[0] == literal_key[0]
    assert quotient_key[1] != literal_key[1]
    assert first.phase_mode != literal.phase_mode
