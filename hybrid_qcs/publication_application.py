"""Nonlearned compute--phase--uncompute reference for the BNN application.

This reference is deliberately separate from policy training and discovery. It
is built from the declared truth function, never inserted into a search frontier.
Its resource improvement relative to direct phase synthesis is a representation
comparison, not a measurement of the effect of learning.
"""
from __future__ import annotations

import time
import numpy as np
from .certify import unitary_from_gates
from .model import Budget, Gate, HybridState
from .oracle_synthesis import OracleLayout, oracle_macro_library, inverse_native_gates
from .publication_corpus import bnn_truth


def evaluator_phase_reference():
    """Certify parity(x) XOR x0*x1*x2 with one flag and one work wire."""
    start, cpu = time.perf_counter(), time.process_time()
    layout = OracleLayout.standard(3, 1)
    library = {(m.family, m.qubits): m for m in oracle_macro_library(layout)}
    specification = [('CNOT', (q, 3)) for q in range(3)] + [
        ('TOFFOLI', (0, 1, 4)), ('TOFFOLI', (2, 4, 3)),
        ('TOFFOLI', (0, 1, 4))]
    evaluator = tuple(g for key in specification for g in library[key].native_gates)
    native = evaluator + (Gate('S', (3,)), Gate('S', (3,))) + inverse_native_gates(evaluator)
    u_eval = unitary_from_gates(5, evaluator)
    # Both values of the coherent output bit are checked, not only y=0.
    expected_eval = np.zeros((32, 16), dtype=complex)
    truth = bnn_truth()
    for basis in range(16):
        expected_eval[basis ^ (truth[basis & 7] << 3), basis] = 1
    eval_error = float(np.max(np.abs(u_eval[:, :16] - expected_eval)))
    u_oracle = unitary_from_gates(5, native)
    expected = np.zeros((32, 8), dtype=complex)
    expected[np.arange(8), np.arange(8)] = [(-1) ** v for v in truth]
    error = float(np.max(np.abs(u_oracle[:, :8] - expected)))
    leakage = float(np.sum(np.abs(u_oracle[8:, :8]) ** 2) / 8)
    budget = Budget(100, 200, 300, 300)
    state = HybridState.identity(5, budget)
    t_depths = [0] * 5
    for gate in native:
        state = state.apply(gate, partial_order_reduction=False)
        if state is None:
            raise AssertionError('reference exceeded its explicit native budget')
        layer = max(t_depths[q] for q in gate.qubits) + int(gate.name in ('T', 'TDG'))
        for q in gate.qubits:
            t_depths[q] = layer
    state.validate()
    dag = state.materialize_dag()
    if eval_error > 1e-9 or error > 1e-9 or leakage > 1e-20:
        raise AssertionError('application reference failed coherent certification')
    return {
        'source': 'deterministic_ANF_evaluator_reference',
        'success': True, 'truth_table': truth,
        'algebraic_function': 'x0 XOR x1 XOR x2 XOR (x0 AND x1 AND x2)',
        'macro_witness': [[name, list(qs)] for name, qs in specification],
        'native': [[g.name, list(g.qubits)] for g in native],
        'evaluator_both_flag_values_error': eval_error,
        'native_isometry_error': error, 'workspace_leakage': leakage,
        'dag_validated': len(dag.gates) == len(native),
        'resources': {'t_count': state.t_count, 'cnot': state.cnot_count,
                      'depth': state.depth, 'gates': state.gate_count,
                      't_depth': max(t_depths), 'auxiliary_qubits': 2,
                      'temporary_flag_qubits': 1, 'clean_scratch_qubits': 1},
        'wall_seconds': time.perf_counter() - start,
        'cpu_seconds': time.process_time() - cpu,
        'claim': 'fixed native lowering reference; not a learned or optimal evaluator',
        'comparison_scope': 'direct-phase versus evaluator representation, not an RL ablation',
    }
