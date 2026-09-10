"""Native hybrid-frontier contracts, transitions and independent certification.

Every search prefix is HybridState=(persistent DAG, Clifford tableau, ordered
Pauli rotations, phase, resources). A numerical isometry is an optional scoring
cache, never a semantic key. No target phase obligations restrict the grammar.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from functools import cached_property
import hashlib
import json
import numpy as np
from .ancilla_contract import AncillaContract, PhaseMode
from .ancilla_search import apply_gate_to_isometry
from .certify import gate_matrix
from .model import Budget, Gate, HybridState, generate_gates

SCHEMA = 'native-hybrid-frontier-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, eq=False)
class NativeProblem:
    name: str
    contract: AncillaContract
    budget: Budget
    unitary: np.ndarray
    family: str = 'native-unitary'
    split: str = 'test'
    max_t_depth: int | None = None
    tolerance: float = 1e-9

    def __post_init__(self):
        if self.contract.total_qubits > 8:
            raise ValueError('dense qualification cache limited to eight physical qubits')
        u = np.array(self.unitary, dtype=np.complex128, copy=True)
        d = self.contract.logical_dimension
        if u.shape != (d, d) or not np.isfinite(u).all():
            raise ValueError('target must be a finite logical square matrix')
        if not np.allclose(u.conj().T @ u, np.eye(d), atol=1e-10, rtol=0):
            raise ValueError('target matrix must be unitary')
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance <= 1e-5:
            raise ValueError('invalid terminal comparison tolerance')
        td = self.budget.max_t_count if self.max_t_depth is None else self.max_t_depth
        if type(td) is not int or td < 0:
            raise ValueError('max_t_depth must be a nonnegative integer')
        object.__setattr__(self, 'max_t_depth', td)
        u.setflags(write=False)
        object.__setattr__(self, 'unitary', u)

    @cached_property
    def target_isometry(self):
        return self.contract.target_isometry(self.unitary)

    @property
    def width(self):
        return self.contract.total_qubits

    @cached_property
    def actions(self):
        return generate_gates(self.width)

    def manifest(self):
        return {'schema': SCHEMA, 'contract': self.contract.canonical_payload(),
                'budget': asdict(self.budget), 'max_t_depth': self.max_t_depth,
                'unitary_real': self.unitary.real.tolist(), 'unitary_imag': self.unitary.imag.tolist(),
                'tolerance': self.tolerance, 'grammar': [[g.name, list(g.qubits)] for g in self.actions],
                'frontier': ['persistent_DAG', 'Clifford_tableau', 'ordered_Pauli_rotations',
                             'global_phase_pi_over_8', 'consumed_resources']}

    @cached_property
    def digest(self):
        return digest(self.manifest())

    def cap(self, objective, value):
        fields = {'t_count': 'max_t_count', 'cnot': 'max_cnot_count',
                  'gates': 'max_gates', 'depth': 'max_depth'}
        if objective == 't_depth':
            return replace(self, max_t_depth=value)
        if objective not in fields:
            raise ValueError('unsupported native objective')
        return replace(self, budget=replace(self.budget, **{fields[objective]: value}))


def phase_target(name, n, coefficients, ancillas=0, *, constant=0, budget=None, max_t_depth=None):
    """Convert a phase specification to a logical matrix ONLY; no search obligations."""
    exponents = [(constant + sum(int(c) * ((int(m) & x).bit_count() % 2)
                  for m, c in coefficients)) % 8 for x in range(1 << n)]
    contract = AncillaContract(n + ancillas, tuple(range(n)), tuple(range(n, n + ancillas)),
                              phase_mode=PhaseMode.EXACT)
    return NativeProblem(name, contract, budget or Budget(16, 24, 64, 48),
                         np.diag(np.exp(1j * np.pi * np.array(exponents) / 4)),
                         'phase-oracle', max_t_depth=max_t_depth)


def archive_key(p, state):
    # Conservatively compare full physical operators, not different-width
    # isometries or approximate hashes. Equal full operators imply equal UJ.
    key = state.exact_key if p.contract.phase_mode is PhaseMode.EXACT else state.canonical_key
    return p.contract.canonical_payload(), key


def next_t_depths(values, gate):
    out = list(values)
    level = max(out[q] for q in gate.qubits) + int(gate.is_non_clifford)
    for q in gate.qubits:
        out[q] = level
    return tuple(out)


def resources(state, t_depths):
    return (*state.resource_vector(), *t_depths)


def legal(p, state, t_depths, gate):
    b = p.budget
    return (state.gate_count < b.max_gates
            and state.t_count + int(gate.is_non_clifford) <= b.max_t_count
            and state.cnot_count + int(gate.is_two_qubit) <= b.max_cnot_count
            and 1 + max(state.wire_depths[q] for q in gate.qubits) <= b.max_depth
            and max(next_t_depths(t_depths, gate), default=0) <= p.max_t_depth)


def distance(p, isometry):
    inner = np.vdot(p.target_isometry, isometry) / p.contract.domain_dimension
    match = abs(inner) if p.contract.phase_mode is PhaseMode.PROJECTIVE else inner.real
    return float(np.sqrt(max(0., 1. - min(1., match))))


def matrix_error(p, actual):
    expected = p.target_isometry
    phase = 1. + 0j
    if p.contract.phase_mode is PhaseMode.PROJECTIVE:
        overlap = np.vdot(expected, actual)
        if abs(overlap):
            phase = overlap / abs(overlap)
    return float(np.max(np.abs(actual - phase * expected)))


def certify_native(p, state, *, provenance='native_frontier_discovery'):
    """Reconstruct the persistent DAG; independently replay matrices and resources.

    General floating-point targets use an explicitly declared numerical
    isometry predicate. This witness certifies feasibility to tolerance, not
    optimality or exact algebraic equality to an arbitrary numerical matrix.
    """
    if not isinstance(state, HybridState) or state.num_qubits != p.width:
        raise ValueError('certifier requires a native HybridState of the declared width')
    dag = state.materialize_dag()
    replay = HybridState.identity(p.width, p.budget)
    actual = np.array(p.contract.input_embedding, copy=True)
    td = (0,) * p.width
    feasible = True
    for g in dag.gates:
        feasible &= legal(p, replay, td, g)
        child = replay.apply(g, partial_order_reduction=False)
        if child is None:
            feasible = False
            break
        replay = child
        td = next_t_depths(td, g)
        actual = gate_matrix(p.width, g) @ actual
    equality = (replay.exact_key == state.exact_key
                and replay.clifford_lift == state.clifford_lift
                and replay.resource_vector() == state.resource_vector())
    error = matrix_error(p, actual)
    invalid = p.contract.invalid_clean_output_rows
    leakage = float(np.sum(np.abs(actual[invalid]) ** 2) / p.contract.domain_dimension)
    used = {q for g in dag.gates for q in g.qubits if q in p.contract.clean_ancillas}
    return {'schema': SCHEMA, 'problem_digest': p.digest, 'source': provenance,
            'success': bool(feasible and equality and error <= p.tolerance and leakage <= p.tolerance ** 2),
            'phase_mode': p.contract.phase_mode.value, 'isometry_error': error,
            'workspace_leakage': leakage, 'exact_symbolic_replay': bool(equality),
            'dag_validated': True, 'global_phase_eighths': state.global_phase_eighths,
            'native': [[g.name, list(g.qubits)] for g in dag.gates],
            'resources': {'t_count': state.t_count, 'cnot': state.cnot_count,
                          'gates': state.gate_count, 'depth': state.depth,
                          'wire_depths': list(state.wire_depths), 't_depth': max(td),
                          't_wire_depths': list(td),
                          'available_ancillas': len(p.contract.clean_ancillas), 'used_ancillas': len(used)},
            'optimality': 'not_established', 'terminal_predicate': 'independent_numerical_isometry_to_tolerance'}
