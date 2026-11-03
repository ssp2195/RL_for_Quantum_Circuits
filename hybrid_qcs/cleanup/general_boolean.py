"""Exact general Boolean phase-oracle constructions in the existing protocol IR.

Each ANF monomial is compiled either by a four-T temporary-AND chain plus
a CZ phase, or by a shorter chain and an exact seven-T direct CCZ phase.
Choices and resource comparisons are exhaustive, not heuristically ranked.
All ancilla qubits are measured, corrected and reused between monomials.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product

from .contract import Hardware, Limits, Polynomial, digest
from .consumer import bits
from .ir import Instruction
from .phase_protocol import PhaseProtocol, verify_phase
from .causal_protocol import causalize_immediate_cleanup


@dataclass(frozen=True)
class BooleanPhaseProblem:
    name: str
    n: int
    target_polynomial: Polynomial
    limits: Limits = field(default_factory=Limits)
    hardware: Hardware = field(default_factory=Hardware)

    def __post_init__(self):
        if not self.name or type(self.n) is not int or not 1 <= self.n <= 64:
            raise ValueError("invalid Boolean phase problem")
        if not isinstance(self.target_polynomial, frozenset) or any(
            type(m) is not int or not 0 <= m < (1 << self.n) for m in self.target_polynomial
        ):
            raise ValueError("ANF monomials must be masks of logical input wires")

    @property
    def digest(self) -> str:
        return digest({'name': self.name, 'n': self.n,
                       'anf': sorted(self.target_polynomial),
                       'limits': asdict(self.limits),
                       'hardware': asdict(self.hardware)})

    @classmethod
    def from_truth_table(cls, name: str, table: tuple[int, ...], *,
                         limits: Limits | None = None) -> 'BooleanPhaseProblem':
        length = len(table)
        if length < 2 or length & (length - 1) or any(v not in (0, 1) for v in table):
            raise ValueError("truth table must contain 2^n Boolean entries")
        n = length.bit_length() - 1
        coefficients = list(table)
        for i in range(n):
            for mask in range(length):
                if mask & (1 << i):
                    coefficients[mask] ^= coefficients[mask ^ (1 << i)]
        return cls(name, n, frozenset(i for i, v in enumerate(coefficients) if v),
                   limits or Limits())


def _monomials(problem: BooleanPhaseProblem) -> tuple[int, ...]:
    return tuple(sorted(problem.target_polynomial, key=lambda m: (m.bit_count(), m)))


def _compile_term(mask: int, mode: str, n: int, outcome: int) -> tuple[Instruction, ...]:
    wires = bits(mask)
    degree = len(wires)
    if degree == 0:
        return (Instruction('MINUS', (0,), 'consumer'),)
    if degree == 1:
        return (Instruction('Z', wires, 'consumer'),)
    if degree == 2:
        return (Instruction('CZ', wires, 'consumer'),)
    if mode not in ('measured', 'direct'):
        raise ValueError("nonlinear terms require 'measured' or 'direct' mode")

    # Direct: CCZ on three available values, with d-3 ANDs (zero for d=3).
    # Measured: CZ on two available values, with d-2 ANDs.
    number_and = degree - (3 if mode == 'direct' else 2)
    ops = []
    for j in range(number_and):
        operands = (wires[0], wires[1]) if j == 0 else (n + j - 1, wires[j + 1])
        ops.append(Instruction('AND', (*operands, n + j), 'prepare'))
    if mode == 'direct':
        controls = wires if degree == 3 else (n + number_and - 1, wires[-2], wires[-1])
        ops.append(Instruction('CCZ', controls, 'consumer'))
    else:
        ops.append(Instruction('CZ', (n + number_and - 1, wires[-1]), 'consumer'))
    for j in reversed(range(number_and)):
        operands = (wires[0], wires[1]) if j == 0 else (n + j - 1, wires[j + 1])
        ops.append(Instruction('MX', (n + j,), f'measure_{outcome}_{j}', outcome=outcome))
        ops.append(Instruction('CZ', operands, f'correct_{outcome}_{j}', guard=outcome))
        outcome += 1
    return tuple(ops)


def compile_boolean_phase(
    problem: BooleanPhaseProblem,
    modes: tuple[str, ...] | None = None,
    *, m1_guided: bool = False,
) -> PhaseProtocol:
    """Build an exact complete oracle; modes index degree>=3 monomials."""
    terms = _monomials(problem)
    nonlinear = tuple(m for m in terms if m.bit_count() >= 3)
    if modes is None:
        modes = ('measured',) * len(nonlinear)
    if len(modes) != len(nonlinear) or any(m not in ('measured', 'direct') for m in modes):
        raise ValueError("one declared construction mode required for each nonlinear term")
    ops = []
    outcome = mode_index = 0
    peak = 0
    for term in terms:
        if term.bit_count() < 3:
            emitted = _compile_term(term, '', problem.n, outcome)
        else:
            mode = modes[mode_index]
            mode_index += 1
            peak = max(peak, term.bit_count() - (3 if mode == 'direct' else 2))
            emitted = _compile_term(term, mode, problem.n, outcome)
        ops.extend(emitted)
        outcome += sum(op.kind == 'MX' for op in emitted)
    if m1_guided:
        ops = list(causalize_immediate_cleanup(tuple(ops), problem.n))
    return PhaseProtocol(problem.digest, problem.n + peak, tuple(ops),
                         ('generic_m1_causal_' if m1_guided else 'generic_boolean_anf_exact_')
                         + '_'.join(modes))


def optimize_boolean_phase(
    problem: BooleanPhaseProblem,
    *,
    objective: tuple[str, ...] = ('peak_aux', 't_count', 'cnot', 'native_depth',
                                   'measurement_rounds'),
    max_nonlinear_terms: int = 12,
) -> dict:
    """Exact finite construction-menu optimizer, no guessed resource costs.

    Never reports global Clifford+T optimality. A cap on search size raises
    explicitly instead of silently skipping combinations.
    """
    allowed = ('peak_aux', 't_count', 'cnot', 'native_gates', 'native_depth',
               't_depth', 'measurement_rounds', 'worst_case_ticks')
    if not objective or len(objective) != len(set(objective)) or any(k not in allowed for k in objective):
        raise ValueError("invalid lexicographic optimization objective")
    degree = [m for m in _monomials(problem) if m.bit_count() >= 3]
    if len(degree) > max_nonlinear_terms:
        raise ValueError("exact construction menu exceeds declared nonlinear-term limit")
    candidates = []
    for modes in product(('measured', 'direct'), repeat=len(degree)):
        circuit = compile_boolean_phase(problem, modes)
        result = verify_phase(problem, circuit)
        if result['valid']:
            circuit.persistent_tail()
            candidates.append({'modes': modes, 'protocol': circuit,
                               'resources': result['resources']})
        elif result.get('reason') not in ('resource contract exceeded',):
            raise AssertionError("candidate failed exact independent verification: " + result['reason'])
    if not candidates:
        return {'status': 'infeasible_in_enumerated_menu', 'candidates': (),
                'global_oracle_optimality': False}
    def dominates(a, b):
        keys = allowed
        return (all(a['resources'][k] <= b['resources'][k] for k in keys) and
                any(a['resources'][k] < b['resources'][k] for k in keys))
    pareto = tuple(a for a in candidates if not any(dominates(b, a) for b in candidates))
    best = min(pareto, key=lambda row: (tuple(row['resources'][k] for k in objective), row['modes']))
    return {'status': 'certified_finite_menu_optimum', 'selected': best,
            'pareto': pareto, 'candidate_count': len(candidates),
            'global_oracle_optimality': False, 'objective': objective}
