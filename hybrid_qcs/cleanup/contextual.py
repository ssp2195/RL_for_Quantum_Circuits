"""Exact causal Clifford-correction analysis for general Boolean temporary values.

Boolean functions are square-free ANF polynomials from cleanup.contract.
This module establishes correction feasibility and necessary ancilla-qubit bounds
for a declared sequence of live contexts. It does not claim global oracle
optimality or charge zero for preparing the contexts.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .contract import Polynomial, plus, times


@dataclass(frozen=True)
class LiveStage:
    """Values held on live wires during one correction stage, in wire order."""
    values: tuple[Polynomial, ...]

    def __post_init__(self):
        if not all(isinstance(p, frozenset) and all(type(m) is int and m >= 0 for m in p)
                   for p in self.values):
            raise ValueError("live values must be exact ANF polynomials")


@dataclass(frozen=True)
class MeasuredValue:
    """Garbage value and first stage where its classical measurement is known."""
    polynomial: Polynomial
    first_stage: int


@dataclass(frozen=True)
class PhaseGate:
    """One stage-local Z or CZ correction, conditioned on a measurement bit."""
    stage: int
    measurement: int
    kind: str
    wires: tuple[int, ...]


def _vector(poly: Polynomial, monomials: dict[int, int]) -> int:
    result = 0
    for mask in poly:
        result ^= 1 << monomials[mask]
    return result


def _basis(columns: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    """Exact GF(2) elimination; each basis vector retains its source mask."""
    pivots: dict[int, tuple[int, int]] = {}
    for index, vector in enumerate(columns):
        coeff = 1 << index
        while vector:
            pivot = vector.bit_length() - 1
            if pivot in pivots:
                old, old_coeff = pivots[pivot]
                vector ^= old
                coeff ^= old_coeff
            else:
                pivots[pivot] = (vector, coeff)
                break
    return pivots


def _solve(pivots: dict[int, tuple[int, int]], vector: int) -> int | None:
    coeff = 0
    while vector:
        pivot = vector.bit_length() - 1
        if pivot not in pivots:
            return None
        old, old_coeff = pivots[pivot]
        vector ^= old
        coeff ^= old_coeff
    return coeff


def rank(polynomials: tuple[Polynomial, ...]) -> int:
    """Dimension of the span of exact Boolean functions."""
    monomials = {m: i for i, m in enumerate(sorted(set().union(*polynomials)))}
    return len(_basis(tuple(_vector(p, monomials) for p in polynomials)))


def phase_generators(stage: LiveStage) -> tuple[tuple[str, tuple[int, ...], Polynomial], ...]:
    """All wire-local phase generators, including a constant global sign."""
    values = stage.values
    result = [('MINUS', (), frozenset({0}))]
    result.extend(('Z', (i,), v) for i, v in enumerate(values))
    result.extend(('CZ', (i, j), times(values[i], values[j]))
                  for i, j in combinations(range(len(values)), 2))
    return tuple(result)


def causal_corrections(
    stages: tuple[LiveStage, ...], measured: tuple[MeasuredValue, ...],
) -> tuple[PhaseGate, ...] | None:
    """Exact M1 test and witness; None means no permitted causal correction.

    Constants are irrelevant branch-global signs. They are omitted from emitted
    gates, but the linear system retains them when deciding feasibility.
    """
    if not stages:
        if measured:
            raise ValueError("measurements require a correction stage")
        return ()
    for item in measured:
        if not 0 <= item.first_stage < len(stages):
            raise ValueError("measurement availability is outside the stages")
    generators = tuple(phase_generators(s) for s in stages)
    gates = []
    for bit, item in enumerate(measured):
        options = tuple((j, kind, wires, poly)
                        for j in range(item.first_stage, len(stages))
                        for kind, wires, poly in generators[j])
        polys = tuple(poly for _, _, _, poly in options)
        monomials = {m: i for i, m in enumerate(
            sorted(set().union(item.polynomial, *polys)))}
        columns = tuple(_vector(poly, monomials) for poly in polys)
        solution = _solve(_basis(columns), _vector(item.polynomial, monomials))
        if solution is None:
            return None
        for index, (stage, kind, wires, _) in enumerate(options):
            if solution >> index & 1 and kind != 'MINUS':
                gates.append(PhaseGate(stage, bit, kind, wires))
    return tuple(sorted(gates, key=lambda op: (op.stage, op.measurement, op.kind, op.wires)))


def correction_rank(
    baseline: LiveStage, measured: tuple[MeasuredValue, ...], first_stage: int = 0,
) -> int:
    """Rank of future byproducts modulo phases from fixed original wires."""
    base = tuple(p for _, _, p in phase_generators(baseline))
    future = tuple(m.polynomial for m in measured if m.first_stage >= first_stage)
    return rank(base + future) - rank(base)


def necessary_peak_ancillas(
    baseline: LiveStage, measured: tuple[MeasuredValue, ...], stages: int,
) -> int:
    """M2 necessary peak bound, NOT a sufficient feasibility criterion.

    Counts added live ancilla qubits, excluding the fixed baseline wires and
    all ancilla qubits used before these correction stages.
    """
    if stages <= 0:
        raise ValueError("positive number of correction stages required")
    if any(not 0 <= m.first_stage < stages for m in measured):
        raise ValueError("invalid measurement availability")
    t = len(baseline.values)
    answer = 0
    for ell in range(stages):
        required = correction_rank(baseline, measured, ell)
        remaining = stages - ell
        ancillas = 0
        while remaining * (ancillas * (t + 1) + ancillas * (ancillas - 1) // 2) < required:
            ancillas += 1
        answer = max(answer, ancillas)
    return answer
