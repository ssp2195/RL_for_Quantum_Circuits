"""Exact M3 correction-stage construction, with all garbage already measured.

This diagnostic counts *additional correction ancilla qubits*. It does not
include the (possibly large) circuit required to create and measure the
original cubic garbage; it is not a full-oracle ancilla lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contract import times
from .contextual import LiveStage, MeasuredValue, causal_corrections, correction_rank, necessary_peak_ancillas


@dataclass(frozen=True)
class M3Result:
    q: int
    stages: int
    peak_correction_ancillas: int
    lower_bound_ancillas: int
    preparation_t_count: int
    conditional_cz_count: int
    ancilla_measurements: int
    causal_corrections_certified: bool


def construct_m3_corrections(q: int, stages: int) -> M3Result:
    """Explicit grouping of q quadratic factors into L correction stages."""
    if type(q) is not int or not 1 <= q <= 6 or type(stages) is not int or not 1 <= stages <= q:
        raise ValueError('M3 diagnostic admits 1 <= stages <= q <= 6')
    x = tuple(frozenset({1 << i}) for i in range(q))
    y = tuple(frozenset({1 << (q+i)}) for i in range(2*q))
    quadratic = tuple(times(y[2*j], y[2*j+1]) for j in range(q))
    baseline = LiveStage(x + y)
    byproducts = tuple(MeasuredValue(times(x[i], quadratic[j]), 0)
                       for j in range(q) for i in range(q))
    # Contiguous balanced blocks are exact predetermined construction choices,
    # not a learning score or heuristic omission of alternatives.
    groups = tuple(tuple(j for j in range(q) if j*stages//q == s)
                   for s in range(stages))
    contexts = tuple(LiveStage(baseline.values + tuple(quadratic[j] for j in group))
                     for group in groups)
    witness = causal_corrections(contexts, byproducts)
    if witness is None:
        raise AssertionError('M3 staged Clifford corrections did not span all byproducts')
    if correction_rank(baseline, byproducts) != q*q:
        raise AssertionError('M3 cubic monomials lost independence')
    peak = max(len(group) for group in groups)
    minimum = necessary_peak_ancillas(baseline, byproducts, stages)
    if minimum > peak:
        raise AssertionError('necessary ancilla-qubit bound contradicts construction')
    return M3Result(q, stages, peak, minimum, 4*q, q*q+q, q, True)
