"""Permutation-friendly features for frontier-record scheduling.

The learner deliberately receives a *candidate record* (plus optional
frontier context), never a gate action or a mutable frontier-array index.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from circuit.circuit_state import CircuitState


_EPS = 1e-8


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _rotation_statistics(state: CircuitState) -> tuple[float, float, float]:
    rotations = tuple(getattr(state, "rotations", ()))
    if not rotations:
        return 0.0, 0.0, 0.0

    n = max(1, state.dag.num_qubits)
    weights = np.asarray([rotation.axis.weight for rotation in rotations], dtype=float)

    anticommuting_pairs = 0
    for i, left in enumerate(rotations):
        for right in rotations[i + 1 :]:
            if not left.axis.commutes_with(right.axis):
                anticommuting_pairs += 1

    possible_pairs = max(1, len(rotations) * (len(rotations) - 1) // 2)
    return (
        _safe_div(len(rotations), max(1, state.budget.max_t_count)),
        float(np.mean(weights)) / n,
        _safe_div(anticommuting_pairs, possible_pairs),
    )


def _base_features(state: CircuitState) -> np.ndarray:
    budget = state.budget
    max_two_qubit = (
        budget.max_two_qubit_count
        if getattr(budget, "max_two_qubit_count", None) is not None
        else max(1, budget.max_gates)
    )
    t_count = float(getattr(state, "t_count", 0))
    two_qubit_count = float(getattr(state, "two_qubit_count", 0))
    gate_count = float(getattr(state, "num_gates", 0))
    depth = float(getattr(state, "depth", state.dag.depth()))
    wire_depths = np.asarray(getattr(state, "wire_depths", ()), dtype=float)
    if not len(wire_depths):
        wire_depths = np.zeros(state.dag.num_qubits, dtype=float)

    rotation_count, mean_pauli_weight, anticommutation_density = _rotation_statistics(state)
    depth_balance = float(np.std(wire_depths)) / max(1.0, float(budget.max_depth))
    two_qubit_ratio = _safe_div(two_qubit_count, max_two_qubit)

    return np.asarray(
        [
            _safe_div(t_count, budget.max_t_count),
            two_qubit_ratio,
            _safe_div(gate_count, budget.max_gates),
            _safe_div(depth, budget.max_depth),
            _safe_div(budget.max_t_count - t_count, budget.max_t_count),
            _safe_div(max_two_qubit - two_qubit_count, max_two_qubit),
            _safe_div(budget.max_gates - gate_count, budget.max_gates),
            _safe_div(budget.max_depth - depth, budget.max_depth),
            rotation_count,
            mean_pauli_weight,
            anticommutation_density,
            depth_balance,
        ],
        dtype=np.float32,
    )


def extract_features(
    state: CircuitState,
    frontier: Optional[Iterable[object]] = None,
) -> np.ndarray:
    """Return candidate features with optional frontier-relative context.

    ``frontier`` may contain either ``CircuitState`` values or search nodes.
    Context is invariant to the input ordering, which is crucial because
    frontier positions have no semantic identity.
    """
    base = _base_features(state)
    if frontier is None:
        return np.concatenate((base, np.zeros(4, dtype=np.float32)))

    states: list[CircuitState] = []
    for item in frontier:
        candidate = getattr(item, "state", item)
        if isinstance(candidate, CircuitState):
            states.append(candidate)

    if not states:
        return np.concatenate((base, np.zeros(4, dtype=np.float32)))

    all_base = np.vstack([_base_features(candidate) for candidate in states])
    mean = np.mean(all_base, axis=0)
    std = np.std(all_base, axis=0)
    z = (base - mean) / (std + _EPS)

    # Compact shared-context signal: resource pressure, rotation complexity,
    # and Pareto-relevant depth imbalance relative to the active frontier.
    context = np.asarray([z[0], z[1], z[8], z[11]], dtype=np.float32)
    return np.concatenate((base, context))


def feature_dimension(state: CircuitState | None = None) -> int:
    """Derive (rather than duplicate) the policy input dimension."""
    if state is None:
        # Base features plus four context coordinates; keeping this expression
        # tied to the extractor makes changes fail loudly in callers/tests.
        return 12 + 4
    return int(extract_features(state).shape[0])
