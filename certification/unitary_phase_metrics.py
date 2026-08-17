"""Shared raw projective metrics for finite dense unitary matrices."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


PROJECTIVE_UNITARY_METRICS_SCHEMA = "projective-unitary-metrics-v2"
DEFAULT_UNITARITY_TOLERANCE = 1e-9


def _tolerance(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("unitarity_tolerance must be a finite non-negative real")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("unitarity_tolerance must be a finite non-negative real")
    return result


def _validated_matrix(value: Any, *, name: str, tolerance: float) -> tuple[np.ndarray, float]:
    try:
        matrix = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a complex matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    dimension = int(matrix.shape[0])
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError(f"{name} dimension must be a positive power of two")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    identity = np.eye(dimension, dtype=np.complex128)
    error = float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro"))
    if error > tolerance:
        raise ValueError(
            f"{name} must be unitary; Frobenius unitarity error {error:.17g} "
            f"exceeds tolerance {tolerance:.17g}"
        )
    matrix.setflags(write=False)
    return matrix, error


@dataclass(frozen=True, slots=True)
class ProjectiveUnitaryMetrics:
    schema_version: str
    dimension: int
    trace_overlap: complex
    normalized_trace_magnitude_raw: float
    normalized_trace_magnitude: float
    process_fidelity: float
    process_infidelity: float
    phase_frobenius_discrepancy: float
    optimal_global_phase: complex
    phase_aligned_frobenius_norm: float
    maximum_phase_aligned_entry_error: float
    candidate_unitarity_error: float
    target_unitarity_error: float


def projective_unitary_metrics(
    candidate: Any,
    target: Any,
    *,
    unitarity_tolerance: float = DEFAULT_UNITARITY_TOLERANCE,
) -> ProjectiveUnitaryMetrics:
    """Compute projective metrics without rescaling or projecting either input."""

    tolerance = _tolerance(unitarity_tolerance)
    candidate_matrix, candidate_error = _validated_matrix(
        candidate, name="candidate", tolerance=tolerance
    )
    target_matrix, target_error = _validated_matrix(
        target, name="target", tolerance=tolerance
    )
    if candidate_matrix.shape != target_matrix.shape:
        raise ValueError("candidate and target dimensions must match")
    dimension = int(candidate_matrix.shape[0])
    overlap = complex(np.trace(target_matrix.conj().T @ candidate_matrix))
    raw = float(abs(overlap) / dimension)
    clipped = min(1.0, max(0.0, raw))
    fidelity = min(1.0, max(0.0, clipped * clipped))
    discrepancy = math.sqrt(max(0.0, 1.0 - clipped))
    phase = 1.0 + 0.0j if abs(overlap) == 0.0 else complex(overlap / abs(overlap))
    aligned = candidate_matrix - phase * target_matrix
    return ProjectiveUnitaryMetrics(
        schema_version=PROJECTIVE_UNITARY_METRICS_SCHEMA,
        dimension=dimension,
        trace_overlap=overlap,
        normalized_trace_magnitude_raw=raw,
        normalized_trace_magnitude=clipped,
        process_fidelity=fidelity,
        process_infidelity=min(1.0, max(0.0, 1.0 - fidelity)),
        phase_frobenius_discrepancy=discrepancy,
        optimal_global_phase=phase,
        phase_aligned_frobenius_norm=float(np.linalg.norm(aligned, ord="fro")),
        maximum_phase_aligned_entry_error=float(np.max(np.abs(aligned))),
        candidate_unitarity_error=candidate_error,
        target_unitarity_error=target_error,
    )


def phase_frobenius_discrepancy(
    candidate: Any,
    target: Any,
    *,
    unitarity_tolerance: float = DEFAULT_UNITARITY_TOLERANCE,
) -> float:
    return projective_unitary_metrics(
        candidate, target, unitarity_tolerance=unitarity_tolerance
    ).phase_frobenius_discrepancy


__all__ = [
    "DEFAULT_UNITARITY_TOLERANCE",
    "PROJECTIVE_UNITARY_METRICS_SCHEMA",
    "ProjectiveUnitaryMetrics",
    "phase_frobenius_discrepancy",
    "projective_unitary_metrics",
]
