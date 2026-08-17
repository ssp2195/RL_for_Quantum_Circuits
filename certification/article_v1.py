"""Article-V1 dense certification for exact small-scale synthesis.

This module deliberately coexists with :mod:`certification.simulator`.  The
legacy simulator certifier uses a phase anchor followed by entrywise
``allclose``; the publication profile instead uses the trace/Frobenius
discrepancy specified by Article Eq. (129).  Both reconstruct candidates from
the authoritative DAG witness and neither consults symbolic or learned state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from certification.base import CertResult, CertStatus
from certification.base_engine import CertificationEngine
from certification.simulator import SynthesisTarget, unitary_from_gates
from certification.unitary_phase_metrics import (
    DEFAULT_UNITARITY_TOLERANCE,
    projective_unitary_metrics,
)


ARTICLE_V1_CERTIFICATION_SCHEMA = "phase-frobenius-raw-v2"
DEFAULT_TAU_CERT = 1e-6


def _nonnegative_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-negative real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be a finite non-negative real number"
        ) from error
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return normalized


@dataclass(frozen=True, slots=True)
class _ValidatedUnitary:
    matrix: np.ndarray
    dimension: int
    num_qubits: int
    finite: bool
    unitary: bool
    unitarity_error: float


def _validated_unitary(
    value: Any,
    *,
    name: str,
    unitarity_tolerance: float,
) -> _ValidatedUnitary:
    """Return a private immutable validation record or reject clearly."""

    try:
        matrix = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a complex matrix") from error

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    dimension = int(matrix.shape[0])
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError(f"{name} dimension must be a positive power of two")

    finite = bool(np.isfinite(matrix).all())
    if not finite:
        raise ValueError(f"{name} must contain only finite values")

    identity = np.eye(dimension, dtype=np.complex128)
    unitarity_error = float(
        np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro")
    )
    unitary = bool(unitarity_error <= unitarity_tolerance)
    if not unitary:
        raise ValueError(
            f"{name} must be unitary; Frobenius unitarity error "
            f"{unitarity_error:.17g} exceeds tolerance "
            f"{unitarity_tolerance:.17g}"
        )

    matrix.setflags(write=False)
    return _ValidatedUnitary(
        matrix=matrix,
        dimension=dimension,
        num_qubits=dimension.bit_length() - 1,
        finite=finite,
        unitary=unitary,
        unitarity_error=unitarity_error,
    )


@dataclass(frozen=True, slots=True)
class ArticleV1CertificationDiagnostics:
    """Immutable diagnostics containing only JSON-serializable values."""

    schema_version: str
    normalized_trace_magnitude: float
    process_fidelity: float
    process_infidelity: float
    phase_frobenius_discrepancy: float
    tau_cert: float
    passed: bool
    optimal_global_phase: tuple[float, float]
    phase_aligned_matrix_error: float
    phase_aligned_frobenius_norm: float
    maximum_phase_aligned_entry_error: float
    candidate_unitarity_error: float
    target_unitarity_error: float
    unitarity_tolerance: float
    candidate_dimension: int
    target_dimension: int
    candidate_num_qubits: int
    target_num_qubits: int
    candidate_finite: bool
    target_finite: bool
    candidate_unitary: bool
    target_unitary: bool

    @property
    def c_phi(self) -> float:
        return self.normalized_trace_magnitude

    @property
    def delta_phi(self) -> float:
        return self.phase_frobenius_discrepancy

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-ready mapping, including formula aliases."""

        payload = asdict(self)
        payload["optimal_global_phase"] = list(self.optimal_global_phase)
        payload["c_phi"] = self.c_phi
        payload["delta_phi"] = self.delta_phi
        return payload


def article_v1_certification_diagnostics(
    candidate: Any,
    target: Any,
    *,
    tau_cert: float = DEFAULT_TAU_CERT,
    unitarity_tolerance: float = DEFAULT_UNITARITY_TOLERANCE,
) -> ArticleV1CertificationDiagnostics:
    """Evaluate the Article-V1 trace/Frobenius certification predicate.

    Both matrices must be finite, square, same-dimensional, power-of-two
    unitaries.  Malformed inputs raise a clear ``TypeError`` or ``ValueError``;
    they are never converted into a successful numerical comparison.
    """

    tau = _nonnegative_finite_float(tau_cert, name="tau_cert")
    validation_tolerance = _nonnegative_finite_float(
        unitarity_tolerance,
        name="unitarity_tolerance",
    )
    metrics = projective_unitary_metrics(
        candidate, target, unitarity_tolerance=validation_tolerance
    )
    dimension = metrics.dimension
    c_phi = metrics.normalized_trace_magnitude
    delta_phi = metrics.phase_frobenius_discrepancy
    optimal_phase = metrics.optimal_global_phase
    frobenius_norm = metrics.phase_aligned_frobenius_norm
    normalized_matrix_error = float(frobenius_norm / math.sqrt(2.0 * dimension))

    return ArticleV1CertificationDiagnostics(
        schema_version=ARTICLE_V1_CERTIFICATION_SCHEMA,
        normalized_trace_magnitude=c_phi,
        process_fidelity=metrics.process_fidelity,
        process_infidelity=metrics.process_infidelity,
        phase_frobenius_discrepancy=delta_phi,
        tau_cert=tau,
        passed=bool(delta_phi <= tau),
        optimal_global_phase=(float(optimal_phase.real), float(optimal_phase.imag)),
        phase_aligned_matrix_error=normalized_matrix_error,
        phase_aligned_frobenius_norm=frobenius_norm,
        maximum_phase_aligned_entry_error=metrics.maximum_phase_aligned_entry_error,
        candidate_unitarity_error=metrics.candidate_unitarity_error,
        target_unitarity_error=metrics.target_unitarity_error,
        unitarity_tolerance=validation_tolerance,
        candidate_dimension=dimension,
        target_dimension=dimension,
        candidate_num_qubits=dimension.bit_length() - 1,
        target_num_qubits=dimension.bit_length() - 1,
        candidate_finite=True,
        target_finite=True,
        candidate_unitary=True,
        target_unitary=True,
    )


def phase_frobenius_discrepancy(candidate: Any, target: Any) -> float:
    """Return Article Eq. (129), with strict finite/unitary validation."""

    return article_v1_certification_diagnostics(
        candidate,
        target,
        # The tolerance cannot affect the discrepancy; zero makes that
        # separation explicit for callers using this metric directly.
        tau_cert=0.0,
    ).phase_frobenius_discrepancy


class ArticleV1CertificationEngine(CertificationEngine):
    """Independent DAG-replay certifier using Article Eq. (129) only."""

    schema_version = ARTICLE_V1_CERTIFICATION_SCHEMA

    def __init__(
        self,
        target: SynthesisTarget | np.ndarray | None = None,
        *,
        tau_cert: float = DEFAULT_TAU_CERT,
        unitarity_tolerance: float = DEFAULT_UNITARITY_TOLERANCE,
    ) -> None:
        self.tau_cert = _nonnegative_finite_float(tau_cert, name="tau_cert")
        self.unitarity_tolerance = _nonnegative_finite_float(
            unitarity_tolerance,
            name="unitarity_tolerance",
        )
        if isinstance(target, SynthesisTarget) and not target.quotient_global_phase:
            raise ValueError(
                "ArticleV1CertificationEngine supports global-phase quotienting only"
            )
        if target is None:
            self.target: SynthesisTarget | None = None
            return

        target_matrix = target.unitary if isinstance(target, SynthesisTarget) else target
        validated = _validated_unitary(
            target_matrix,
            name="target",
            unitarity_tolerance=self.unitarity_tolerance,
        )
        # Preserve the repository's exact-target interface so existing target
        # context factories can consume this engine without a compatibility
        # adapter.  Article V1 always quotients one global phase.
        self.target = SynthesisTarget(validated.matrix, quotient_global_phase=True)

    def certify(self, state: Any) -> CertResult:
        if self.target is None:
            return CertResult(
                CertStatus.INCONCLUSIVE,
                score=0.0,
                info={
                    "schema_version": self.schema_version,
                    "reason": "no_target_configured",
                    "tau_cert": self.tau_cert,
                    "passed": False,
                },
            )

        try:
            dag = state.dag
            candidate = unitary_from_gates(dag.num_qubits, dag.gates)
        except (AttributeError, TypeError, ValueError) as error:
            return CertResult(
                CertStatus.FAILURE,
                score=0.0,
                info={
                    "schema_version": self.schema_version,
                    "reason": "candidate_dag_reconstruction_rejected",
                    "error": str(error),
                    "tau_cert": self.tau_cert,
                    "passed": False,
                },
            )

        try:
            diagnostics = article_v1_certification_diagnostics(
                candidate,
                self.target.unitary,
                tau_cert=self.tau_cert,
                unitarity_tolerance=self.unitarity_tolerance,
            )
        except (TypeError, ValueError) as error:
            return CertResult(
                CertStatus.FAILURE,
                score=0.0,
                info={
                    "schema_version": self.schema_version,
                    "reason": "candidate_matrix_validation_rejected",
                    "error": str(error),
                    "tau_cert": self.tau_cert,
                    "passed": False,
                },
            )

        info = diagnostics.to_dict()
        info["reason"] = (
            "equivalent_phase_frobenius" if diagnostics.passed else "not_target"
        )
        return CertResult(
            CertStatus.SUCCESS if diagnostics.passed else CertStatus.INCONCLUSIVE,
            score=1.0 if diagnostics.passed else 0.0,
            info=info,
        )


__all__ = [
    "ARTICLE_V1_CERTIFICATION_SCHEMA",
    "DEFAULT_TAU_CERT",
    "DEFAULT_UNITARITY_TOLERANCE",
    "ArticleV1CertificationDiagnostics",
    "ArticleV1CertificationEngine",
    "article_v1_certification_diagnostics",
    "phase_frobenius_discrepancy",
]
