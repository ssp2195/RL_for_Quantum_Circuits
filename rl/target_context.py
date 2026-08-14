"""Target-relative dense metrics for small frontier-ranking experiments.

The search engine remains authoritative for legality, canonicalisation, and
terminal certification.  This module only supplies bounded, target-relative
observations for a learner and potential-based reward diagnostics.  It is
therefore deliberately small-instance only: dense simulation is useful for
the labelled GHZ-3 experiment, but should not silently be used for larger
searches.

Qubit ``0`` is the least-significant computational-basis bit, matching
``certification.simulator`` and the circuit DAG.  The default probe is
``|0...0>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import exp, isclose, log, sqrt
from typing import Any, Literal, TypeAlias

import numpy as np

from certification.simulator import (
    SynthesisTarget,
    unitary_from_gates,
)
from circuit.circuit_state import CircuitState


DEFAULT_MAX_DENSE_QUBITS = 3
"""Largest register supported by the target-aware linear GHZ experiment."""

DENSE_TARGET_CONTEXT_SCHEMA_VERSION = "dense-target-context-v1"
"""Stable identifier for target-metric consumers and saved policy metadata."""

PhaseMode: TypeAlias = Literal["quotient_global_phase", "literal_phase"]
_ROUND_OFF_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class TargetProgressWeights:
    """Convex weights for the bounded target-progress potential.

    Keeping the weights together makes the shaping/feature configuration
    inspectable rather than scattering GHZ-specific constants across the
    environment and learner.  Requiring a convex combination preserves the
    potential's ``[0, 1]`` range.
    """

    process_fidelity: float = 0.60
    support_match: float = 0.15
    entanglement_match: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.process_fidelity,
            self.support_match,
            self.entanglement_match,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("target-progress weights must be finite and non-negative")
        total = float(sum(values))
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "target-progress weights must sum to 1.0 to keep the potential bounded"
            )

    @property
    def total(self) -> float:
        """Return the validated sum of the three potential weights."""

        return float(
            self.process_fidelity + self.support_match + self.entanglement_match
        )


@dataclass(frozen=True, slots=True)
class TargetMetrics:
    """Cached bounded diagnostics for one authoritative DAG witness.

    ``candidate_fingerprint`` describes the full DAG witness, never a frontier
    record ID.  That choice is intentionally conservative: semantically
    identical witnesses can be simulated more than once, but phase-distinct
    witnesses cannot be conflated by a cache shortcut.
    """

    target_fingerprint: str
    candidate_fingerprint: str
    phase_mode: PhaseMode
    process_fidelity: float
    phase_aligned_frobenius_distance: float
    probe_state_fidelity: float
    effective_support_size: float
    support_match: float
    one_qubit_linear_entropies: tuple[float, ...]
    entanglement_match: float
    potential: float

    @property
    def linear_entropies(self) -> tuple[float, ...]:
        """Short alias useful to feature extractors and reporting code."""

        return self.one_qubit_linear_entropies


WitnessIdentity: TypeAlias = tuple[int, tuple[tuple[str, tuple[int, ...]], ...]]
MetricCacheKey: TypeAlias = tuple[str, PhaseMode, WitnessIdentity]


def _readonly_complex_array(value: Any, *, name: str) -> np.ndarray:
    """Return a finite, one-dimensional complex array which callers cannot edit."""

    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional statevector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite amplitudes")
    copied = np.array(array, dtype=np.complex128, copy=True)
    copied.setflags(write=False)
    return copied


def _normalized_probe_state(value: Any, dimension: int) -> np.ndarray:
    """Validate a user probe or build the native ``|0...0>`` probe."""

    if value is None:
        probe = np.zeros(dimension, dtype=np.complex128)
        probe[0] = 1.0
    else:
        probe = _readonly_complex_array(value, name="probe_state")
        if probe.shape != (dimension,):
            raise ValueError(
                f"probe_state must have shape ({dimension},), got {probe.shape!r}"
            )

    norm = float(np.linalg.norm(probe))
    if not np.isclose(norm, 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError("probe_state must be normalized")
    readonly = np.array(probe, dtype=np.complex128, copy=True)
    readonly.setflags(write=False)
    return readonly


def _phase_mode(quotient_global_phase: bool) -> PhaseMode:
    return "quotient_global_phase" if quotient_global_phase else "literal_phase"


def _matrix_fingerprint(unitary: np.ndarray) -> str:
    """Return a stable digest of a target matrix, independent of object identity."""

    contiguous = np.ascontiguousarray(unitary, dtype=np.complex128)
    digest = sha256()
    digest.update(DENSE_TARGET_CONTEXT_SCHEMA_VERSION.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _gate_witness_identity(state: CircuitState) -> WitnessIdentity:
    """Return a complete, authoritative DAG-witness cache identity.

    The cache does not use record IDs, priorities, feature vectors, or a
    symbolic-only summary.  The immutable tuple of ordered DAG operations is
    derived directly from the same witness passed to dense simulation.
    """

    if not isinstance(state, CircuitState):
        raise TypeError("target metrics require a CircuitState")
    try:
        gates = state.dag.gates
        num_qubits = int(state.dag.num_qubits)
    except AttributeError as exc:  # pragma: no cover - CircuitState contract
        raise TypeError("target metrics require a state with a circuit DAG") from exc

    operations: list[tuple[str, tuple[int, ...]]] = []
    for gate in gates:
        gate_type = getattr(gate, "gate_type", None)
        name = getattr(gate_type, "name", gate_type)
        if not isinstance(name, str):
            raise TypeError(f"DAG gate has no stable gate name: {gate!r}")
        qubits = tuple(getattr(gate, "qubits", ()))
        if any(isinstance(qubit, bool) or not isinstance(qubit, int) for qubit in qubits):
            raise TypeError(f"DAG gate has invalid qubit labels: {gate!r}")
        operations.append((name.upper(), qubits))
    return num_qubits, tuple(operations)


def _witness_fingerprint(identity: WitnessIdentity) -> str:
    digest = sha256()
    digest.update(repr(identity).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _bounded_unit_interval(value: float, *, name: str) -> float:
    """Clamp only harmless floating-point overshoot of a physical metric."""

    if not np.isfinite(value):
        raise ArithmeticError(f"{name} is not finite")
    if value < -_ROUND_OFF_TOLERANCE or value > 1.0 + _ROUND_OFF_TOLERANCE:
        raise ArithmeticError(f"{name}={value!r} is outside its physical [0, 1] range")
    return min(1.0, max(0.0, float(value)))


def _one_qubit_linear_entropies(
    statevector: np.ndarray,
    num_qubits: int,
) -> tuple[float, ...]:
    """Return ``2 * (1 - Tr(rho_i^2))`` in native least-significant order."""

    if num_qubits == 0:
        return ()

    # In a C-order reshape, axis n - 1 - q corresponds to the q-th
    # least-significant bit of a computational-basis index.
    tensor = np.asarray(statevector, dtype=np.complex128).reshape((2,) * num_qubits)
    entropies: list[float] = []
    for qubit in range(num_qubits):
        axis = num_qubits - 1 - qubit
        amplitudes = np.moveaxis(tensor, axis, 0).reshape(2, -1)
        reduced_density = amplitudes @ amplitudes.conj().T
        purity = float(np.real(np.trace(reduced_density @ reduced_density)))
        entropy = 2.0 * (1.0 - purity)
        entropies.append(_bounded_unit_interval(entropy, name=f"linear entropy q{qubit}"))
    return tuple(entropies)


def _effective_support_size(statevector: np.ndarray) -> float:
    probabilities = np.abs(statevector) ** 2
    inverse_participation = float(np.sum(probabilities**2))
    if not np.isfinite(inverse_participation) or inverse_participation <= 0.0:
        raise ArithmeticError("probe state has invalid inverse participation ratio")
    return float(1.0 / inverse_participation)


@dataclass(frozen=True, slots=True, init=False)
class DenseTargetContext:
    """Immutable target-relative metric provider for a small dense target.

    The public arrays are defensive, read-only copies.  The only mutable
    object is a private per-context memoization dictionary; it is an execution
    optimisation and never changes target semantics or learner-visible state.
    """

    target: SynthesisTarget
    target_unitary: np.ndarray
    num_qubits: int
    probe_state: np.ndarray
    target_probe_state: np.ndarray
    fingerprint: str
    phase_mode: PhaseMode
    weights: TargetProgressWeights
    target_effective_support_size: float
    target_one_qubit_linear_entropies: tuple[float, ...]
    _cache: dict[MetricCacheKey, TargetMetrics] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        target: SynthesisTarget | np.ndarray,
        *,
        probe_state: Any | None = None,
        weights: TargetProgressWeights | None = None,
        potential_weights: TargetProgressWeights | None = None,
        max_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
        quotient_global_phase: bool | None = None,
    ) -> None:
        """Build a context from an exact synthesis target or a dense unitary.

        ``potential_weights`` is retained as an explicit spelling for callers
        configuring reward shaping; it is mutually exclusive with ``weights``.
        A raw matrix defaults to global-phase-quotient semantics, matching
        :class:`SynthesisTarget`.
        """

        if isinstance(max_qubits, bool) or not isinstance(max_qubits, int) or max_qubits < 0:
            raise ValueError("max_qubits must be a non-negative integer")
        if weights is not None and potential_weights is not None:
            raise ValueError("pass either weights or potential_weights, not both")
        configured_weights = weights or potential_weights or TargetProgressWeights()
        if not isinstance(configured_weights, TargetProgressWeights):
            raise TypeError("weights must be a TargetProgressWeights instance")

        if isinstance(target, SynthesisTarget):
            if (
                quotient_global_phase is not None
                and quotient_global_phase != target.quotient_global_phase
            ):
                raise ValueError(
                    "quotient_global_phase conflicts with the supplied SynthesisTarget"
                )
            canonical_target = SynthesisTarget(
                target.unitary,
                quotient_global_phase=target.quotient_global_phase,
            )
        else:
            phase_quotient = True if quotient_global_phase is None else quotient_global_phase
            if not isinstance(phase_quotient, bool):
                raise TypeError("quotient_global_phase must be a bool")
            canonical_target = SynthesisTarget(
                np.asarray(target, dtype=np.complex128),
                quotient_global_phase=phase_quotient,
            )

        num_qubits = canonical_target.num_qubits
        if num_qubits > max_qubits:
            raise ValueError(
                "DenseTargetContext supports at most "
                f"{max_qubits} qubits; received a {num_qubits}-qubit target"
            )
        target_unitary = np.array(canonical_target.unitary, dtype=np.complex128, copy=True)
        target_unitary.setflags(write=False)
        probe = _normalized_probe_state(probe_state, target_unitary.shape[0])
        target_probe = np.array(target_unitary @ probe, dtype=np.complex128, copy=True)
        target_probe.setflags(write=False)

        object.__setattr__(self, "target", canonical_target)
        object.__setattr__(self, "target_unitary", target_unitary)
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "probe_state", probe)
        object.__setattr__(self, "target_probe_state", target_probe)
        object.__setattr__(self, "fingerprint", _matrix_fingerprint(target_unitary))
        object.__setattr__(self, "phase_mode", _phase_mode(canonical_target.quotient_global_phase))
        object.__setattr__(self, "weights", configured_weights)
        object.__setattr__(
            self,
            "target_effective_support_size",
            _effective_support_size(target_probe),
        )
        object.__setattr__(
            self,
            "target_one_qubit_linear_entropies",
            _one_qubit_linear_entropies(target_probe, num_qubits),
        )
        object.__setattr__(self, "_cache", {})

    @classmethod
    def from_synthesis_target(
        cls,
        target: SynthesisTarget,
        **kwargs: Any,
    ) -> "DenseTargetContext":
        """Explicit factory for the certifier's exact target type."""

        if not isinstance(target, SynthesisTarget):
            raise TypeError("target must be a SynthesisTarget")
        return cls(target, **kwargs)

    @classmethod
    def from_certification_engine(
        cls,
        certification_engine: Any,
        **kwargs: Any,
    ) -> "DenseTargetContext":
        """Derive a context from a configured dense simulator certifier.

        The helper intentionally rejects engines without an exposed exact
        :class:`SynthesisTarget`; silently guessing a target would undermine
        the separation between target-aware observations and certification.
        """

        target = getattr(certification_engine, "target", None)
        if not isinstance(target, SynthesisTarget):
            raise ValueError(
                "certification engine has no configured exact SynthesisTarget"
            )
        return cls(target, **kwargs)

    @property
    def schema_version(self) -> str:
        """Version for context consumers and persisted learned-policy metadata."""

        return DENSE_TARGET_CONTEXT_SCHEMA_VERSION

    @property
    def target_metrics_schema_version(self) -> str:
        """Compatibility-friendly explicit name for the metrics schema."""

        return DENSE_TARGET_CONTEXT_SCHEMA_VERSION

    @property
    def cache_size(self) -> int:
        """Number of witness metrics memoized by this context only."""

        return len(self._cache)

    def cache_key(self, state: CircuitState) -> MetricCacheKey:
        """Return the safe per-context cache key for ``state``.

        It includes the target fingerprint, full authoritative DAG witness, and
        phase-sensitivity mode.  In particular, literal-phase targets cannot
        share a cache entry with global-phase-quotient evaluation.
        """

        identity = _gate_witness_identity(state)
        if identity[0] != self.num_qubits:
            raise ValueError(
                "candidate DAG width does not match the dense target: "
                f"{identity[0]} != {self.num_qubits}"
            )
        return self.fingerprint, self.phase_mode, identity

    def metrics(self, state: CircuitState) -> TargetMetrics:
        """Return target-relative diagnostics for an authoritative DAG witness."""

        key = self.cache_key(state)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        candidate = unitary_from_gates(self.num_qubits, state.dag.gates)
        target_overlap = np.trace(self.target_unitary.conj().T @ candidate)
        dimension = candidate.shape[0]
        process_fidelity = _bounded_unit_interval(
            float(abs(target_overlap) ** 2 / (dimension * dimension)),
            name="process fidelity",
        )

        if abs(target_overlap) > _ROUND_OFF_TOLERANCE:
            phase = target_overlap / abs(target_overlap)
        else:
            phase = 1.0 + 0.0j
        residual = candidate - phase * self.target_unitary
        phase_aligned_frobenius_distance = _bounded_unit_interval(
            float(np.linalg.norm(residual, ord="fro") / sqrt(2.0 * dimension)),
            name="phase-aligned Frobenius distance",
        )

        candidate_probe = candidate @ self.probe_state
        probe_state_fidelity = _bounded_unit_interval(
            float(abs(np.vdot(self.target_probe_state, candidate_probe)) ** 2),
            name="probe-state fidelity",
        )
        effective_support_size = _effective_support_size(candidate_probe)
        support_match = _bounded_unit_interval(
            exp(
                -abs(
                    log(effective_support_size)
                    - log(self.target_effective_support_size)
                )
            ),
            name="support match",
        )
        entropies = _one_qubit_linear_entropies(candidate_probe, self.num_qubits)
        if self.num_qubits:
            entanglement_match = 1.0 - sum(
                abs(value - target_value)
                for value, target_value in zip(
                    entropies,
                    self.target_one_qubit_linear_entropies,
                )
            ) / self.num_qubits
        else:
            entanglement_match = 1.0
        entanglement_match = _bounded_unit_interval(
            entanglement_match,
            name="entanglement match",
        )
        potential = _bounded_unit_interval(
            self.weights.process_fidelity * process_fidelity
            + self.weights.support_match * support_match
            + self.weights.entanglement_match * entanglement_match,
            name="target-progress potential",
        )

        metrics = TargetMetrics(
            target_fingerprint=self.fingerprint,
            candidate_fingerprint=_witness_fingerprint(key[2]),
            phase_mode=self.phase_mode,
            process_fidelity=process_fidelity,
            phase_aligned_frobenius_distance=phase_aligned_frobenius_distance,
            probe_state_fidelity=probe_state_fidelity,
            effective_support_size=effective_support_size,
            support_match=support_match,
            one_qubit_linear_entropies=entropies,
            entanglement_match=entanglement_match,
            potential=potential,
        )
        self._cache[key] = metrics
        return metrics

    def potential(self, state: CircuitState) -> float:
        """Return the configured target-progress potential for ``state``."""

        return self.metrics(state).potential


def target_context_from_certification_engine(
    certification_engine: Any,
    **kwargs: Any,
) -> DenseTargetContext:
    """Convenience wrapper around :meth:`DenseTargetContext.from_certification_engine`."""

    return DenseTargetContext.from_certification_engine(certification_engine, **kwargs)


def target_context_from_synthesis_target(
    target: SynthesisTarget,
    **kwargs: Any,
) -> DenseTargetContext:
    """Convenience wrapper around :meth:`DenseTargetContext.from_synthesis_target`."""

    return DenseTargetContext.from_synthesis_target(target, **kwargs)


__all__ = [
    "DEFAULT_MAX_DENSE_QUBITS",
    "DENSE_TARGET_CONTEXT_SCHEMA_VERSION",
    "DenseTargetContext",
    "TargetMetrics",
    "TargetProgressWeights",
    "target_context_from_certification_engine",
    "target_context_from_synthesis_target",
]
