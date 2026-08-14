"""Deterministic features for frontier-record scheduling.

The legacy feature vector deliberately describes only a candidate's resource
state and order-invariant resource context.  Passing a dense target context
opt-in appends a *labelled*, target-relative block for small circuits.  The
learner still receives a frontier record rather than a gate action, and none
of the features has access to a frontier position, heap position, or record
identifier.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING

import numpy as np

from circuit.circuit_state import CircuitState
from enums import GateType

if TYPE_CHECKING:  # Avoid importing dense simulation on the legacy path.
    from rl.target_context import DenseTargetContext, TargetMetrics


_EPS = 1e-8
_METRIC_TOLERANCE = 1e-7

# ``extract_features(..., target_context=None)`` intentionally remains this
# exact 16-dimensional representation.  Existing experiments and checkpoints
# can therefore continue to use the target-free policy without migration.
_BASE_FEATURE_NAMES = (
    "t_count_fraction",
    "two_qubit_fraction",
    "gate_count_fraction",
    "depth_fraction",
    "t_count_remaining_fraction",
    "two_qubit_remaining_fraction",
    "gate_count_remaining_fraction",
    "depth_remaining_fraction",
    "rotation_count_fraction",
    "mean_pauli_weight_fraction",
    "rotation_anticommutation_density",
    "wire_depth_imbalance_fraction",
)
_BASE_CONTEXT_FEATURE_NAMES = (
    "frontier_z_t_count_fraction",
    "frontier_z_two_qubit_fraction",
    "frontier_z_rotation_count_fraction",
    "frontier_z_wire_depth_imbalance_fraction",
)
LEGACY_FEATURE_SCHEMA_VERSION = "frontier-resource-v1"
LEGACY_FEATURE_DIMENSION = len(_BASE_FEATURE_NAMES) + len(_BASE_CONTEXT_FEATURE_NAMES)

# Target-aware features are intentionally fixed-width for the GHZ-3
# experiment.  Labels are meaningful because the target unitary is itself
# qubit-labelled; frontier *storage* order remains semantically irrelevant.
TARGET_AWARE_FEATURE_SCHEMA_VERSION = "frontier-target-aware-v1"
TARGET_FEATURE_QUBIT_CAPACITY = 3
TARGET_SEMANTIC_FEATURE_NAMES = (
    "target_process_fidelity",
    "target_sqrt_process_fidelity",
    "target_phase_aligned_frobenius_distance",
    "target_probe_state_fidelity",
    "target_support_match",
    "target_entanglement_match",
    "target_progress_potential",
)
DIRECTED_CNOT_PAIRS = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)
# This is the sole gate-type ordering used by the feature schema.  It is not
# derived from a set or a dictionary, so saved policy weights remain legible.
LAST_OPERATION_GATE_ORDER = (
    GateType.H,
    GateType.S,
    GateType.SDG,
    GateType.T,
    GateType.TDG,
    GateType.X,
    GateType.CNOT,
)

_PER_QUBIT_FEATURE_NAMES = tuple(
    f"qubit_{qubit}_gate_touch_fraction"
    for qubit in range(TARGET_FEATURE_QUBIT_CAPACITY)
) + tuple(
    f"qubit_{qubit}_hadamard_fraction"
    for qubit in range(TARGET_FEATURE_QUBIT_CAPACITY)
) + tuple(
    f"qubit_{qubit}_wire_depth_fraction"
    for qubit in range(TARGET_FEATURE_QUBIT_CAPACITY)
) + tuple(
    f"qubit_{qubit}_linear_entropy"
    for qubit in range(TARGET_FEATURE_QUBIT_CAPACITY)
)
_DIRECTED_CNOT_FEATURE_NAMES = tuple(
    f"cnot_{control}_to_{target}_fraction"
    for control, target in DIRECTED_CNOT_PAIRS
)
_LAST_OPERATION_FEATURE_NAMES = tuple(
    f"last_gate_{gate_type.name.lower()}" for gate_type in LAST_OPERATION_GATE_ORDER
) + tuple(
    f"last_first_operand_{qubit}"
    for qubit in range(TARGET_FEATURE_QUBIT_CAPACITY)
) + tuple(
    f"last_second_operand_{qubit}"
    for qubit in range(TARGET_FEATURE_QUBIT_CAPACITY)
)
_TARGET_FRONTIER_CONTEXT_FEATURE_NAMES = (
    "target_frontier_candidate_potential",
    "target_frontier_potential_minus_mean",
    "target_frontier_potential_minus_max",
    "target_frontier_process_fidelity_minus_mean",
    "target_frontier_entanglement_match_minus_mean",
)
_TARGET_AWARE_FEATURE_NAMES = (
    TARGET_SEMANTIC_FEATURE_NAMES
    + _PER_QUBIT_FEATURE_NAMES
    + _DIRECTED_CNOT_FEATURE_NAMES
    + _LAST_OPERATION_FEATURE_NAMES
    + _TARGET_FRONTIER_CONTEXT_FEATURE_NAMES
    + ("bias",)
)
TARGET_AWARE_FEATURE_DIMENSION = LEGACY_FEATURE_DIMENSION + len(_TARGET_AWARE_FEATURE_NAMES)


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


def _frontier_states(frontier: Optional[Iterable[object]]) -> list[CircuitState]:
    if frontier is None:
        return []
    states: list[CircuitState] = []
    for item in frontier:
        candidate = getattr(item, "state", item)
        if isinstance(candidate, CircuitState):
            states.append(candidate)
    return states


def _legacy_context_features(
    base: np.ndarray, frontier_states: Sequence[CircuitState]
) -> np.ndarray:
    if not frontier_states:
        return np.zeros(len(_BASE_CONTEXT_FEATURE_NAMES), dtype=np.float32)

    all_base = np.vstack([_base_features(candidate) for candidate in frontier_states])
    # Sorting each scalar column makes the reduction bitwise stable when a
    # heap or array supplies the same records in a different order.
    sorted_base = np.sort(all_base, axis=0)
    mean = np.mean(sorted_base, axis=0)
    std = np.std(sorted_base, axis=0)
    z = (base - mean) / (std + _EPS)

    # Compact shared-context signal: resource pressure, rotation complexity,
    # and Pareto-relevant depth imbalance relative to the active frontier.
    return np.asarray([z[0], z[1], z[8], z[11]], dtype=np.float32)


def _target_num_qubits(target_context: object) -> int:
    try:
        num_qubits = getattr(target_context, "num_qubits")
    except AttributeError as exc:  # pragma: no cover - defensive API message
        raise TypeError("target_context must expose an integer num_qubits") from exc
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, (int, np.integer)):
        raise TypeError("target_context.num_qubits must be an integer")
    num_qubits = int(num_qubits)
    if num_qubits < 1:
        raise ValueError("target_context.num_qubits must be positive")
    if num_qubits > TARGET_FEATURE_QUBIT_CAPACITY:
        raise ValueError(
            "target-aware feature capacity is three qubits; "
            f"received target_context.num_qubits={num_qubits}"
        )
    return num_qubits


def _validate_target_state(state: CircuitState, target_context: object) -> int:
    target_qubits = _target_num_qubits(target_context)
    if state.dag.num_qubits != target_qubits:
        raise ValueError(
            "target-aware features require a state and target context with the "
            f"same qubit count; got state={state.dag.num_qubits}, "
            f"target={target_qubits}"
        )
    return target_qubits


def _unit_interval_metric(metrics: object, name: str) -> float:
    """Read a finite unit-interval semantic metric without hiding bad data."""
    try:
        value = float(getattr(metrics, name))
    except AttributeError as exc:  # pragma: no cover - explicit integration error
        raise TypeError(f"target metrics must expose {name!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"target metric {name!r} must be finite, got {value!r}")
    if value < -_METRIC_TOLERANCE or value > 1.0 + _METRIC_TOLERANCE:
        raise ValueError(
            f"target metric {name!r} must be normalized to [0, 1], got {value!r}"
        )
    # The target context may differ from an endpoint by harmless dense
    # roundoff.  Clipping only that tolerance preserves the intended metric.
    return float(np.clip(value, 0.0, 1.0))


def _target_metrics(
    state: CircuitState, target_context: object
) -> "TargetMetrics":
    _validate_target_state(state, target_context)
    metrics_fn = getattr(target_context, "metrics", None)
    if not callable(metrics_fn):
        raise TypeError("target_context must expose metrics(state)")
    return metrics_fn(state)


def _per_qubit_features(
    state: CircuitState, metrics: object, num_qubits: int
) -> np.ndarray:
    touches = np.zeros(TARGET_FEATURE_QUBIT_CAPACITY, dtype=np.float64)
    hadamards = np.zeros(TARGET_FEATURE_QUBIT_CAPACITY, dtype=np.float64)
    for gate in state.dag.gates:
        for qubit in gate.qubits:
            if qubit < TARGET_FEATURE_QUBIT_CAPACITY:
                touches[qubit] += 1.0
        if gate.gate_type is GateType.H:
            hadamards[gate.qubits[0]] += 1.0

    depths = np.zeros(TARGET_FEATURE_QUBIT_CAPACITY, dtype=np.float64)
    wire_depths = tuple(getattr(state, "wire_depths", ()))
    for qubit in range(num_qubits):
        if qubit < len(wire_depths):
            depths[qubit] = float(wire_depths[qubit])

    try:
        entropies = np.asarray(
            getattr(metrics, "one_qubit_linear_entropies"), dtype=np.float64
        ).reshape(-1)
    except AttributeError as exc:  # pragma: no cover - explicit integration error
        raise TypeError("target metrics must expose one_qubit_linear_entropies") from exc
    if entropies.size != num_qubits:
        raise ValueError(
            "target metrics one_qubit_linear_entropies must have one value per "
            f"target qubit; expected {num_qubits}, got {entropies.size}"
        )
    if not np.all(np.isfinite(entropies)):
        raise ValueError("target one-qubit linear entropies must be finite")
    if np.any(entropies < -_METRIC_TOLERANCE) or np.any(
        entropies > 1.0 + _METRIC_TOLERANCE
    ):
        raise ValueError("target one-qubit linear entropies must be in [0, 1]")
    padded_entropies = np.zeros(TARGET_FEATURE_QUBIT_CAPACITY, dtype=np.float64)
    padded_entropies[:num_qubits] = np.clip(entropies, 0.0, 1.0)

    return np.concatenate(
        (
            touches / max(1.0, float(state.budget.max_gates)),
            hadamards / max(1.0, float(state.budget.max_gates)),
            depths / max(1.0, float(state.budget.max_depth)),
            padded_entropies,
        )
    )


def _directed_cnot_features(state: CircuitState) -> np.ndarray:
    counts = {pair: 0.0 for pair in DIRECTED_CNOT_PAIRS}
    for gate in state.dag.gates:
        if gate.gate_type is GateType.CNOT:
            pair = tuple(gate.qubits)
            if pair in counts:
                counts[pair] += 1.0
    maximum = getattr(state.budget, "max_two_qubit_count", None)
    if maximum is None:
        maximum = state.budget.max_gates
    return np.asarray(
        [counts[pair] / max(1.0, float(maximum)) for pair in DIRECTED_CNOT_PAIRS],
        dtype=np.float64,
    )


def _last_operation_features(state: CircuitState) -> np.ndarray:
    result = np.zeros(len(_LAST_OPERATION_FEATURE_NAMES), dtype=np.float64)
    gates = state.dag.gates
    if not gates:
        return result

    last_gate = gates[-1]
    try:
        gate_index = LAST_OPERATION_GATE_ORDER.index(last_gate.gate_type)
    except ValueError as exc:  # pragma: no cover - CircuitState rejects unknown gates
        raise ValueError(f"unsupported last gate for feature schema: {last_gate!r}") from exc
    result[gate_index] = 1.0

    operand_offset = len(LAST_OPERATION_GATE_ORDER)
    first_operand = last_gate.qubits[0]
    if first_operand < TARGET_FEATURE_QUBIT_CAPACITY:
        result[operand_offset + first_operand] = 1.0
    if len(last_gate.qubits) == 2:
        second_operand = last_gate.qubits[1]
        if second_operand < TARGET_FEATURE_QUBIT_CAPACITY:
            result[operand_offset + TARGET_FEATURE_QUBIT_CAPACITY + second_operand] = 1.0
    return result


def _target_semantic_features(metrics: object) -> np.ndarray:
    process_fidelity = _unit_interval_metric(metrics, "process_fidelity")
    return np.asarray(
        [
            process_fidelity,
            np.sqrt(process_fidelity),
            _unit_interval_metric(metrics, "phase_aligned_frobenius_distance"),
            _unit_interval_metric(metrics, "probe_state_fidelity"),
            _unit_interval_metric(metrics, "support_match"),
            _unit_interval_metric(metrics, "entanglement_match"),
            _unit_interval_metric(metrics, "potential"),
        ],
        dtype=np.float64,
    )


def _target_frontier_context_features(
    candidate_metrics: object,
    frontier_states: Sequence[CircuitState],
    target_context: object,
) -> np.ndarray:
    candidate_potential = _unit_interval_metric(candidate_metrics, "potential")
    candidate_process = _unit_interval_metric(candidate_metrics, "process_fidelity")
    candidate_entanglement = _unit_interval_metric(candidate_metrics, "entanglement_match")

    if not frontier_states:
        return np.asarray(
            [candidate_potential, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

    frontier_metrics = [_target_metrics(candidate, target_context) for candidate in frontier_states]
    # Sort scalar values before reductions so context is invariant even at the
    # floating-point-bit level under arbitrary frontier container ordering.
    potentials = np.sort(
        np.asarray([_unit_interval_metric(item, "potential") for item in frontier_metrics])
    )
    process_fidelities = np.sort(
        np.asarray(
            [_unit_interval_metric(item, "process_fidelity") for item in frontier_metrics]
        )
    )
    entanglement_matches = np.sort(
        np.asarray(
            [
                _unit_interval_metric(item, "entanglement_match")
                for item in frontier_metrics
            ]
        )
    )
    return np.asarray(
        [
            candidate_potential,
            candidate_potential - float(np.mean(potentials)),
            candidate_potential - float(np.max(potentials)),
            candidate_process - float(np.mean(process_fidelities)),
            candidate_entanglement - float(np.mean(entanglement_matches)),
        ],
        dtype=np.float64,
    )


def _target_aware_features(
    state: CircuitState,
    frontier_states: Sequence[CircuitState],
    target_context: object,
) -> np.ndarray:
    num_qubits = _validate_target_state(state, target_context)
    metrics = _target_metrics(state, target_context)
    return np.concatenate(
        (
            _target_semantic_features(metrics),
            _per_qubit_features(state, metrics, num_qubits),
            _directed_cnot_features(state),
            _last_operation_features(state),
            _target_frontier_context_features(metrics, frontier_states, target_context),
            np.asarray([1.0], dtype=np.float64),
        )
    )


def feature_names(target_context: Optional[object] = None) -> tuple[str, ...]:
    """Return the authoritative ordered feature labels for a policy schema."""
    if target_context is None:
        return _BASE_FEATURE_NAMES + _BASE_CONTEXT_FEATURE_NAMES
    _target_num_qubits(target_context)
    return _BASE_FEATURE_NAMES + _BASE_CONTEXT_FEATURE_NAMES + _TARGET_AWARE_FEATURE_NAMES


def feature_schema_version(target_context: Optional[object] = None) -> str:
    """Return a durable schema label suitable for checkpoints and reports."""
    return (
        LEGACY_FEATURE_SCHEMA_VERSION
        if target_context is None
        else TARGET_AWARE_FEATURE_SCHEMA_VERSION
    )


def feature_metadata(target_context: Optional[object] = None) -> dict[str, object]:
    """Describe the active extractor without exposing mutable feature state."""
    names = feature_names(target_context)
    metadata: dict[str, object] = {
        "feature_schema_version": feature_schema_version(target_context),
        "feature_dim": len(names),
        "feature_names": names,
        "target_aware": target_context is not None,
    }
    if target_context is not None:
        metadata.update(
            {
                "target_fingerprint": str(getattr(target_context, "fingerprint", "")),
                "target_context_schema_version": str(
                    getattr(target_context, "schema_version", "")
                ),
                "target_feature_qubit_capacity": TARGET_FEATURE_QUBIT_CAPACITY,
            }
        )
    return metadata


def extract_features(
    state: CircuitState,
    frontier: Optional[Iterable[object]] = None,
    target_context: Optional[object] = None,
) -> np.ndarray:
    """Return candidate features with optional order-invariant context.

    ``frontier`` may contain either ``CircuitState`` values or search nodes.
    Without ``target_context`` this returns the historical 16-coordinate
    target-free vector.  Supplying a :class:`DenseTargetContext` appends the
    fixed three-qubit target-aware block and an explicit bias coordinate.
    """
    base = _base_features(state)
    states = _frontier_states(frontier)
    legacy = np.concatenate((base, _legacy_context_features(base, states)))
    if target_context is None:
        return legacy

    target_aware = _target_aware_features(state, states, target_context)
    features = np.concatenate((legacy, target_aware)).astype(np.float32, copy=False)
    if features.shape != (TARGET_AWARE_FEATURE_DIMENSION,):  # pragma: no cover
        raise AssertionError("target-aware feature schema dimension drifted")
    return features


def feature_dimension(
    state: CircuitState | None = None,
    target_context: Optional[object] = None,
) -> int:
    """Derive the policy input dimension from the active extractor schema."""
    if state is None:
        if target_context is None:
            return LEGACY_FEATURE_DIMENSION
        _target_num_qubits(target_context)
        return TARGET_AWARE_FEATURE_DIMENSION
    return int(extract_features(state, target_context=target_context).shape[0])


class FeatureProvider(Protocol):
    """Stable opt-in feature-provider contract for frontier record policies.

    Providers receive a :class:`CircuitState` and an optional frontier of
    states or nodes, and must return a deterministic fixed-width vector.  A
    provider may use labelled target/problem information, but it must never
    use a heap position, record ID, or any other scheduler implementation
    detail as a learned feature.

    The protocol deliberately mirrors the module-level legacy helpers so the
    original target-free and ``DenseTargetContext`` paths remain available via
    :class:`LegacyFeatureProvider` and
    :class:`TargetContextFeatureProvider` without changing their schemas.
    """

    @property
    def schema_version(self) -> str:
        """Durable schema label used to bind policy weights."""

    @property
    def dimension(self) -> int:
        """Exact number of coordinates returned by :meth:`extract`."""

    @property
    def names(self) -> tuple[str, ...]:
        """Ordered, human-readable coordinate names."""

    def extract(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> np.ndarray:
        """Return one fixed-width feature vector for ``state``."""

    def metadata(self) -> Mapping[str, object]:
        """Return immutable-schema metadata suitable for a policy report."""


class LegacyFeatureProvider:
    """Adapter for the existing target-free/dense-target feature functions.

    This is intentionally thin: it delegates to the public module functions
    above, so existing 16-D target-free and 60-D target-aware vectors remain
    exactly unchanged.  It is the default provider used by
    :class:`rl.policy.LinearQPolicy` when callers do not supply a custom one.
    """

    def __init__(self, target_context: Optional[object] = None) -> None:
        self.target_context = target_context

    @property
    def schema_version(self) -> str:
        return feature_schema_version(self.target_context)

    @property
    def dimension(self) -> int:
        return feature_dimension(target_context=self.target_context)

    @property
    def names(self) -> tuple[str, ...]:
        return feature_names(self.target_context)

    def extract(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> np.ndarray:
        return extract_features(state, frontier, target_context=self.target_context)

    def metadata(self) -> Mapping[str, object]:
        # Return the historical metadata verbatim.  In particular, callers
        # using a DenseTargetContext see the same target fields as before the
        # provider abstraction was introduced.
        return feature_metadata(self.target_context)

    def bind(self, target_context: object) -> "TargetContextFeatureProvider":
        """Return a target-aware adapter without mutating this provider."""

        if target_context is None:
            raise TypeError("target_context must not be None")
        return TargetContextFeatureProvider(target_context)


class TargetContextFeatureProvider(LegacyFeatureProvider):
    """Named adapter for the existing ``DenseTargetContext`` feature path."""

    def __init__(self, target_context: object) -> None:
        if target_context is None:
            raise TypeError("target_context must not be None")
        super().__init__(target_context)


def validate_feature_provider(provider: object) -> FeatureProvider:
    """Validate the small runtime contract used by ``LinearQPolicy``.

    Structural validation gives integration errors at policy construction
    rather than after a partially completed scheduling transition.  A custom
    provider need not inherit a project base class; matching the documented
    attributes and methods is sufficient.
    """

    required = ("schema_version", "dimension", "names", "extract", "metadata")
    missing = [name for name in required if not hasattr(provider, name)]
    if missing:
        raise TypeError(
            "feature_provider must expose " + ", ".join(required) + "; missing " + ", ".join(missing)
        )
    if not callable(getattr(provider, "extract")) or not callable(
        getattr(provider, "metadata")
    ):
        raise TypeError("feature_provider.extract and feature_provider.metadata must be callable")

    schema_version = getattr(provider, "schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise TypeError("feature_provider.schema_version must be a non-empty string")
    dimension = getattr(provider, "dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or int(dimension) < 1:
        raise ValueError("feature_provider.dimension must be a positive integer")
    names = tuple(getattr(provider, "names"))
    if len(names) != int(dimension) or not all(isinstance(name, str) and name for name in names):
        raise ValueError(
            "feature_provider.names must contain one non-empty string per feature dimension"
        )
    return provider  # type: ignore[return-value]


__all__ = [
    "DIRECTED_CNOT_PAIRS",
    "LAST_OPERATION_GATE_ORDER",
    "LEGACY_FEATURE_DIMENSION",
    "LEGACY_FEATURE_SCHEMA_VERSION",
    "TARGET_AWARE_FEATURE_DIMENSION",
    "TARGET_AWARE_FEATURE_SCHEMA_VERSION",
    "TARGET_FEATURE_QUBIT_CAPACITY",
    "TARGET_SEMANTIC_FEATURE_NAMES",
    "FeatureProvider",
    "LegacyFeatureProvider",
    "TargetContextFeatureProvider",
    "extract_features",
    "feature_dimension",
    "feature_metadata",
    "feature_names",
    "feature_schema_version",
    "validate_feature_provider",
]
