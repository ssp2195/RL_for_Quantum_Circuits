"""Independent dense-unitary certification for small circuit instances.

The symbolic representation used by search is deliberately not consulted here:
the authoritative DAG witness is simulated directly.  This keeps certification
independent from canonicalisation and the Clifford/phase-polynomial machinery.

Qubit ``0`` is the least-significant bit of a computational-basis index, which
matches the bit-mask convention used elsewhere in this repository.  Gates are
applied in ``dag.gates`` order, so a later gate left-multiplies the accumulated
unitary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from certification.base import CertResult, CertStatus
from certification.base_engine import CertificationEngine


_SQRT2_INV = 1.0 / np.sqrt(2.0)

# These matrices use the conventional local basis ordering.  In particular the
# CNOT basis is |control, target> = |00>, |01>, |10>, |11>.  ``_embed_gate``
# maps that local ordering onto the repository's least-significant-qubit global
# index convention.
H_MATRIX = np.array(
    [[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]],
    dtype=np.complex128,
)
S_MATRIX = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
SDG_MATRIX = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
T_MATRIX = np.array(
    [[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]],
    dtype=np.complex128,
)
TDG_MATRIX = np.array(
    [[1.0, 0.0], [0.0, np.exp(-1.0j * np.pi / 4.0)]],
    dtype=np.complex128,
)
X_MATRIX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
CNOT_MATRIX = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 0.0]],
    dtype=np.complex128,
)

_GATE_MATRICES = {
    "H": H_MATRIX,
    "S": S_MATRIX,
    "SDG": SDG_MATRIX,
    "T": T_MATRIX,
    "TDG": TDG_MATRIX,
    "X": X_MATRIX,
    "CNOT": CNOT_MATRIX,
}

# Accept commonly used aliases when consuming an older serialized witness.  The
# current gate library should use SDG and TDG, but accepting aliases here does
# not create a second semantic gate type.
_GATE_ALIASES = {
    "S_DAG": "SDG",
    "S†": "SDG",
    "SDAG": "SDG",
    "T_DAG": "TDG",
    "T†": "TDG",
    "TDAG": "TDG",
}


@dataclass(frozen=True, slots=True)
class SynthesisTarget:
    """Dense target unitary for exact synthesis certification.

    ``quotient_global_phase`` defaults to true because synthesis ordinarily
    treats two unitaries differing only by a global phase as equivalent.
    """

    unitary: np.ndarray
    quotient_global_phase: bool = True

    def __post_init__(self) -> None:
        unitary = np.array(self.unitary, dtype=np.complex128, copy=True)
        if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
            raise ValueError("A synthesis target must be a square matrix")

        dimension = unitary.shape[0]
        if dimension < 1 or dimension & (dimension - 1):
            raise ValueError(
                "A synthesis target dimension must be a power of two"
            )
        if not np.isfinite(unitary).all():
            raise ValueError("A synthesis target must contain only finite values")
        if not np.allclose(
            unitary.conj().T @ unitary,
            np.eye(dimension, dtype=np.complex128),
            atol=1e-9,
            rtol=1e-9,
        ):
            raise ValueError("A synthesis target must be unitary")
        if not isinstance(self.quotient_global_phase, bool):
            raise TypeError("quotient_global_phase must be a bool")

        # Avoid accidental mutation of a target while a search is in progress.
        unitary.setflags(write=False)
        object.__setattr__(self, "unitary", unitary)

    @property
    def num_qubits(self) -> int:
        """The register width implied by the target matrix dimension."""
        return self.unitary.shape[0].bit_length() - 1


def _gate_name(gate_or_type: Any) -> str:
    """Return a normalized gate name from a Gate, GateType, or string."""
    gate_type = getattr(
        gate_or_type,
        "gate_type",
        getattr(gate_or_type, "gate", gate_or_type),
    )
    name = getattr(gate_type, "name", gate_type)
    if not isinstance(name, str):
        raise TypeError(f"Unsupported gate type {gate_type!r}")
    normalized = name.upper()
    return _GATE_ALIASES.get(normalized, normalized)


def gate_matrix(gate_or_type: Any) -> np.ndarray:
    """Return a defensive copy of the dense local matrix for ``gate_or_type``.

    Supported synthesis gates are ``H``, ``S``, ``SDG``, ``T``, ``TDG``, and
    ``CNOT``.  ``X`` is accepted as well because it is an existing Clifford
    enum member and may occur in a manually supplied DAG witness.
    """

    name = _gate_name(gate_or_type)
    try:
        return _GATE_MATRICES[name].copy()
    except KeyError as exc:
        supported = ", ".join(sorted(_GATE_MATRICES))
        raise ValueError(
            f"Unsupported gate {name!r}; supported gates are {supported}"
        ) from exc


def _local_index(basis_index: int, qubits: Sequence[int]) -> int:
    """Extract qubit bits in conventional operand order into a local index."""
    local_index = 0
    num_operands = len(qubits)
    for operand_index, qubit in enumerate(qubits):
        bit = (basis_index >> qubit) & 1
        local_index |= bit << (num_operands - operand_index - 1)
    return local_index


def _replace_local_index(
    basis_index: int,
    qubits: Sequence[int],
    local_index: int,
) -> int:
    """Replace ``qubits`` in ``basis_index`` using conventional local order."""
    output_index = basis_index
    num_operands = len(qubits)
    for operand_index, qubit in enumerate(qubits):
        bit = (local_index >> (num_operands - operand_index - 1)) & 1
        mask = 1 << qubit
        if bit:
            output_index |= mask
        else:
            output_index &= ~mask
    return output_index


def _validate_gate_qubits(
    name: str,
    qubits: Sequence[int],
    num_qubits: int,
) -> tuple[int, ...]:
    expected_arity = 2 if name == "CNOT" else 1
    normalized_qubits = tuple(qubits)
    if len(normalized_qubits) != expected_arity:
        raise ValueError(
            f"{name} expects {expected_arity} qubit(s), got {normalized_qubits!r}"
        )
    if len(set(normalized_qubits)) != len(normalized_qubits):
        raise ValueError(f"{name} cannot act on the same qubit twice")
    for qubit in normalized_qubits:
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError(f"Qubit indices must be integers, got {qubit!r}")
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError(
                f"Qubit {qubit} is outside a {num_qubits}-qubit register"
            )
    return normalized_qubits


def _embed_gate(
    local_matrix: np.ndarray,
    num_qubits: int,
    qubits: Sequence[int],
) -> np.ndarray:
    """Embed a one- or two-qubit local matrix in a dense register unitary."""
    dimension = 1 << num_qubits
    local_dimension = 1 << len(qubits)
    full_matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    for input_index in range(dimension):
        local_input = _local_index(input_index, qubits)
        for local_output in range(local_dimension):
            amplitude = local_matrix[local_output, local_input]
            if amplitude != 0:
                output_index = _replace_local_index(
                    input_index,
                    qubits,
                    local_output,
                )
                full_matrix[output_index, input_index] = amplitude

    return full_matrix


def unitary_from_gates(num_qubits: int, gates: Iterable[Any]) -> np.ndarray:
    """Return the full dense unitary implemented by a gate witness.

    The iterable order is circuit execution order: for gates ``[g1, g2]`` the
    returned matrix is ``U(g2) @ U(g1)``.  Independent DAG gates commute, so
    using the DAG's deterministic topological order is valid.
    """

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")

    dimension = 1 << num_qubits
    unitary = np.eye(dimension, dtype=np.complex128)

    for gate in gates:
        name = _gate_name(gate)
        local_matrix = gate_matrix(name)
        try:
            qubits = getattr(gate, "qubits")
        except AttributeError as exc:
            raise TypeError(
                "Each simulated gate must expose a 'qubits' sequence"
            ) from exc
        qubits = _validate_gate_qubits(name, qubits, num_qubits)
        unitary = _embed_gate(local_matrix, num_qubits, qubits) @ unitary

    return unitary


def equivalent_up_to_global_phase(
    candidate: np.ndarray,
    target: np.ndarray,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> bool:
    """Whether two matrices agree numerically after removing one global phase."""
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")

    try:
        candidate_array = np.asarray(candidate, dtype=np.complex128)
        target_array = np.asarray(target, dtype=np.complex128)
    except (TypeError, ValueError):
        return False

    if (
        candidate_array.ndim != 2
        or target_array.ndim != 2
        or candidate_array.shape != target_array.shape
        or candidate_array.shape[0] != candidate_array.shape[1]
    ):
        return False
    if not np.isfinite(candidate_array).all() or not np.isfinite(target_array).all():
        return False

    # A true unitary has a non-zero entry.  Choosing the largest target entry
    # gives the most stable phase anchor in the presence of numerical noise.
    anchor = np.unravel_index(
        np.argmax(np.abs(target_array)),
        target_array.shape,
    )
    denominator = target_array[anchor]
    if abs(denominator) <= atol:
        return False

    ratio = candidate_array[anchor] / denominator
    if abs(ratio) <= atol:
        return False
    phase = ratio / abs(ratio)

    return bool(
        np.allclose(
            candidate_array,
            phase * target_array,
            atol=atol,
            rtol=rtol,
        )
    )


class SimulatorCertificationEngine(CertificationEngine):
    """Certify a DAG witness by independently simulating its dense unitary."""

    def __init__(
        self,
        target: SynthesisTarget | np.ndarray | None = None,
        *,
        atol: float = 1e-9,
        rtol: float = 1e-9,
    ) -> None:
        if atol < 0 or rtol < 0:
            raise ValueError("atol and rtol must be non-negative")
        self.target = (
            None
            if target is None
            else (
                target
                if isinstance(target, SynthesisTarget)
                else SynthesisTarget(np.asarray(target, dtype=np.complex128))
            )
        )
        self.atol = atol
        self.rtol = rtol

    def certify(self, state: Any) -> CertResult:
        if self.target is None:
            return CertResult(
                CertStatus.INCONCLUSIVE,
                score=0.0,
                info={"reason": "no_target_configured"},
            )
        candidate = unitary_from_gates(
            state.dag.num_qubits,
            state.dag.gates,
        )

        if candidate.shape != self.target.unitary.shape:
            equivalent = False
        elif self.target.quotient_global_phase:
            equivalent = equivalent_up_to_global_phase(
                candidate,
                self.target.unitary,
                atol=self.atol,
                rtol=self.rtol,
            )
        else:
            equivalent = bool(
                np.allclose(
                    candidate,
                    self.target.unitary,
                    atol=self.atol,
                    rtol=self.rtol,
                )
            )

        if equivalent:
            return CertResult(
                CertStatus.SUCCESS,
                score=1.0,
                info={"reason": "equivalent_dense_unitary"},
            )

        # A non-target prefix may become a target after further legal gates;
        # only a complete target match is terminal in the general search.
        return CertResult(
            CertStatus.INCONCLUSIVE,
            score=0.0,
            info={"reason": "not_target"},
        )


__all__ = [
    "CNOT_MATRIX",
    "H_MATRIX",
    "SDG_MATRIX",
    "S_MATRIX",
    "SynthesisTarget",
    "SimulatorCertificationEngine",
    "TDG_MATRIX",
    "T_MATRIX",
    "X_MATRIX",
    "equivalent_up_to_global_phase",
    "gate_matrix",
    "unitary_from_gates",
]
