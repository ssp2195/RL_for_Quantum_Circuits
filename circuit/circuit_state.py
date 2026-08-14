"""Authoritative DAG witnesses plus exact Clifford-frame/Pauli semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from algebra.pauli import PauliAxis
from algebra.pauli_rotation import PauliRotation, normalize_rotation_word
from algebra.tableau import CliffordFrame
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget


_CLIFFORD_NAMES = {"H", "S", "SDG", "X", "CNOT"}
_T_NAMES = {"T", "TDG"}
_SINGLE_QUBIT_NAMES = {"H", "S", "SDG", "T", "TDG", "X"}


@dataclass
class CircuitState:
    """A prefix witness and a sound symbolic description of its unitary.

    The invariant (up to the frame's projective Clifford convention) is

    ``U = exp(i * phase * pi/8) * frame * rotations[0] * ...``.

    New gates are appended in circuit order, therefore their matrices
    left-multiply the existing witness.  Clifford gates update the residual
    frame and T/T† gates prepend the appropriately transported Pauli rotation.
    """

    dag: CircuitDAG
    budget: ResourceBudget

    t_count: int = 0
    two_qubit_count: int = 0
    num_gates: int = 0
    wire_depths: Tuple[int, ...] | Sequence[int] = field(default_factory=tuple)
    depth: int = 0

    frame: Optional[CliffordFrame] = None
    rotations: Tuple[PauliRotation, ...] | Sequence[PauliRotation] = field(
        default_factory=tuple
    )
    global_phase_eighths: int = 0

    # Retained as an inert compatibility slot.  The legacy phase polynomial is
    # intentionally excluded from canonicalisation and general certification.
    phase_poly: Optional[object] = None

    # Static legal-continuation contract for the initial all-to-all/no-ancilla
    # implementation.  Consumed resources stay in the Pareto resource vector.
    continuation_interface: Tuple[object, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # The DAG is the single public authoritative witness.  Replay it even
        # when it is empty: otherwise a caller could attach a projectively
        # identity-but-literal-phase Clifford frame to an empty witness and
        # produce a false literal semantic state.  ``copy`` uses the private
        # trusted constructor below to avoid this O(length) replay on every
        # search child.
        self._replay_dag_witness()

    def _replay_dag_witness(self) -> None:
        """Reconstruct an exact state from its public DAG witness."""
        original_gates = tuple(self.dag.gates)
        n = self.dag.num_qubits
        self.dag = CircuitDAG(n)
        self.t_count = 0
        self.two_qubit_count = 0
        self.num_gates = 0
        self.wire_depths = tuple(0 for _ in range(n))
        self.depth = 0
        self.frame = CliffordFrame(n)
        self.rotations = ()
        self.global_phase_eighths = 0
        self.phase_poly = None
        self.continuation_interface = (
            tuple(self.continuation_interface)
            if self.continuation_interface
            else self._default_continuation_interface()
        )
        for gate in original_gates:
            if not self.apply_gate(gate):
                raise ValueError("DAG witness exceeds the supplied ResourceBudget")

    def _default_continuation_interface(self) -> Tuple[object, ...]:
        """Immutable legal-continuation contract for this search instance."""
        return (
            "all-to-all",
            "no-ancilla",
            "H",
            "S",
            "SDG",
            "T",
            "TDG",
            "CNOT",
            self.budget.max_t_count,
            self.budget.max_depth,
            self.budget.max_gates,
            self.budget.max_two_qubit_count,
        )

    @classmethod
    def _from_trusted_snapshot(
        cls,
        *,
        dag: CircuitDAG,
        budget: ResourceBudget,
        t_count: int,
        two_qubit_count: int,
        num_gates: int,
        wire_depths: Tuple[int, ...],
        depth: int,
        frame: CliffordFrame,
        rotations: Tuple[PauliRotation, ...],
        global_phase_eighths: int,
        continuation_interface: Tuple[object, ...],
    ) -> "CircuitState":
        """Internal fast path used only to copy an already-validated state."""
        state = object.__new__(cls)
        state.dag = dag
        state.budget = budget
        state.t_count = t_count
        state.two_qubit_count = two_qubit_count
        state.num_gates = num_gates
        state.wire_depths = wire_depths
        state.depth = depth
        state.frame = frame
        state.rotations = rotations
        state.global_phase_eighths = global_phase_eighths
        state.phase_poly = None
        state.continuation_interface = continuation_interface
        return state

    @property
    def tableau(self) -> CliffordFrame:
        """Backward-compatible name for the now-complete Clifford frame."""
        assert self.frame is not None
        return self.frame

    @property
    def rotation_word(self) -> Tuple[PauliRotation, ...]:
        return tuple(self.rotations)

    def resource_vector(self) -> Tuple[int, ...]:
        return (
            self.t_count,
            self.two_qubit_count,
            self.num_gates,
            *self.wire_depths,
        )

    def symbolic_unitary(self):
        """Materialise the semantic invariant for small-instance tests only."""
        import numpy as np

        assert self.frame is not None
        unitary = self.frame.to_unitary()
        for rotation in self.rotations:
            unitary = unitary @ rotation.to_matrix()
        return np.exp(1j * np.pi * self.global_phase_eighths / 8.0) * unitary

    def can_apply(self, gate: Gate) -> bool:
        """Return whether a legal one-gate continuation fits the budget."""
        if not self._valid_gate(gate):
            return False
        return self._check_budget(gate)

    def apply_gate(self, gate: Gate) -> bool:
        """Append ``gate`` iff it is legal and continuation-budget safe."""
        if not self.can_apply(gate):
            return False

        # Resources are calculated before mutating the DAG so independent
        # gates can share a layer even when appended at different times.
        affected = tuple(gate.qubits)
        layer = 1 + max(self.wire_depths[q] for q in affected)

        self.dag.add_gate(gate)
        self._update_resources(gate, layer)
        self._update_semantics(gate)

        if self.depth != self.dag.depth():  # catches resource/DAG regressions
            raise AssertionError("per-wire resource depth diverged from CircuitDAG")
        return True

    def _valid_gate(self, gate: Gate) -> bool:
        name = gate.gate_type.name
        if name not in _CLIFFORD_NAMES | _T_NAMES:
            return False
        if not gate.qubits or any(
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or qubit < 0
            or qubit >= self.dag.num_qubits
            for qubit in gate.qubits
        ):
            return False
        if len(set(gate.qubits)) != len(gate.qubits):
            return False
        if name in _SINGLE_QUBIT_NAMES:
            return len(gate.qubits) == 1
        return name == "CNOT" and len(gate.qubits) == 2

    def _check_budget(self, gate: Gate) -> bool:
        name = gate.gate_type.name
        if self.num_gates + 1 > self.budget.max_gates:
            return False

        layer = 1 + max(self.wire_depths[q] for q in gate.qubits)
        if layer > self.budget.max_depth:
            return False

        if name in _T_NAMES and self.t_count + 1 > self.budget.max_t_count:
            return False

        if name == "CNOT":
            maximum = getattr(self.budget, "max_two_qubit_count", None)
            if maximum is not None and self.two_qubit_count + 1 > maximum:
                return False
        return True

    def _update_resources(self, gate: Gate, layer: int) -> None:
        self.num_gates += 1
        mutable_depths = list(self.wire_depths)
        for qubit in gate.qubits:
            # All wires touched by a multiqubit operation synchronize to the
            # operation's layer; untouched wires remain independently shallow.
            mutable_depths[qubit] = layer
        self.wire_depths = tuple(mutable_depths)
        self.depth = max(self.wire_depths, default=0)

        if gate.gate_type.name in _T_NAMES:
            self.t_count += 1
        if gate.gate_type.name == "CNOT":
            self.two_qubit_count += 1

    def _update_semantics(self, gate: Gate) -> None:
        assert self.frame is not None
        name = gate.gate_type.name
        if name in _CLIFFORD_NAMES:
            self._apply_clifford(name, gate.qubits)
            return

        # T = exp(i*pi/8) R_Z(pi/4), and T† has both signs reversed.
        axis = PauliAxis.z_axis(self.dag.num_qubits, gate.qubits[0])
        transported = self.frame.inverse_conjugate(axis)
        turns = 1 if name == "T" else -1
        normalized, phase_delta = normalize_rotation_word(
            (PauliRotation(transported, turns), *self.rotations)
        )
        self.rotations = tuple(normalized)
        self.global_phase_eighths = (
            self.global_phase_eighths + turns + int(phase_delta)
        ) % 16

    def _apply_clifford(self, name: str, qubits: Tuple[int, ...]) -> None:
        assert self.frame is not None
        if name == "H":
            self.frame.apply_H(qubits[0])
        elif name == "S":
            self.frame.apply_S(qubits[0])
        elif name == "SDG":
            self.frame.apply_SDG(qubits[0])
        elif name == "X":
            self.frame.apply_X(qubits[0])
        elif name == "CNOT":
            self.frame.apply_CNOT(qubits[0], qubits[1])
        else:  # pragma: no cover - _valid_gate makes this unreachable
            raise ValueError(f"unsupported Clifford gate {name}")

    def copy(self) -> "CircuitState":
        assert self.frame is not None
        return CircuitState._from_trusted_snapshot(
            dag=self.dag.copy(),
            budget=self.budget,
            t_count=self.t_count,
            two_qubit_count=self.two_qubit_count,
            num_gates=self.num_gates,
            wire_depths=tuple(self.wire_depths),
            depth=self.depth,
            frame=self.frame.copy(),
            rotations=tuple(self.rotations),
            global_phase_eighths=self.global_phase_eighths,
            continuation_interface=tuple(self.continuation_interface),
        )

    def __repr__(self) -> str:
        return (
            "CircuitState("
            f"gates={self.num_gates}, depth={self.depth}, T={self.t_count}, "
            f"two_qubit={self.two_qubit_count}, rotations={len(self.rotations)})\n"
            f"{self.dag}"
        )
