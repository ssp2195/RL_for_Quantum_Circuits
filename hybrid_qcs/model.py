"""Hybrid frontier state: persistent DAG + Clifford tableau + Pauli rotations."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

from .pauli import Pauli
from .rotation import (
    PauliRotation,
    anticommuting_pair_count,
    insert_rotation,
    mean_pauli_weight,
)
from .tableau import CliffordTableau

CLIFFORD_GATES = frozenset({"H", "S", "SDG", "CNOT"})
NON_CLIFFORD_GATES = frozenset({"T", "TDG"})
SINGLE_QUBIT_GATES = frozenset({"H", "S", "SDG", "T", "TDG"})
GATE_ORDER = {
    name: index
    for index, name in enumerate(("H", "S", "SDG", "T", "TDG", "CNOT"))
}
INVERSE_GATE = {
    "H": "H",
    "S": "SDG",
    "SDG": "S",
    "T": "TDG",
    "TDG": "T",
    "CNOT": "CNOT",
}


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).upper())
        object.__setattr__(self, "qubits", tuple(self.qubits))

    @property
    def is_clifford(self) -> bool:
        return self.name in CLIFFORD_GATES

    @property
    def is_non_clifford(self) -> bool:
        return self.name in NON_CLIFFORD_GATES

    @property
    def is_two_qubit(self) -> bool:
        return self.name == "CNOT"

    def sort_key(self) -> tuple[int, tuple[int, ...]]:
        return GATE_ORDER[self.name], self.qubits

    def label(self) -> str:
        return f"{self.name}({','.join(str(q) for q in self.qubits)})"


@dataclass(frozen=True, slots=True)
class Budget:
    max_t_count: int
    max_cnot_count: int
    max_gates: int
    max_depth: int

    def __post_init__(self) -> None:
        for name in ("max_t_count", "max_cnot_count", "max_gates", "max_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True, eq=False)
class WitnessStep:
    """One persistent gate event in the lossless dependency DAG."""

    previous: Optional["WitnessStep"]
    gate: Gate
    parents: tuple["WitnessStep", ...]
    level: int
    index: int


@dataclass(frozen=True, slots=True)
class DAGNode:
    index: int
    gate: Gate
    parents: tuple[int, ...]
    level: int


@dataclass(frozen=True, slots=True)
class MaterializedDAG:
    num_qubits: int
    nodes: tuple[DAGNode, ...]
    wire_depths: tuple[int, ...]
    depth: int

    @property
    def gates(self) -> tuple[Gate, ...]:
        return tuple(node.gate for node in self.nodes)

    def validate(self) -> None:
        latest: list[int | None] = [None] * self.num_qubits
        wire_depths = [0] * self.num_qubits
        for expected_index, node in enumerate(self.nodes):
            if node.index != expected_index:
                raise AssertionError("DAG node indices are not contiguous")
            expected_parents = tuple(
                sorted(
                    {
                        latest[q]
                        for q in node.gate.qubits
                        if latest[q] is not None
                    }
                )
            )
            if node.parents != expected_parents:
                raise AssertionError("DAG parents do not match wire dependencies")
            expected_level = 1 + max(
                (self.nodes[parent].level for parent in expected_parents), default=0
            )
            if node.level != expected_level:
                raise AssertionError("DAG level recurrence failed")
            for q in node.gate.qubits:
                latest[q] = node.index
                wire_depths[q] = node.level
        if tuple(wire_depths) != self.wire_depths:
            raise AssertionError("DAG wire-depth cache is inconsistent")
        if max(wire_depths, default=0) != self.depth:
            raise AssertionError("DAG depth cache is inconsistent")


@dataclass(slots=True)
class TransitionProfile:
    gate_validation_ns: int = 0
    dag_append_ns: int = 0
    tableau_update_ns: int = 0
    inverse_pauli_transport_ns: int = 0
    rotation_insertion_ns: int = 0
    canonical_key_ns: int = 0
    attempted: int = 0
    accepted: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "gate_validation_ns": self.gate_validation_ns,
            "dag_append_ns": self.dag_append_ns,
            "tableau_update_ns": self.tableau_update_ns,
            "inverse_pauli_transport_ns": self.inverse_pauli_transport_ns,
            "rotation_insertion_ns": self.rotation_insertion_ns,
            "canonical_key_ns": self.canonical_key_ns,
            "attempted": self.attempted,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class HybridState:
    """One exact symbolic circuit prefix, modulo global phase in its key."""

    num_qubits: int
    budget: Budget
    tail: Optional[WitnessStep]
    wire_tails: tuple[Optional[WitnessStep], ...]
    tableau: CliffordTableau
    rotations: tuple[PauliRotation, ...]
    global_phase_eighths: int
    t_count: int
    cnot_count: int
    gate_count: int
    wire_depths: tuple[int, ...]
    depth: int
    canonical_key: tuple[object, ...]

    @classmethod
    def identity(cls, num_qubits: int, budget: Budget) -> "HybridState":
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or not 1 <= num_qubits <= 3:
            raise ValueError("num_qubits must lie in 1..3")
        tableau = CliffordTableau.identity(num_qubits)
        rotations: tuple[PauliRotation, ...] = ()
        return cls(
            num_qubits=num_qubits,
            budget=budget,
            tail=None,
            wire_tails=tuple(None for _ in range(num_qubits)),
            tableau=tableau,
            rotations=rotations,
            global_phase_eighths=0,
            t_count=0,
            cnot_count=0,
            gate_count=0,
            wire_depths=tuple(0 for _ in range(num_qubits)),
            depth=0,
            canonical_key=_canonical_key(num_qubits, tableau, rotations),
        )

    def resource_vector(self) -> tuple[int, ...]:
        return (self.t_count, self.cnot_count, self.gate_count, *self.wire_depths)

    @property
    def last_gate(self) -> Gate | None:
        return None if self.tail is None else self.tail.gate

    @property
    def mean_pauli_weight(self) -> float:
        return mean_pauli_weight(self.rotations)

    @property
    def anticommuting_pairs(self) -> int:
        return anticommuting_pair_count(self.rotations)

    def apply(
        self,
        gate: Gate,
        *,
        partial_order_reduction: bool = True,
        profile: TransitionProfile | None = None,
    ) -> "HybridState | None":
        """Return one legal child; legality is target-independent."""

        validation_started = time.perf_counter_ns()
        if profile is not None:
            profile.attempted += 1
        if not _valid_gate(gate, self.num_qubits):
            if profile is not None:
                profile.gate_validation_ns += time.perf_counter_ns() - validation_started
            return None

        if partial_order_reduction:
            wire_parents = {self.wire_tails[q] for q in gate.qubits}
            if len(wire_parents) == 1:
                previous = next(iter(wire_parents))
                if previous is not None and INVERSE_GATE[previous.gate.name] == gate.name and previous.gate.qubits == gate.qubits:
                    if profile is not None:
                        profile.gate_validation_ns += time.perf_counter_ns() - validation_started
                    return None
            if self.tail is not None:
                last = self.tail.gate
                if set(last.qubits).isdisjoint(gate.qubits) and gate.sort_key() < last.sort_key():
                    if profile is not None:
                        profile.gate_validation_ns += time.perf_counter_ns() - validation_started
                    return None

        layer = 1 + max(self.wire_depths[q] for q in gate.qubits)
        next_t = self.t_count + int(gate.is_non_clifford)
        next_cnot = self.cnot_count + int(gate.is_two_qubit)
        if (
            self.gate_count + 1 > self.budget.max_gates
            or layer > self.budget.max_depth
            or next_t > self.budget.max_t_count
            or next_cnot > self.budget.max_cnot_count
        ):
            if profile is not None:
                profile.gate_validation_ns += time.perf_counter_ns() - validation_started
            return None
        if profile is not None:
            profile.gate_validation_ns += time.perf_counter_ns() - validation_started

        tableau = self.tableau
        rotations = self.rotations
        phase = self.global_phase_eighths
        if gate.is_clifford:
            started = time.perf_counter_ns()
            tableau = tableau.left_multiply(gate.name, gate.qubits)
            if profile is not None:
                profile.tableau_update_ns += time.perf_counter_ns() - started
        else:
            started = time.perf_counter_ns()
            transported = tableau.inverse_conjugate(
                Pauli.z_axis(self.num_qubits, gate.qubits[0])
            )
            if profile is not None:
                profile.inverse_pauli_transport_ns += time.perf_counter_ns() - started
            turns = 1 if gate.name == "T" else -1
            started = time.perf_counter_ns()
            rotations, phase_delta = insert_rotation(
                rotations, PauliRotation(transported, turns)
            )
            phase = (phase + turns + phase_delta) % 16
            if profile is not None:
                profile.rotation_insertion_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        parents = tuple(
            sorted(
                {
                    parent
                    for q in gate.qubits
                    if (parent := self.wire_tails[q]) is not None
                },
                key=lambda step: step.index,
            )
        )
        step = WitnessStep(self.tail, gate, parents, layer, self.gate_count)
        wire_tails = list(self.wire_tails)
        wire_depths = list(self.wire_depths)
        for q in gate.qubits:
            wire_tails[q] = step
            wire_depths[q] = layer
        if profile is not None:
            profile.dag_append_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        key = _canonical_key(self.num_qubits, tableau, rotations)
        if profile is not None:
            profile.canonical_key_ns += time.perf_counter_ns() - started
            profile.accepted += 1

        return HybridState(
            num_qubits=self.num_qubits,
            budget=self.budget,
            tail=step,
            wire_tails=tuple(wire_tails),
            tableau=tableau,
            rotations=rotations,
            global_phase_eighths=phase,
            t_count=next_t,
            cnot_count=next_cnot,
            gate_count=self.gate_count + 1,
            wire_depths=tuple(wire_depths),
            depth=max(wire_depths, default=0),
            canonical_key=key,
        )

    def reconstruct_gates(self) -> tuple[Gate, ...]:
        gates: list[Gate] = []
        current = self.tail
        while current is not None:
            gates.append(current.gate)
            current = current.previous
        gates.reverse()
        return tuple(gates)

    def materialize_dag(self) -> MaterializedDAG:
        steps: list[WitnessStep] = []
        current = self.tail
        while current is not None:
            steps.append(current)
            current = current.previous
        steps.reverse()
        identity_to_index = {id(step): index for index, step in enumerate(steps)}
        nodes = tuple(
            DAGNode(
                index=index,
                gate=step.gate,
                parents=tuple(sorted(identity_to_index[id(parent)] for parent in step.parents)),
                level=step.level,
            )
            for index, step in enumerate(steps)
        )
        dag = MaterializedDAG(self.num_qubits, nodes, self.wire_depths, self.depth)
        dag.validate()
        return dag

    def validate(self) -> None:
        self.tableau.validate()
        dag = self.materialize_dag()
        replay = HybridState.identity(self.num_qubits, self.budget)
        for gate in dag.gates:
            child = replay.apply(gate, partial_order_reduction=False)
            if child is None:
                raise AssertionError("witness cannot be replayed under its budget")
            replay = child
        if replay.canonical_key != self.canonical_key:
            raise AssertionError("symbolic state disagrees with the DAG witness")
        if replay.resource_vector() != self.resource_vector():
            raise AssertionError("resources disagree with the DAG witness")


def _canonical_key(
    num_qubits: int,
    tableau: CliffordTableau,
    rotations: tuple[PauliRotation, ...],
) -> tuple[object, ...]:
    return (
        "hybrid-clifford-pauli-incremental-v1",
        num_qubits,
        tableau.canonical_payload(),
        tuple(rotation.canonical_payload() for rotation in rotations),
    )


def _valid_gate(gate: Gate, num_qubits: int) -> bool:
    if not isinstance(gate, Gate) or gate.name not in CLIFFORD_GATES | NON_CLIFFORD_GATES:
        return False
    if not gate.qubits or any(
        isinstance(q, bool) or not isinstance(q, int) or q < 0 or q >= num_qubits
        for q in gate.qubits
    ):
        return False
    if len(set(gate.qubits)) != len(gate.qubits):
        return False
    if gate.name in SINGLE_QUBIT_GATES:
        return len(gate.qubits) == 1
    return gate.name == "CNOT" and len(gate.qubits) == 2


def generate_gates(num_qubits: int) -> tuple[Gate, ...]:
    gates: list[Gate] = []
    for q in range(num_qubits):
        for name in ("H", "S", "SDG", "T", "TDG"):
            gates.append(Gate(name, (q,)))
    for control in range(num_qubits):
        for target in range(num_qubits):
            if control != target:
                gates.append(Gate("CNOT", (control, target)))
    return tuple(gates)


__all__ = [
    "Budget",
    "DAGNode",
    "Gate",
    "HybridState",
    "MaterializedDAG",
    "TransitionProfile",
    "WitnessStep",
    "generate_gates",
]
