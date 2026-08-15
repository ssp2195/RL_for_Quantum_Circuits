from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Set

from circuit.gate import Gate


@dataclass(frozen=True, slots=True)
class GateNode:
    """A gate instance node in the dependency DAG.

    Attributes:
        id: Unique integer ID of this gate instance (not the gate type).
        gate: The ``Gate`` object stored at this node.
        parents: IDs of immediate dependency predecessors.
        children: IDs of immediate dependency successors.
        level: Dependency depth; source/root nodes are at level 1.
    """

    id: int
    gate: Gate
    parents: frozenset[int] = field(default_factory=frozenset)
    children: frozenset[int] = field(default_factory=frozenset)
    level: int = 1

    def __repr__(self):
        return f"GateNode(id={self.id}, gate={self.gate!r}, level={self.level})"


class CircuitDAG:
    """
    True dependency DAG for a quantum circuit.

    Nodes are gate instances; directed edges represent data dependencies
    between gates acting on shared qubits. Independent gates have no edges
    between them, even if one was inserted later.

    Invariants (see ``validate``):
      - unique node IDs
      - valid, symmetric parent/child edges
      - acyclicity (from construction)
      - topological order contains every node exactly once, respecting edges
      - ``last_gate_on_qubit`` tracks the most recent gate on each qubit
      - node levels follow the dependency recurrence
    """

    def __init__(self, num_qubits: int):
        if (
            isinstance(num_qubits, bool)
            or not isinstance(num_qubits, int)
            or num_qubits < 0
        ):
            raise ValueError("num_qubits must be a non-negative integer")
        self._num_qubits = num_qubits
        self._nodes: Dict[int, GateNode] = {}
        self._topological_order: List[int] = []
        self._last_gate_on_qubit: List[Optional[int]] = [None] * num_qubits
        self._next_node_id: int = 0
        self._depth: int = 0

    @property
    def num_qubits(self) -> int:
        """Fixed witness arity."""

        return self._num_qubits

    @property
    def nodes(self) -> Mapping[int, GateNode]:
        """Read-only node mapping for structural inspection."""

        return MappingProxyType(self._nodes)

    @property
    def topological_order(self) -> List[int]:
        """A defensive copy of the stable topological node order."""

        return list(self._topological_order)

    @property
    def last_gate_on_qubit(self) -> List[Optional[int]]:
        """A defensive copy of the last-gate cache."""

        return list(self._last_gate_on_qubit)

    @property
    def gates(self) -> List[Gate]:
        """Gates in topological order, derived from the DAG state."""
        return [self._nodes[node_id].gate for node_id in self._topological_order]

    def add_gate(self, gate: Gate) -> int:
        """Reject public raw mutation of an authoritative witness.

        Use :meth:`CircuitState.apply_gate` for a live synthesis state or
        :meth:`from_gates` when constructing a detached witness for replay or
        certification.  Keeping this former API as a hard failure makes stale
        semantic/resource caches immediately visible to existing callers.
        """

        raise RuntimeError(
            "CircuitDAG.add_gate is not a supported mutation boundary; "
            "use CircuitState.apply_gate or CircuitDAG.from_gates"
        )

    @classmethod
    def from_gates(cls, num_qubits: int, gates: Iterable[Gate]) -> "CircuitDAG":
        """Build and validate a detached DAG witness from circuit-order gates."""

        dag = cls(num_qubits)
        for gate in gates:
            dag._append_gate_unchecked(gate)
        dag.validate()
        return dag

    def _append_gate_unchecked(self, gate: Gate) -> int:
        """Internal structural append used by ``CircuitState`` and replay.

        This method performs full gate-shape validation but deliberately does
        not know about semantic summaries or resource budgets.  Public code
        must therefore use ``CircuitState.apply_gate``.
        """

        if not gate.qubits:
            raise ValueError(f"Gate {gate!r} must act on at least one qubit")

        for q in gate.qubits:
            if isinstance(q, bool) or not isinstance(q, int) or q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Invalid qubit {q!r} for gate {gate!r} on a "
                    f"{self.num_qubits}-qubit circuit"
                )

        if len(set(gate.qubits)) != len(gate.qubits):
            raise ValueError(f"Gate {gate!r} has duplicate qubit indices")

        node_id = self._next_node_id
        self._next_node_id += 1

        parents: Set[int] = set()
        for q in gate.qubits:
            last = self._last_gate_on_qubit[q]
            if last is not None:
                parents.add(last)

        level = 1 + max(
            (self._nodes[p].level for p in parents),
            default=0,
        )

        node = GateNode(
            id=node_id,
            gate=gate,
            parents=frozenset(parents),
            level=level,
        )
        self._nodes[node_id] = node

        for p in parents:
            parent = self._nodes[p]
            self._nodes[p] = GateNode(
                id=parent.id,
                gate=parent.gate,
                parents=parent.parents,
                children=parent.children | {node_id},
                level=parent.level,
            )

        for q in gate.qubits:
            self._last_gate_on_qubit[q] = node_id

        self._topological_order.append(node_id)
        self._depth = max(self._depth, level)

        return node_id

    def depth(self) -> int:
        return self._depth

    def size(self) -> int:
        return len(self._nodes)

    def copy(self):
        """Deep structural copy with independent node objects and edge sets."""
        new_dag = CircuitDAG(self.num_qubits)
        new_dag._nodes = {
            nid: GateNode(
                id=node.id,
                gate=node.gate,
                parents=frozenset(node.parents),
                children=frozenset(node.children),
                level=node.level,
            )
            for nid, node in self._nodes.items()
        }
        new_dag._topological_order = list(self._topological_order)
        new_dag._last_gate_on_qubit = list(self._last_gate_on_qubit)
        new_dag._next_node_id = self._next_node_id
        new_dag._depth = self._depth
        return new_dag

    def predecessors(self, node_id: int) -> Set[int]:
        """Immediate dependency predecessors (a copy)."""
        return set(self._nodes[node_id].parents)

    def successors(self, node_id: int) -> Set[int]:
        """Immediate dependency successors (a copy)."""
        return set(self._nodes[node_id].children)

    def roots(self) -> List[int]:
        """IDs of nodes with no parents."""
        return [nid for nid in self._topological_order if not self._nodes[nid].parents]

    def leaves(self) -> List[int]:
        """IDs of nodes with no children."""
        return [nid for nid in self._topological_order if not self._nodes[nid].children]

    def topological_nodes(self) -> List[GateNode]:
        """GateNode objects in topological order."""
        return [self._nodes[nid] for nid in self._topological_order]

    def ready_nodes(self, expanded: Optional[Set[int]] = None) -> List[int]:
        """IDs of nodes whose parents are all in ``expanded``."""
        expanded = set() if expanded is None else expanded
        return [
            nid
            for nid in self._topological_order
            if self._nodes[nid].parents.issubset(expanded)
        ]

    def validate(self) -> None:
        """Assert all DAG invariants. Raises AssertionError on violation."""
        nodes = self._nodes

        assert self._next_node_id == len(nodes)
        assert len(self._last_gate_on_qubit) == self.num_qubits

        # Unique node IDs and topo completeness/uniqueness.
        assert len(nodes) == len(set(nodes.keys()))
        assert len(self._topological_order) == len(set(self._topological_order))
        assert len(self._topological_order) == len(nodes)
        for nid in self._topological_order:
            assert nid in nodes, f"topological id {nid} missing from nodes"

        topo_index = {nid: i for i, nid in enumerate(self._topological_order)}

        for nid, node in nodes.items():
            assert node.id == nid
            # Valid qubit indices.
            assert node.gate.qubits, f"node {nid} has a gate with no qubits"
            for q in node.gate.qubits:
                assert not isinstance(q, bool) and isinstance(q, int), (
                    f"node {nid} has non-integer qubit {q!r}"
                )
                assert 0 <= q < self.num_qubits, (
                    f"node {nid} qubit {q} out of range"
                )
            assert len(set(node.gate.qubits)) == len(node.gate.qubits)
            # No self loops.
            assert nid not in node.parents
            assert nid not in node.children
            # Parent/child symmetry.
            for p in node.parents:
                assert p in nodes, f"node {nid} parent {p} missing"
                assert nid in nodes[p].children, (
                    f"parent {p} of {nid} has no matching child"
                )
            for c in node.children:
                assert c in nodes, f"node {nid} child {c} missing"
                assert nid in nodes[c].parents, (
                    f"child {c} of {nid} has no matching parent"
                )
            # Topological order respects edges.
            for p in node.parents:
                assert topo_index[p] < topo_index[nid], (
                    f"edge {p} -> {nid} violates topological order"
                )
            # Level recurrence.
            if node.parents:
                expected = 1 + max(nodes[p].level for p in node.parents)
            else:
                expected = 1
            assert node.level == expected, (
                f"node {nid} level {node.level} != expected {expected}"
            )

        # Reconstruct dependencies directly from circuit order.  This proves
        # both that every edge is the immediate shared-wire dependency and
        # that last_gate_on_qubit points to the *latest* gate, rather than
        # merely to some valid gate touching that wire.
        latest: List[Optional[int]] = [None] * self.num_qubits
        for nid in self._topological_order:
            node = nodes[nid]
            expected_parents = frozenset(
                parent
                for q in node.gate.qubits
                if (parent := latest[q]) is not None
            )
            assert node.parents == expected_parents, (
                f"node {nid} parents {sorted(node.parents)} != immediate "
                f"wire dependencies {sorted(expected_parents)}"
            )
            for q in node.gate.qubits:
                latest[q] = nid
        assert self._last_gate_on_qubit == latest, (
            "last_gate_on_qubit does not track the latest topological gate"
        )

        # Cached depth equals max level.
        max_level = max((node.level for node in nodes.values()), default=0)
        assert self._depth == max_level, (
            f"cached depth {self._depth} != max level {max_level}"
        )

    def __repr__(self):
        lines = [f"CircuitDAG(num_qubits={self.num_qubits}, depth={self._depth})"]
        lines.append("  nodes=[")
        for nid in self._topological_order:
            node = self._nodes[nid]
            lines.append(f"    {nid}: {node.gate!r},")
        lines.append("  ]")
        lines.append("  edges=[")
        for nid in self._topological_order:
            for c in sorted(self._nodes[nid].children):
                lines.append(f"    {nid} -> {c},")
        lines.append("  ]")
        return "\n".join(lines)
