from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from circuit.gate import Gate


@dataclass
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
    parents: Set[int] = field(default_factory=set)
    children: Set[int] = field(default_factory=set)
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
        self.num_qubits = num_qubits
        self.nodes: Dict[int, GateNode] = {}
        self.topological_order: List[int] = []
        self.last_gate_on_qubit: List[Optional[int]] = [None] * num_qubits
        self._next_node_id: int = 0
        self._depth: int = 0

    @property
    def gates(self) -> List[Gate]:
        """Gates in topological order, derived from the DAG state."""
        return [self.nodes[node_id].gate for node_id in self.topological_order]

    def add_gate(self, gate: Gate) -> int:
        """Add a gate, wiring true dependency edges, and return its node ID."""
        if not gate.qubits:
            raise ValueError(f"Gate {gate!r} must act on at least one qubit")

        for q in gate.qubits:
            if not isinstance(q, int) or q < 0 or q >= self.num_qubits:
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
            last = self.last_gate_on_qubit[q]
            if last is not None:
                parents.add(last)

        level = 1 + max(
            (self.nodes[p].level for p in parents),
            default=0,
        )

        node = GateNode(id=node_id, gate=gate, parents=parents, level=level)
        self.nodes[node_id] = node

        for p in parents:
            self.nodes[p].children.add(node_id)

        for q in gate.qubits:
            self.last_gate_on_qubit[q] = node_id

        self.topological_order.append(node_id)
        self._depth = max(self._depth, level)

        return node_id

    def depth(self) -> int:
        return self._depth

    def size(self) -> int:
        return len(self.nodes)

    def copy(self):
        """Deep structural copy with independent node objects and edge sets."""
        new_dag = CircuitDAG(self.num_qubits)
        new_dag.nodes = {
            nid: GateNode(
                id=node.id,
                gate=node.gate,
                parents=set(node.parents),
                children=set(node.children),
                level=node.level,
            )
            for nid, node in self.nodes.items()
        }
        new_dag.topological_order = list(self.topological_order)
        new_dag.last_gate_on_qubit = list(self.last_gate_on_qubit)
        new_dag._next_node_id = self._next_node_id
        new_dag._depth = self._depth
        return new_dag

    def predecessors(self, node_id: int) -> Set[int]:
        """Immediate dependency predecessors (a copy)."""
        return set(self.nodes[node_id].parents)

    def successors(self, node_id: int) -> Set[int]:
        """Immediate dependency successors (a copy)."""
        return set(self.nodes[node_id].children)

    def roots(self) -> List[int]:
        """IDs of nodes with no parents."""
        return [nid for nid in self.topological_order if not self.nodes[nid].parents]

    def leaves(self) -> List[int]:
        """IDs of nodes with no children."""
        return [nid for nid in self.topological_order if not self.nodes[nid].children]

    def topological_nodes(self) -> List[GateNode]:
        """GateNode objects in topological order."""
        return [self.nodes[nid] for nid in self.topological_order]

    def ready_nodes(self, expanded: Optional[Set[int]] = None) -> List[int]:
        """IDs of nodes whose parents are all in ``expanded``."""
        expanded = set() if expanded is None else expanded
        return [
            nid
            for nid in self.topological_order
            if self.nodes[nid].parents.issubset(expanded)
        ]

    def validate(self) -> None:
        """Assert all DAG invariants. Raises AssertionError on violation."""
        nodes = self.nodes

        assert self._next_node_id == len(nodes)

        # Unique node IDs and topo completeness/uniqueness.
        assert len(nodes) == len(set(nodes.keys()))
        assert len(self.topological_order) == len(set(self.topological_order))
        assert len(self.topological_order) == len(nodes)
        for nid in self.topological_order:
            assert nid in nodes, f"topological id {nid} missing from nodes"

        topo_index = {nid: i for i, nid in enumerate(self.topological_order)}

        for nid, node in nodes.items():
            assert node.id == nid
            # Valid qubit indices.
            assert node.gate.qubits, f"node {nid} has a gate with no qubits"
            for q in node.gate.qubits:
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

        # last_gate_on_qubit points to valid, gate-qubit-consistent nodes.
        for q in range(self.num_qubits):
            last = self.last_gate_on_qubit[q]
            if last is not None:
                assert last in nodes, f"last gate on qubit {q} missing"
                assert q in nodes[last].gate.qubits

        # Cached depth equals max level.
        max_level = max((node.level for node in nodes.values()), default=0)
        assert self._depth == max_level, (
            f"cached depth {self._depth} != max level {max_level}"
        )

    def __repr__(self):
        lines = [f"CircuitDAG(num_qubits={self.num_qubits}, depth={self._depth})"]
        lines.append("  nodes=[")
        for nid in self.topological_order:
            node = self.nodes[nid]
            lines.append(f"    {nid}: {node.gate!r},")
        lines.append("  ]")
        lines.append("  edges=[")
        for nid in self.topological_order:
            for c in sorted(self.nodes[nid].children):
                lines.append(f"    {nid} -> {c},")
        lines.append("  ]")
        return "\n".join(lines)
