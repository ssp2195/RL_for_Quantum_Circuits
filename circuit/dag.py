from typing import List
from circuit.gate import Gate


class CircuitDAG:
    """
    Minimal DAG representation:
    - For now: sequential list (topological order)
    - Later: extend to full dependency graph if needed
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: List[Gate] = []

    def add_gate(self, gate: Gate):
        # No dependency checks yet (Stage 4 will handle canonicalization)
        self.gates.append(gate)

    def depth(self) -> int:
        # Simplified depth = number of layers (sequential for now)
        return len(self.gates)

    def size(self) -> int:
        return len(self.gates)

    def copy(self):
        new_dag = CircuitDAG(self.num_qubits)
        new_dag.gates = list(self.gates)
        return new_dag

    def __repr__(self):
        return " -> ".join(str(g) for g in self.gates)
