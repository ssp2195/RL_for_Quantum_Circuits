from dataclasses import dataclass, field
from typing import Optional

from circuit.dag import CircuitDAG
from circuit.gate import Gate
from enums import GateType
from ckt_types import ResourceBudget


@dataclass
class CircuitState:
    dag: CircuitDAG
    budget: ResourceBudget

    t_count: int = 0
    depth: int = 0
    num_gates: int = 0

    # placeholders for later stages
    phase_poly: Optional[object] = None
    tableau: Optional[object] = None

    def apply_gate(self, gate: Gate) -> bool:
        """
        Apply a gate if within budget.
        Returns True if applied, False otherwise.
        """

        # Check qubit bounds
        if any(q >= self.dag.num_qubits or q < 0 for q in gate.qubits):
            return False

        # Budget checks
        if not self._check_budget(gate):
            return False

        # Apply
        self.dag.add_gate(gate)
        self._update_resources(gate)


        # Phase polynomial updates (Stage 2)
        if self.phase_poly is None:
            from algebra.phase_polynomial import PhasePolynomial
            self.phase_poly = PhasePolynomial(self.dag.num_qubits)

        if gate.gate_type.name == "T":
            self.phase_poly.apply_T(gate.qubits[0])

        elif gate.gate_type.name == "S":
            self.phase_poly.apply_S(gate.qubits[0])

        elif gate.gate_type.name == "CNOT":
            self.phase_poly.apply_CNOT(gate.qubits[0], gate.qubits[1])


        # Tableau updates (Stage 3)
        if self.tableau is None:
            from algebra.tableau import CliffordTableau
            self.tableau = CliffordTableau(self.dag.num_qubits)

        if gate.gate_type.name == "H":
            self.tableau.apply_H(gate.qubits[0])

        elif gate.gate_type.name == "S":
            self.tableau.apply_S(gate.qubits[0])

        elif gate.gate_type.name == "CNOT":
            self.tableau.apply_CNOT(gate.qubits[0], gate.qubits[1])
        

        return True

    def _check_budget(self, gate: Gate) -> bool:
        # Gate count
        if self.num_gates + 1 > self.budget.max_gates:
            return False

        # Depth (simplified)
        if self.depth + 1 > self.budget.max_depth:
            return False

        # T-count
        if gate.gate_type == GateType.T:
            if self.t_count + 1 > self.budget.max_t_count:
                return False

        return True

    def _update_resources(self, gate: Gate):
        self.num_gates += 1
        self.depth = self.dag.depth()

        if gate.gate_type == GateType.T:
            self.t_count += 1

    def copy(self):
        return CircuitState(
            dag=self.dag.copy(),
            budget=self.budget,
            t_count=self.t_count,
            depth=self.depth,
            num_gates=self.num_gates,
            phase_poly=self.phase_poly.copy() if self.phase_poly is not None else None,
            tableau=self.tableau.copy() if self.tableau is not None else None,
        )

    def __repr__(self):
        return (
            f"CircuitState("
            f"gates={self.num_gates}, "
            f"depth={self.depth}, "
            f"T={self.t_count})\n"
            f"{self.dag}"
        )
