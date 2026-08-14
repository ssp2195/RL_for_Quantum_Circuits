from dataclasses import dataclass
from enums import GateType
from typing import Tuple


@dataclass(frozen=True)
class Gate:
    gate_type: GateType
    qubits: Tuple[int, ...]  # (target,) or (control, target)

    def is_two_qubit(self) -> bool:
        return self.gate_type == GateType.CNOT

    def is_non_clifford(self) -> bool:
        """Whether the gate contributes one Clifford+T non-Clifford unit."""
        return self.gate_type.name in {"T", "TDG"}

    def __repr__(self):
        return f"{self.gate_type.name}{self.qubits}"
