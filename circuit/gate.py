from dataclasses import dataclass
from enums import GateType
from typing import Tuple


@dataclass(frozen=True)
class Gate:
    gate_type: GateType
    qubits: Tuple[int, ...]  # (target,) or (control, target)

    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2

    def __repr__(self):
        return f"{self.gate_type.name}{self.qubits}"
