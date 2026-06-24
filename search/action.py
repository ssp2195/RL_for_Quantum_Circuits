from dataclasses import dataclass
from enums import GateType


@dataclass(frozen=True)
class Action:
    gate_type: GateType
    qubits: tuple  # (q,) or (control, target)

    def __repr__(self):
        return f"{self.gate_type.name}{self.qubits}"
