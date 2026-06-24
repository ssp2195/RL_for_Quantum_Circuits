from typing import List

from enums import GateType
from search.action import Action


def generate_actions(num_qubits: int) -> List[Action]:
    actions = []

    # Single-qubit gates
    for q in range(num_qubits):
        actions.append(Action(GateType.H, (q,)))
        actions.append(Action(GateType.S, (q,)))
        actions.append(Action(GateType.T, (q,)))

    # Two-qubit gates (CNOT)
    for control in range(num_qubits):
        for target in range(num_qubits):
            if control != target:
                actions.append(Action(GateType.CNOT, (control, target)))

    return actions
