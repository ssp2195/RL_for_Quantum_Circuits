from enum import Enum, auto


class GateType(Enum):
    H = auto()
    S = auto()
    T = auto()
    X = auto()
    CNOT = auto()


class ActionType(Enum):
    ADD_GATE = auto()
    NO_OP = auto()
