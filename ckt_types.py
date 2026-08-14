from dataclasses import dataclass
from typing import Optional, Tuple
from enums import GateType


@dataclass(frozen=True)
class GateSpec:
    gate: GateType
    targets: Tuple[int, ...]          # e.g., (q,) or (control, target)
    controls: Optional[Tuple[int, ...]] = None


@dataclass
class ResourceBudget:
    """Monotone resource limits for a synthesis witness.

    ``max_two_qubit_count`` is optional to preserve the original three-field
    constructor while allowing the frontier archive to retain an entangling
    gate objective independently of total gate count.
    """

    max_t_count: int
    max_depth: int
    max_gates: int
    max_two_qubit_count: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("max_t_count", "max_depth", "max_gates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        value = self.max_two_qubit_count
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("max_two_qubit_count must be None or a non-negative integer")
