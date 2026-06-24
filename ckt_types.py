from dataclasses import dataclass
from typing import Tuple, Optional
from enums import GateType


@dataclass(frozen=True)
class GateSpec:
    gate: GateType
    targets: Tuple[int, ...]          # e.g., (q,) or (control, target)
    controls: Optional[Tuple[int, ...]] = None


@dataclass
class ResourceBudget:
    max_t_count: int
    max_depth: int
    max_gates: int
