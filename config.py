from dataclasses import dataclass
from ckt_types import ResourceBudget


@dataclass
class Config:
    num_qubits: int
    budget: ResourceBudget

    # RL-related placeholders (used later)
    max_steps: int = 100
    discount: float = 0.99
