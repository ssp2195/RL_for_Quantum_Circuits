from dataclasses import dataclass
from typing import Optional
from ckt_types import ResourceBudget


@dataclass
class Config:
    num_qubits: int
    budget: ResourceBudget

    # RL-related placeholders (used later)
    max_steps: int = 100
    discount: float = 1.0
    max_frontier: int = 100
    # Every Kth selection can be forced to the oldest open record.  Zero
    # disables the fairness interleave for purely learned experiments.
    fairness_interval: int = 0
    seed: Optional[int] = None
