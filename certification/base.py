from enum import Enum, auto
from dataclasses import dataclass


class CertStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    INCONCLUSIVE = auto()


@dataclass
class CertResult:
    status: CertStatus
    score: float = 0.0   # optional confidence / reward signal
    info: dict = None
