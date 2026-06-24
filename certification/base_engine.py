from abc import ABC, abstractmethod
from certification.base import CertResult


class CertificationEngine(ABC):
    """
    Abstract interface for certification engines
    """

    @abstractmethod
    def certify(self, state) -> CertResult:
        pass
