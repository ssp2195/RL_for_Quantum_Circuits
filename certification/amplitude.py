from certification.base import CertResult, CertStatus
from certification.base_engine import CertificationEngine


class AmplitudeCertificationEngine(CertificationEngine):
    """
    Placeholder for VA-QPE / amplitude amplification
    """

    def certify(self, state) -> CertResult:
        return CertResult(
            CertStatus.INCONCLUSIVE,
            score=0.0,
            info={"reason": "amplitude_not_implemented"}
        )
