from certification.base import CertResult, CertStatus
from certification.base_engine import CertificationEngine


class AlgebraicCertificationEngine(CertificationEngine):
    """
    Fast algebraic check based on phase polynomial
    """

    def __init__(self, target_phase_terms):
        """
        target_phase_terms: canonical tuple from phase polynomial
        """
        self.target = target_phase_terms

    def certify(self, state) -> CertResult:
        if state.phase_poly is None:
            return CertResult(CertStatus.INCONCLUSIVE)

        current = tuple(sorted(
            (m, c % 8) for m, c in state.phase_poly.terms.items()
        ))

        if current == self.target:
            return CertResult(CertStatus.SUCCESS, score=1.0)

        return CertResult(CertStatus.FAILURE, score=0.0)
