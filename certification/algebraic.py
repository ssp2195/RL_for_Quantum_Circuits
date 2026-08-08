from certification.base import CertResult, CertStatus
from certification.base_engine import CertificationEngine


class AlgebraicCertificationEngine(CertificationEngine):
    """
    Fast algebraic check based on phase polynomial
    """

    def __init__(self, target_phase_terms):
        """
        target_phase_terms: canonical tuple of (mask, coeff) pairs
        """
        self.target = dict(target_phase_terms)

    def certify(self, state) -> CertResult:
        if state.phase_poly is None:
            return CertResult(CertStatus.INCONCLUSIVE)

        current = dict(
            (m, c % 8)
            for m, c in state.phase_poly.terms.items()
            if c % 8 != 0
        )

        if current == self.target:
            return CertResult(CertStatus.SUCCESS, score=1.0)

        # Partial progress: if every current term is a compatible subterm
        # of the target, the branch may still reach it → INCONCLUSIVE.
        # Otherwise the state is provably wrong → FAILURE.
        for mask, coeff in current.items():
            if self.target.get(mask) != coeff:
                return CertResult(CertStatus.FAILURE, score=0.0)

        return CertResult(CertStatus.INCONCLUSIVE, score=0.5)
