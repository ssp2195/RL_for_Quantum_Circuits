"""Legacy algebraic certification hints.

The phase-polynomial data structure is not a complete representation of an
arbitrary Clifford+T circuit: in particular it cannot account for interleaved
Hadamards or the full Clifford action.  It must therefore never be used as a
terminal success oracle.  The independent dense simulator is responsible for
all general Clifford+T certification.
"""

from certification.base import CertResult, CertStatus
from certification.base_engine import CertificationEngine


class AlgebraicCertificationEngine(CertificationEngine):
    """Conservative legacy hint engine for phase-polynomial search metadata.

    ``target_phase_terms`` remains accepted for compatibility with callers
    built around the original API.  Matching terms are deliberately not
    interpreted as unitary equality, because the representation omits enough
    information to make that conclusion unsound even for some CNOT-containing
    circuits.  This engine consequently returns only ``INCONCLUSIVE``.
    """

    def __init__(self, target_phase_terms):
        self.target = dict(target_phase_terms)

    def certify(self, state) -> CertResult:
        return CertResult(
            CertStatus.INCONCLUSIVE,
            score=0.0,
            info={
                "reason": "phase_polynomial_is_not_a_complete_clifford_t_oracle"
            },
        )
