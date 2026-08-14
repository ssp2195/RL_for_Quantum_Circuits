from certification.base import CertResult, CertStatus


class CompositeCertificationEngine:
    """
    Runs multiple certification engines in sequence
    """

    def __init__(self, engines):
        self.engines = engines

    def certify(self, state):
        result = CertResult(CertStatus.INCONCLUSIVE, score=0.0)
        for engine in self.engines:
            result = engine.certify(state)

            if result.status == CertStatus.SUCCESS:
                return result

            if result.status == CertStatus.FAILURE:
                return result

        # If all inconclusive
        return result
