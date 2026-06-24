from certification.base import CertResult, CertStatus
from certification.base_engine import CertificationEngine


class SimulatorCertificationEngine(CertificationEngine):
    """
    Placeholder for statevector / unitary simulation
    """

    def certify(self, state) -> CertResult:
        # TODO: integrate Qiskit / custom simulator
        return CertResult(
            CertStatus.INCONCLUSIVE,
            score=0.0,
            info={"reason": "simulator_not_implemented"}
        )
