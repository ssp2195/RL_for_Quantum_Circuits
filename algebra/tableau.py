import numpy as np


class CliffordTableau:
    """
    Stabilizer tableau (binary symplectic form)

    Stores:
        X: (n, n)
        Z: (n, n)
        phase: (n,)
    """

    def __init__(self, num_qubits: int):
        self.n = num_qubits

        self.X = np.eye(self.n, dtype=np.uint8)
        self.Z = np.zeros((self.n, self.n), dtype=np.uint8)
        self.phase = np.zeros(self.n, dtype=np.uint8)

    # ---------- Gate Updates ----------

    def apply_H(self, q: int):
        # swap X and Z columns
        self.X[:, q], self.Z[:, q] = self.Z[:, q].copy(), self.X[:, q].copy()

    def apply_S(self, q: int):
        # Z = Z ⊕ X
        self.Z[:, q] ^= self.X[:, q]

    def apply_CNOT(self, control: int, target: int):
        # X_target ^= X_control
        self.X[:, target] ^= self.X[:, control]

        # Z_control ^= Z_target
        self.Z[:, control] ^= self.Z[:, target]

    # ---------- Utilities ----------

    def copy(self):
        new_tab = CliffordTableau(self.n)
        new_tab.X = self.X.copy()
        new_tab.Z = self.Z.copy()
        new_tab.phase = self.phase.copy()
        return new_tab

    def __repr__(self):
        return f"X=\n{self.X}\nZ=\n{self.Z}\nphase={self.phase}"
