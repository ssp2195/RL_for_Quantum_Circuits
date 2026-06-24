from dataclasses import dataclass
from typing import Dict


def parity(x: int) -> int:
    """Compute parity (mod 2) of integer bitmask."""
    return bin(x).count("1") % 2


@dataclass
class PhasePolynomial:
    """
    Represents phase polynomial:
        f(x) = sum coeff[mask] * parity(mask & x) mod 8
    """

    num_qubits: int

    # mask (int) -> coefficient (mod 8)
    terms: Dict[int, int]

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.terms = {}

    # ---------- Core Operations ----------

    def add_term(self, mask: int, coeff: int):
        coeff = coeff % 8
        if coeff == 0:
            return

        if mask in self.terms:
            new_coeff = (self.terms[mask] + coeff) % 8
            if new_coeff == 0:
                del self.terms[mask]
            else:
                self.terms[mask] = new_coeff
        else:
            self.terms[mask] = coeff

    # ---------- Gate Updates ----------

    def apply_T(self, qubit: int):
        """
        T gate adds phase term:
            +1 * x_q
        """
        mask = 1 << qubit
        self.add_term(mask, 1)

    def apply_S(self, qubit: int):
        """
        S gate = T^2
        """
        mask = 1 << qubit
        self.add_term(mask, 2)

    def apply_CNOT(self, control: int, target: int):
        """
        Substitute:
            x_target ← x_target ⊕ x_control
        Which transforms all masks.
        """
        new_terms = {}

        for mask, coeff in self.terms.items():
            bit_t = (mask >> target) & 1

            if bit_t:
                # flip control bit
                new_mask = mask ^ (1 << control)
            else:
                new_mask = mask

            if new_mask in new_terms:
                new_terms[new_mask] = (new_terms[new_mask] + coeff) % 8
            else:
                new_terms[new_mask] = coeff

        # clean zeros
        self.terms = {m: c for m, c in new_terms.items() if c % 8 != 0}

    # ---------- Utilities ----------

    def evaluate(self, x: int) -> int:
        """Evaluate polynomial at bitstring x."""
        total = 0
        for mask, coeff in self.terms.items():
            total += coeff * parity(mask & x)
        return total % 8

    def copy(self):
        new_pp = PhasePolynomial(self.num_qubits)
        new_pp.terms = dict(self.terms)
        return new_pp

    def __repr__(self):
        if not self.terms:
            return "0"
        return " + ".join(f"{c}*[{bin(m)}]" for m, c in self.terms.items())
