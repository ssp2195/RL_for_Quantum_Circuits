"""Exact scalar lift of a projective Clifford tableau (small-register cache).

A Clifford is fixed uniquely by its signed tableau and the convention that the
first nonzero amplitude of C|0> is positive real. Gaussian-integer amplitudes
with a common power-of-sqrt(2) denominator implement that convention without
floating point. This derived column cache is not a phase polynomial.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CliffordColumn:
    values: tuple[tuple[int, int], ...]
    denominator_power: int = 0

    @classmethod
    def identity(cls, n: int) -> "CliffordColumn":
        return cls(((1, 0),) + ((0, 0),) * ((1 << n) - 1))

    def apply(self, name: str, qubits: tuple[int, ...]) -> tuple["CliffordColumn", int]:
        """Return canonical G C|0> and the extracted phase in units pi/8."""
        values = list(self.values)
        power = self.denominator_power
        if name == 'CNOT':
            c, t = qubits
            values = [self.values[i ^ (((i >> c) & 1) << t)] for i in range(len(values))]
        elif name in ('S', 'SDG'):
            sign = 1 if name == 'S' else -1
            for i, (a, b) in enumerate(values):
                if i & (1 << qubits[0]):
                    values[i] = (-sign * b, sign * a)
        elif name == 'H':
            bit = 1 << qubits[0]
            for i in range(len(values)):
                if not i & bit:
                    a, b = self.values[i]
                    c, d = self.values[i | bit]
                    values[i], values[i | bit] = (a + c, b + d), (a - c, b - d)
            power += 1
        else:
            raise ValueError('only native Clifford gates have a Clifford scalar lift')
        a, b = next(z for z in values if z != (0, 0))
        if b == 0:
            k = 0 if a > 0 else 4
        elif a == 0:
            k = 2 if b > 0 else 6
        elif abs(a) == abs(b):
            k = {(True, True): 1, (False, True): 3,
                 (False, False): 5, (True, False): 7}[a > 0, b > 0]
        else:
            raise AssertionError('Clifford amplitude has a non-eighth-root phase')
        # Multiply by exp(-i*k*pi/4), with exact Gaussian arithmetic.
        u, v = ((1, 0), (1, -1), (0, -1), (-1, -1),
                (-1, 0), (-1, 1), (0, 1), (1, 1))[k]
        values = [(u*a - v*b, v*a + u*b) for a, b in values]
        power += k % 2
        while power >= 2 and all(a % 2 == b % 2 == 0 for a, b in values):
            values = [(a // 2, b // 2) for a, b in values]
            power -= 2
        return CliffordColumn(tuple(values), power), (2 * k) % 16

    def vector(self):
        """Numerical reference rendering; never used as an archive key."""
        import numpy as np
        return np.array([complex(a, b) for a, b in self.values]) / (2. ** (self.denominator_power / 2))


def symbolic_unitary(state):
    """Reference rendering of exp(i*pi*phi/8) C(Theta) prod R_P, not DAG replay."""
    import numpy as np
    column = state.clifford_lift.vector()
    matrix = np.empty((len(column), len(column)), dtype=complex)
    xs = [axis.to_matrix() for axis in state.tableau.forward_x]
    for j in range(len(column)):
        v = column.copy()
        for q, x in enumerate(xs):
            if j & (1 << q):
                v = x @ v
        matrix[:, j] = v
    matrix *= np.exp(1j * np.pi * state.global_phase_eighths / 8)
    for rotation in state.rotations:
        matrix = matrix @ rotation.to_matrix()
    return matrix
