import numpy as np

from circuit.circuit_state import CircuitState
from enums import GateType


# ---------- Utility ----------

def _safe_div(x, y):
    return x / y if y > 0 else 0.0


def _hamming_weight(x: int) -> int:
    return bin(x).count("1")


# ---------- Main Feature Extractor ----------

def extract_features(state: CircuitState) -> np.ndarray:
    """
    Converts CircuitState → feature vector φ(s)
    """

    features = []

    budget = state.budget

    # =========================================================
    # 1. Resource Features (normalized)
    # =========================================================

    features.append(_safe_div(state.t_count, budget.max_t_count))
    features.append(_safe_div(state.depth, budget.max_depth))
    features.append(_safe_div(state.num_gates, budget.max_gates))

    # =========================================================
    # 2. Remaining Budget (pressure signals)
    # =========================================================

    remaining_t = budget.max_t_count - state.t_count
    remaining_depth = budget.max_depth - state.depth
    remaining_gates = budget.max_gates - state.num_gates

    features.append(_safe_div(remaining_t, budget.max_t_count))
    features.append(_safe_div(remaining_depth, budget.max_depth))
    features.append(_safe_div(remaining_gates, budget.max_gates))

    # =========================================================
    # 3. Phase Polynomial Features
    # =========================================================

    if state.phase_poly is None or not state.phase_poly.terms:
        features.extend([0.0, 0.0, 0.0, 0.0])
    else:
        terms = state.phase_poly.terms

        num_terms = len(terms)

        coeffs = np.array([c % 8 for c in terms.values()], dtype=float)

        avg_coeff = np.mean(coeffs)
        max_coeff = np.max(coeffs)

        # degree = hamming weight of masks
        degrees = np.array(
            [_hamming_weight(mask) for mask in terms.keys()],
            dtype=float
        )

        avg_degree = np.mean(degrees)

        # normalize
        features.append(num_terms / (2 ** state.dag.num_qubits))
        features.append(avg_coeff / 8.0)
        features.append(max_coeff / 8.0)
        features.append(avg_degree / state.dag.num_qubits)

    # =========================================================
    # 4. Structural Features
    # =========================================================

    dag = state.dag

    # gate counts
    num_two_qubit = sum(
        1 for g in dag.gates if len(g.qubits) == 2
    )

    features.append(_safe_div(num_two_qubit, max(1, state.num_gates)))

    # depth ratio (redundant but useful signal)
    features.append(_safe_div(dag.depth(), budget.max_depth))

    # =========================================================
    # Final vector
    # =========================================================

    return np.array(features, dtype=np.float32)
