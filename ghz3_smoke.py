"""Deterministic GHZ-3 state-preparation smoke test and artifact generator.

This module deliberately separates two claims:

* the reference circuit is validated as preparation of the GHZ-3 *state*;
* the existing exact frontier baseline is asked to rediscover the same native
  witness under its tight known resource budget.

It does not train an RL policy and does not claim a general learned-synthesis
benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from certification.simulator import (
    equivalent_up_to_global_phase,
    state_fidelity,
    statevector_from_gates,
    unitary_from_gates,
)
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from evaluate import evaluate
from reporting import save_ghz3_artifacts


MAX_STATE_INFIDELITY = 1e-12
GHZ3_GATES: tuple[Gate, ...] = (
    Gate(GateType.H, (0,)),
    Gate(GateType.CNOT, (0, 1)),
    Gate(GateType.CNOT, (0, 2)),
)
GHZ3_BUDGET = ResourceBudget(
    max_t_count=0,
    max_depth=3,
    max_gates=3,
    max_two_qubit_count=2,
)


def expected_ghz3_state() -> np.ndarray:
    """Return the analytical positive GHZ-3 state in the native basis order."""

    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1.0 / np.sqrt(2.0)
    expected[7] = 1.0 / np.sqrt(2.0)
    return expected


def build_state(gates: Sequence[Gate], *, budget: ResourceBudget) -> CircuitState:
    """Build a circuit through the public state/DAG transition path."""

    state = CircuitState(CircuitDAG(3), budget)
    for gate in gates:
        if not state.apply_gate(gate):
            raise ValueError(f"GHZ-3 witness gate is illegal under its budget: {gate!r}")
        state.dag.validate()
    return state


def build_reference_ghz3_state() -> CircuitState:
    """Construct the canonical native GHZ-3 reference witness."""

    return build_state(GHZ3_GATES, budget=GHZ3_BUDGET)


def _operations_to_gates(operations: Sequence[dict[str, Any]]) -> list[Gate]:
    return [
        Gate(GateType[str(operation["gate"])], tuple(operation["qubits"]))
        for operation in operations
    ]


def _directed_structure_matches(gates: Sequence[Gate]) -> bool:
    if len(gates) != 3:
        return False
    h_gates = [gate for gate in gates if gate.gate_type is GateType.H]
    cnot_pairs = {
        gate.qubits for gate in gates if gate.gate_type is GateType.CNOT
    }
    return (
        gates[0] == Gate(GateType.H, (0,))
        and h_gates == [Gate(GateType.H, (0,))]
        and cnot_pairs == {(0, 1), (0, 2)}
        and all(gate.gate_type not in {GateType.T, GateType.TDG} for gate in gates)
    )


def validate_ghz3_state_preparation(state: CircuitState) -> tuple[np.ndarray, dict[str, Any]]:
    """Independently validate a public DAG witness against analytical GHZ+."""

    state.dag.validate()
    actual = statevector_from_gates(3, state.dag.gates)
    expected = expected_ghz3_state()
    fidelity = state_fidelity(expected, actual, atol=MAX_STATE_INFIDELITY)
    probabilities = np.abs(actual) ** 2
    dense_witness = unitary_from_gates(3, state.dag.gates)
    symbolic_agrees = equivalent_up_to_global_phase(
        state.symbolic_unitary(), dense_witness, atol=1e-9, rtol=1e-9
    )
    return actual, {
        "passed": bool(fidelity >= 1.0 - MAX_STATE_INFIDELITY),
        "fidelity": float(fidelity),
        "norm": float(np.linalg.norm(actual)),
        "all_finite": bool(np.isfinite(actual).all()),
        "probabilities": [float(value) for value in probabilities],
        "symbolic_agrees_with_dense_witness": bool(symbolic_agrees),
        "resources": {
            "num_gates": state.num_gates,
            "two_qubit_count": state.two_qubit_count,
            "t_count": state.t_count,
            "depth": state.depth,
            "wire_depths": list(state.wire_depths),
            # The continuation contract explicitly has no ancilla support.
            "ancilla_count": 0,
        },
    }


def run_ghz3_smoke(
    output_dir: str | Path,
    *,
    max_steps: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the reference validation and deterministic native-witness search."""

    reference_state = build_reference_ghz3_state()
    _, reference_preparation = validate_ghz3_state_preparation(reference_state)

    # ``evaluate`` uses its existing dense unitary target only to locate a
    # witness.  The state-preparation evidence below is independently checked
    # on |000> and must not be mistaken for a general state target API.
    search = evaluate(
        num_qubits=3,
        target_gates=GHZ3_GATES,
        budget=GHZ3_BUDGET,
        max_steps=max_steps,
        seed=seed,
        scheduler="fifo",
        collect_trace=True,
    )
    generated_gates = _operations_to_gates(search["witness_operations"])
    generated_preparation: dict[str, Any]
    generated_statevector: np.ndarray
    if search["certified"]:
        generated_state = build_state(generated_gates, budget=GHZ3_BUDGET)
        generated_statevector, generated_preparation = validate_ghz3_state_preparation(
            generated_state
        )
    else:
        generated_statevector = np.zeros(8, dtype=np.complex128)
        generated_preparation = {
            "passed": False,
            "reason": "search did not return a certified witness",
            "resources": {},
        }

    expected_resources = {
        "num_gates": 3,
        "two_qubit_count": 2,
        "t_count": 0,
        "depth": 3,
        "wire_depths": [3, 2, 3],
        "ancilla_count": 0,
    }
    resource_match = generated_preparation.get("resources") == expected_resources
    structure_match = _directed_structure_matches(generated_gates)
    correct = bool(
        search["certified"]
        and generated_preparation.get("passed", False)
        and generated_preparation.get("symbolic_agrees_with_dense_witness", False)
        and structure_match
        and resource_match
    )
    report: dict[str, Any] = {
        "correct": correct,
        "scope": (
            "GHZ-3 state-preparation smoke test plus deterministic exact "
            "frontier witness discovery; not a trained-policy benchmark."
        ),
        "reference_state_preparation": reference_preparation,
        "state_preparation": generated_preparation,
        "search": search,
        "optimality": {
            "matches_known_native_resource_baseline": bool(
                structure_match and resource_match
            ),
            "expected_resources": expected_resources,
            "justification": (
                "Two entangling gates are required to connect three qubits; "
                "the native H plus two shared-control CNOTs has three gates "
                "and depth three."
            ),
        },
    }
    report["artifacts"] = save_ghz3_artifacts(
        output_dir,
        report=report,
        # A failed search must never receive a diagram of the reference
        # witness: that would falsely imply that a circuit was discovered.
        gates=generated_gates,
        statevector=generated_statevector,
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs") / "ghz3-smoke",
        help="directory for JSON, CSV, SVG, and Markdown evaluation artifacts",
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = run_ghz3_smoke(
        args.artifacts_dir, max_steps=args.max_steps, seed=args.seed
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["correct"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
