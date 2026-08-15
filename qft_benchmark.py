"""Run the QFT-3 reference/capability/AQFT benchmark and save artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from benchmarks.qft import (
    ExactCapability,
    analytical_qft_matrix,
    aqft3_metrics,
    assess_native_exact_capability,
    declared_qft_target_matrix,
    prepare_native_exact_search,
    qft_reference,
    reference_unitary,
)
from reporting.qft import save_qft_benchmark_artifacts
from search.action_space import generate_actions


MATRIX_TOLERANCE = 2e-12


def _maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def run_qft_benchmark(output_dir: str | Path) -> dict[str, Any]:
    """Validate every declared QFT boundary and save a compact report."""

    exact_forward = qft_reference(3)
    exact_inverse = qft_reference(3, inverse=True)
    no_swap_forward = qft_reference(3, include_final_swaps=False)
    no_swap_inverse = qft_reference(3, inverse=True, include_final_swaps=False)

    analytical_forward = analytical_qft_matrix(3)
    analytical_inverse = analytical_qft_matrix(3, inverse=True)
    operation_errors = {
        "forward": _maximum_error(reference_unitary(exact_forward), analytical_forward),
        "inverse": _maximum_error(reference_unitary(exact_inverse), analytical_inverse),
        "forward_no_swaps": _maximum_error(
            reference_unitary(no_swap_forward),
            declared_qft_target_matrix(no_swap_forward),
        ),
        "inverse_no_swaps": _maximum_error(
            reference_unitary(no_swap_inverse),
            declared_qft_target_matrix(no_swap_inverse),
        ),
    }
    inverse_adjoint_error = _maximum_error(
        analytical_inverse, analytical_forward.conj().T
    )
    analytical_unitarity_error = _maximum_error(
        analytical_forward.conj().T @ analytical_forward, np.eye(8)
    )
    swap_targets_are_distinct = not np.allclose(
        declared_qft_target_matrix(no_swap_forward),
        analytical_forward,
        atol=MATRIX_TOLERANCE,
        rtol=MATRIX_TOLERANCE,
    )

    capability = assess_native_exact_capability(exact_forward)
    exact_request = prepare_native_exact_search(exact_forward)
    qft1_capability = assess_native_exact_capability(qft_reference(1))
    qft2_capability = assess_native_exact_capability(qft_reference(2))

    aqft = aqft3_metrics()
    omitted_angles = [
        str(operation.angle_pi) for operation in aqft.reference.omitted_operations
    ]
    expected_aqft_process_fidelity = float((10.0 + 3.0 * np.sqrt(2.0)) / 16.0)
    native_action_names = sorted(
        {action.gate_type.name for action in generate_actions(3)}
    )
    expected_native_actions = ["CNOT", "H", "S", "SDG", "T", "TDG"]

    checks = {
        "analytical_forward_is_unitary": analytical_unitarity_error
        <= MATRIX_TOLERANCE,
        "analytical_inverse_is_adjoint": inverse_adjoint_error <= MATRIX_TOLERANCE,
        "high_level_forward_matches_analytical": operation_errors["forward"]
        <= MATRIX_TOLERANCE,
        "high_level_inverse_matches_analytical": operation_errors["inverse"]
        <= MATRIX_TOLERANCE,
        "no_swap_forward_matches_declared_permutation": operation_errors[
            "forward_no_swaps"
        ]
        <= MATRIX_TOLERANCE,
        "no_swap_inverse_matches_declared_permutation": operation_errors[
            "inverse_no_swaps"
        ]
        <= MATRIX_TOLERANCE,
        "swap_and_no_swap_targets_are_distinct": swap_targets_are_distinct,
        "qft1_is_exact_native": qft1_capability.classification
        is ExactCapability.EXACT_NATIVE,
        "qft2_is_exact_decomposable": qft2_capability.classification
        is ExactCapability.EXACT_DECOMPOSABLE,
        "qft3_requires_approximation": capability.classification
        is ExactCapability.APPROXIMATION_REQUIRED,
        "qft3_exact_target_not_submitted": not exact_request.accepted
        and exact_request.target is None,
        "aqft3_omits_only_pi_over_4": omitted_angles == ["1/4"],
        "aqft3_process_fidelity_matches_declared_omission": np.isclose(
            aqft.process_fidelity,
            expected_aqft_process_fidelity,
            atol=MATRIX_TOLERANCE,
            rtol=MATRIX_TOLERANCE,
        ),
        "native_search_grammar_is_unchanged": native_action_names
        == expected_native_actions,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "qft3-reference-benchmark-v1",
        "passed": passed,
        "scope": (
            "Analytical and high-level QFT-3 reference validation, exact native "
            "capability guard, and separately labelled AQFT-3 scoring; no search "
            "or synthesis claim."
        ),
        "checks": checks,
        "matrix_tolerance": MATRIX_TOLERANCE,
        "exact_qft3": {
            "reference": exact_forward.metadata(),
            "inverse_reference": exact_inverse.metadata(),
            "no_swap_reference": no_swap_forward.metadata(),
            "no_swap_inverse_reference": no_swap_inverse.metadata(),
            "operation_matrix_max_errors": operation_errors,
            "analytical_inverse_adjoint_max_error": inverse_adjoint_error,
            "analytical_unitarity_max_error": analytical_unitarity_error,
            "capability": capability.metadata(),
            "native_search_target_created": exact_request.target is not None,
        },
        "aqft3": aqft.metadata(),
        "native_search_grammar": {
            "actions": native_action_names,
            "parameterized_reference_operations_present": False,
        },
        "limitations": [
            "The diagrams are high-level references, not native circuit witnesses.",
            "AQFT fidelity metrics are not exact-certification results.",
            "No RL policy or frontier search is run by this benchmark.",
        ],
    }
    report["artifacts"] = save_qft_benchmark_artifacts(
        output_dir,
        report=report,
        exact_reference=exact_forward,
        approximate_reference=aqft.reference,
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs") / "qft3-reference",
        help="directory for summary JSON/Markdown and high-level SVG diagrams",
    )
    args = parser.parse_args(argv)
    report = run_qft_benchmark(args.artifacts_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
