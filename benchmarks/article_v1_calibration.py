"""Deterministic numerical calibration for the Article V1 raw certifier."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from benchmarks.article_native_corpus import load_article_v1_config, native_gate_grammar
from benchmarks.native_corpus import ccz_reference_benchmark
from benchmarks.toffoli import KNOWN_TOFFOLI_GATES, toffoli_reference_unitary
from certification.simulator import unitary_from_gates
from certification.unitary_phase_metrics import projective_unitary_metrics
from circuit.gate import Gate
from enums import GateType


CALIBRATION_SCHEMA = "article-v1-raw-metric-calibration-v1"


def _measurement(name: str, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    metrics = projective_unitary_metrics(left, right)
    return {
        "name": name,
        "delta": metrics.phase_frobenius_discrepancy,
        "normalized_trace_magnitude_raw": metrics.normalized_trace_magnitude_raw,
    }


def _next_simple_decimal(value: float) -> float:
    if value <= 0.0:
        return 1e-12
    return 10.0 ** math.ceil(math.log10(value))


def calibrate_certifier(config: str | Path, output: str | Path) -> dict[str, Any]:
    resolved = load_article_v1_config(config)
    equivalent: list[dict[str, Any]] = []
    distinct: list[dict[str, Any]] = []
    rng = random.Random(104729)
    maximum_generator_length = max(
        item.max_generator_length for item in resolved.difficulties
    )

    for width in (1, 2, 3):
        identity = np.eye(2**width, dtype=np.complex128)
        equivalent.append(_measurement(f"identity-{width}", identity, identity))
        grammar = native_gate_grammar(width)
        for gate in grammar:
            matrix = unitary_from_gates(width, (gate,))
            equivalent.append(_measurement(f"self-{width}-{gate.gate_type.name}-{gate.qubits}", matrix, matrix))
            equivalent.append(_measurement(f"global-phase-{width}-{gate.gate_type.name}-{gate.qubits}", np.exp(0.371j) * matrix, matrix))

        metamorphic = [
            (Gate(GateType.H, (0,)), Gate(GateType.H, (0,))),
            (Gate(GateType.T, (0,)), Gate(GateType.TDG, (0,))),
            (Gate(GateType.S, (0,)), Gate(GateType.SDG, (0,))),
        ]
        if width >= 2:
            metamorphic.append((Gate(GateType.CNOT, (0, 1)), Gate(GateType.CNOT, (0, 1))))
        for index, witness in enumerate(metamorphic):
            equivalent.append(_measurement(f"inverse-{width}-{index}", unitary_from_gates(width, witness), identity))
        equivalent.append(_measurement(f"tt-is-s-{width}", unitary_from_gates(width, (Gate(GateType.T, (0,)), Gate(GateType.T, (0,)))), unitary_from_gates(width, (Gate(GateType.S, (0,)),))))
        if width >= 2:
            left = unitary_from_gates(
                width, (Gate(GateType.T, (0,)), Gate(GateType.S, (1,)))
            )
            right = unitary_from_gates(
                width, (Gate(GateType.S, (1,)), Gate(GateType.T, (0,)))
            )
            equivalent.append(_measurement(f"safe-commuting-reorder-{width}", left, right))

        for index in range(64):
            length = rng.randint(1, maximum_generator_length)
            witness = tuple(rng.choice(grammar) for _ in range(length))
            matrix = unitary_from_gates(width, witness)
            equivalent.append(_measurement(f"random-self-{width}-{index}-length-{length}", matrix, matrix))

        t_matrix = unitary_from_gates(width, (Gate(GateType.T, (0,)),))
        tdg_matrix = unitary_from_gates(width, (Gate(GateType.TDG, (0,)),))
        h_matrix = unitary_from_gates(width, (Gate(GateType.H, (0,)),))
        distinct.append(_measurement(f"wrong-relative-phase-{width}", t_matrix, identity))
        distinct.append(_measurement(f"wrong-t-tdg-sign-{width}", t_matrix, tdg_matrix))
        distinct.append(_measurement(f"omitted-h-{width}", h_matrix, identity))
        distinct.append(_measurement(f"distinct-short-h-vs-t-{width}", h_matrix, t_matrix))
        localized = np.eye(2**width, dtype=np.complex128)
        localized[-1, -1] = np.exp(1.0j * 1e-3)
        distinct.append(_measurement(f"localized-unitary-perturbation-{width}", localized, identity))
        if width >= 2:
            distinct.append(_measurement(f"reversed-cnot-{width}", unitary_from_gates(width, (Gate(GateType.CNOT, (0, 1)),)), unitary_from_gates(width, (Gate(GateType.CNOT, (1, 0)),))))

    ghz_witness = (
        Gate(GateType.H, (0,)),
        Gate(GateType.CNOT, (0, 1)),
        Gate(GateType.CNOT, (1, 2)),
    )
    ghz = unitary_from_gates(3, ghz_witness)
    equivalent.append(_measurement("known-ghz", ghz, ghz))
    equivalent.append(
        _measurement(
            "known-toffoli",
            unitary_from_gates(3, KNOWN_TOFFOLI_GATES),
            toffoli_reference_unitary(),
        )
    )
    ccz = ccz_reference_benchmark()
    equivalent.append(
        _measurement(
            "known-ccz",
            unitary_from_gates(3, ccz.witness),
            ccz.unitary,
        )
    )

    floor = max(float(row["delta"]) for row in equivalent)
    provisional = _next_simple_decimal(10.0 * floor)
    minimum_distinct = min(float(row["delta"]) for row in distinct)
    report = {
        "schema_version": CALIBRATION_SCHEMA,
        "metric_schema": "projective-unitary-metrics-v2",
        "config_digest": resolved.digest,
        "equivalent_pair_count": len(equivalent),
        "non_equivalent_pair_count": len(distinct),
        "equivalent_floor": floor,
        "provisional_tau_cert": provisional,
        "frozen_tau_cert": float(resolved.experiment["certification_tolerance"]),
        "minimum_non_equivalent_delta": minimum_distinct,
        "maximum_generator_length": maximum_generator_length,
        "frozen_tau_identity": resolved.tau_identity,
        "identity_equivalent_floor_covered": floor <= resolved.tau_identity,
        "identity_separation_at_least_100x": minimum_distinct >= 100.0 * resolved.tau_identity,
        "tau_at_most_1e_6": provisional <= 1e-6,
        "separation_at_least_100x": minimum_distinct >= 100.0 * provisional,
        "passed": provisional <= 1e-6 and minimum_distinct >= 100.0 * provisional and float(resolved.experiment["certification_tolerance"]) >= provisional and floor <= resolved.tau_identity and minimum_distinct >= 100.0 * resolved.tau_identity,
        "equivalent_pairs": equivalent,
        "non_equivalent_pairs": distinct,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["CALIBRATION_SCHEMA", "calibrate_certifier"]
