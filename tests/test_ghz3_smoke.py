"""End-to-end deterministic GHZ-3 search and artifact smoke test."""

import json
from pathlib import Path

import numpy as np

from ghz3_smoke import run_ghz3_smoke


def test_ghz3_frontier_baseline_discovers_the_known_optimal_witness(tmp_path: Path):
    report = run_ghz3_smoke(tmp_path, max_steps=256, seed=0)

    assert report["correct"]
    assert report["search"]["certified"]
    assert report["search"]["terminated"]
    assert not report["search"]["truncated"]
    assert report["search"]["witness_operations"] == [
        {"gate": "H", "qubits": [0]},
        {"gate": "CNOT", "qubits": [0, 1]},
        {"gate": "CNOT", "qubits": [0, 2]},
    ]
    assert report["state_preparation"]["fidelity"] >= 1.0 - 1e-12
    assert report["resource_baseline"]["matches_known_native_resource_baseline"]
    assert report["optimality"] == report["resource_baseline"]
    assert "optimality" in report["deprecated_report_fields"]

    artifacts = report["artifacts"]
    for artifact_path in artifacts.values():
        assert Path(artifact_path).is_file()

    summary = json.loads(Path(artifacts["summary"]).read_text(encoding="utf-8"))
    assert summary["correct"]
    np.testing.assert_allclose(
        summary["state_preparation"]["probabilities"],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        atol=1e-12,
    )
    circuit_svg = Path(artifacts["circuit_diagram"]).read_text(encoding="utf-8")
    assert "CNOT 0-&gt;1" in circuit_svg
    assert "CNOT 0-&gt;2" in circuit_svg


def test_failed_search_does_not_render_the_reference_as_a_generated_circuit(
    tmp_path: Path,
):
    report = run_ghz3_smoke(tmp_path, max_steps=1, seed=0)

    assert not report["correct"]
    assert not report["search"]["certified"]
    assert report["search"]["truncated"]
    circuit_svg = Path(report["artifacts"]["circuit_diagram"]).read_text(encoding="utf-8")
    assert "No certified GHZ-3 witness" in circuit_svg
    assert "CNOT 0-&gt;1" not in circuit_svg
