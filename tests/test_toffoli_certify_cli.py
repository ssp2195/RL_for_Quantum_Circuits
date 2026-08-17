"""End-to-end artifact coverage for the public Toffoli certification runner."""

from __future__ import annotations

import json
from pathlib import Path

from toffoli_certify import main, run_toffoli_certification


_ARTIFACT_FILENAMES = {
    "summary": "summary.json",
    "report": "summary.md",
    "truth_table": "truth_table.csv",
    "resource_summary": "resource_summary.json",
    "candidate_unitary": "candidate_unitary.csv",
    "target_unitary": "target_unitary.csv",
    "circuit_diagram": "circuit.svg",
}


def test_toffoli_certification_runner_writes_complete_verifiable_artifacts(tmp_path: Path):
    report = run_toffoli_certification(tmp_path)
    semantic = report["candidate"]["semantic_validation"]
    resources = report["candidate"]["resource_validation"]

    assert report["correct"]
    assert semantic["exact_certified"]
    assert semantic["global_phase_equivalent"]
    assert semantic["truth_table_correct"]
    assert semantic["column_phase_consistent"]
    assert semantic["symbolic_agrees_with_dense"]
    assert resources["resource_accounting_correct"]
    mandatory = report["positive_checks"]
    assert mandatory["matches_published_t_lower_bound"]
    assert mandatory["matches_published_cnot_lower_bound"]
    assert report["truth_table"]
    assert report["negative_controls"]
    assert all(
        result["passed"] for result in report["negative_controls"].values()
    )

    assert set(_ARTIFACT_FILENAMES) <= set(report["artifacts"])
    for key, expected_name in _ARTIFACT_FILENAMES.items():
        path = Path(report["artifacts"][key])
        assert path.is_file(), f"missing {key} artifact"
        assert path.name == expected_name

    summary_path = Path(report["artifacts"]["summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["correct"]
    assert summary["candidate"]["semantic_validation"]["exact_certified"]
    assert summary["target"]

    truth_table_lines = Path(report["artifacts"]["truth_table"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(truth_table_lines) == 9  # header plus the eight input columns
    assert "input" in truth_table_lines[0].lower()
    assert "output" in truth_table_lines[0].lower()

    svg = Path(report["artifacts"]["circuit_diagram"]).read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Toffoli" in svg


def test_corrupt_mandatory_check_never_reports_a_successful_certification(tmp_path: Path):
    artifact_dir = tmp_path / "corrupt"
    report = run_toffoli_certification(artifact_dir, corrupt_mandatory_check=True)

    assert not report["correct"]
    assert report["candidate"]
    assert report["negative_controls"]
    for artifact_path in report["artifacts"].values():
        assert Path(artifact_path).is_file()

    assert main(
        [
            "--artifacts-dir",
            str(tmp_path / "corrupt-cli"),
            "--corrupt-mandatory-check",
        ]
    ) == 1
