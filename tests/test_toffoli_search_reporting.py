"""Contract tests for learned-Toffoli artifact presentation only."""

from __future__ import annotations

import json
from pathlib import Path

from circuit.gate import Gate
from enums import GateType
from reporting.toffoli_search import save_toffoli_search_artifacts


def _report(*, certified: bool) -> dict[str, object]:
    return {
        "correct": certified,
        "seed": 23,
        "problem": {
            "name": "ToffoliParityNetworkProblem",
            "schema_version": "toffoli-parity-network-v1",
        },
        "phase_identity": {"passed": certified, "mode": "literal"},
        "fifo": {"trace": [{"expansion": 1, "frontier_size": 2, "stage": 0}]},
        "uniform": {"trace": []},
        "random": {"trace": [{"expansion": 1, "frontier_size": 2}]},
        "zero_policy": {"trace": []},
        "training_history": [{"episode": 1, "reward": 0.5, "steps": 2}],
        "learned": {
            "certified": certified,
            "expansions": 1,
            "trace": [
                {
                    "expansion": 1,
                    "frontier_size": 3,
                    "stage": 1,
                    "basis_rows": 4,
                    "emitted_terms": 2,
                }
            ],
        },
        "policy": {"feature_names": ["bias", "depth"], "weights": [0.0, 1.5]},
        "truth_table": [{"input_index": 0, "expected_output_index": 0}],
        "resource_summary": {"num_gates": 1, "depth": 1},
        "search_metrics": {"learned_expansions": 1},
    }


def _required_paths(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir() if path.is_file()}


def test_toffoli_search_artifacts_are_complete_and_deterministic(tmp_path: Path) -> None:
    report = _report(certified=True)
    gates = (Gate(GateType.H, (0,)),)
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_toffoli_search_artifacts(first, report=report, learned_witness_gates=gates, seed=23)
    save_toffoli_search_artifacts(second, report=report, learned_witness_gates=gates, seed=23)

    expected = {
        "summary.json",
        "summary.md",
        "phase_identity.json",
        "fifo_trace.csv",
        "uniform_trace.csv",
        "random_trace_seed_23.csv",
        "zero_policy_trace.csv",
        "training_history.csv",
        "learned_trace.csv",
        "policy.json",
        "policy_weights.csv",
        "truth_table.csv",
        "resource_summary.json",
        "search_metrics.csv",
        "circuit.svg",
        "frontier_size.svg",
    }
    assert _required_paths(first) == expected
    assert _required_paths(second) == expected
    for name in sorted(expected):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["learned_witness"]["operations"] == [
        {"gate": "H", "index": 1, "qubits": [0]}
    ]
    diagram = (first / "circuit.svg").read_text(encoding="utf-8")
    assert "Generated learned Toffoli witness" in diagram
    assert "Actual gate sequence reconstructed from the learned evaluation" in diagram


def test_uncertified_toffoli_artifacts_never_draw_a_candidate_prefix(tmp_path: Path) -> None:
    report = _report(certified=False)
    # Supplying a gate here simulates a buggy caller retaining a nonterminal
    # prefix.  The presentation boundary must still render the no-witness SVG.
    save_toffoli_search_artifacts(
        tmp_path,
        report=report,
        learned_witness_gates=(Gate(GateType.H, (0,)),),
        seed=23,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["learned_witness"]["operations"] == []
    diagram = (tmp_path / "circuit.svg").read_text(encoding="utf-8")
    assert "No certified learned Toffoli witness" in diagram
    assert "No certified learned witness was returned; no reference circuit is shown." in diagram
    assert "Actual gate sequence reconstructed from the learned evaluation" not in diagram
