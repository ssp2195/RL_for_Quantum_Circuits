"""End-to-end direct training coverage for the learned GHZ-3 scheduler.

The policy action is deliberately a persistent frontier record.  These tests
therefore inspect the selected-record trace and the witness reconstructed from
the dense-certified ``solution_node``; they never hand a gate to the policy or
substitute the known GHZ reference on failure.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pytest

from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from evaluate import evaluate
from ghz3_rl import main, run_ghz3_rl
from rl.policy import LinearQPolicy
from rl.target_context import DenseTargetContext


def _run(output_dir: Path, **kwargs):
    """Run the intentionally verbose trainer without polluting pytest output."""

    with redirect_stdout(StringIO()):
        return run_ghz3_rl(output_dir, **kwargs)


@pytest.fixture(scope="module")
def reproducible_reports(tmp_path_factory: pytest.TempPathFactory):
    """Two independent, same-seed direct-GHZ training runs.

    The runner's defaults are the calibrated direct protocol: 50 episodes,
    a four-selection training horizon, and a common 32-selection horizon for
    both frozen zero and learned evaluation.  Keeping this fixture module
    scoped avoids repeatedly paying the small dense-training cost.
    """

    root = tmp_path_factory.mktemp("ghz3_rl")
    first = _run(root / "first")
    second = _run(root / "second")
    return first, second


def test_direct_ghz_training_is_reproducible_and_improves_the_fair_baseline(
    reproducible_reports,
):
    first, second = reproducible_reports

    assert first["training_history"] == second["training_history"]
    np.testing.assert_allclose(first["policy"]["weights"], second["policy"]["weights"])
    assert first["policy"]["weight_digest"] == second["policy"]["weight_digest"]
    assert first["evaluation"]["trace"] == second["evaluation"]["trace"]
    assert first["evaluation"]["witness_operations"] == second["evaluation"]["witness_operations"]

    assert first["correct"]
    assert first["learning"]["curriculum_used"] is False
    assert first["learning"]["training_max_steps"] == 4
    assert first["learning"]["evaluation_max_steps"] == 32
    # Both policies receive the same target/budget/evaluation horizon.  The
    # stable-ID zero baseline is allowed to finish and needs 25 expansions;
    # this is not a deliberately truncated comparison.
    assert first["zero_policy"]["certified"]
    assert first["zero_policy_expansions"] == 25
    assert first["evaluation"]["expansions"] == 3
    assert first["evaluation"]["expansions"] < first["zero_policy_expansions"]


def test_frozen_learned_policy_selects_ghz_prefix_records_and_certifies_solution_node(
    reproducible_reports,
):
    report, _ = reproducible_reports
    evaluation = report["evaluation"]
    trace = evaluation["trace"]

    assert evaluation["certified"]
    assert evaluation["search_solution_present"]
    assert evaluation["exact_unitary_certification"]
    assert not evaluation["truncated"]
    assert not evaluation["fairness_override_observed"]
    assert all(not row["selected_by_fairness"] for row in trace)
    assert len(trace) == 3
    assert trace[0]["selected_record_id"] == 0
    assert trace[0]["selected_prefix"] == []
    assert trace[1]["selected_prefix"] == [{"gate": "H", "qubits": [0]}]
    assert trace[2]["selected_prefix"] in (
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CNOT", "qubits": [0, 1]},
        ],
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CNOT", "qubits": [0, 2]},
        ],
    )

    witness = evaluation["witness_operations"]
    assert witness[0] == {"gate": "H", "qubits": [0]}
    assert len(witness) == 3
    assert {tuple(operation["qubits"]) for operation in witness if operation["gate"] == "CNOT"} == {
        (0, 1),
        (0, 2),
    }
    assert all(operation["gate"] not in {"T", "TDG"} for operation in witness)

    state = report["state_preparation"]
    assert state["passed"]
    assert state["symbolic_agrees_with_dense_witness"]
    assert state["fidelity"] >= 1.0 - 1e-12
    np.testing.assert_allclose(
        state["probabilities"],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        atol=1e-12,
    )
    assert state["resources"]["num_gates"] == 3
    assert state["resources"]["two_qubit_count"] == 2
    assert state["resources"]["t_count"] == 0
    assert state["resources"]["depth"] == 3
    assert report["validation"]["negative_relative_phase_rejected"]
    assert report["validation"]["negative_relative_phase_fidelity"] < 1e-12


def test_learned_artifacts_are_complete_and_do_not_hide_the_target_metadata(
    reproducible_reports,
):
    report, _ = reproducible_reports
    artifacts = report["artifacts"]
    required = {
        "summary",
        "training_history",
        "evaluation_trace",
        "policy_weights",
        "circuit_diagram",
        "frontier_progress",
        "report",
    }
    assert required <= artifacts.keys()
    for path in artifacts.values():
        assert Path(path).is_file()

    summary = json.loads(Path(artifacts["summary"]).read_text(encoding="utf-8"))
    assert summary["correct"]
    assert summary["learning"]["target_fingerprint"].startswith("sha256:")
    assert summary["learning"]["feature_schema_version"] == "frontier-target-aware-v1"
    assert summary["evaluation"]["witness_operations"] == report["evaluation"]["witness_operations"]
    assert "CNOT 0-&gt;1" in Path(artifacts["circuit_diagram"]).read_text(encoding="utf-8")
    assert "CNOT 0-&gt;2" in Path(artifacts["circuit_diagram"]).read_text(encoding="utf-8")


def test_failed_learned_evaluation_never_substitutes_the_reference_witness(
    tmp_path: Path,
):
    """A one-step uniform cap makes both training and evaluation fail safely."""

    report = _run(tmp_path / "failure", episodes=1, max_steps=1)

    assert not report["correct"]
    assert not report["evaluation"]["certified"]
    assert report["evaluation"]["truncated"]
    assert report["evaluation"]["witness_operations"] == []
    circuit_svg = Path(report["artifacts"]["circuit_diagram"]).read_text(encoding="utf-8")
    assert "No certified GHZ-3 witness" in circuit_svg
    assert "CNOT 0-&gt;1" not in circuit_svg
    assert "CNOT 0-&gt;2" not in circuit_svg

    # The CLI's process-success contract follows the report rather than
    # emitting a known reference witness for an unsuccessful run.
    with redirect_stdout(StringIO()):
        assert main(
            [
                "--artifacts-dir",
                str(tmp_path / "failure-cli"),
                "--episodes",
                "1",
                "--max-steps",
                "1",
            ]
        ) == 1


def test_evaluator_learned_scheduler_requires_explicit_nonzero_policy():
    """The evaluator cannot quietly turn ``learned`` into a zero baseline."""

    target_gates = (Gate(GateType.H, (0,)),)
    budget = ResourceBudget(max_t_count=0, max_depth=1, max_gates=1)
    with pytest.raises(ValueError, match="explicitly supplied"):
        evaluate(
            num_qubits=1,
            target_gates=target_gates,
            budget=budget,
            scheduler="learned",
        )

    target = SynthesisTarget(unitary_from_gates(1, target_gates))
    policy = LinearQPolicy(target_context=DenseTargetContext(target), seed=2)
    policy.theta[0] = 0.01  # Explicit nonzero trained/checkpoint-like state.
    report = evaluate(
        num_qubits=1,
        target_gates=target_gates,
        budget=budget,
        max_steps=1,
        scheduler="learned",
        policy=policy,
        target_aware_features=True,
        reward_mode="target_progress",
        fairness_interval=0,
        collect_trace=True,
    )
    assert report["certified"]
    assert report["feature_schema"]["target_fingerprint"] == policy.target_fingerprint
    assert not report["trace"][0]["selected_by_fairness"]
