from __future__ import annotations

import time

import numpy as np

from rl.compact_sarsa import (
    CompactFeatureExtractor,
    CompactLinearSarsaRanker,
    CompactOnlineEpisode,
    train_compact_online,
)
from search.compact_parity import (
    CompactFrontier,
    CompactNode,
    CompactParityProblem,
    CompactState,
    basis_distance_to_identity,
    phase_truth_table,
    reconstruct_core_witness,
    toffoli_compact_target,
)


def _problem() -> CompactParityProblem:
    return CompactParityProblem(toffoli_compact_target(), max_cnot=6)


def test_ccz_phase_identity_is_exact() -> None:
    phases = phase_truth_table(toffoli_compact_target())
    expected = tuple(4 if assignment == 0b111 else 0 for assignment in range(8))
    assert phases == expected


def test_three_qubit_cnot_basis_graph_has_168_vertices() -> None:
    distance = basis_distance_to_identity(3)
    assert len(distance) == 168
    assert distance[(0b001, 0b010, 0b100)] == 0
    assert distance[(0b001, 0b011, 0b100)] == 1


def test_hot_state_excludes_general_symbolic_pipeline() -> None:
    state = CompactState.initial(3)
    assert state.semantic_key() == ((1, 2, 4), 0)
    for name in ("dag", "tableau", "frame", "rotations", "phase_poly"):
        assert not hasattr(state, name)


def test_policy_selects_record_and_problem_enumerates_every_legal_child() -> None:
    problem = _problem()
    root = CompactNode(problem.initial_state())
    legal = problem.legal_actions(root.state)
    assert {action.name for action in legal} == {"T", "CNOT"}
    children = problem.expand(root)
    assert children
    assert all(child.parent is root for child in children)

    frontier = CompactFrontier()
    for child in children:
        frontier.insert(child)
    ranker = CompactLinearSarsaRanker(seed=7)
    selected = ranker.select(
        frontier.nodes(),
        CompactFeatureExtractor(problem),
        epsilon=0.0,
    )
    assert selected.node in frontier.nodes()


def test_scalar_dominance_rejects_costlier_semantic_duplicate() -> None:
    frontier = CompactFrontier()
    cheap = CompactNode(CompactState((1, 2, 4), 0, 0))
    expensive = CompactNode(CompactState((1, 2, 4), 0, 2))
    assert frontier.insert(cheap)
    assert not frontier.insert(expensive)
    assert frontier.duplicate_rejected == 1


def test_parent_links_reconstruct_witness_without_dag_copy() -> None:
    problem = _problem()
    root = CompactNode(problem.initial_state())
    child = next(child for child in problem.expand(root) if child.action.name == "T")
    assert reconstruct_core_witness(child) == (child.action,)


def test_online_sarsa_update_changes_finite_weights() -> None:
    problem = _problem()
    ranker = CompactLinearSarsaRanker(seed=5)
    before = ranker.theta.copy()
    episode = CompactOnlineEpisode(problem).run(
        ranker,
        epsilon=0.5,
        max_expansions=64,
        learn=True,
    )
    assert ranker.update_count == episode.expansions
    assert np.isfinite(ranker.theta).all()
    assert not np.array_equal(before, ranker.theta)


def test_compact_online_training_profile_is_fast_and_finds_goal() -> None:
    problem = _problem()
    started = time.process_time()
    result = train_compact_online(
        problem,
        episodes=64,
        training_max_expansions=256,
        evaluation_max_expansions=3_000,
        checkpoint_interval=4,
        cpu_limit_seconds=30.0,
        seed=23,
    )
    elapsed = time.process_time() - started
    assert result.completed
    assert result.final_evaluation.success
    assert result.final_evaluation.solution_node is not None
    assert problem.is_goal(result.final_evaluation.solution_node.state)
    witness = result.final_evaluation.core_witness
    assert len(witness) == 13
    assert sum(operation.name == "CNOT" for operation in witness) == 6
    assert sum(operation.name in {"T", "TDG"} for operation in witness) == 7
    assert result.final_evaluation.expansions < result.initial_evaluation.expansions
    assert elapsed < 30.0


def test_learned_witness_passes_existing_authoritative_certifier() -> None:
    import pytest

    simulator = pytest.importorskip("certification.simulator")
    toffoli = pytest.importorskip("benchmarks.toffoli")
    base = pytest.importorskip("certification.base")
    from search.compact_parity import materialize_authoritative_state

    problem = _problem()
    result = train_compact_online(
        problem,
        episodes=64,
        training_max_expansions=256,
        evaluation_max_expansions=3_000,
        checkpoint_interval=4,
        cpu_limit_seconds=30.0,
        seed=23,
    )
    assert result.final_evaluation.solution_node is not None
    authoritative = materialize_authoritative_state(
        result.final_evaluation.solution_node,
        problem.target,
    )
    certifier = simulator.SimulatorCertificationEngine(
        simulator.SynthesisTarget(
            toffoli.toffoli_reference_unitary(),
            quotient_global_phase=True,
        )
    )
    certificate = certifier.certify(authoritative)
    assert certificate.status is base.CertStatus.SUCCESS
    assert authoritative.num_gates == 15
    assert authoritative.t_count == 7
    assert authoritative.two_qubit_count == 6
