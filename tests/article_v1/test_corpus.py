from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.article_native_corpus import (
    ARTICLE_V1_CONFIG_SCHEMA,
    COMPLETE_TRAINING_SCOPE,
    DIFFICULTY_ORDER,
    NATIVE_GATE_NAMES,
    OOD_LENGTH_CHECKPOINT_FAMILY,
    PARTIAL_SMOKE_TRAINING_SCOPE,
    SPLIT_ORDER,
    STANDARD_CHECKPOINT_FAMILY,
    ArticleV1EvaluationTarget,
    _is_semantic_duplicate,
    article_delta_phi,
    build_article_v1_corpus,
    dense_target_digest,
    load_article_v1_config,
    native_gate_grammar,
)
from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.gate import Gate
from enums import GateType
from certification.unitary_phase_metrics import phase_frobenius_discrepancy


def test_checked_in_profiles_fix_article_v1_counts_and_strata():
    pilot = load_article_v1_config("pilot")
    publication = load_article_v1_config("publication")

    assert pilot.to_dict()["schema_version"] == ARTICLE_V1_CONFIG_SCHEMA
    assert pilot.profile == "pilot"
    assert publication.profile == "publication"
    assert pilot.qubits == publication.qubits == (2, 3)
    assert tuple(item.name for item in pilot.difficulties) == DIFFICULTY_ORDER
    assert (
        pilot.difficulty("easy").min_generator_length,
        pilot.difficulty("easy").max_generator_length,
    ) == (2, 3)
    assert (
        pilot.difficulty("medium").min_generator_length,
        pilot.difficulty("medium").max_generator_length,
    ) == (4, 5)
    assert pilot.difficulty("hard").min_generator_length == 6

    # The publication test set is the article's default 75-target corpus:
    # exactly 25 targets in each declared difficulty stratum.
    assert dict(publication.split("test").counts) == {
        "easy": 25,
        "medium": 25,
        "hard": 25,
    }
    assert dict(publication.split("ood_test").counts) == {
        "easy": 0,
        "medium": 25,
        "hard": 25,
    }
    pilot_primary_counts = {
        difficulty: sum(
            dict(pilot.split(split).counts)[difficulty]
            for split in ("train", "validation", "test")
        )
        for difficulty in DIFFICULTY_ORDER
    }
    assert pilot_primary_counts == {"easy": 5, "medium": 5, "hard": 5}
    assert len(publication.experiment["training_seeds"]) >= 5
    assert len(publication.experiment["random_scheduler_seeds"]) >= 10
    assert publication.experiment["gamma"] == 1.0
    assert publication.experiment["certification_tolerance"] == 1e-6


def test_pre_v2_config_schema_is_rejected(tmp_path: Path) -> None:
    payload = load_article_v1_config("pilot").to_dict()
    payload["schema_version"] = "article-v1-corpus-config-v1"
    path = tmp_path / "old-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config schema must be"):
        load_article_v1_config(path)


def test_manifest_exposes_article_facing_aliases() -> None:
    corpus = build_article_v1_corpus(load_article_v1_config("pilot"))
    manifest = corpus.manifest()
    case = manifest["cases"][0]

    assert manifest["identity_tolerance"] == manifest["tau_identity"]
    assert case["stratum"] == case["difficulty"]
    assert case["generator_witness"] == case["generator"]["witness_operations"]


def test_native_grammar_is_exact_and_all_to_all_directed():
    for num_qubits in (2, 3):
        grammar = native_gate_grammar(num_qubits)
        assert len(grammar) == 5 * num_qubits + num_qubits * (num_qubits - 1)
        assert {gate.gate_type.name for gate in grammar} == set(NATIVE_GATE_NAMES)
        cnot_pairs = {
            gate.qubits for gate in grammar if gate.gate_type is GateType.CNOT
        }
        assert cnot_pairs == {
            (control, target)
            for control in range(num_qubits)
            for target in range(num_qubits)
            if control != target
        }


def test_pilot_corpus_is_deterministic_unique_nonidentity_and_replayable():
    first = build_article_v1_corpus("pilot")
    second = build_article_v1_corpus("pilot")
    manifest = first.manifest()

    assert manifest == second.manifest()
    # The in-distribution pilot is exactly five targets per difficulty
    # (15 total), plus four separately labelled length-OOD targets.
    assert manifest["target_count"] == 19
    assert manifest["target_ids_are_globally_unique"]
    assert manifest["semantic_duplicates_rejected_with_delta_phi"]
    assert manifest["projective_identity_targets_rejected"]
    assert not manifest["target_specific_reachability_oracle"]
    assert tuple(manifest["split_order"]) == SPLIT_ORDER
    assert manifest["split_seeds"] == {
        "train": 201729,
        "validation": 202753,
        "test": 203769,
        "ood_test": 204783,
    }
    assert manifest["counts"] == {
        "train": {"easy": 2, "medium": 2, "hard": 2},
        "validation": {"easy": 1, "medium": 1, "hard": 1},
        "test": {"easy": 2, "medium": 2, "hard": 2},
        "ood_test": {"easy": 0, "medium": 2, "hard": 2},
    }

    tolerance = first.config.tau_identity
    for case in first.targets:
        assert case.num_qubits in {2, 3}
        stratum = first.config.difficulty(case.difficulty)
        assert stratum.min_generator_length <= case.generator_length
        assert case.generator_length <= stratum.max_generator_length
        assert case.generator_length <= case.budget.max_gates
        assert case.target_id == dense_target_digest(
            case.unitary, decimals=first.config.digest_decimals
        )
        identity = np.eye(1 << case.num_qubits, dtype=np.complex128)
        assert article_delta_phi(case.unitary, identity) > tolerance
        assert np.allclose(
            unitary_from_gates(case.num_qubits, case.generator_witness),
            case.unitary,
        )
        assert all(
            gate.gate_type.name in NATIVE_GATE_NAMES
            for gate in case.generator_witness
        )
        record = next(
            item for item in manifest["cases"] if item["target_id"] == case.target_id
        )
        assert record["target_identity_digest"] == case.target_id
        assert record["target_unitary_digest"].startswith("sha256:")
        assert record["target_matrix_shape"] == list(case.unitary.shape)
        assert record["generation_schema_version"] == manifest["schema_version"]
        assert record["resource_budget"] == {
            "max_t_count": case.budget.max_t_count,
            "max_two_qubit_count": case.budget.max_two_qubit_count,
            "max_gates": case.budget.max_gates,
            "max_depth": case.budget.max_depth,
        }
        assert record["expansion_budget_grid"]
        assert record["expansion_budget_grid"] == sorted(
            record["expansion_budget_grid"]
        )

    for left, right in combinations(first.targets, 2):
        if left.unitary.shape == right.unitary.shape:
            assert article_delta_phi(left.unitary, right.unitary) > tolerance


def test_pilot_and_publication_corpora_are_disjoint_and_internally_unique():
    pilot = build_article_v1_corpus("pilot")
    publication = build_article_v1_corpus("publication")
    pilot_target_ids = [case.target_id for case in pilot.targets]
    publication_target_ids = [case.target_id for case in publication.targets]

    assert len(pilot_target_ids) == len(set(pilot_target_ids)) == 19
    assert len(publication_target_ids) == len(set(publication_target_ids)) == 230
    assert set(pilot_target_ids).isdisjoint(publication_target_ids)
    identity_tolerance = max(
        pilot.config.tau_identity,
        publication.config.tau_identity,
    )
    cross_profile_distances = [
        article_delta_phi(pilot_case.unitary, publication_case.unitary)
        for pilot_case in pilot.targets
        for publication_case in publication.targets
        if pilot_case.unitary.shape == publication_case.unitary.shape
    ]
    assert cross_profile_distances
    assert min(cross_profile_distances) > identity_tolerance


def test_ood_length_partition_is_disjoint_and_manifested():
    corpus = build_article_v1_corpus("pilot")
    manifest = corpus.manifest()
    definition = corpus.config.ood_length_split
    short_training = tuple(
        case
        for case in corpus.cases(split=definition.training_source_split)
        if case.generator_length <= definition.training_max_generator_length
    )
    long_evaluation = corpus.cases(split=definition.evaluation_split)

    assert short_training
    assert long_evaluation
    assert max(case.generator_length for case in short_training) <= 4
    assert min(case.generator_length for case in long_evaluation) >= 5
    assert max(case.generator_length for case in long_evaluation) <= 8
    assert not ({case.target_id for case in short_training} & {
        case.target_id for case in long_evaluation
    })
    assert manifest["ood_length_split"]["semantic_overlap"] is False
    assert manifest["ood_length_split"]["training_target_ids"] == [
        case.target_id for case in short_training
    ]
    assert manifest["ood_length_split"]["evaluation_target_ids"] == [
        case.target_id for case in long_evaluation
    ]


def test_checkpoint_scopes_bind_standard_and_ood_partitions() -> None:
    corpus = build_article_v1_corpus("pilot")
    standard = corpus.checkpoint_scope(
        checkpoint_family=STANDARD_CHECKPOINT_FAMILY
    )
    ood = corpus.checkpoint_scope(
        checkpoint_family=OOD_LENGTH_CHECKPOINT_FAMILY
    )
    train_ids = {case.target_id for case in corpus.cases(split="train")}
    validation_ids = {
        case.target_id for case in corpus.cases(split="validation")
    }
    test_ids = {case.target_id for case in corpus.cases(split="test")}
    ood_ids = {case.target_id for case in corpus.cases(split="ood_test")}

    assert set(standard.allowed_training_target_ids) == train_ids
    assert set(standard.permitted_evaluation_target_ids) == (
        validation_ids | test_ids
    )
    assert not (set(standard.permitted_evaluation_target_ids) & ood_ids)
    assert set(ood.allowed_training_target_ids) < train_ids
    assert ood_ids <= set(ood.permitted_evaluation_target_ids)
    assert set(standard.held_out_target_ids) == (
        validation_ids | test_ids | ood_ids
    )
    assert standard.checkpoint_family != ood.checkpoint_family
    assert standard.training_scope_mode == COMPLETE_TRAINING_SCOPE
    assert standard.expected_training_beta == corpus.config.experiment["beta"]
    assert standard.expected_certification_tolerance == (
        corpus.config.experiment["certification_tolerance"]
    )
    assert standard.expected_episodes_per_target == (
        corpus.config.experiment["training_episodes_per_target"]
    )
    assert standard.expected_learning_rate == corpus.config.experiment["learning_rate"]
    assert dict(standard.expected_epsilon_schedule) == dict(
        corpus.config.experiment["epsilon"]
    )
    assert standard.allowed_training_seeds == tuple(
        corpus.config.experiment["training_seeds"]
    )
    assert standard.expected_expansion_cap is None
    assert tuple(
        target_id
        for target_id, _budget in standard.expected_training_expansion_budgets
    ) == standard.allowed_training_target_ids

    selected = corpus.cases(split="train")[:1]
    smoke = corpus.checkpoint_scope(
        checkpoint_family=STANDARD_CHECKPOINT_FAMILY,
        training_scope_mode=PARTIAL_SMOKE_TRAINING_SCOPE,
        training_target_ids=tuple(case.target_id for case in selected),
        allowed_training_seeds=(int(corpus.config.experiment["training_seeds"][0]),),
        expected_episodes_per_target=1,
        expected_expansion_cap=16,
    )
    assert smoke.training_scope_mode == PARTIAL_SMOKE_TRAINING_SCOPE
    assert smoke.allowed_training_target_ids == tuple(
        case.target_id for case in selected
    )
    assert smoke.expected_episodes_per_target == 1
    assert smoke.expected_expansion_cap == 16
    assert len(smoke.allowed_training_seeds) == 1


def test_evaluation_surface_exposes_dense_target_but_not_replay_witness():
    corpus = build_article_v1_corpus("pilot")
    case = corpus.cases(split="test", difficulty="easy")[0]
    target = case.synthesis_target()
    evaluation = case.evaluation_target()
    metadata = case.metadata()

    assert isinstance(target, SynthesisTarget)
    assert isinstance(evaluation, ArticleV1EvaluationTarget)
    assert isinstance(evaluation.target, SynthesisTarget)
    assert np.array_equal(evaluation.target.unitary, case.unitary)
    assert not hasattr(evaluation, "generator_witness")
    assert not evaluation.target_specific_reachability_oracle
    assert metadata["target_specific_reachability_oracle"] is False
    assert metadata["generator_witness_evaluation_prohibited"] is True
    assert metadata["generator_witness_provenance"] == (
        "reachability-and-replay-audit-only"
    )


def test_delta_phi_and_dense_digest_quotient_only_global_phase():
    corpus = build_article_v1_corpus("pilot")
    case = corpus.targets[0]
    shifted = np.exp(0.371j) * case.unitary
    identity = np.eye(case.unitary.shape[0], dtype=np.complex128)
    two_t = unitary_from_gates(
        2,
        (Gate(GateType.T, (0,)), Gate(GateType.T, (0,))),
    )
    one_s = unitary_from_gates(2, (Gate(GateType.S, (0,)),))

    assert article_delta_phi(case.unitary, shifted) <= 1e-7
    assert dense_target_digest(case.unitary) == dense_target_digest(shifted)
    assert article_delta_phi(two_t, one_s) <= corpus.config.tau_identity
    assert dense_target_digest(two_t) == dense_target_digest(one_s)
    assert article_delta_phi(case.unitary, identity) > corpus.config.tau_identity


def test_corpus_deduplication_calls_shared_projective_metric(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_metric(left, right):
        calls.append((np.asarray(left), np.asarray(right)))
        return phase_frobenius_discrepancy(left, right)

    monkeypatch.setattr(
        "benchmarks.article_native_corpus.phase_frobenius_discrepancy",
        recording_metric,
    )
    identity = np.eye(2, dtype=np.complex128)

    assert _is_semantic_duplicate(
        identity,
        [SimpleNamespace(unitary=identity)],
        tau_identity=1e-7,
    )
    assert len(calls) == 1
