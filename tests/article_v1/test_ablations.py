from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from canonical.canonicalizer import Canonicalizer
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from experiments.article_v1_ablations import (
    ARTICLE_V1_ABLATION_PROTOCOL_SCHEMA_VERSION,
    ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION,
    ARTICLE_V1_ABLATION_REGISTRY,
    FULL_VALIDATION_SCOPE,
    REQUIRED_ABLATION_IDS,
    SUBSET_VALIDATION_SCOPE,
    SUPPLEMENTARY_METHOD_IDS,
    SUPPLEMENTARY_SCOPE,
    preregister_validation_subset,
    run_article_v1_ablations,
    select_preregistered_validation_cases,
    write_ablations_csv,
)
from experiments.profiles import ARTICLE_V1_PROFILE
from rl.article_features import (
    ARTICLE_V1_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION,
    ArticleTargetContext,
    ArticleV1NoTargetFeatureProvider,
    ArticleV1NoZFeatureProvider,
)


@dataclass(frozen=True, slots=True)
class FakeBudget:
    expansion_budget: int = 100


@dataclass(frozen=True, slots=True)
class FakeCase:
    target_id: str
    split: str
    difficulty: str
    budget: FakeBudget = FakeBudget()


@dataclass(frozen=True, slots=True)
class FakeCheckpoint:
    training_seed: int
    feature_schema: str
    beta: float

    @property
    def weight_digest(self) -> str:
        return f"checkpoint:{self.training_seed}:{self.feature_schema}:{self.beta}"


class FakeCorpus:
    def __init__(self, *, validation_split: str = "validation") -> None:
        self.requested_splits: list[str] = []
        self.config = SimpleNamespace(
            digest="sha256:fixed-corpus-config",
            experiment={
                "training_seeds": (17,),
                "training_episodes_per_target": 2,
                "learning_rate": 0.001,
                "epsilon": {"start": 0.2, "minimum": 0.05, "decay": 0.995},
                "beta": 1.0,
                "certification_tolerance": 1e-9,
            },
        )
        self._cases = {
            "train": (FakeCase("train-0", "train", "easy"),),
            "validation": tuple(
                FakeCase(f"validation-{difficulty}-{index}", validation_split, difficulty)
                for difficulty in ("easy", "medium", "hard")
                for index in range(2)
            ),
        }

    def evaluation_targets(self, *, split: str):
        self.requested_splits.append(split)
        return self._cases[split]


def validation_cases() -> tuple[FakeCase, ...]:
    return tuple(
        FakeCase(f"v-{difficulty}-{index}", "validation", difficulty)
        for difficulty in ("easy", "medium", "hard")
        for index in range(3)
    )


def test_required_registry_freezes_schemas_dimensions_rewards_and_toggles():
    assert tuple(ARTICLE_V1_ABLATION_REGISTRY) == (
        REQUIRED_ABLATION_IDS + SUPPLEMENTARY_METHOD_IDS
    )

    no_target = ARTICLE_V1_ABLATION_REGISTRY["no_target_feature"]
    no_z = ARTICLE_V1_ABLATION_REGISTRY["no_frontier_context"]
    no_shaping = ARTICLE_V1_ABLATION_REGISTRY["no_reward_shaping"]
    direct = ARTICLE_V1_ABLATION_REGISTRY["direct_target_distance"]
    pareto_off = ARTICLE_V1_ABLATION_REGISTRY["pareto_pruning_off"]
    absorption_off = ARTICLE_V1_ABLATION_REGISTRY[
        "enhanced_pauli_canonicalization_off"
    ]

    assert (no_target.feature_schema_version, no_target.feature_dimension) == (
        ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION,
        28,
    )
    assert (no_z.feature_schema_version, no_z.feature_dimension) == (
        ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION,
        21,
    )
    assert ArticleV1NoTargetFeatureProvider().dimension == 28
    assert ArticleV1NoZFeatureProvider(
        ArticleTargetContext(np.eye(2, dtype=np.complex128))
    ).dimension == 21

    assert no_shaping.beta == 0.0
    assert no_shaping.reward_schema_version == ARTICLE_V1_PROFILE.reward_schema
    assert no_shaping.feature_schema_version == ARTICLE_V1_FEATURE_SCHEMA_VERSION
    assert direct.role == "primary_baseline"
    assert direct.scheduler == "article_target_distance"
    assert direct.direct_target_distance_primary_baseline is True
    assert direct.checkpoint_mode == "none"

    assert pareto_off.evaluation_scope == SUBSET_VALIDATION_SCOPE
    assert pareto_off.pareto_dominance_enabled is False
    assert pareto_off.absorb_clifford_angles is True
    assert absorption_off.evaluation_scope == SUBSET_VALIDATION_SCOPE
    assert absorption_off.pareto_dominance_enabled is True
    assert absorption_off.absorb_clifford_angles is False
    assert absorption_off.canonicalization_mode == "raw_witness"
    assert all(
        ARTICLE_V1_ABLATION_REGISTRY[name].config_schema_version.endswith("-v1")
        for name in REQUIRED_ABLATION_IDS
    )


def test_supplementary_modes_are_registered_but_not_article_v1_runs():
    expected_profiles = {
        "extended_target_aware_37d": "extended-target-aware-37d-v1",
        "composite_target_progress": "composite-target-progress-v1",
        "ghz3_direct_protocol": "ghz3-direct-v1",
        "toffoli_parity_protocol": "toffoli-parity-v1",
    }
    for identifier, profile_name in expected_profiles.items():
        entry = ARTICLE_V1_ABLATION_REGISTRY[identifier]
        assert entry.role == "supplementary_case_study"
        assert entry.profile_name == profile_name
        assert entry.evaluation_scope == SUPPLEMENTARY_SCOPE
        assert entry.enabled_in_article_v1_protocol is False

    extended = ARTICLE_V1_ABLATION_REGISTRY["extended_target_aware_37d"]
    assert extended.feature_schema_version == "extended-target-aware-37d-v1"
    assert extended.feature_dimension == 37


def test_raw_witness_canonicalization_is_sound_and_disables_enhanced_rewrites():
    budget = ResourceBudget(4, 4, 4, 4)
    tt = CircuitState(
        CircuitDAG.from_gates(
            1, (Gate(GateType.T, (0,)), Gate(GateType.T, (0,)))
        ),
        budget,
    )
    s = CircuitState(CircuitDAG.from_gates(1, (Gate(GateType.S, (0,)),)), budget)
    enhanced = Canonicalizer()
    raw = Canonicalizer(
        absorb_clifford_angles=False,
        normalization_mode="raw_witness",
    )

    assert enhanced.semantic_key(tt) == enhanced.semantic_key(s)
    assert raw.semantic_key(tt) != raw.semantic_key(s)
    assert raw.semantic_key(tt) == raw.semantic_key(
        CircuitState(tt.dag.copy(), budget)
    )

    left_then_right = CircuitState(
        CircuitDAG.from_gates(
            2, (Gate(GateType.H, (0,)), Gate(GateType.H, (1,)))
        ),
        budget,
    )
    right_then_left = CircuitState(
        CircuitDAG.from_gates(
            2, (Gate(GateType.H, (1,)), Gate(GateType.H, (0,)))
        ),
        budget,
    )
    assert enhanced.semantic_key(left_then_right) == enhanced.semantic_key(
        right_then_left
    )
    assert raw.semantic_key(left_then_right) != raw.semantic_key(right_then_left)


def test_preregistered_subset_is_balanced_permutation_invariant_and_outcome_free():
    cases = validation_cases()
    first = preregister_validation_subset(
        cases,
        corpus_config_digest="sha256:corpus",
        per_difficulty=1,
    )
    second = preregister_validation_subset(
        reversed(cases),
        corpus_config_digest="sha256:corpus",
        per_difficulty=1,
    )

    assert first == second
    assert len(first.target_ids) == 3
    selected = select_preregistered_validation_cases(cases, first)
    assert tuple(case.target_id for case in selected) == first.target_ids
    assert tuple(case.difficulty for case in selected) == ("easy", "medium", "hard")
    assert first.metadata()["outcomes_consulted"] is False
    assert first.selection_digest.startswith("sha256:")


@pytest.mark.parametrize("leaked_split", ["train", "test", "ood_test"])
def test_preregistered_subset_fails_closed_on_non_validation_input(leaked_split):
    cases = validation_cases() + (FakeCase("leak", leaked_split, "easy"),)

    with pytest.raises(ValueError, match="leakage is prohibited"):
        preregister_validation_subset(
            cases,
            corpus_config_digest="sha256:corpus",
        )


def test_protocol_executes_only_validation_and_records_exact_variants(tmp_path):
    corpus = FakeCorpus()
    training_calls: list[dict[str, object]] = []
    evaluation_calls: list[dict[str, object]] = []

    def fake_trainer(cases, **kwargs):
        assert {case.split for case in cases} == {"train"}
        training_calls.append(dict(kwargs))
        return FakeCheckpoint(
            training_seed=int(kwargs["training_seed"]),
            feature_schema=str(kwargs["feature_schema"]),
            beta=float(kwargs["beta"]),
        )

    def fake_evaluator(case, **kwargs):
        assert case.split == "validation"
        evaluation_calls.append({"target_id": case.target_id, **kwargs})
        return {
            "schema_version": "article-v1-raw-run-v3",
            "target_id": case.target_id,
            "split": case.split,
            "difficulty": case.difficulty,
            "evaluation_seed": kwargs["evaluation_seed"],
            "certified": True,
            "expansions": 7,
            "runtime_seconds": 0.01,
        }

    csv_path = tmp_path / "ablations.csv"
    payload = run_article_v1_ablations(
        corpus,
        output_csv=csv_path,
        expansion_cap=12,
        trainer=fake_trainer,
        evaluator=fake_evaluator,
    )

    assert corpus.requested_splits == ["train", "validation"]
    assert payload["schema_version"] == ARTICLE_V1_ABLATION_PROTOCOL_SCHEMA_VERSION
    assert payload["record_schema_version"] == ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION
    assert payload["evaluation_split"] == "validation"
    assert payload["test_targets_observed"] is False
    assert len(payload["records"]) == 30
    assert {record["split"] for record in payload["records"]} == {"validation"}
    assert not (
        set(SUPPLEMENTARY_METHOD_IDS)
        & {record["ablation_id"] for record in payload["records"]}
    )

    # Three variants plus one primary checkpoint for both restricted toggles.
    assert len(training_calls) == 4
    assert {
        (call["feature_schema"], call["beta"])
        for call in training_calls
    } == {
        (ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION, 1.0),
        (ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION, 1.0),
        (ARTICLE_V1_FEATURE_SCHEMA_VERSION, 0.0),
        (ARTICLE_V1_FEATURE_SCHEMA_VERSION, 1.0),
    }
    assert all(
        call["training_scope_mode"] == "complete_train_partition"
        for call in training_calls
    )
    beta_zero_calls = [
        call
        for call in evaluation_calls
        if getattr(call["checkpoint"], "beta", None) == 0.0
    ]
    assert beta_zero_calls
    assert all(call["beta"] == 0.0 for call in beta_zero_calls)
    assert all(
        call["checkpoint_scope"].expected_training_beta == 0.0
        for call in beta_zero_calls
    )
    assert all(
        call["checkpoint_scope"].expected_certification_tolerance == 1e-9
        for call in evaluation_calls
    )

    no_shaping = [
        record
        for record in payload["records"]
        if record["ablation_id"] == "no_reward_shaping"
    ]
    assert no_shaping and all(record["exact_base_reward_only"] for record in no_shaping)
    direct = [
        record
        for record in payload["records"]
        if record["ablation_id"] == "direct_target_distance"
    ]
    assert len(direct) == 6
    assert all(record["checkpoint_digest"] == "none" for record in direct)
    assert all(record["direct_target_distance_primary_baseline"] for record in direct)

    subset_ids = set(payload["validation_subset"]["target_ids"])
    pareto_calls = [
        call
        for call in evaluation_calls
        if call["pareto_dominance_enabled"] is False
    ]
    absorption_calls = [
        call
        for call in evaluation_calls
        if call["absorb_clifford_angles"] is False
    ]
    assert all(
        call["canonicalization_mode"] == "raw_witness"
        for call in absorption_calls
    )
    assert {call["target_id"] for call in pareto_calls} == subset_ids
    assert {call["target_id"] for call in absorption_calls} == subset_ids
    assert len(pareto_calls) == len(absorption_calls) == 3

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(payload["records"])
    assert {row["split"] for row in rows} == {"validation"}
    assert {row["test_targets_observed"] for row in rows} == {"False"}


def test_protocol_rejects_leaky_validation_corpus_before_training_or_evaluation():
    corpus = FakeCorpus(validation_split="test")
    called = False

    def must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("campaign function must not be reached")

    with pytest.raises(ValueError, match="leakage is prohibited"):
        run_article_v1_ablations(
            corpus,
            trainer=must_not_run,
            evaluator=must_not_run,
        )
    assert called is False


def test_csv_writer_rejects_test_record(tmp_path):
    record = {
        "schema_version": ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION,
        "ablation_id": "no_target_feature",
        "split": "test",
        "test_targets_observed": False,
    }

    with pytest.raises(ValueError, match="test leakage"):
        write_ablations_csv(Path(tmp_path) / "ablations.csv", (record,))
