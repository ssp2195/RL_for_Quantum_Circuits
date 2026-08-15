from __future__ import annotations

from benchmarks.native_corpus import build_native_target_corpus
from experiments.article_benchmark import (
    evaluate_native_corpus,
    run_tiny_ablations,
    select_article_policy,
    train_article_policy,
)


def test_shared_scheduler_harness_reports_individual_seeds_and_aggregates():
    corpus = build_native_target_corpus()
    one_qubit_case = next(
        case
        for case in corpus.cases(suite="bounded_synthesis", split="test")
        if case.num_qubits == 1
    )

    report = evaluate_native_corpus(
        [one_qubit_case],
        seeds=(3, 7),
        max_steps=4,
        schedulers=("fifo", "lifo", "uniform_cost", "random", "zero_policy"),
    )

    assert report["reward_mode"] == "expansion_cost"
    assert report["no_reference_witness_used_for_search"]
    for scheduler in ("fifo", "lifo", "uniform_cost", "random", "zero_policy"):
        summary = report["schedulers"][scheduler]
        assert summary["runs"] == 2
        assert summary["success_rate"] == 1.0
        assert [row["seed"] for row in summary["individual_runs"]] == [3, 7]
        assert all(row["search_metrics"]["expanded"] == 1 for row in summary["individual_runs"])
        assert summary["runtime_seconds_std"] >= 0.0
        assert summary["runtime_seconds_median"] >= 0.0
        assert summary["solution_resource_quality"] is not None
        assert all(
            row["archive_size_final"] == row["search_metrics"]["archive_size"]
            and row["time_to_solution"] is not None
            and row["reward_coefficients"]["article_equation"] == 24
            for row in summary["individual_runs"]
        )


def test_article_policy_training_uses_only_dense_targets_and_records_target_ids():
    corpus = build_native_target_corpus()
    two_qubit_case = next(
        case
        for case in corpus.cases(suite="bounded_synthesis", split="train")
        if case.num_qubits == 2
    )

    trained = train_article_policy(
        [two_qubit_case],
        seed=13,
        episodes_per_target=1,
        max_steps=8,
    )

    assert trained.training_target_ids == (two_qubit_case.target_id,)
    assert trained.metadata()["feature_schema"] == "article-frontier-eq19-v2"
    assert trained.metadata()["feature_dimension"] == 37
    assert trained.metadata()["policy_weight_digest"].startswith("sha256:")
    assert trained.metadata()["policy_weight_norm"] >= 0.0
    assert trained.metadata()["learning_rate"] == 1e-3
    assert len(trained.report()["weights"]) == 37
    assert trained.report()["training_histories"][0]["episodes"][0][
        "mean_absolute_td_error"
    ] >= 0.0
    assert len(trained.weights) == 37
    assert trained.histories[0]["target_id"] == two_qubit_case.target_id
    assert trained.runtime_seconds >= 0.0


def test_tiny_ablation_report_exercises_every_declared_switch(tmp_path):
    corpus = build_native_target_corpus()
    case = next(
        case
        for case in corpus.cases(suite="bounded_synthesis", split="validation")
        if case.num_qubits == 2
    )
    output = tmp_path / "ablations.json"

    report = run_tiny_ablations(case, seed=5, max_steps=3, output_path=output)

    assert output.is_file()
    assert set(report) >= {
        "canonicalization",
        "pareto_dominance",
        "clifford_angle_absorption",
        "target_aware_features",
        "reward",
        "fairness",
        "visit_bonus",
    }
    assert report["canonicalization"]["on"]["canonicalization_enabled"]
    assert not report["canonicalization"]["off"]["canonicalization_enabled"]
    assert report["target_aware_features"]["off"]["target_features"] is False
    assert report["target_aware_features"]["on"]["target_features"] is True
    assert report["reward"]["expansion_cost"]["training_history"]
    assert report["reward"]["target_progress_shaping"]["training_history"]
    assert report["fairness"]["on"]["fairness_interval"] == 2
    assert any(
        row["selected_by_fairness"]
        for row in report["fairness"]["on"]["evaluation"]["trace"]
    )
    assert report["visit_bonus"]["off"]["exploration_beta"] == 0.0
    assert report["visit_bonus"]["on"]["exploration_beta"] > 0.0


def test_validation_selection_never_observes_test_targets():
    corpus = build_native_target_corpus()
    training = tuple(
        case
        for case in corpus.cases(suite="bounded_synthesis", split="train")
        if case.num_qubits == 2
    )
    validation = tuple(
        case
        for case in corpus.cases(suite="bounded_synthesis", split="validation")
        if case.num_qubits == 2
    )

    selected, report = select_article_policy(
        training,
        validation,
        seed=19,
        validation_seeds=(3,),
        episodes_per_target=1,
        training_max_steps=8,
        validation_max_steps=8,
        learning_rates=(1e-3,),
    )

    assert report["test_targets_observed"] is False
    assert report["training_target_ids"] == [training[0].target_id]
    assert report["validation_target_ids"] == [validation[0].target_id]
    assert report["candidates"][0]["selected"]
    assert selected.metadata()["policy_weight_digest"] == report[
        "selected_policy"
    ]["policy_weight_digest"]
