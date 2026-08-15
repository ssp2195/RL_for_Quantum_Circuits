"""Shared article-compatible scheduler evaluation on native Clifford+T targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

from certification.simulator import SimulatorCertificationEngine, SynthesisTarget
from ckt_types import ResourceBudget
from config import Config
from env.rl_env import CircuitSynthesisEnv
from evaluate import evaluate
from rl.article_features import ARTICLE_FEATURE_SCHEMA_VERSION, ArticleFeatureProvider
from rl.baselines import LinearContextualBanditPolicy, LinearExpectedSarsaPolicy
from rl.policy import LinearQPolicy
from rl.target_context import DenseTargetContext
from train import Trainer


DEFAULT_SCHEDULERS = (
    "fifo",
    "lifo",
    "uniform_cost",
    "random",
    "target_potential",
    "zero_policy",
)


def _case_value(case: object, name: str):
    try:
        return getattr(case, name)
    except AttributeError as exc:
        raise TypeError(f"native target case must expose {name!r}") from exc


def _budget(num_qubits: int, maximum_gates: int = 4) -> ResourceBudget:
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")
    return ResourceBudget(
        max_t_count=maximum_gates,
        max_two_qubit_count=maximum_gates,
        max_gates=maximum_gates,
        max_depth=maximum_gates,
    )


def _provider(case: object) -> ArticleFeatureProvider:
    target = SynthesisTarget(np.asarray(_case_value(case, "unitary"), dtype=np.complex128))
    context = DenseTargetContext.from_synthesis_target(target)
    return ArticleFeatureProvider(context)


def _policy_class(name: str):
    choices = {
        "sarsa": LinearQPolicy,
        "expected_sarsa": LinearExpectedSarsaPolicy,
        "contextual_bandit": LinearContextualBanditPolicy,
    }
    try:
        return choices[name]
    except KeyError as exc:
        raise ValueError(f"learner must be one of {tuple(choices)}") from exc


@dataclass(frozen=True)
class TrainedArticlePolicy:
    """Transferable Eq. (19) weights plus auditable training metadata."""

    weights: tuple[float, ...]
    learner: str
    seed: int
    learning_rate: float
    discount: float
    episodes_per_target: int
    max_steps: int
    epsilon_start: float
    epsilon_minimum: float
    epsilon_decay: float
    training_target_ids: tuple[str, ...]
    histories: tuple[dict[str, object], ...]
    runtime_seconds: float

    def policy_for(
        self,
        case: object,
        *,
        feature_provider: ArticleFeatureProvider | None = None,
    ) -> LinearQPolicy:
        provider = _provider(case) if feature_provider is None else feature_provider
        policy = _policy_class(self.learner)(
            feature_provider=provider,
            lr=self.learning_rate,
            gamma=self.discount,
            seed=self.seed,
        )
        if len(self.weights) != policy.feature_dim:
            raise ValueError("stored weights do not match the article feature schema")
        policy.theta[:] = np.asarray(self.weights, dtype=np.float64)
        return policy

    def metadata(self) -> dict[str, object]:
        payload = np.asarray(self.weights, dtype="<f8").tobytes(order="C")
        return {
            "learner": self.learner,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "episodes_per_target": self.episodes_per_target,
            "max_steps": self.max_steps,
            "epsilon_start": self.epsilon_start,
            "epsilon_minimum": self.epsilon_minimum,
            "epsilon_decay": self.epsilon_decay,
            "feature_schema": ARTICLE_FEATURE_SCHEMA_VERSION,
            "feature_dimension": len(self.weights),
            "policy_weight_digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            "policy_weight_norm": float(np.linalg.norm(self.weights)),
            "training_target_ids": self.training_target_ids,
            "runtime_seconds": self.runtime_seconds,
        }

    def report(self) -> dict[str, object]:
        """Return checkpoint weights and deterministic TD diagnostics."""

        return {
            **self.metadata(),
            "weights": list(self.weights),
            "training_histories": list(self.histories),
        }


def train_article_policy(
    cases: Sequence[object],
    *,
    seed: int,
    episodes_per_target: int = 2,
    max_steps: int = 32,
    learner: str = "sarsa",
    learning_rate: float = 1e-3,
) -> TrainedArticlePolicy:
    """Train one transferable linear schema without exposing replay witnesses."""

    if not cases:
        raise ValueError("at least one training target is required")
    if episodes_per_target < 1 or max_steps < 1:
        raise ValueError("episodes_per_target and max_steps must be positive")
    weights: np.ndarray | None = None
    histories: list[dict[str, object]] = []
    runtime = 0.0
    target_ids: list[str] = []

    for target_index, case in enumerate(cases):
        num_qubits = int(_case_value(case, "num_qubits"))
        unitary = np.asarray(_case_value(case, "unitary"), dtype=np.complex128)
        provider = _provider(case)
        policy = _policy_class(learner)(
            feature_provider=provider,
            lr=learning_rate,
            gamma=1.0,
            seed=seed + target_index,
        )
        if weights is not None:
            policy.theta[:] = weights
        config = Config(
            num_qubits=num_qubits,
            budget=_budget(num_qubits),
            max_steps=max_steps,
            max_frontier=64,
            discount=1.0,
            seed=seed + target_index,
            fairness_interval=0,
            reward_mode="expansion_cost",
        )
        environment = CircuitSynthesisEnv(
            config,
            SimulatorCertificationEngine(SynthesisTarget(unitary)),
            feature_provider=provider,
        )
        trainer = Trainer(environment, policy=policy)
        trainer.epsilon = 0.1
        trainer.min_epsilon = 0.02
        trainer.epsilon_decay = 0.995
        target_history = trainer.train(episodes_per_target)
        runtime += trainer.last_training_runtime_seconds
        target_id = str(_case_value(case, "target_id"))
        target_ids.append(target_id)
        histories.append({"target_id": target_id, "episodes": target_history})
        weights = np.array(policy.theta, copy=True)

    assert weights is not None
    return TrainedArticlePolicy(
        weights=tuple(float(value) for value in weights),
        learner=learner,
        seed=seed,
        learning_rate=float(learning_rate),
        discount=1.0,
        episodes_per_target=int(episodes_per_target),
        max_steps=int(max_steps),
        epsilon_start=0.1,
        epsilon_minimum=0.02,
        epsilon_decay=0.995,
        training_target_ids=tuple(target_ids),
        histories=tuple(histories),
        runtime_seconds=float(runtime),
    )


def select_article_policy(
    training_cases: Sequence[object],
    validation_cases: Sequence[object],
    *,
    seed: int,
    validation_seeds: Sequence[int] = (5, 13),
    episodes_per_target: int = 3,
    training_max_steps: int = 32,
    validation_max_steps: int = 64,
    learners: Sequence[str] = ("sarsa",),
    learning_rates: Sequence[float] = (1e-3, 5e-4),
) -> tuple[TrainedArticlePolicy, dict[str, object]]:
    """Select hyperparameters on validation targets before test evaluation."""

    if not training_cases or not validation_cases:
        raise ValueError("training_cases and validation_cases must be non-empty")
    if not learners or not learning_rates or not validation_seeds:
        raise ValueError("candidate learners/rates and validation seeds are required")

    candidates: list[tuple[tuple[float, float, int], TrainedArticlePolicy, dict]] = []
    for candidate_index, (learner, learning_rate) in enumerate(
        (learner, rate) for learner in learners for rate in learning_rates
    ):
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning rates must be finite and positive")
        trained = train_article_policy(
            training_cases,
            seed=seed,
            episodes_per_target=episodes_per_target,
            max_steps=training_max_steps,
            learner=learner,
            learning_rate=float(learning_rate),
        )
        validation = evaluate_native_corpus(
            validation_cases,
            seeds=validation_seeds,
            max_steps=validation_max_steps,
            schedulers=("learned",),
            trained_policy=trained,
        )
        aggregate = validation["schedulers"]["learned"]
        mean_expansions = aggregate["successful_expansions_mean"]
        score = (
            float(aggregate["success_rate"]),
            -float("inf") if mean_expansions is None else -float(mean_expansions),
            -candidate_index,
        )
        candidates.append((score, trained, aggregate))

    _, selected, _ = max(candidates, key=lambda item: item[0])
    report: dict[str, object] = {
        "schema": "article-validation-model-selection-v1",
        "selection_rule": (
            "maximize validation success rate, then minimize mean successful "
            "expansions, then preserve declared candidate order"
        ),
        "training_target_ids": [
            str(_case_value(case, "target_id")) for case in training_cases
        ],
        "validation_target_ids": [
            str(_case_value(case, "target_id")) for case in validation_cases
        ],
        "validation_seeds": [int(value) for value in validation_seeds],
        "test_targets_observed": False,
        "candidates": [
            {
                "learner": trained.learner,
                "learning_rate": trained.learning_rate,
                "policy_weight_digest": trained.metadata()["policy_weight_digest"],
                "validation": aggregate,
                "selected": trained is selected,
            }
            for _, trained, aggregate in candidates
        ],
        "selected_policy": selected.metadata(),
    }
    return selected, report


def summarize_runs(runs: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(runs)
    successes = [row for row in rows if bool(row["certified"])]
    expansions = [int(row["expansions"]) for row in successes]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    resource_vectors = [
        tuple(int(value) for value in row["solution_resource_vector"])
        for row in successes
        if row.get("solution_resource_vector") is not None
    ]
    resource_quality = None
    if resource_vectors:
        resource_quality = {
            "mean_t_count": float(np.mean([vector[0] for vector in resource_vectors])),
            "mean_two_qubit_count": float(
                np.mean([vector[1] for vector in resource_vectors])
            ),
            "mean_gate_count": float(
                np.mean([vector[2] for vector in resource_vectors])
            ),
            "mean_depth": float(
                np.mean([max(vector[3:], default=0) for vector in resource_vectors])
            ),
        }
    return {
        "runs": len(rows),
        "successes": len(successes),
        "success_rate": float(len(successes) / len(rows)) if rows else 0.0,
        "successful_expansions_mean": (
            float(np.mean(expansions)) if expansions else None
        ),
        "successful_expansions_std": (
            float(np.std(expansions)) if expansions else None
        ),
        "successful_expansions_median": (
            float(median(expansions)) if expansions else None
        ),
        "runtime_seconds_mean": float(np.mean(runtimes)) if runtimes else 0.0,
        "runtime_seconds_std": float(np.std(runtimes)) if runtimes else 0.0,
        "runtime_seconds_median": float(median(runtimes)) if runtimes else 0.0,
        "solution_resource_quality": resource_quality,
        "individual_runs": rows,
    }


def evaluate_native_corpus(
    cases: Sequence[object],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    max_steps: int = 64,
    schedulers: Sequence[str] = DEFAULT_SCHEDULERS,
    trained_policy: TrainedArticlePolicy | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate all schedulers with one grammar, archive, budget, and certifier."""

    if not cases or not seeds:
        raise ValueError("cases and seeds must be non-empty")
    declared = list(schedulers)
    if trained_policy is not None and "learned" not in declared:
        declared.append("learned")
    runs_by_scheduler: dict[str, list[dict[str, object]]] = {
        scheduler: [] for scheduler in declared
    }

    for case in cases:
        num_qubits = int(_case_value(case, "num_qubits"))
        unitary = np.asarray(_case_value(case, "unitary"), dtype=np.complex128)
        target_id = str(_case_value(case, "target_id"))
        for seed in seeds:
            for scheduler in declared:
                provider = _provider(case) if scheduler in {"zero_policy", "learned"} else None
                if provider is not None:
                    provider.bind_search_horizon(max_steps)
                policy = (
                    trained_policy.policy_for(case, feature_provider=provider)
                    if scheduler == "learned" and trained_policy is not None
                    else None
                )
                report = evaluate(
                    num_qubits=num_qubits,
                    target_gates=(),
                    target_unitary=unitary,
                    budget=_budget(num_qubits),
                    max_steps=max_steps,
                    seed=int(seed),
                    scheduler=scheduler,
                    collect_trace=False,
                    policy=policy,
                    target_aware_features=scheduler == "target_potential",
                    reward_mode="expansion_cost",
                    fairness_interval=0,
                    feature_provider=provider,
                )
                report["target_id"] = target_id
                report["seed"] = int(seed)
                runs_by_scheduler[scheduler].append(report)

    result: dict[str, object] = {
        "schema": "article-native-corpus-evaluation-v1",
        "seeds": [int(seed) for seed in seeds],
        "max_steps": int(max_steps),
        "resource_budget": {
            "max_t_count": 4,
            "max_two_qubit_count": 4,
            "max_gates": 4,
            "max_depth": 4,
        },
        "native_gate_grammar": ["H", "S", "SDG", "T", "TDG", "CNOT"],
        "reward_mode": "expansion_cost",
        "target_cases": [
            (
                case.metadata()
                if callable(getattr(case, "metadata", None))
                else {
                    "target_id": str(_case_value(case, "target_id")),
                    "num_qubits": int(_case_value(case, "num_qubits")),
                }
            )
            for case in cases
        ],
        "schedulers": {
            scheduler: summarize_runs(runs)
            for scheduler, runs in runs_by_scheduler.items()
        },
        "trained_policy": None if trained_policy is None else trained_policy.report(),
        "no_reference_witness_used_for_search": True,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return result


def run_tiny_ablations(
    case: object,
    *,
    seed: int = 0,
    max_steps: int = 32,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Run the declared reduction/reward/fairness switches on one tiny target.

    This function records raw runs; it deliberately makes no claim that one
    target establishes a statistically significant component effect.
    """

    num_qubits = int(_case_value(case, "num_qubits"))
    unitary = np.asarray(_case_value(case, "unitary"), dtype=np.complex128)

    def run(**overrides):
        options = {
            "num_qubits": num_qubits,
            "target_gates": (),
            "target_unitary": unitary,
            "budget": _budget(num_qubits),
            "max_steps": max_steps,
            "seed": seed,
            "scheduler": "fifo",
            "collect_trace": True,
            "reward_mode": "expansion_cost",
        }
        options.update(overrides)
        return evaluate(**options)

    def train_behavioral_ablation(
        *,
        target_features: bool = True,
        reward_mode: str = "expansion_cost",
        fairness_interval: int = 0,
    ) -> dict[str, object]:
        provider = _provider(case) if target_features else ArticleFeatureProvider()
        config = Config(
            num_qubits=num_qubits,
            budget=_budget(num_qubits),
            max_steps=max_steps,
            max_frontier=64,
            discount=1.0,
            seed=seed,
            fairness_interval=fairness_interval,
            reward_mode=reward_mode,
        )
        environment = CircuitSynthesisEnv(
            config,
            SimulatorCertificationEngine(SynthesisTarget(unitary)),
            feature_provider=provider,
        )
        trainer = Trainer(
            environment,
            policy=LinearQPolicy(feature_provider=provider, gamma=1.0, seed=seed),
        )
        trainer.epsilon = 0.1
        trainer.min_epsilon = 0.1
        trainer.epsilon_decay = 1.0
        history = trainer.train(2)
        evaluation_provider = (
            _provider(case) if target_features else ArticleFeatureProvider()
        )
        evaluation = evaluate(
            num_qubits=num_qubits,
            target_gates=(),
            target_unitary=unitary,
            budget=_budget(num_qubits),
            max_steps=max_steps,
            seed=seed,
            scheduler="greedy",
            collect_trace=True,
            policy=trainer.policy,
            reward_mode=reward_mode,
            fairness_interval=fairness_interval,
            feature_provider=evaluation_provider,
        )
        return {
            "reward_mode": config.reward_mode,
            "target_features": target_features,
            "fairness_interval": fairness_interval,
            "exploration_beta": trainer.exploration_beta,
            "training_history": history,
            "policy": trainer.policy.metadata(),
            "evaluation": evaluation,
        }

    result: dict[str, object] = {
        "schema": "article-tiny-ablation-v1",
        "target_id": str(_case_value(case, "target_id")),
        "seed": seed,
        "canonicalization": {
            "on": run(canonicalization_enabled=True),
            "off": run(canonicalization_enabled=False),
        },
        "pareto_dominance": {
            "on": run(pareto_dominance_enabled=True),
            "off": run(pareto_dominance_enabled=False),
        },
        "clifford_angle_absorption": {
            "on": run(absorb_clifford_angles=True),
            "off": run(absorb_clifford_angles=False),
        },
        "target_aware_features": {
            "off": train_behavioral_ablation(target_features=False),
            "on": train_behavioral_ablation(target_features=True),
        },
        "reward": {
            "expansion_cost": train_behavioral_ablation(
                reward_mode="expansion_cost"
            ),
            "target_progress_shaping": train_behavioral_ablation(
                reward_mode="target_progress_shaping",
            ),
        },
        "fairness": {
            "off": train_behavioral_ablation(fairness_interval=0),
            "on": train_behavioral_ablation(fairness_interval=2),
        },
        "visit_bonus": {
            "off": train_behavioral_ablation(reward_mode="expansion_cost"),
            "on": train_behavioral_ablation(
                reward_mode="expansion_cost_plus_visit_bonus"
            ),
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return result


__all__ = [
    "DEFAULT_SCHEDULERS",
    "TrainedArticlePolicy",
    "evaluate_native_corpus",
    "run_tiny_ablations",
    "select_article_policy",
    "summarize_runs",
    "train_article_policy",
]
