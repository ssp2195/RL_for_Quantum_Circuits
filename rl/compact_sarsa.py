"""Online linear SARSA ranker for compact persistent frontier records."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Sequence

import numpy as np

from search.compact_parity import (
    CompactFrontier,
    CompactGate,
    CompactNode,
    CompactParityProblem,
    reconstruct_core_witness,
)


FEATURE_NAMES = (
    "bias",
    "emitted_fraction",
    "remaining_fraction",
    "cnot_fraction",
    "basis_distance_fraction",
    "cnot_slack_fraction",
    "exposed_remaining_fraction",
    "identity_row_fraction",
    "mean_row_weight_fraction",
    "last_is_phase",
    "last_is_cnot",
)
FEATURE_DIMENSION = len(FEATURE_NAMES)


class CompactFeatureExtractor:
    """Cheap bounded features independent of frontier storage order."""

    schema_version = "compact-parity-record-features-v1"
    names = FEATURE_NAMES
    dimension = FEATURE_DIMENSION

    def __init__(self, problem: CompactParityProblem) -> None:
        self.problem = problem

    def one(self, node: CompactNode) -> np.ndarray:
        cached = node.cached_features
        if cached is not None:
            return cached  # type: ignore[return-value]

        state = node.state
        term_count = max(1, len(self.problem.target.terms))
        max_cnot = max(1, self.problem.max_cnot)
        emitted = state.phase_count
        distance = self.problem.basis_distance(state)
        identity_rows = sum(
            actual == expected
            for actual, expected in zip(
                state.basis_rows,
                self.problem.target.identity_basis,
                strict=True,
            )
        )
        mean_row_weight = float(
            np.mean([row.bit_count() for row in state.basis_rows])
        )
        features = np.asarray(
            [
                1.0,
                emitted / term_count,
                (term_count - emitted) / term_count,
                state.cnot_count / max_cnot,
                distance / max_cnot,
                (self.problem.max_cnot - state.cnot_count - distance) / max_cnot,
                self.problem.exposed_remaining_terms(state) / max(1, state.num_qubits),
                identity_rows / max(1, state.num_qubits),
                mean_row_weight / max(1, state.num_qubits),
                float(state.last_action_code == 1),
                float(state.last_action_code == 2),
            ],
            dtype=np.float64,
        )
        features.setflags(write=False)
        node.cached_features = features
        return features

    def batch(self, nodes: Sequence[CompactNode]) -> np.ndarray:
        if not nodes:
            return np.empty((0, self.dimension), dtype=np.float64)
        return np.vstack([self.one(node) for node in nodes])


@dataclass(frozen=True, slots=True)
class SelectedRecord:
    node: CompactNode = field(compare=False)
    features: np.ndarray = field(compare=False, repr=False)
    value: float


class CompactLinearSarsaRanker:
    """Shared linear scorer trained by bounded semi-gradient SARSA."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        discount: float = 0.95,
        seed: int = 23,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        self.theta = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.rng = np.random.default_rng(seed)
        self.selection_count = 0
        self.update_count = 0

    def select(
        self,
        nodes: Sequence[CompactNode],
        extractor: CompactFeatureExtractor,
        *,
        epsilon: float,
    ) -> SelectedRecord:
        if not nodes:
            raise ValueError("cannot select from an empty compact frontier")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        matrix = extractor.batch(nodes)
        scores = matrix @ self.theta
        if self.rng.random() < epsilon:
            index = int(self.rng.integers(len(nodes)))
        else:
            best = float(np.max(scores))
            tied = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=1e-12))
            # Nodes are supplied in record-ID order, making the tie rule stable.
            index = int(tied[0])
        self.selection_count += 1
        return SelectedRecord(nodes[index], matrix[index].copy(), float(scores[index]))

    def update(
        self,
        selected: SelectedRecord,
        reward: float,
        next_selected: SelectedRecord | None,
    ) -> float:
        bootstrap = 0.0 if next_selected is None else next_selected.value
        td_error = float(reward + self.discount * bootstrap - selected.value)
        bounded_error = float(np.clip(td_error, -5.0, 5.0))
        normalizer = 1.0 + float(selected.features @ selected.features)
        self.theta += (
            self.learning_rate
            * bounded_error
            * selected.features
            / normalizer
        )
        if not np.isfinite(self.theta).all():
            raise FloatingPointError("compact SARSA produced non-finite weights")
        self.update_count += 1
        return td_error


@dataclass(frozen=True, slots=True)
class CompactEpisodeResult:
    success: bool
    expansions: int
    generated: int
    accepted: int
    peak_frontier: int
    archive_size: int
    duplicate_rejected: int
    dominated_retired: int
    reopened: int
    total_reward: float
    mean_absolute_td_error: float
    solution_node: CompactNode | None = field(default=None, compare=False, repr=False)

    @property
    def core_witness(self) -> tuple[CompactGate, ...]:
        if self.solution_node is None:
            return ()
        return reconstruct_core_witness(self.solution_node)


class CompactOnlineEpisode:
    """One MDP episode whose action selects a persistent frontier record."""

    def __init__(self, problem: CompactParityProblem) -> None:
        self.problem = problem
        self.extractor = CompactFeatureExtractor(problem)

    def run(
        self,
        ranker: CompactLinearSarsaRanker,
        *,
        epsilon: float,
        max_expansions: int,
        learn: bool,
    ) -> CompactEpisodeResult:
        if max_expansions <= 0:
            raise ValueError("max_expansions must be positive")

        frontier = CompactFrontier()
        root = CompactNode(self.problem.initial_state())
        assert frontier.insert(root)
        selected = ranker.select(frontier.nodes(), self.extractor, epsilon=epsilon)
        generated = 0
        accepted = 1
        total_reward = 0.0
        td_errors: list[float] = []

        for expansion in range(1, max_expansions + 1):
            frontier.remove_for_expansion(selected.node)
            children = self.problem.expand(selected.node)
            generated += len(children)
            solution: CompactNode | None = None
            for child in children:
                if frontier.insert(child):
                    accepted += 1
                    if self.problem.is_goal(child.state):
                        solution = child
                        break

            terminated = solution is not None
            exhausted = len(frontier) == 0
            term_count = max(1, len(self.problem.target.terms))
            # Dense online reward: favor expansion of records that have emitted
            # more required terms, retain a per-expansion search cost, and give
            # an explicit terminal bonus.  No reference witness is consulted.
            reward = (
                selected.node.state.phase_count / term_count
                - 1.0 / max_expansions
            )
            if terminated:
                reward += 5.0
            elif exhausted:
                reward -= 5.0
            total_reward += reward

            next_selected: SelectedRecord | None = None
            if not terminated and not exhausted and expansion < max_expansions:
                next_selected = ranker.select(
                    frontier.nodes(),
                    self.extractor,
                    epsilon=epsilon,
                )
            if learn:
                td_errors.append(ranker.update(selected, reward, next_selected))

            if terminated or exhausted or expansion == max_expansions:
                return CompactEpisodeResult(
                    success=terminated,
                    expansions=expansion,
                    generated=generated,
                    accepted=accepted,
                    peak_frontier=frontier.peak_size,
                    archive_size=frontier.archive_size,
                    duplicate_rejected=frontier.duplicate_rejected,
                    dominated_retired=frontier.dominated_retired,
                    reopened=frontier.reopened,
                    total_reward=total_reward,
                    mean_absolute_td_error=(
                        float(np.mean(np.abs(td_errors))) if td_errors else 0.0
                    ),
                    solution_node=solution,
                )
            assert next_selected is not None
            selected = next_selected

        raise AssertionError("bounded compact episode must return")


@dataclass(frozen=True, slots=True)
class CheckpointEvaluation:
    episode: int
    success: bool
    expansions: int
    weight_norm: float


@dataclass(frozen=True, slots=True)
class CompactTrainingResult:
    completed: bool
    cpu_seconds: float
    episodes_requested: int
    episodes_completed: int
    training_successes: int
    initial_evaluation: CompactEpisodeResult
    final_evaluation: CompactEpisodeResult
    checkpoint_evaluations: tuple[CheckpointEvaluation, ...]
    history: tuple[CompactEpisodeResult, ...]
    weights: tuple[float, ...]


def _better(
    candidate: CompactEpisodeResult,
    incumbent: CompactEpisodeResult,
) -> bool:
    if candidate.success != incumbent.success:
        return candidate.success
    return candidate.expansions < incumbent.expansions


def train_compact_online(
    problem: CompactParityProblem,
    *,
    episodes: int = 64,
    training_max_expansions: int = 256,
    evaluation_max_expansions: int = 3_000,
    checkpoint_interval: int = 4,
    cpu_limit_seconds: float = 1_800.0,
    seed: int = 23,
    learning_rate: float = 0.05,
    epsilon_start: float = 0.50,
    epsilon_end: float = 0.02,
) -> CompactTrainingResult:
    """Train online and retain the best frozen checkpoint within a CPU guard."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if training_max_expansions <= 0 or evaluation_max_expansions <= 0:
        raise ValueError("expansion caps must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if cpu_limit_seconds <= 0.0:
        raise ValueError("cpu_limit_seconds must be positive")

    ranker = CompactLinearSarsaRanker(
        learning_rate=learning_rate,
        seed=seed,
    )
    runner = CompactOnlineEpisode(problem)
    started = time.process_time()

    initial = runner.run(
        ranker,
        epsilon=0.0,
        max_expansions=evaluation_max_expansions,
        learn=False,
    )
    best_evaluation = initial
    best_weights = ranker.theta.copy()
    checkpoints = [
        CheckpointEvaluation(
            episode=0,
            success=initial.success,
            expansions=initial.expansions,
            weight_norm=float(np.linalg.norm(ranker.theta)),
        )
    ]
    history: list[CompactEpisodeResult] = []

    for episode in range(episodes):
        if time.process_time() - started >= cpu_limit_seconds:
            break
        fraction = episode / max(1, episodes - 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        history.append(
            runner.run(
                ranker,
                epsilon=epsilon,
                max_expansions=training_max_expansions,
                learn=True,
            )
        )

        completed_episodes = episode + 1
        if (
            completed_episodes % checkpoint_interval == 0
            or completed_episodes == episodes
        ):
            frozen = runner.run(
                ranker,
                epsilon=0.0,
                max_expansions=evaluation_max_expansions,
                learn=False,
            )
            checkpoints.append(
                CheckpointEvaluation(
                    episode=completed_episodes,
                    success=frozen.success,
                    expansions=frozen.expansions,
                    weight_norm=float(np.linalg.norm(ranker.theta)),
                )
            )
            if _better(frozen, best_evaluation):
                best_evaluation = frozen
                best_weights = ranker.theta.copy()

    ranker.theta[:] = best_weights
    final = runner.run(
        ranker,
        epsilon=0.0,
        max_expansions=evaluation_max_expansions,
        learn=False,
    )
    cpu_seconds = time.process_time() - started
    return CompactTrainingResult(
        completed=len(history) == episodes,
        cpu_seconds=cpu_seconds,
        episodes_requested=episodes,
        episodes_completed=len(history),
        training_successes=sum(result.success for result in history),
        initial_evaluation=initial,
        final_evaluation=final,
        checkpoint_evaluations=tuple(checkpoints),
        history=tuple(history),
        weights=tuple(float(value) for value in ranker.theta),
    )
