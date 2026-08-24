"""Transparent online semi-gradient SARSA frontier-record ranker."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import time
from typing import Callable, Sequence

import numpy as np

from .benchmarks import SynthesisTarget
from .certify import certify_state
from .search import HybridSearch, SearchRecord, symbolic_distance_components

FEATURE_NAMES = (
    "bias",
    "t_fraction",
    "cnot_fraction",
    "gate_fraction",
    "depth_fraction",
    "wire_depth_q0",
    "wire_depth_q1",
    "wire_depth_q2",
    "rotation_fraction",
    "anticommuting_pair_fraction",
    "mean_pauli_weight",
    "tableau_mismatch",
    "rotation_sequence_mismatch",
    "rotation_multiset_mismatch",
    "total_symbolic_distance",
    "target_rotation_fraction",
    "target_tableau_nonidentity",
    "last_H",
    "last_S",
    "last_SDG",
    "last_T",
    "last_TDG",
    "last_CNOT",
    "register_fraction",
)
FEATURE_DIM = len(FEATURE_NAMES)


def record_features(record: SearchRecord, target: SynthesisTarget) -> np.ndarray:
    state = record.state
    total, tableau_mismatch, sequence_mismatch, multiset_mismatch = (
        symbolic_distance_components(state, target)
    )
    maximum_pairs = max(1, math.comb(max(2, state.budget.max_t_count), 2))
    wire_depths = list(state.wire_depths) + [0] * (3 - state.num_qubits)
    last_name = None if state.last_gate is None else state.last_gate.name
    target_nonidentity = sum(
        payload != identity
        for payload, identity in zip(
            target.tableau_payload,
            _identity_tableau_payload(target.num_qubits),
            strict=True,
        )
    )
    values = np.asarray(
        [
            1.0,
            state.t_count / max(1, state.budget.max_t_count),
            state.cnot_count / max(1, state.budget.max_cnot_count),
            state.gate_count / max(1, state.budget.max_gates),
            state.depth / max(1, state.budget.max_depth),
            *(depth / max(1, state.budget.max_depth) for depth in wire_depths[:3]),
            len(state.rotations) / max(1, state.budget.max_t_count),
            state.anticommuting_pairs / maximum_pairs,
            state.mean_pauli_weight / state.num_qubits,
            tableau_mismatch / max(1, 2 * state.num_qubits),
            sequence_mismatch / max(1, state.budget.max_t_count + len(target.rotation_payloads)),
            multiset_mismatch / max(1, state.budget.max_t_count + len(target.rotation_payloads)),
            total / max(1, 4 * state.num_qubits + 2 * state.budget.max_t_count),
            len(target.rotation_payloads) / max(1, state.budget.max_t_count),
            target_nonidentity / max(1, 2 * state.num_qubits),
            *(1.0 if last_name == name else 0.0 for name in ("H", "S", "SDG", "T", "TDG", "CNOT")),
            state.num_qubits / 3.0,
        ],
        dtype=np.float64,
    )
    if values.shape != (FEATURE_DIM,):
        raise AssertionError(f"feature schema produced {values.shape}")
    return values


def _identity_tableau_payload(n: int) -> tuple[tuple[int, int, int], ...]:
    result = []
    for q in range(n):
        result.append((1 << q, 0, 1))
    for q in range(n):
        result.append((0, 1 << q, 1))
    return tuple(result)


@dataclass(slots=True)
class LinearSarsaRanker:
    learning_rate: float = 0.01
    gamma: float = 1.0
    seed: int = 0
    theta: np.ndarray = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    updates: int = field(init=False, default=0)
    feature_time_ns: int = field(init=False, default=0)
    scoring_time_ns: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.theta = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def snapshot(
        self, records: Sequence[SearchRecord], target: SynthesisTarget
    ) -> tuple[tuple[SearchRecord, ...], np.ndarray, np.ndarray]:
        nodes = tuple(records)
        if not nodes:
            return nodes, np.empty((0, FEATURE_DIM)), np.empty(0)
        started = time.perf_counter_ns()
        rows: list[np.ndarray] = []
        for record in nodes:
            if record.features is None:
                record.features = record_features(record, target)
            rows.append(np.asarray(record.features, dtype=np.float64))
        matrix = np.asarray(rows, dtype=np.float64)
        self.feature_time_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        scores = matrix @ self.theta
        self.scoring_time_ns += time.perf_counter_ns() - started
        return nodes, matrix, scores

    def choose(
        self,
        records: Sequence[SearchRecord],
        target: SynthesisTarget,
        epsilon: float,
    ) -> tuple[int, np.ndarray, float]:
        nodes, matrix, scores = self.snapshot(records, target)
        if not nodes:
            raise RuntimeError("cannot choose from an empty frontier")
        if self.rng.random() < epsilon:
            index = int(self.rng.integers(len(nodes)))
        else:
            best = float(np.max(scores))
            tied = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=1e-12))
            index = min((int(i) for i in tied), key=lambda i: nodes[i].record_id)
        return nodes[index].record_id, matrix[index].copy(), float(scores[index])

    def update(
        self,
        features: np.ndarray,
        q_value: float,
        reward: float,
        next_q_value: float | None,
    ) -> float:
        target = reward if next_q_value is None else reward + self.gamma * next_q_value
        td_error = float(target - q_value)
        self.theta += self.learning_rate * td_error * features
        np.clip(self.theta, -50.0, 50.0, out=self.theta)
        self.updates += 1
        return td_error


@dataclass(frozen=True, slots=True)
class EpisodeLog:
    episode: int
    target: str
    success: bool
    certified: bool
    expansions: int
    total_reward: float
    mean_abs_td: float
    epsilon: float
    cpu_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    episodes: tuple[EpisodeLog, ...]
    total_expansions: int
    cpu_seconds: float
    wall_seconds: float
    deadline_hit: bool
    profile_totals: dict[str, int]
    peak_frontier: int
    maximum_rotation_length: int


ProgressCallback = Callable[[dict[str, float | int | str | bool]], None]


def train_online_sarsa(
    policy: LinearSarsaRanker,
    targets: Sequence[SynthesisTarget],
    *,
    episodes: int = 36,
    max_expansions: int = 256,
    epsilon_start: float = 0.35,
    epsilon_end: float = 0.03,
    deadline_seconds: float = 1650.0,
    progress: ProgressCallback | None = None,
) -> TrainingResult:
    if not targets:
        raise ValueError("training requires at least one target")
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    logs: list[EpisodeLog] = []
    total_expansions = 0
    deadline_hit = False
    profile_totals: Counter[str] = Counter()
    peak_frontier = 0
    maximum_rotation_length = 0

    for episode in range(episodes):
        if _deadline_hit(cpu_start, wall_start, deadline_seconds):
            deadline_hit = True
            break
        fraction = episode / max(1, episodes - 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        target = targets[episode % len(targets)]
        env = HybridSearch(target, max_expansions=max_expansions)
        record_id, features, q_value = policy.choose(env.open_records(), target, epsilon)
        total_reward = 0.0
        td_errors: list[float] = []
        episode_cpu = time.process_time()
        episode_wall = time.perf_counter()

        while True:
            if _deadline_hit(cpu_start, wall_start, deadline_seconds):
                deadline_hit = True
                break
            transition = env.expand(record_id)
            total_reward += transition.reward
            done = transition.terminated or transition.truncated
            if done:
                td_errors.append(policy.update(features, q_value, transition.reward, None))
                break
            next_record_id, next_features, next_q = policy.choose(
                env.open_records(), target, epsilon
            )
            td_errors.append(policy.update(features, q_value, transition.reward, next_q))
            record_id, features = next_record_id, next_features
            q_value = float(features @ policy.theta)

        state = env.solution_state()
        certified = False
        if state is not None:
            certification_started = time.perf_counter_ns()
            certified = certify_state(target, state).success
            profile_totals["certification_ns"] += (
                time.perf_counter_ns() - certification_started
            )
            if not certified:
                raise AssertionError("symbolic terminal state failed dense certification")
        total_expansions += env.expansions
        metrics = env.metrics()
        peak_frontier = max(peak_frontier, int(metrics["frontier_peak"]))
        maximum_rotation_length = max(
            maximum_rotation_length,
            int(metrics["max_rotation_length"]),
        )
        for name, value in dict(metrics["profile"]).items():
            profile_totals[str(name)] += int(value)
        logs.append(
            EpisodeLog(
                episode=episode,
                target=target.name,
                success=state is not None,
                certified=certified,
                expansions=env.expansions,
                total_reward=total_reward,
                mean_abs_td=float(np.mean(np.abs(td_errors))) if td_errors else 0.0,
                epsilon=epsilon,
                cpu_seconds=time.process_time() - episode_cpu,
                wall_seconds=time.perf_counter() - episode_wall,
            )
        )
        if progress is not None and (episode == 0 or (episode + 1) % 4 == 0 or deadline_hit):
            cpu_elapsed = time.process_time() - cpu_start
            wall_elapsed = time.perf_counter() - wall_start
            recent = logs[-min(8, len(logs)) :]
            progress(
                {
                    "phase": "training",
                    "episode": episode + 1,
                    "episodes": episodes,
                    "target": target.name,
                    "recent_success_rate": sum(log.success for log in recent) / len(recent),
                    "total_expansions": total_expansions,
                    "expansions_per_cpu_second": total_expansions / max(cpu_elapsed, 1e-9),
                    "cpu_seconds": cpu_elapsed,
                    "wall_seconds": wall_elapsed,
                    "projected_cpu_seconds": cpu_elapsed * episodes / max(1, episode + 1),
                    "deadline_seconds": deadline_seconds,
                    "deadline_hit": deadline_hit,
                }
            )
        if deadline_hit:
            break

    return TrainingResult(
        episodes=tuple(logs),
        total_expansions=total_expansions,
        cpu_seconds=time.process_time() - cpu_start,
        wall_seconds=time.perf_counter() - wall_start,
        deadline_hit=deadline_hit,
        profile_totals=dict(profile_totals),
        peak_frontier=peak_frontier,
        maximum_rotation_length=maximum_rotation_length,
    )


def _deadline_hit(cpu_start: float, wall_start: float, limit: float) -> bool:
    return max(time.process_time() - cpu_start, time.perf_counter() - wall_start) >= limit


__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "EpisodeLog",
    "LinearSarsaRanker",
    "TrainingResult",
    "record_features",
    "train_online_sarsa",
]
