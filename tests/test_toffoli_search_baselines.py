"""Deterministic integration coverage for constrained Toffoli baselines.

These tests drive persistent frontier records directly.  They intentionally do
not instantiate an RL policy or choose gates: after a scheduler picks one
record, :class:`ToffoliParityNetworkProblem` owns exhaustive legal child
generation and the environment owns independent dense certification.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO
from typing import Literal
import pytest

from benchmarks.toffoli import (
    KNOWN_TOFFOLI_BUDGET,
    KNOWN_TOFFOLI_GATES,
    TOFFOLI_NUM_QUBITS,
    toffoli_reference_unitary,
)
from certification.simulator import SimulatorCertificationEngine, SynthesisTarget
from ckt_types import ResourceBudget
from config import Config
from enums import GateType
from env.rl_env import CircuitSynthesisEnv
from rl.policy import LinearQPolicy
from rl.toffoli_parity import ToffoliParityFeatureProvider, ToffoliParityRewardModel
from search.problems.toffoli_parity import ToffoliParityNetworkProblem
from train import Trainer


Scheduler = Literal["fifo", "uniform"]
_EXHAUSTIVE_STEP_CAP = 100_000


@dataclass(frozen=True)
class BaselineRun:
    """Small, equality-friendly trace returned by the direct scheduler helper."""

    scheduler: Scheduler
    seed: int
    terminated: bool
    truncated: bool
    certified: bool
    expansions: int
    trace: tuple[tuple[object, ...], ...]
    solution_actions: tuple[object, ...]
    search_metrics: tuple[tuple[str, int], ...]


class _RecordingToffoliEnv(CircuitSynthesisEnv):
    """Retain actual record IDs expanded through the Gym-index adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expanded_record_ids: list[int | None] = []

    def step(self, action):
        result = super().step(action)
        self.expanded_record_ids.append(result[-1].get("selected_record_id"))
        return result


class _ChooseToffoliCnotRecordPolicy(LinearQPolicy):
    """Test-only ranker that chooses a non-first existing core record.

    It never creates a gate or a child.  Its sole purpose is to make a tied
    frontier selection observable, so the test can detect an accidental
    equality-based ``list.index`` lookup in ``Trainer``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proposed_record_ids: list[int | None] = []
        self.chose_nonfirst_core_record = False

    def select_node(self, nodes, epsilon=0.1):
        if len(nodes) == 1:
            selected = nodes[0]
        else:
            selected = next(
                node
                for node in nodes
                if node.action is not None
                and node.action.gate_type is GateType.CNOT
                and node.action.qubits == (0, 1)
            )
            self.chose_nonfirst_core_record = selected is not nodes[0]
        self.proposed_record_ids.append(selected.record_id)
        return selected


def _copy_budget(
    *,
    max_t_count: int = KNOWN_TOFFOLI_BUDGET.max_t_count,
    max_two_qubit_count: int = KNOWN_TOFFOLI_BUDGET.max_two_qubit_count,
    max_gates: int = KNOWN_TOFFOLI_BUDGET.max_gates,
    max_depth: int = KNOWN_TOFFOLI_BUDGET.max_depth,
) -> ResourceBudget:
    """Build a fresh budget so tests cannot mutate benchmark constants."""

    return ResourceBudget(
        max_t_count=max_t_count,
        max_two_qubit_count=max_two_qubit_count,
        max_gates=max_gates,
        max_depth=max_depth,
    )


def run_toffoli_search_baseline(
    *,
    scheduler: Scheduler,
    seed: int,
    budget: ResourceBudget,
    max_steps: int,
) -> BaselineRun:
    """Run a direct deterministic record scheduler against the real problem.

    ``fifo`` means the oldest stable archive record.  ``uniform`` is the
    deterministic uniform-cost baseline: consumed two-qubit count, total gate
    count, depth, then stable record ID.  Neither mode inspects or emits a
    gate; all child generation remains inside the supplied constrained search
    problem.

    The returned witness is reconstructed *only* from ``env.solution_node``.
    In particular, a failed run yields an empty tuple rather than a known
    Toffoli reference witness.
    """

    if scheduler not in {"fifo", "uniform"}:
        raise ValueError(f"unsupported scheduler {scheduler!r}")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    config = Config(
        num_qubits=TOFFOLI_NUM_QUBITS,
        budget=budget,
        max_steps=max_steps,
        # This Gym compatibility mask does not limit the core frontier; the
        # helper adapts a selected concrete record back to its current index.
        max_frontier=64,
        fairness_interval=0,
        seed=seed,
    )
    environment = CircuitSynthesisEnv(
        config,
        SimulatorCertificationEngine(SynthesisTarget(toffoli_reference_unitary())),
        problem=ToffoliParityNetworkProblem(),
    )
    environment.reset(seed=seed)
    trace: list[tuple[object, ...]] = []
    terminated = environment.solution_node is not None
    truncated = False

    while not (terminated or truncated):
        nodes = environment.current_nodes()
        if not nodes:  # pragma: no cover - environment should report terminal
            raise AssertionError("nonterminal Toffoli environment exposed no frontier records")
        if scheduler == "fifo":
            selected = min(nodes, key=lambda node: int(node.record_id or 0))
        else:
            selected = min(
                nodes,
                key=lambda node: (
                    int(node.state.two_qubit_count),
                    int(node.state.num_gates),
                    int(node.state.depth),
                    int(node.record_id or 0),
                ),
            )
        index = next(index for index, node in enumerate(nodes) if node is selected)
        prefix = tuple(repr(action) for action in selected.reconstruct_actions())
        _, reward, terminated, truncated, info = environment.step(index)
        # This check guards against accidentally using a positional frontier
        # index as a semantic action when the underlying ordering changes.
        assert info["selected_record_id"] == selected.record_id
        assert not info["selected_by_fairness"]
        trace.append(
            (
                int(selected.record_id or 0),
                prefix,
                int(info["num_children"]),
                int(info["num_accepted"]),
                int(info["num_pruned"]),
                bool(info["num_certified"]),
                float(reward),
            )
        )

    solution_actions: tuple[object, ...]
    if environment.solution_node is None:
        solution_actions = ()
    else:
        solution_actions = tuple(environment.solution_node.reconstruct_actions())
    return BaselineRun(
        scheduler=scheduler,
        seed=seed,
        terminated=bool(terminated),
        truncated=bool(truncated),
        certified=environment.solution_node is not None,
        expansions=environment.steps,
        trace=tuple(trace),
        solution_actions=solution_actions,
        search_metrics=tuple(sorted(environment.search_metrics.items())),
    )


@pytest.mark.parametrize("scheduler", ("fifo", "uniform"))
def test_toffoli_baseline_schedulers_are_seed_reproducible(scheduler: Scheduler):
    """Both direct record schedulers reproduce the same bounded trace by seed."""

    first = run_toffoli_search_baseline(
        scheduler=scheduler,
        seed=37,
        budget=_copy_budget(),
        max_steps=32,
    )
    second = run_toffoli_search_baseline(
        scheduler=scheduler,
        seed=37,
        budget=_copy_budget(),
        max_steps=32,
    )

    # Instrumentation deliberately records wall-clock nanoseconds, which are
    # observable but cannot be reproducible.  Compare the complete scheduler
    # result after removing only those timing counters, and separately retain
    # a contract check that every timing sample is present and nonnegative.
    deterministic_first = replace(
        first,
        search_metrics=tuple(
            item for item in first.search_metrics if not item[0].endswith("_time_ns")
        ),
    )
    deterministic_second = replace(
        second,
        search_metrics=tuple(
            item for item in second.search_metrics if not item[0].endswith("_time_ns")
        ),
    )
    assert deterministic_first == deterministic_second
    first_timings = {
        name: value for name, value in first.search_metrics if name.endswith("_time_ns")
    }
    second_timings = {
        name: value for name, value in second.search_metrics if name.endswith("_time_ns")
    }
    assert first_timings.keys() == second_timings.keys()
    assert first_timings
    assert all(value >= 0 for value in (*first_timings.values(), *second_timings.values()))
    assert first.trace
    assert first.expansions == len(first.trace)


@pytest.mark.parametrize(
    "budget",
    (
        _copy_budget(max_t_count=6),
        _copy_budget(max_two_qubit_count=5),
        _copy_budget(max_gates=14),
    ),
    ids=("t-count", "two-qubit-count", "gate-count"),
)
def test_toffoli_negative_budgets_exhaust_without_a_certified_witness(
    budget: ResourceBudget,
):
    """The finite constrained graph must exhaust rather than fake success."""

    result = run_toffoli_search_baseline(
        scheduler="fifo",
        seed=11,
        budget=budget,
        max_steps=_EXHAUSTIVE_STEP_CAP,
    )

    assert result.terminated
    assert not result.truncated
    assert not result.certified
    assert result.solution_actions == ()
    assert result.expansions < _EXHAUSTIVE_STEP_CAP
    # Each of these limits is below the declared normal-form lower bound, so
    # the initial frontier record itself is infeasible.  This proves terminal
    # exhaustion rather than merely reaching the generous test cap.
    assert result.expansions == 1
    assert len(result.trace) == 1
    assert result.trace[0][1] == ()  # root has no prefix actions
    assert result.trace[0][2] == 0  # root expansion generated no children


def test_failed_toffoli_search_never_substitutes_the_reference_witness():
    """Failure reports no ``solution_node`` actions, never the known reference."""

    result = run_toffoli_search_baseline(
        scheduler="fifo",
        seed=5,
        budget=_copy_budget(max_gates=0),
        max_steps=_EXHAUSTIVE_STEP_CAP,
    )

    assert result.terminated
    assert not result.truncated
    assert not result.certified
    assert result.solution_actions == ()
    assert result.solution_actions != KNOWN_TOFFOLI_GATES


def test_trainer_expands_the_identity_selected_toffoli_provider_record():
    """Trainer keeps a concrete normal-form record through its index adapter.

    The root and outer-H expansions are forced by the problem.  Once the
    normal-form core exposes several tied records, the test policy returns the
    existing ``CNOT(0, 1)`` object rather than a positional action.  The
    actual third expansion must therefore carry that exact record ID.
    """

    problem = ToffoliParityNetworkProblem()
    provider = ToffoliParityFeatureProvider(
        problem,
        target_fingerprint="toffoli-trainer-identity-test",
    )
    environment = _RecordingToffoliEnv(
        Config(
            num_qubits=TOFFOLI_NUM_QUBITS,
            budget=_copy_budget(),
            max_steps=3,
            max_frontier=64,
            fairness_interval=0,
            seed=13,
        ),
        SimulatorCertificationEngine(SynthesisTarget(toffoli_reference_unitary())),
        problem=problem,
        feature_provider=provider,
        reward_model=ToffoliParityRewardModel(provider),
    )
    policy = _ChooseToffoliCnotRecordPolicy(feature_provider=provider, seed=13)
    trainer = Trainer(environment, policy=policy)
    trainer.epsilon = 0.0
    trainer.min_epsilon = 0.0

    with redirect_stdout(StringIO()):
        history = trainer.train(1)

    assert environment.feature_dim == provider.dimension == 66
    assert policy.feature_provider is provider
    assert policy.chose_nonfirst_core_record
    assert history[0]["truncated"]
    assert not history[0]["certified"]
    # The policy proposed persistent records [root, outer H, CNOT], and the
    # environment expanded that exact sequence under the normal-form problem.
    assert environment.expanded_record_ids == policy.proposed_record_ids
    assert environment.expanded_record_ids[-1] not in {
        environment.expanded_record_ids[0],
        environment.expanded_record_ids[1],
    }
