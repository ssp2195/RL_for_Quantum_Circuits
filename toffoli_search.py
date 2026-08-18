"""Train and evaluate a learned frontier scheduler for exact Toffoli-3.

The policy selects an already-open frontier record.  The constrained
``ToffoliParityNetworkProblem`` remains solely responsible for legal child
generation and terminal structure.  In particular, this runner never embeds
or replays a reference gate sequence: its analytical target is the CCX basis
permutation, and its diagram is reconstructed only from a fresh learned
``solution_node``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from benchmarks.toffoli import TOFFOLI_NUM_QUBITS, toffoli_reference_unitary, validate_toffoli_unitary
from circuit.gate import Gate
from reporting.toffoli import stable_matrix_digest
from reporting.toffoli_search import save_toffoli_search_artifacts


DEFAULT_SEED = 23
DEFAULT_EPISODES = 5
# FIFO reaches the constrained exact witness at 2734 expansions in the
# calibrated normal form.  The compact five-episode training protocol keeps
# exploration bounded while frozen schedulers share a sufficient 3000-step
# evaluation horizon.
DEFAULT_TRAINING_MAX_STEPS = 100
DEFAULT_EVALUATION_MAX_STEPS = 3_000
# Random is an evidence-only reproducibility probe.  It is deliberately not
# a second broad synthesis baseline, so two short same-seed traces suffice.
DEFAULT_RANDOM_REPRODUCIBILITY_MAX_STEPS = 32
DEFAULT_RANDOM_REPRODUCIBILITY_SEED_OFFSETS = (0, 1, 2)
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPSILON_START = 0.25
DEFAULT_EPSILON_MINIMUM = 0.03
DEFAULT_EPSILON_DECAY = 0.97

# These are resource constraints, not a circuit description.  The runner
# never contains a sequence of native gates from which Toffoli could be
# reconstructed; every candidate comes from its own frontier solution node.
TOFFOLI_BUDGET = {
    "max_t_count": 7,
    "max_two_qubit_count": 6,
    "max_gates": 15,
    "max_depth": 12,
}
EXPECTED_GATE_PROFILE = {
    "t_count": 7,
    "two_qubit_count": 6,
    "num_gates": 15,
    "h_count": 2,
}


def _stage3_dependencies() -> SimpleNamespace:
    """Load Stage 3 core pieces only when the runner is actually invoked.

    Keeping these imports late makes the CLI module inspectable while the
    constrained problem is being developed, and gives users an actionable
    integration error rather than an unrelated package-import cascade.
    """

    try:
        from certification.base import CertStatus
        from certification.simulator import (
            SimulatorCertificationEngine,
            SynthesisTarget,
            unitary_from_gates,
        )
        from ckt_types import ResourceBudget
        from config import Config
        from env.rl_env import CircuitSynthesisEnv
        from rl.policy import LinearQPolicy
        from rl.toffoli_parity import (
            ToffoliParityFeatureProvider,
            ToffoliParityRewardModel,
        )
        from search.problems.toffoli_parity import ToffoliParityNetworkProblem
        from search.problems.toffoli_parity import (
            phase_identity_holds,
            phase_identity_rows,
        )
        from train import Trainer
    except ImportError as exc:  # pragma: no cover - installation diagnostic
        raise RuntimeError(
            "Stage 3 requires the constrained Toffoli problem, parity feature "
            "provider, reward adapter, and provider-aware LinearQPolicy."
        ) from exc
    return SimpleNamespace(
        CertStatus=CertStatus,
        Config=Config,
        CircuitSynthesisEnv=CircuitSynthesisEnv,
        LinearQPolicy=LinearQPolicy,
        ResourceBudget=ResourceBudget,
        SimulatorCertificationEngine=SimulatorCertificationEngine,
        SynthesisTarget=SynthesisTarget,
        ToffoliParityFeatureProvider=ToffoliParityFeatureProvider,
        ToffoliParityNetworkProblem=ToffoliParityNetworkProblem,
        ToffoliParityRewardModel=ToffoliParityRewardModel,
        Trainer=Trainer,
        phase_identity_holds=phase_identity_holds,
        phase_identity_rows=phase_identity_rows,
        unitary_from_gates=unitary_from_gates,
    )


def _as_gate(action: object) -> Gate:
    """Convert a public reconstructed search action into a concrete gate."""

    if isinstance(action, Gate):
        return action
    try:
        gate_type = getattr(action, "gate_type")
        qubits = tuple(getattr(action, "qubits"))
    except AttributeError as exc:  # pragma: no cover - SearchNode contract
        raise TypeError(f"reconstructed action is not a circuit gate: {action!r}") from exc
    return Gate(gate_type, qubits)


def _operation(gate: Gate) -> dict[str, object]:
    return {"gate": gate.gate_type.name, "qubits": [int(qubit) for qubit in gate.qubits]}


def _stage_name(progress: object) -> str:
    raw = getattr(progress, "stage", "UNKNOWN")
    value = getattr(raw, "name", raw)
    return str(value).upper().split(".")[-1]


def _progress_fields(problem: object, node: object) -> dict[str, object]:
    """Expose the constrained analyzer's required review fields per selection."""

    progress = problem.analyze(getattr(node, "state"))
    rows = tuple(getattr(progress, "basis_rows", ()))
    emitted_terms = getattr(progress, "emitted_terms", 0)
    return {
        "stage": _stage_name(progress),
        "basis_rows": [int(row) for row in rows],
        "emitted_terms": int(emitted_terms),
        "is_terminal_candidate": bool(problem.is_terminal_candidate(node)),
    }


def _new_environment(
    deps: SimpleNamespace,
    *,
    problem: object,
    feature_provider: object | None,
    reward_model: object | None,
    certification_engine: object,
    seed: int,
    max_steps: int,
    budget: Mapping[str, int] = TOFFOLI_BUDGET,
) -> object:
    resource_budget = deps.ResourceBudget(
        max_t_count=int(budget["max_t_count"]),
        max_two_qubit_count=int(budget["max_two_qubit_count"]),
        max_gates=int(budget["max_gates"]),
        max_depth=int(budget["max_depth"]),
    )
    config = deps.Config(
        num_qubits=TOFFOLI_NUM_QUBITS,
        budget=resource_budget,
        max_steps=max_steps,
        # The runner selects the concrete current record by identity and
        # passes its positional adapter index through Gym's compatibility
        # adapter, so keep the advertised action space large enough to cover
        # every calibrated frontier record.
        max_frontier=131_072,
        discount=1.0,
        fairness_interval=0,
        seed=seed,
        target_aware_features=False,
        reward_mode="legacy",
    )
    return deps.CircuitSynthesisEnv(
        config,
        certification_engine,
        problem=problem,
        feature_provider=feature_provider,
        reward_model=reward_model,
        # Rollout decisions use persistent frontier records directly.  Avoid
        # materialising an unused Gym observation across every broad baseline
        # expansion; policy scoring still calls its provider explicitly.
        observation_features=False,
    )


def _new_policy(
    deps: SimpleNamespace,
    feature_provider: object,
    *,
    seed: int,
    learning_rate: float,
) -> object:
    """Create a provider-bound policy, never silently falling back to legacy features."""

    try:
        return deps.LinearQPolicy(
            feature_provider=feature_provider,
            lr=learning_rate,
            gamma=1.0,
            seed=seed,
        )
    except TypeError as exc:  # pragma: no cover - core integration diagnostic
        raise RuntimeError(
            "LinearQPolicy must support feature_provider= for the Toffoli parity schema"
        ) from exc


def _choose_node(
    scheduler: str,
    nodes: Sequence[object],
    *,
    policy: object | None,
    rng: np.random.Generator | None,
) -> object:
    if not nodes:
        raise ValueError("cannot choose from an empty frontier")
    if scheduler == "fifo":
        return min(nodes, key=lambda node: int(getattr(node, "record_id", 0) or 0))
    if scheduler == "uniform":
        return min(
            nodes,
            key=lambda node: (
                int(getattr(getattr(node, "state"), "two_qubit_count", 0)),
                int(getattr(getattr(node, "state"), "num_gates", 0)),
                int(getattr(getattr(node, "state"), "depth", 0)),
                int(getattr(node, "record_id", 0) or 0),
            ),
        )
    if scheduler == "random":
        if rng is None:
            raise ValueError("random scheduling requires a seeded RNG")
        return nodes[int(rng.integers(len(nodes)))]
    if scheduler in {"zero_policy", "learned"}:
        if policy is None:
            raise ValueError(f"{scheduler} scheduling requires a policy")
        selected = policy.select_node(nodes, epsilon=0.0)
        if selected is None:  # pragma: no cover - policy contract
            raise RuntimeError("policy returned no node for a nonempty frontier")
        return selected
    raise ValueError(f"unsupported Toffoli scheduler {scheduler!r}")


def _archive_metrics(environment: object) -> dict[str, int]:
    """Return final archive-state counts without relying on a private cache.

    ``Frontier`` retains expanded records for Pareto dominance, so the active
    archive population and the selectable frontier population are different
    quantities.  Preserve both in every scheduler report to make a short
    learned trace auditable against the broad deterministic baselines.
    """

    frontier = getattr(environment, "frontier", None)
    archive = getattr(frontier, "archive", None)
    all_records = getattr(archive, "all_records", None)
    if not callable(all_records):
        return {
            "archive_record_count": 0,
            "archive_active_records": 0,
            "archive_tombstoned_records": 0,
            "archive_expanded_records": 0,
            "archive_queued_records": 0,
        }
    records = tuple(all_records())
    return {
        "archive_record_count": len(records),
        "archive_active_records": sum(bool(getattr(record, "active", False)) for record in records),
        "archive_tombstoned_records": sum(
            bool(getattr(record, "tombstoned", False)) for record in records
        ),
        "archive_expanded_records": sum(
            bool(getattr(record, "expanded", False)) for record in records
        ),
        "archive_queued_records": sum(
            bool(getattr(record, "queued", False)) for record in records
        ),
    }


def _rollout(
    environment: object,
    *,
    problem: object,
    scheduler: str,
    seed: int,
    policy: object | None = None,
) -> tuple[dict[str, Any], tuple[Gate, ...]]:
    """Run one fresh scheduler evaluation and retain an auditable trace."""

    if policy is not None:
        # The scheduler is passed explicitly below, but retain the public
        # environment association for callers inspecting a frozen rollout.
        environment.policy = policy
    _, reset_info = environment.reset(seed=seed)
    terminated = bool(reset_info.get("initial_certified", False))
    truncated = False
    rng = np.random.default_rng(seed) if scheduler == "random" else None
    trace: list[dict[str, Any]] = []

    while not (terminated or truncated):
        nodes = environment.current_nodes()
        if not nodes:
            break
        selected = _choose_node(scheduler, nodes, policy=policy, rng=rng)
        # SearchNode equality intentionally does not distinguish all records;
        # preserve the concrete object selected by this scheduler.
        identity_index = next(index for index, candidate in enumerate(nodes) if candidate is selected)
        selected_q_value = (
            float(policy.node_value(selected, nodes))
            if policy is not None and scheduler in {"zero_policy", "learned"}
            else None
        )
        selected_progress = _progress_fields(problem, selected)
        prefix = tuple(_as_gate(action) for action in selected.reconstruct_actions())
        _, reward, terminated, truncated, info = environment.step(identity_index)
        selected_id = info.get("selected_record_id")
        actual = next(
            (node for node in nodes if getattr(node, "record_id", None) == selected_id),
            selected,
        )
        if actual is not selected and not bool(info.get("selected_by_fairness", False)):
            raise RuntimeError("environment expanded a different selected frontier record")
        trace.append(
            {
                "expansion": int(environment.steps),
                "selected_record_id": selected_id,
                "selected_by_fairness": bool(info.get("selected_by_fairness", False)),
                "selected_q_value": selected_q_value,
                "selected_prefix": [_operation(gate) for gate in prefix],
                "frontier_size": int(info.get("frontier_size", 0)),
                "num_children": int(info.get("num_children", 0)),
                "num_accepted": int(info.get("num_accepted", 0)),
                "num_pruned": int(info.get("num_pruned", 0)),
                "reward": float(reward),
                **selected_progress,
            }
        )

    solution = environment.solution_node
    gates = (
        tuple(_as_gate(action) for action in solution.reconstruct_actions())
        if solution is not None
        else ()
    )
    cert_engine = getattr(environment, "cert_engine")
    cert_status = (
        cert_engine.certify(solution.state).status.name if solution is not None else "NONE"
    )
    search_metrics = dict(getattr(environment, "search_metrics", {}))
    search_metrics.update(_archive_metrics(environment))
    return (
        {
            "scheduler": scheduler,
            "seed": int(seed),
            "step_cap": int(getattr(getattr(environment, "config"), "max_steps")),
            "certified": bool(solution is not None and cert_status == "SUCCESS"),
            "search_solution_present": solution is not None,
            "terminal_candidate": bool(solution is not None and problem.is_terminal_candidate(solution)),
            "terminal_stage": _stage_name(problem.analyze(solution.state)) if solution is not None else None,
            "environment_certification_status": cert_status,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "expansions": int(environment.steps),
            "frontier_size": len(environment.current_nodes()),
            "fairness_override_observed": any(row["selected_by_fairness"] for row in trace),
            "trace": trace,
            "witness_operations": [_operation(gate) for gate in gates],
            "search_metrics": search_metrics,
            "problem": dict(problem.metadata()),
        },
        gates,
    )


def _resource_summary(state: object | None) -> dict[str, object]:
    if state is None:
        return {"available": False, "reason": "no learned solution node"}
    dag = getattr(state, "dag")
    dag.validate()
    gates = tuple(dag.gates)
    counts = {name: 0 for name in ("H", "S", "SDG", "T", "TDG", "X", "CNOT")}
    for gate in gates:
        counts[gate.gate_type.name] = counts.get(gate.gate_type.name, 0) + 1
    t_like = counts.get("T", 0) + counts.get("TDG", 0)
    cnot_count = counts.get("CNOT", 0)
    accounting_correct = bool(
        int(getattr(state, "t_count")) == t_like
        and int(getattr(state, "two_qubit_count")) == cnot_count
        and int(getattr(state, "num_gates")) == len(gates)
        and int(getattr(state, "depth")) == max(tuple(getattr(state, "wire_depths")), default=0)
    )
    actual = {
        "t_count": int(getattr(state, "t_count")),
        "two_qubit_count": int(getattr(state, "two_qubit_count")),
        "num_gates": int(getattr(state, "num_gates")),
        "h_count": counts.get("H", 0),
    }
    return {
        "available": True,
        "resources": {
            **actual,
            "depth": int(getattr(state, "depth")),
            "wire_depths": [int(value) for value in getattr(state, "wire_depths")],
        },
        "gate_counts": counts,
        "expected_gate_profile": dict(EXPECTED_GATE_PROFILE),
        "matches_exact_gate_profile": actual == EXPECTED_GATE_PROFILE,
        "resource_budget": dict(TOFFOLI_BUDGET),
        "resource_accounting_correct": accounting_correct,
    }


def _truth_table_rows(diagnostics: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    phase_identity = bool(getattr(diagnostics, "column_phase_consistent", False))
    for row in getattr(diagnostics, "truth_table", ()):
        input_index = int(getattr(row, "input_index"))
        expected_index = int(getattr(row, "expected_output_index"))
        observed_index = int(getattr(row, "observed_output_index"))
        amplitude = complex(getattr(row, "amplitude"))
        rows.append(
            {
                "input_index": input_index,
                "input_bits": format(input_index, "03b"),
                "expected_output_index": expected_index,
                "expected_output_bits": format(expected_index, "03b"),
                "candidate_output_index": observed_index,
                "candidate_output_bits": format(observed_index, "03b"),
                "expected_output_probability": float(getattr(row, "expected_output_probability")),
                "maximum_off_target_probability": float(getattr(row, "maximum_off_target_probability")),
                "expected_output_amplitude_real": float(amplitude.real),
                "expected_output_amplitude_imag": float(amplitude.imag),
                "mapping_correct": bool(getattr(row, "correct")),
                "phase_identity": phase_identity,
            }
        )
    return rows


def _independent_validation(
    deps: SimpleNamespace,
    state: object | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate only the returned candidate against the analytical CCX oracle."""

    if state is None:
        return (
            {
                "available": False,
                "passed": False,
                "reason": "no learned solution node",
            },
            [],
        )
    candidate = deps.unitary_from_gates(TOFFOLI_NUM_QUBITS, state.dag.gates)
    diagnostics = validate_toffoli_unitary(candidate)
    return (
        {
            "available": True,
            "passed": bool(
                diagnostics.global_phase_equivalent
                and diagnostics.truth_table_correct
                and diagnostics.column_phase_consistent
            ),
            "global_phase_equivalent": bool(diagnostics.global_phase_equivalent),
            "truth_table_correct": bool(diagnostics.truth_table_correct),
            "column_phase_consistent": bool(diagnostics.column_phase_consistent),
            "max_phase_aligned_matrix_error": float(diagnostics.max_phase_aligned_matrix_error),
            "process_fidelity": float(diagnostics.process_fidelity),
            "candidate_matrix_digest": stable_matrix_digest(candidate),
        },
        _truth_table_rows(diagnostics),
    )


def _trace_signature(run: Mapping[str, object]) -> str:
    """Stable comparison payload for the duplicated seeded-random rollout."""

    return json.dumps(run.get("trace", ()), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _run_negative_control(
    deps: SimpleNamespace,
    *,
    problem: object,
    feature_provider: object | None,
    reward_model: object | None,
    certification_engine: object,
    seed: int,
    max_steps: int,
    name: str,
    budget: Mapping[str, int],
) -> dict[str, object]:
    environment = _new_environment(
        deps,
        problem=problem,
        feature_provider=feature_provider,
        reward_model=reward_model,
        certification_engine=certification_engine,
        seed=seed,
        max_steps=max_steps,
        budget=budget,
    )
    run, _ = _rollout(environment, problem=problem, scheduler="fifo", seed=seed)
    return {
        "name": name,
        "passed": not bool(run["certified"]),
        "budget": dict(budget),
        "run": run,
    }


def _validate_options(
    *,
    episodes: int,
    seed: int,
    training_max_steps: int,
    evaluation_max_steps: int,
    random_reproducibility_max_steps: int,
    learning_rate: float,
    epsilon_start: float,
    epsilon_minimum: float,
    epsilon_decay: float,
) -> None:
    for name, value in (
        ("episodes", episodes),
        ("seed", seed),
        ("training_max_steps", training_max_steps),
        ("evaluation_max_steps", evaluation_max_steps),
        ("random_reproducibility_max_steps", random_reproducibility_max_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        training_max_steps <= 0
        or evaluation_max_steps <= 0
        or random_reproducibility_max_steps <= 0
    ):
        raise ValueError("search horizons must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= epsilon_minimum <= epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= minimum <= start <= 1")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0, 1]")


def run_toffoli_search(
    output_dir: str | Path,
    *,
    train: bool = False,
    episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    training_max_steps: int = DEFAULT_TRAINING_MAX_STEPS,
    evaluation_max_steps: int = DEFAULT_EVALUATION_MAX_STEPS,
    random_reproducibility_max_steps: int = DEFAULT_RANDOM_REPRODUCIBILITY_MAX_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon_start: float = DEFAULT_EPSILON_START,
    epsilon_minimum: float = DEFAULT_EPSILON_MINIMUM,
    epsilon_decay: float = DEFAULT_EPSILON_DECAY,
) -> dict[str, Any]:
    """Run the reproducible Stage 3 benchmark and save its full artifact set.

    A non-training invocation still writes all artifacts, but it intentionally
    cannot claim a learned witness.  The CLI returns zero only when every
    declared Stage 3 acceptance gate has passed.
    """

    _validate_options(
        episodes=episodes,
        seed=seed,
        training_max_steps=training_max_steps,
        evaluation_max_steps=evaluation_max_steps,
        random_reproducibility_max_steps=random_reproducibility_max_steps,
        learning_rate=learning_rate,
        epsilon_start=epsilon_start,
        epsilon_minimum=epsilon_minimum,
        epsilon_decay=epsilon_decay,
    )
    deps = _stage3_dependencies()
    target_unitary = toffoli_reference_unitary()
    target_fingerprint = stable_matrix_digest(target_unitary)
    target = deps.SynthesisTarget(target_unitary)
    certification_engine = deps.SimulatorCertificationEngine(target)
    problem = deps.ToffoliParityNetworkProblem()
    provider = deps.ToffoliParityFeatureProvider(
        problem,
        target_fingerprint=target_fingerprint,
        problem_schema_version=problem.schema_version,
    )
    reward_model = deps.ToffoliParityRewardModel(provider)

    def environment(
        *,
        run_seed: int,
        max_steps: int,
        budget: Mapping[str, int] = TOFFOLI_BUDGET,
        with_provider: bool = False,
        with_reward: bool = False,
    ):
        if with_reward and not with_provider:
            raise ValueError("a Toffoli reward model requires its parity feature provider")
        return _new_environment(
            deps,
            problem=problem,
            feature_provider=provider if with_provider else None,
            reward_model=reward_model if with_reward else None,
            certification_engine=certification_engine,
            seed=run_seed,
            max_steps=max_steps,
            budget=budget,
        )

    fifo, _ = _rollout(
        environment(run_seed=seed + 1, max_steps=evaluation_max_steps),
        problem=problem,
        scheduler="fifo",
        seed=seed + 1,
    )
    uniform, _ = _rollout(
        environment(run_seed=seed + 2, max_steps=evaluation_max_steps),
        problem=problem,
        scheduler="uniform",
        seed=seed + 2,
    )
    random_horizon = min(evaluation_max_steps, random_reproducibility_max_steps)
    random_seed_checks: dict[str, dict[str, object]] = {}
    random: dict[str, Any] | None = None
    for offset in DEFAULT_RANDOM_REPRODUCIBILITY_SEED_OFFSETS:
        random_seed = seed + offset
        random_run, _ = _rollout(
            environment(run_seed=random_seed, max_steps=random_horizon),
            problem=problem,
            scheduler="random",
            seed=random_seed,
        )
        random_repeat, _ = _rollout(
            environment(run_seed=random_seed, max_steps=random_horizon),
            problem=problem,
            scheduler="random",
            seed=random_seed,
        )
        reproducible = bool(
            _trace_signature(random_run) == _trace_signature(random_repeat)
            and random_run["certified"] == random_repeat["certified"]
        )
        random_run["reproducible"] = reproducible
        random_run["reproducibility_horizon"] = random_horizon
        random_seed_checks[str(random_seed)] = {
            "seed": random_seed,
            "reproducible": reproducible,
            "certified": bool(random_run["certified"]),
            "expansions": int(random_run["expansions"]),
            "truncated": bool(random_run["truncated"]),
            "step_cap": random_horizon,
        }
        if offset == 0:
            random = random_run
    if random is None:  # pragma: no cover - fixed nonempty offset contract
        raise AssertionError("at least one random reproducibility seed is required")
    zero_policy = _new_policy(deps, provider, seed=seed + 3, learning_rate=learning_rate)
    zero, _ = _rollout(
        environment(
            run_seed=seed + 3,
            max_steps=evaluation_max_steps,
            with_provider=True,
        ),
        problem=problem,
        scheduler="zero_policy",
        seed=seed + 3,
        policy=zero_policy,
    )

    training_history: list[dict[str, Any]] = []
    learned: dict[str, Any] = {
        "scheduler": "learned",
        "trained": False,
        "certified": False,
        "search_solution_present": False,
        "trace": [],
        "expansions": 0,
        "fairness_override_observed": False,
        "witness_operations": [],
    }
    learned_gates: tuple[Gate, ...] = ()
    learned_state: object | None = None
    learned_policy = _new_policy(deps, provider, seed=seed, learning_rate=learning_rate)
    # ``--train --episodes 0`` is the public, bounded failure-path probe used
    # by the CLI regression test.  It must still write the complete negative
    # artifact bundle instead of asking ``Trainer`` to execute an invalid
    # zero-episode training loop.
    if train and episodes > 0:
        training_environment = environment(
            run_seed=seed,
            max_steps=training_max_steps,
            with_provider=True,
            with_reward=True,
        )
        trainer = deps.Trainer(training_environment, policy=learned_policy)
        trainer.epsilon = float(epsilon_start)
        trainer.min_epsilon = float(epsilon_minimum)
        trainer.epsilon_decay = float(epsilon_decay)
        training_history = trainer.train(episodes)
        learned_environment = environment(
            run_seed=seed + 4,
            max_steps=evaluation_max_steps,
            with_provider=True,
        )
        learned, learned_gates = _rollout(
            learned_environment,
            problem=problem,
            scheduler="learned",
            seed=seed + 4,
            policy=learned_policy,
        )
        learned["trained"] = True
        learned_state = (
            learned_environment.solution_node.state
            if learned_environment.solution_node is not None
            else None
        )

    oracle_diagnostics = validate_toffoli_unitary(target_unitary)
    parity_phase_rows = [dict(row) for row in deps.phase_identity_rows()]
    parity_phase_identity_valid = bool(deps.phase_identity_holds())
    oracle_valid = bool(
        oracle_diagnostics.global_phase_equivalent
        and oracle_diagnostics.truth_table_correct
        and oracle_diagnostics.column_phase_consistent
    )
    learned_phase, truth_table = _independent_validation(deps, learned_state)
    resources = _resource_summary(learned_state)
    learned["independent_validation"] = learned_phase
    learned["resource_summary"] = resources
    learned["certified"] = bool(learned.get("certified", False) and learned_phase["passed"])

    negative_budgets = {
        "max_t_count_6": {**TOFFOLI_BUDGET, "max_t_count": 6},
        "max_two_qubit_count_5": {**TOFFOLI_BUDGET, "max_two_qubit_count": 5},
        "max_gates_14": {**TOFFOLI_BUDGET, "max_gates": 14},
    }
    negative_controls = {
        name: _run_negative_control(
            deps,
            problem=problem,
            feature_provider=None,
            reward_model=None,
            certification_engine=certification_engine,
            seed=seed + 10 + index,
            max_steps=evaluation_max_steps,
            name=name,
            budget=budget,
        )
        for index, (name, budget) in enumerate(negative_budgets.items())
    }

    policy_metadata = dict(learned_policy.metadata())
    policy_metadata["feature_names"] = list(getattr(provider, "names", ()))
    policy_metadata["weights"] = [float(value) for value in learned_policy.theta]
    policy_metadata["provider"] = dict(provider.metadata())
    policy_metadata["reward_model"] = dict(reward_model.metadata())
    trained_nonzero = bool(np.any(np.abs(learned_policy.theta) > 0.0)) if train else False
    stage_gates = {
        "seven_term_parity_phase_identity_valid": parity_phase_identity_valid,
        "analytical_oracle_and_phase_identity_valid": oracle_valid,
        "fifo_certified": bool(fifo["certified"]),
        "uniform_certified": bool(uniform["certified"]),
        "random_seeded_reproducible": all(
            bool(check["reproducible"]) for check in random_seed_checks.values()
        ),
        "zero_policy_certified": bool(zero["certified"]),
        "negative_t_budget_rejected": bool(negative_controls["max_t_count_6"]["passed"]),
        "negative_cnot_budget_rejected": bool(
            negative_controls["max_two_qubit_count_5"]["passed"]
        ),
        "negative_gate_budget_rejected": bool(
            negative_controls["max_gates_14"]["passed"]
        ),
        "learned_run_requested": bool(train),
        "learned_policy_nonzero": trained_nonzero,
        "learned_fresh_certified": bool(learned["certified"]),
        "learned_no_fairness_override": not bool(learned.get("fairness_override_observed", False)),
        "learned_terminal_structure": bool(
            learned.get("terminal_candidate", False) and learned.get("terminal_stage") == "DONE"
        ),
        "learned_independent_phase_identity": bool(learned_phase.get("passed", False)),
        "learned_exact_resource_profile": bool(resources.get("matches_exact_gate_profile", False)),
        "learned_resource_accounting": bool(resources.get("resource_accounting_correct", False)),
        "learned_improves_zero_policy": bool(
            learned["certified"]
            and zero["certified"]
            and int(learned["expansions"]) < int(zero["expansions"])
        ),
    }
    phase_identity = {
        # This file's top-level result is the seven-term CCZ identity plus
        # the analytical target self-check.  Candidate validation is reported
        # separately so an unsuccessful learned run does not make a proved
        # phase identity appear false.
        "passed": bool(parity_phase_identity_valid and oracle_valid),
        "seven_term_parity_identity": {
            "passed": parity_phase_identity_valid,
            "rows": parity_phase_rows,
        },
        "oracle": {
            "global_phase_equivalent": bool(oracle_diagnostics.global_phase_equivalent),
            "truth_table_correct": bool(oracle_diagnostics.truth_table_correct),
            "column_phase_consistent": bool(oracle_diagnostics.column_phase_consistent),
            "target_matrix_digest": target_fingerprint,
        },
        "learned_candidate": learned_phase,
    }
    report: dict[str, Any] = {
        "correct": False,
        "all_stage3_gates_passed": False,
        "scope": (
            "Exact Toffoli synthesis within the fixed seven-term CCZ parity-network "
            "normal form; not a proof of general unconstrained Clifford+T "
            "synthesis. The policy selects frontier records only, while the problem "
            "enumerates legal continuations."
        ),
        "seed": seed,
        "training": {
            "requested": bool(train),
            "episodes": episodes if train else 0,
            "seed": seed,
            "learning_rate": float(learning_rate),
            "epsilon_start": float(epsilon_start),
            "epsilon_minimum": float(epsilon_minimum),
            "epsilon_decay": float(epsilon_decay),
            "training_max_steps": training_max_steps,
            "evaluation_max_steps": evaluation_max_steps,
            "random_reproducibility_max_steps": random_horizon,
            "random_reproducibility_seeds": [
                seed + offset for offset in DEFAULT_RANDOM_REPRODUCIBILITY_SEED_OFFSETS
            ],
            "fairness_interval": 0,
        },
        "problem": dict(problem.metadata()),
        "target": {
            "name": "analytical CCX(q0,q1 -> q2)",
            "num_qubits": TOFFOLI_NUM_QUBITS,
            "target_matrix_digest": target_fingerprint,
            "quotient_global_phase": True,
        },
        "phase_identity": phase_identity,
        "fifo": fifo,
        "uniform": uniform,
        "random": random,
        "random_seed_checks": random_seed_checks,
        "zero_policy": zero,
        "training_history": training_history,
        "learned": learned,
        "policy": policy_metadata,
        "truth_table": truth_table,
        "resource_summary": resources,
        "negative_controls": negative_controls,
        "search_metrics": {
            "fifo": fifo.get("search_metrics", {}),
            "uniform": uniform.get("search_metrics", {}),
            "random": random.get("search_metrics", {}),
            "zero_policy": zero.get("search_metrics", {}),
            "learned": learned.get("search_metrics", {}),
        },
        "acceptance_gates": stage_gates,
    }
    report["correct"] = bool(all(stage_gates.values()))
    report["all_stage3_gates_passed"] = report["correct"]

    # Save once to establish every required path, then record the successful
    # write as an explicit acceptance gate and refresh summary artifacts.
    artifacts = save_toffoli_search_artifacts(
        output_dir,
        report=report,
        learned_witness_gates=learned_gates,
        seed=seed,
        num_qubits=TOFFOLI_NUM_QUBITS,
    )
    artifacts_written = all(Path(path).is_file() for path in artifacts.values())
    stage_gates["artifact_contract_written"] = artifacts_written
    report["artifacts"] = artifacts
    report["correct"] = bool(all(stage_gates.values()))
    report["all_stage3_gates_passed"] = report["correct"]
    report["artifacts"] = save_toffoli_search_artifacts(
        output_dir,
        report=report,
        learned_witness_gates=learned_gates,
        seed=seed,
        num_qubits=TOFFOLI_NUM_QUBITS,
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs") / "toffoli-search",
        help="directory for deterministic learned-Toffoli JSON, CSV, and SVG artifacts",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="run seeded SARSA training before the fresh frozen learned evaluation",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--training-max-steps", type=int, default=DEFAULT_TRAINING_MAX_STEPS
    )
    parser.add_argument(
        "--evaluation-max-steps", type=int, default=DEFAULT_EVALUATION_MAX_STEPS
    )
    parser.add_argument(
        "--random-reproducibility-max-steps",
        type=int,
        default=DEFAULT_RANDOM_REPRODUCIBILITY_MAX_STEPS,
        help="short same-seed random-trace horizon; random certification is not required",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args(argv)
    try:
        report = run_toffoli_search(
            args.artifacts_dir,
            train=args.train,
            episodes=args.episodes,
            seed=args.seed,
            training_max_steps=args.training_max_steps,
            evaluation_max_steps=args.evaluation_max_steps,
            random_reproducibility_max_steps=args.random_reproducibility_max_steps,
            learning_rate=args.learning_rate,
        )
    except Exception as exc:
        print(f"Toffoli Stage 3 search failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["correct"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["main", "run_toffoli_search"]
