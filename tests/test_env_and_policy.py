import numpy as np

from certification.composite import CompositeCertificationEngine
from certification.simulator import SimulatorCertificationEngine, SynthesisTarget, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import Config
from enums import GateType
from env.rl_env import CircuitSynthesisEnv
from rl.features import extract_features, feature_dimension
from rl.policy import LinearQPolicy
from search.node import SearchNode
from train import Trainer


def _config(*, max_steps=4, max_gates=3):
    return Config(
        num_qubits=1,
        budget=ResourceBudget(max_t_count=3, max_depth=max_gates, max_gates=max_gates),
        max_steps=max_steps,
        max_frontier=2,
        seed=9,
    )


def _target_for(gates):
    return SynthesisTarget(unitary_from_gates(1, gates))


def _state_with(gates=()):
    state = CircuitState(CircuitDAG(1), ResourceBudget(5, 5, 5))
    for gate_type in gates:
        assert state.apply_gate(Gate(gate_type, (0,)))
    return state


def test_env_uses_gymnasium_termination_for_a_certified_witness():
    target = _target_for([Gate(GateType.H, (0,))])
    env = CircuitSynthesisEnv(_config(), SimulatorCertificationEngine(target))

    observation, reset_info = env.reset(seed=4)
    assert observation.shape == (env.feature_dim,)
    assert reset_info["action_mask"][0] == 1

    _, _, terminated, truncated, info = env.step(0)
    assert terminated
    assert not truncated
    assert info["num_children"] == 5
    assert info["num_certified"] == 1
    assert info["num_certification_nonmatches"] == 4
    assert info["search_metrics"]["generated"] == 5
    assert info["search_metrics"]["certification_nonmatch"] == 4
    assert info["search_metrics"]["expanded"] == 1
    assert info["search_metrics"]["frontier_peak"] == 4
    assert info["search_metrics"]["frontier_mean"] == 2.5
    assert info["search_metrics"]["archive_size"] == 5
    assert info["search_metrics"]["pareto_width_peak"] == 1
    assert env.solution_node is not None
    assert [action.gate_type for action in env.solution_node.reconstruct_actions()] == [GateType.H]


def test_identity_target_is_certified_during_reset():
    target = _target_for([])
    env = CircuitSynthesisEnv(_config(), SimulatorCertificationEngine(target))
    _, info = env.reset(seed=4)

    assert info["initial_certified"]
    _, _, terminated, truncated, step_info = env.step(0)
    assert terminated
    assert not truncated
    assert step_info["initial_certified"]


def test_trainer_records_zero_expansions_for_an_initially_certified_target():
    config = _config(max_steps=2, max_gates=0)
    config.budget = ResourceBudget(0, 0, 0)
    env = CircuitSynthesisEnv(config, SimulatorCertificationEngine(_target_for([])))
    result = Trainer(env).train(1)[0]
    assert result["certified"]
    assert result["steps"] == 0


def test_env_returns_truncation_not_termination_at_the_external_step_limit():
    # X is intentionally outside the default library, so no one-gate child
    # reaches the target and the frontier remains nonempty at the time limit.
    target = _target_for([Gate(GateType.X, (0,))])
    env = CircuitSynthesisEnv(_config(max_steps=1), SimulatorCertificationEngine(target))
    env.reset(seed=4)

    _, _, terminated, truncated, info = env.step(0)
    assert not terminated
    assert truncated
    assert info["frontier_size"] > 0


def test_env_expansion_keeps_a_repeated_t_child_legal():
    target = _target_for([Gate(GateType.X, (0,))])
    env = CircuitSynthesisEnv(_config(max_steps=3, max_gates=2), SimulatorCertificationEngine(target))
    env.reset(seed=4)
    env.step(0)

    nodes = env.current_nodes()
    t_index = next(
        index for index, node in enumerate(nodes) if node.action.gate_type is GateType.T
    )
    _, _, _, _, info = env.step(t_index)
    assert info["num_children"] == 5


def test_unbounded_frontier_uses_explicit_stable_record_ids_beyond_gym_mask():
    target = _target_for([Gate(GateType.X, (0,))])
    config = _config(max_steps=3, max_gates=2)
    config.max_frontier = 1
    env = CircuitSynthesisEnv(config, SimulatorCertificationEngine(target))
    env.reset(seed=4)
    _, _, _, _, info = env.step(0)

    assert info["has_action_overflow"]
    assert info["action_mask"].shape == (1,)
    chosen = env.current_nodes()[-1]
    _, _, _, _, selected_info = env.select_record(chosen.record_id)
    assert selected_info["selected_record_id"] == chosen.record_id


def test_literal_phase_target_enables_a_phase_sensitive_archive():
    gates = [Gate(gate_type, (0,)) for gate_type in (GateType.H, GateType.S) * 3]
    target = SynthesisTarget(
        unitary_from_gates(1, gates),
        quotient_global_phase=False,
    )
    config = Config(
        num_qubits=1,
        budget=ResourceBudget(0, 6, 6),
        max_steps=2_000,
        max_frontier=8,
    )
    # The strict simulator is intentionally wrapped: the environment must
    # inherit literal-phase semantics through a composite verifier too.
    env = CircuitSynthesisEnv(
        config,
        CompositeCertificationEngine([SimulatorCertificationEngine(target)]),
    )
    env.reset(seed=2)
    assert env.canonicalizer.phase_sensitive

    terminated = truncated = False
    while not (terminated or truncated):
        nodes = env.current_nodes()
        node = min(nodes, key=lambda candidate: int(candidate.record_id or 0))
        index = next(i for i, candidate in enumerate(nodes) if candidate is node)
        _, _, terminated, truncated, _ = env.step(index)

    assert terminated
    assert not truncated
    assert env.solution_node is not None


def test_feature_context_is_frontier_permutation_invariant():
    root = SearchNode(priority=0.0, state=_state_with())
    t_node = SearchNode(priority=0.0, state=_state_with([GateType.T]))
    policy = LinearQPolicy(feature_dimension(root.state), seed=3)
    policy.theta = np.linspace(-0.2, 0.2, policy.feature_dim)

    assert np.isclose(
        policy.node_value(root, [root, t_node]),
        policy.node_value(root, [t_node, root]),
    )


def test_sarsa_update_uses_the_actual_next_node_not_a_max_bootstrap():
    current = SearchNode(priority=0.0, state=_state_with())
    next_node = SearchNode(priority=0.0, state=_state_with([GateType.T]))
    policy = LinearQPolicy(feature_dimension(current.state), lr=0.25, gamma=0.5, seed=1)
    before = np.linspace(-0.1, 0.1, policy.feature_dim)
    policy.theta = before.copy()

    frontier = [current]
    next_frontier = [next_node]
    phi = extract_features(current.state, frontier).astype(float)
    expected_error = 1.5 + 0.5 * policy.node_value(next_node, next_frontier) - np.dot(before, phi)

    returned_error = policy.update(
        current.state,
        1.5,
        next_frontier=next_frontier,
        done=False,
        next_node=next_node,
        frontier=frontier,
    )

    assert np.isclose(returned_error, expected_error)
    np.testing.assert_allclose(policy.theta, before + 0.25 * expected_error * phi)
