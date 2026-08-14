"""Gymnasium adapter for RL-guided *frontier-record* selection."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from canonical.canonicalizer import Canonicalizer
from certification.base import CertStatus
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from rl.features import extract_features, feature_dimension
from rl.reward import TargetProgressRewardModel
from rl.target_context import (
    TargetProgressWeights,
    target_context_from_certification_engine,
)
from search.action_space import generate_actions
from search.expansion import expand_node
from search.frontier import Frontier
from search.node import SearchNode
from search.problems.native import NativeGateSearchProblem


def _requires_phase_sensitive_archive(certification_engine: Any) -> bool:
    """Return whether any configured verifier needs literal global phase.

    ``CompositeCertificationEngine`` deliberately has no single target: it
    delegates to its child engines.  Treating a composite as quotient-phase
    merely because it lacks ``.target`` would let its archive collapse a
    phase-distinct prefix before the literal verifier can certify it.  The
    conservative ``any`` policy below is safe for mixed composites too:
    retaining extra phase-distinct records can cost search effort, whereas
    quotienting one away can lose a literal solution.
    """

    seen: set[int] = set()

    def visit(engine: Any) -> bool:
        engine_id = id(engine)
        if engine_id in seen:
            return False
        seen.add(engine_id)

        target = getattr(engine, "target", None)
        if target is not None and getattr(target, "quotient_global_phase", True) is False:
            return True

        children = getattr(engine, "engines", ())
        try:
            return any(visit(child) for child in children)
        except TypeError:
            # A non-iterable ``engines`` attribute is not a composite
            # interface.  Its lack of a literal target is safely quotienting.
            return False

    return visit(certification_engine)


class CircuitSynthesisEnv(gym.Env):
    """Search environment whose action selects one persistent open record.

    A selected record is always expanded through *all* legal symbolic gates.
    The learned policy is never consulted for gate legality or child creation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config,
        certification_engine,
        policy=None,
        problem=None,
        feature_provider=None,
        reward_model=None,
        observation_features: bool = True,
    ):
        super().__init__()
        self.config = config
        self.cert_engine = certification_engine
        self.policy = policy
        # ``NativeGateSearchProblem`` is deliberately the default adapter,
        # rather than a new code path.  Its expansion is the historical
        # all-native gate enumeration, so existing GHZ and generic callers
        # retain their exact semantics while constrained problems can supply
        # a continuation-safe normal form.
        self.problem = problem if problem is not None else NativeGateSearchProblem()
        self.feature_provider = feature_provider
        self.reward_model = reward_model
        if not isinstance(observation_features, bool):
            raise TypeError("observation_features must be a bool")
        # Record schedulers call ``current_nodes``/the policy directly; the
        # Gym observation is not their decision input.  Dedicated benchmark
        # runners can therefore suppress repeated frontier-wide observation
        # materialisation while preserving the default public Gym behavior.
        self.observation_features = observation_features
        self.actions = generate_actions(config.num_qubits)
        # A literal-phase verifier cannot safely share a quotient-phase
        # archive: a branch differing only by global phase may be the only
        # exact literal witness.  Composite verifiers inherit the strictest
        # child policy so their archive cannot discard that witness.
        phase_sensitive = _requires_phase_sensitive_archive(certification_engine)
        self.canonicalizer = self.problem.canonicalizer(
            phase_sensitive=phase_sensitive
        )

        prototype = self.problem.initial_state(config)
        # Target-aware metrics are opt-in.  Existing FIFO/random/resource-only
        # searches consequently retain their historical feature and reward
        # behavior.  Target-progress reward shaping also needs the same dense
        # context even when a caller elects not to append target features.
        self.target_context = None
        self._feature_target_context = None
        self.target_progress_reward = None
        reward_mode = getattr(config, "reward_mode", "legacy")
        needs_target_context = bool(
            getattr(config, "target_aware_features", False)
            or reward_mode == "target_progress"
        )
        if needs_target_context:
            reward_config = config.target_progress_reward
            weights = TargetProgressWeights(
                process_fidelity=reward_config.process_fidelity_weight,
                support_match=reward_config.support_match_weight,
                entanglement_match=reward_config.entanglement_match_weight,
            )
            self.target_context = target_context_from_certification_engine(
                certification_engine,
                weights=weights,
            )
            if bool(getattr(config, "target_aware_features", False)):
                self._feature_target_context = self.target_context
        if reward_mode == "target_progress":
            self.target_progress_reward = TargetProgressRewardModel(
                config.target_progress_reward
            )

        if self.feature_provider is None:
            self.feature_dim = feature_dimension(
                prototype,
                target_context=self._feature_target_context,
            )
        else:
            self.feature_dim = int(getattr(self.feature_provider, "dimension"))
            if self.feature_dim <= 0:
                raise ValueError("feature_provider.dimension must be positive")
        self._bind_policy_target_context(policy)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_dim,),
            dtype=np.float32,
        )
        # Gymnasium's fixed Discrete space is retained strictly as a bounded
        # compatibility adapter.  The core Frontier is unbounded; callers
        # needing a record outside this mask use ``select_record(record_id)``.
        self.action_space = spaces.Discrete(max(1, int(config.max_frontier)))

        self.frontier: Optional[Frontier] = None
        self.current_node: Optional[SearchNode] = None
        self.solution_node: Optional[SearchNode] = None
        self.steps = 0
        self.search_metrics: dict[str, int] = {}

        # Reward values affect learning only, not symbolic correctness.
        self.alpha = 0.1
        self.beta = 0.5
        self.gamma_r = 10.0
        self.delta = 1.0

    def _bind_policy_target_context(self, policy: Any) -> None:
        """Ensure an optional supplied policy scores with this environment's context.

        The policy is a ranker over records, so a target-aware environment and
        a target-agnostic scorer would otherwise silently disagree about the
        feature schema.  Policies constructed after the environment can take
        ``env.target_context`` directly; this binding supports the existing
        convenience constructor path as well.
        """

        if policy is None:
            return
        if self.feature_provider is not None:
            bind_provider = getattr(policy, "bind_feature_provider", None)
            if callable(bind_provider):
                bind_provider(self.feature_provider)
            elif getattr(policy, "feature_provider", None) is not self.feature_provider:
                raise ValueError(
                    "policy does not support the environment feature provider"
                )
            if getattr(policy, "feature_dim", self.feature_dim) != self.feature_dim:
                raise ValueError(
                    "policy feature dimension does not match the environment feature schema"
                )
            return

        if self._feature_target_context is None:
            return
        policy_context = getattr(policy, "target_context", None)
        if policy_context is not None:
            if (
                getattr(policy_context, "fingerprint", None)
                != self._feature_target_context.fingerprint
            ):
                raise ValueError("policy target context does not match environment target")
        else:
            bind_target_context = getattr(policy, "bind_target_context", None)
            if not callable(bind_target_context):
                raise ValueError("policy does not support target-context binding")
            bind_target_context(self._feature_target_context)
        if getattr(policy, "feature_dim", self.feature_dim) != self.feature_dim:
            raise ValueError(
                "policy feature dimension does not match the environment feature schema"
            )

    def _features(
        self,
        state: CircuitState,
        frontier: Optional[Sequence[SearchNode]] = None,
    ) -> np.ndarray:
        """Build an observation with this environment's immutable context."""

        if not self.observation_features:
            return np.zeros(self.feature_dim, dtype=np.float32)
        if self.feature_provider is not None:
            values = np.asarray(
                self.feature_provider.extract(state, frontier), dtype=np.float32
            )
            if values.shape != (self.feature_dim,):
                raise ValueError(
                    "feature provider returned an incompatible feature dimension"
                )
            return values
        return extract_features(
            state,
            frontier,
            target_context=self._feature_target_context,
        )

    def _node_potential(self, node: SearchNode) -> float:
        if self.target_context is None:
            return 0.0
        return float(self.target_context.potential(node.state))

    def _frontier_potential(self, nodes: Sequence[SearchNode]) -> float:
        """Return max node potential, using zero for an empty search state."""

        return max((self._node_potential(node) for node in nodes), default=0.0)

    def _cost(self, state: CircuitState) -> float:
        return float(
            state.t_count + state.two_qubit_count + state.depth + state.num_gates
        )

    def current_nodes(self) -> list[SearchNode]:
        return [] if self.frontier is None else self.frontier.nodes()

    def action_info(self, nodes: Optional[Sequence[SearchNode]] = None) -> dict[str, Any]:
        nodes = list(self.current_nodes() if nodes is None else nodes)
        action_mask = np.zeros(self.action_space.n, dtype=np.int8)
        action_mask[: min(len(nodes), self.action_space.n)] = 1
        return {
            "action_mask": action_mask,
            "record_ids": [node.record_id for node in nodes],
            "visible_record_ids": [
                node.record_id for node in nodes[: self.action_space.n]
            ],
            "has_action_overflow": len(nodes) > self.action_space.n,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.solution_node = None
        state = self.problem.initial_state(self.config)
        root = SearchNode(priority=0.0, state=state)
        self.frontier = Frontier(canonicalizer=self.canonicalizer)
        self.frontier.push(root)
        self.current_node = root
        self.search_metrics = {
            "expanded": 0,
            "generated": 0,
            "accepted": 1,
            "canonical_pruned": 0,
            "dominated": 0,
            "reopened": 0,
            "peak_frontier": 1,
            "terminal_candidates": 0,
            "terminal_certification_failures": 0,
        }
        # A constrained problem knows whether its root is structurally
        # eligible for target certification.  The native adapter returns true
        # here, preserving identity-target behavior exactly.
        if self.problem.is_terminal_candidate(root):
            root_result = self.cert_engine.certify(root.state)
            if root_result.status == CertStatus.SUCCESS:
                self.solution_node = root
        info = self.action_info()
        info["initial_certified"] = self.solution_node is not None
        return self._features(state, self.current_nodes()), info

    def _select_node(
        self, action: Any, nodes: Sequence[SearchNode]
    ) -> tuple[Optional[SearchNode], bool]:
        fairness_interval = max(0, int(getattr(self.config, "fairness_interval", 0)))
        if fairness_interval and self.steps % fairness_interval == 0:
            return min(nodes, key=lambda node: int(node.record_id or 0)), True

        # An explicit mapping disambiguates stable record IDs from positional
        # Gym compatibility indices such as 0, 1, ... .
        if isinstance(action, dict):
            record_id = action.get("record_id")
            if not isinstance(record_id, (int, np.integer)):
                return None, False
            return next(
                (node for node in nodes if node.record_id == int(record_id)),
                None,
            ), False

        if not isinstance(action, (int, np.integer)):
            return None, False
        index = int(action)
        if 0 <= index < len(nodes):
            return nodes[index], False
        return None, False

    def select_record(self, record_id: int):
        """Expand a stable frontier record without relying on a fixed mask."""
        return self.step({"record_id": int(record_id)})

    def step(self, action):
        if self.frontier is None or self.current_node is None:
            raise RuntimeError("call reset() before step()")

        if self.solution_node is not None:
            info = self.action_info()
            info.update(
                {
                    "cert_status": "SUCCESS",
                    "initial_certified": True,
                    "selected_record_id": None,
                }
            )
            return (
                self._features(self.current_node.state, self.current_nodes()),
                0.0,
                True,
                False,
                info,
            )

        self.steps += 1
        nodes_before = self.current_nodes()
        if not nodes_before:
            observation = self._features(self.current_node.state)
            return observation, 0.0, True, False, self.action_info(nodes_before)

        node, selected_by_fairness = self._select_node(action, nodes_before)
        if node is None:
            terminated = self.frontier.is_empty()
            truncated = not terminated and self.steps >= self.config.max_steps
            info = self.action_info(nodes_before)
            info.update({"invalid_action": True, "selected_record_id": None})
            return (
                self._features(self.current_node.state, nodes_before),
                -1.0,
                terminated,
                truncated,
                info,
            )

        self.frontier.remove(node)
        self.current_node = node
        parent_cost = self._cost(node.state)
        # The policy selected ``node`` only.  The problem expander performs a
        # deterministic exhaustive enumeration of every legal one-gate child.
        # The native adapter delegates to the historical ``expand_node``.
        children = self.problem.expand(node)

        accepted = 0
        pruned = 0
        certified = 0
        cert_failures = 0
        cost_delta = 0.0
        terminal_children: list[SearchNode] = []
        generated_child_potentials: list[float] = []

        self.search_metrics["expanded"] += 1
        self.search_metrics["generated"] += len(children)

        for child in children:
            if self.target_progress_reward is not None:
                generated_child_potentials.append(self._node_potential(child))
            terminal_candidate = self.problem.is_terminal_candidate(child)
            if terminal_candidate:
                self.search_metrics["terminal_candidates"] += 1
                result = self.cert_engine.certify(child.state)
                if result.status == CertStatus.SUCCESS:
                    certified += 1
                    terminal_children.append(child)
                    if self.solution_node is None:
                        self.solution_node = child
                    # Local expansion remains exhaustive even when a terminal
                    # witness is discovered.
                    continue

                # A normal-form terminal that fails its independent dense
                # certification is a critical structural defect.  Reject it
                # rather than silently treating it as an ordinary prefix.
                if result.status == CertStatus.FAILURE:
                    cert_failures += 1
                # Native search intentionally certifies every child, but a
                # dense non-match is only a prefix mismatch and must remain
                # searchable.  Constrained normal-form problems opt into
                # critical rejection for structurally complete terminals.
                reject_terminal_failure = bool(
                    getattr(
                        self.problem,
                        "reject_failed_terminal",
                        not isinstance(self.problem, NativeGateSearchProblem),
                    )
                )
                if reject_terminal_failure:
                    self.search_metrics["terminal_certification_failures"] += 1
                    continue

            insertion = self.frontier.insert(child)
            if insertion.accepted:
                accepted += 1
                self.search_metrics["accepted"] += 1
                self.search_metrics["dominated"] += len(insertion.dominated)
                if insertion.dominated:
                    self.search_metrics["reopened"] += 1
                cost_delta += parent_cost - self._cost(child.state)
            else:
                pruned += 1
                self.search_metrics["canonical_pruned"] += 1

        dead_end = int(not children)

        terminated = self.solution_node is not None or self.frontier.is_empty()
        truncated = not terminated and self.steps >= self.config.max_steps
        nodes_after = self.current_nodes()
        self.search_metrics["peak_frontier"] = max(
            self.search_metrics["peak_frontier"], len(nodes_after)
        )
        reward_diagnostics: dict[str, float] = {}
        if self.reward_model is not None:
            # Constrained problems supply their own potential over the
            # continuation progress state.  It is evaluated only at the
            # frontier transition level: the scheduler receives no credit for
            # choosing individual generated gates or for archive pruning.
            potential_before = float(self.reward_model.frontier_potential(nodes_before))
            selected_node_potential = float(
                self.reward_model.frontier_potential((node,))
            )
            potential_after = float(
                self.reward_model.frontier_potential(
                    tuple(nodes_after) + tuple(terminal_children)
                )
            )
            best_generated_child_potential = float(
                self.reward_model.frontier_potential(children)
            )
            breakdown = self.reward_model.reward(
                potential_before=potential_before,
                potential_after=potential_after,
                selected_node_potential=selected_node_potential,
                best_generated_child_potential=best_generated_child_potential,
                certified=bool(certified),
                dead_end=bool(dead_end),
            )
            reward = breakdown.reward
            reward_diagnostics = breakdown.info()
        elif self.target_progress_reward is None:
            # Deliberately preserve the legacy reward as the default.  In
            # particular, existing resource/archive experiments retain their
            # historical pruning term and trainer-side reward clipping.
            reward = (
                self.alpha * cost_delta
                + self.beta * pruned
                + self.gamma_r * float(certified > 0)
                - self.delta * dead_end
            )
        else:
            potential_before = self._frontier_potential(nodes_before)
            selected_node_potential = self._node_potential(node)
            # Only active post-transition records can influence Omega.  A
            # certified terminal child is deliberately included despite not
            # entering the frontier.  Dominated/pruned children remain a
            # diagnostic at most and cannot earn a shaping reward.
            potential_after = max(
                [self._node_potential(candidate) for candidate in nodes_after]
                + [self._node_potential(child) for child in terminal_children],
                default=0.0,
            )
            best_generated_child_potential = max(
                generated_child_potentials,
                default=0.0,
            )
            breakdown = self.target_progress_reward.reward(
                potential_before=potential_before,
                potential_after=potential_after,
                selected_node_potential=selected_node_potential,
                best_generated_child_potential=best_generated_child_potential,
                certified=bool(certified),
                dead_end=bool(dead_end),
            )
            reward = breakdown.reward
            reward_diagnostics = breakdown.info()
        info = self.action_info(nodes_after)
        info.update(
            {
                "cert_status": "SUCCESS" if certified else "NONE",
                "num_certified": certified,
                "num_pruned": pruned,
                "num_children": len(children),
                "num_accepted": accepted,
                "num_certification_nonmatches": cert_failures,
                "terminal_certification_failures": self.search_metrics[
                    "terminal_certification_failures"
                ],
                "frontier_size": len(nodes_after),
                "selected_record_id": node.record_id,
                "selected_by_fairness": selected_by_fairness,
                "solution_actions": (
                    self.solution_node.reconstruct_actions()
                    if self.solution_node is not None
                    else None
                ),
                "problem": dict(self.problem.metadata()),
                "search_metrics": dict(self.search_metrics),
            }
        )
        info.update(reward_diagnostics)
        return (
            self._features(node.state, nodes_after),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.current_node is not None:
            print(self.current_node.state)
