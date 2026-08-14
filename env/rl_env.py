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
from search.action_space import generate_actions
from search.expansion import expand_node
from search.frontier import Frontier
from search.node import SearchNode


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

    def __init__(self, config, certification_engine, policy=None):
        super().__init__()
        self.config = config
        self.cert_engine = certification_engine
        self.policy = policy
        self.actions = generate_actions(config.num_qubits)
        # A literal-phase verifier cannot safely share a quotient-phase
        # archive: a branch differing only by global phase may be the only
        # exact literal witness.  Composite verifiers inherit the strictest
        # child policy so their archive cannot discard that witness.
        self.canonicalizer = Canonicalizer(
            phase_sensitive=_requires_phase_sensitive_archive(certification_engine)
        )

        prototype = CircuitState(CircuitDAG(config.num_qubits), config.budget)
        self.feature_dim = feature_dimension(prototype)
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

        # Reward values affect learning only, not symbolic correctness.
        self.alpha = 0.1
        self.beta = 0.5
        self.gamma_r = 10.0
        self.delta = 1.0

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
        state = CircuitState(CircuitDAG(self.config.num_qubits), self.config.budget)
        root = SearchNode(priority=0.0, state=state)
        self.frontier = Frontier(canonicalizer=self.canonicalizer)
        self.frontier.push(root)
        self.current_node = root
        root_result = self.cert_engine.certify(root.state)
        if root_result.status == CertStatus.SUCCESS:
            self.solution_node = root
        info = self.action_info()
        info["initial_certified"] = self.solution_node is not None
        return extract_features(state, self.current_nodes()), info

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
                extract_features(self.current_node.state, self.current_nodes()),
                0.0,
                True,
                False,
                info,
            )

        self.steps += 1
        nodes_before = self.current_nodes()
        if not nodes_before:
            observation = extract_features(self.current_node.state)
            return observation, 0.0, True, False, self.action_info(nodes_before)

        node, selected_by_fairness = self._select_node(action, nodes_before)
        if node is None:
            terminated = self.frontier.is_empty()
            truncated = not terminated and self.steps >= self.config.max_steps
            info = self.action_info(nodes_before)
            info.update({"invalid_action": True, "selected_record_id": None})
            return (
                extract_features(self.current_node.state, nodes_before),
                -1.0,
                terminated,
                truncated,
                info,
            )

        self.frontier.remove(node)
        self.current_node = node
        parent_cost = self._cost(node.state)
        children = expand_node(node, self.actions, policy=None)

        accepted = 0
        pruned = 0
        certified = 0
        cert_failures = 0
        cost_delta = 0.0

        for child in children:
            result = self.cert_engine.certify(child.state)
            if result.status == CertStatus.SUCCESS:
                certified += 1
                if self.solution_node is None:
                    self.solution_node = child
                # We intentionally continue the loop: local expansion remains
                # exhaustive even when a terminal witness is discovered.
                continue

            # A dense non-match proves only that this *prefix* is not already
            # the target, not that it has no legal solution suffix.  It must
            # therefore never prune a general Clifford+T prefix.
            if result.status == CertStatus.FAILURE:
                cert_failures += 1

            insertion = self.frontier.insert(child)
            if insertion.accepted:
                accepted += 1
                cost_delta += parent_cost - self._cost(child.state)
            else:
                pruned += 1

        dead_end = int(not children)
        reward = (
            self.alpha * cost_delta
            + self.beta * pruned
            + self.gamma_r * float(certified > 0)
            - self.delta * dead_end
        )

        terminated = self.solution_node is not None or self.frontier.is_empty()
        truncated = not terminated and self.steps >= self.config.max_steps
        nodes_after = self.current_nodes()
        info = self.action_info(nodes_after)
        info.update(
            {
                "cert_status": "SUCCESS" if certified else "NONE",
                "num_certified": certified,
                "num_pruned": pruned,
                "num_children": len(children),
                "num_accepted": accepted,
                "num_certification_nonmatches": cert_failures,
                "frontier_size": len(nodes_after),
                "selected_record_id": node.record_id,
                "selected_by_fairness": selected_by_fairness,
                "solution_actions": (
                    self.solution_node.reconstruct_actions()
                    if self.solution_node is not None
                    else None
                ),
            }
        )
        return (
            extract_features(node.state, nodes_after),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.current_node is not None:
            print(self.current_node.state)
