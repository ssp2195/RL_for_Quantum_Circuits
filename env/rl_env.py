"""Gymnasium adapter for RL-guided *frontier-record* selection."""

from __future__ import annotations

from dataclasses import asdict
from collections import defaultdict
import time
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from canonical.canonicalizer import Canonicalizer
from certification.base import CertStatus
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from config import normalize_reward_mode
from rl.features import extract_features, feature_dimension
from rl.reward import TargetProgressRewardModel
from rl.article_v1_reward import ArticleV1RewardModel
from rl.target_context import (
    TargetProgressWeights,
    target_context_from_certification_engine,
)
from search.action_space import generate_actions
from search.expansion import expand_node
from search.archive import ArchiveRecord
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
        target_metric=None,
        instrumentation_enabled: bool = True,
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
        self.problem = (
            problem
            if problem is not None
            else NativeGateSearchProblem(
                absorb_clifford_angles=bool(
                    getattr(config, "absorb_clifford_angles", True)
                ),
                canonicalization_mode=str(
                    getattr(config, "canonicalization_mode", "enhanced")
                ),
            )
        )
        self.feature_provider = feature_provider
        self.generation_counts: dict[object, int] = defaultdict(int)
        if self.feature_provider is not None:
            bind_search_horizon = getattr(
                self.feature_provider, "bind_search_horizon", None
            )
            if callable(bind_search_horizon):
                bind_search_horizon(int(config.max_steps))
            bind_generation_counts = getattr(
                self.feature_provider, "bind_generation_counts", None
            )
            if callable(bind_generation_counts):
                bind_generation_counts(self.generation_counts)
        self.reward_model = reward_model
        if not isinstance(observation_features, bool):
            raise TypeError("observation_features must be a bool")
        # Record schedulers call ``current_nodes``/the policy directly; the
        # Gym observation is not their decision input.  Dedicated benchmark
        # runners can therefore suppress repeated frontier-wide observation
        # materialisation while preserving the default public Gym behavior.
        self.observation_features = observation_features
        if not isinstance(instrumentation_enabled, bool):
            raise TypeError("instrumentation_enabled must be a bool")
        self.instrumentation_enabled = instrumentation_enabled
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
        self.reward_mode = normalize_reward_mode(
            getattr(config, "reward_mode", "legacy_archive_shaping")
        )
        self.article_target_metric = target_metric
        if self.article_target_metric is None and self.feature_provider is not None:
            self.article_target_metric = getattr(
                self.feature_provider, "target_context", None
            )
        self.article_v1_reward = None
        if self.reward_mode == "article_v1_expansion_potential":
            if self.article_target_metric is None:
                raise ValueError(
                    "article_v1_expansion_potential requires an Article V1 target metric"
                )
            if reward_model is not None and not isinstance(
                reward_model, ArticleV1RewardModel
            ):
                raise TypeError(
                    "article_v1 reward_mode requires ArticleV1RewardModel or no reward_model"
                )
            self.article_v1_reward = (
                reward_model
                if isinstance(reward_model, ArticleV1RewardModel)
                else ArticleV1RewardModel(
                    self.article_target_metric,
                    expansion_budget=int(config.max_steps),
                    beta=float(getattr(config, "article_v1_beta", 1.0)),
                )
            )
        needs_target_context = bool(
            getattr(config, "target_aware_features", False)
            or self.reward_mode == "target_progress_shaping"
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
        if self.reward_mode == "target_progress_shaping":
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
        self.search_metrics: dict[str, int | float] = {}
        self._frontier_size_sum = 0
        self._frontier_sample_count = 0
        self._feature_time_ns = 0
        self._feature_evaluation_count = 0

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

        started = time.perf_counter_ns() if self.instrumentation_enabled else 0
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
            result = values
        else:
            result = extract_features(
                state,
                frontier,
                target_context=self._feature_target_context,
            )
        self._feature_evaluation_count += 1
        if self.instrumentation_enabled:
            self._feature_time_ns += time.perf_counter_ns() - started
        if self.search_metrics:
            self.search_metrics["feature_evaluation_count"] = self._feature_evaluation_count
            self.search_metrics["feature_time_ns"] = self._feature_time_ns
        return result

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

    def current_records(self) -> list[ArchiveRecord]:
        """Return the authoritative selectable archive records.

        Feature evaluators consume records rather than reconstructing
        canonical/resource identity from their nodes at every decision.  The
        frontier remains the sole owner of membership and ordering.
        """

        return [] if self.frontier is None else self.frontier.active_records()

    def _synchronize_feature_frontier(
        self,
        *,
        generation_count_updates: Optional[dict[object, int]] = None,
    ) -> None:
        synchronize = getattr(self.feature_provider, "synchronize_frontier", None)
        if callable(synchronize):
            synchronize(
                self.current_records(),
                generation_count_updates=generation_count_updates,
            )

    def _article_frontier_potential(
        self,
        nodes: Sequence[SearchNode],
        *,
        terminal: bool = False,
    ) -> float:
        """Return the exact Article potential, using the indexed minimum when available."""

        if terminal:
            return 0.0
        minimum_target_distance = getattr(
            self.feature_provider, "minimum_target_distance", None
        )
        if callable(minimum_target_distance):
            self._synchronize_feature_frontier()
            minimum = minimum_target_distance()
            if minimum is None:
                return 0.0
            value = float(minimum)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("feature index returned an invalid target distance")
            return -value
        assert self.article_v1_reward is not None
        return self.article_v1_reward.frontier_potential(nodes)

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
            "reward_mode": self.reward_mode,
            "action_semantics": "persistent_frontier_record",
        }

    def _reset_search_metrics(self) -> None:
        """Initialize mathematically defined search-event instrumentation.

        Frontier statistics sample ``F_0`` after root insertion and ``F_t``
        after every successful selected-record expansion.  Invalid Gym
        actions do not create a search transition and therefore do not add a
        sample.
        """

        assert self.frontier is not None
        initial_frontier = len(self.current_nodes())
        self._frontier_size_sum = initial_frontier
        self._frontier_sample_count = 1
        archive = self.frontier.archive
        self.search_metrics = {
            "generated": 0,
            "certification_nonmatch": 0,
            "duplicate_rejected": 0,
            "dominated_retired": 0,
            "pareto_incomparable_accepted": 0,
            "reopened": 0,
            "expanded": 0,
            "frontier_peak": initial_frontier,
            "frontier_mean": float(initial_frontier),
            "archive_size": archive.archive_size,
            "pareto_width_peak": archive.pareto_width_peak,
            # Compatibility diagnostics retained for existing reports.  Each
            # is updated from, and therefore exactly mirrors, the named event
            # above rather than carrying an independent definition.
            "accepted": 1,
            "canonical_pruned": 0,
            "dominated": 0,
            "peak_frontier": initial_frontier,
            "terminal_candidates": 0,
            "terminal_certification_failures": 0,
            # Article V1 publication taxonomy.  Legacy names above remain
            # aliases rather than being repurposed.
            "num_expanded": 0,
            "num_gate_attempts": 0,
            "num_generated": 0,
            "num_exact_duplicate_rejections": 0,
            "num_dominance_rejections": 0,
            "num_dominance_replacements": 0,
            "num_pareto_incomparable_acceptances": 0,
            "num_reopenings": 0,
            "frontier_sum": initial_frontier,
            "frontier_observation_count": 1,
            "archive_record_count": archive.archive_record_count,
            "active_archive_peak": archive.active_record_peak,
            "maximum_pareto_antichain_width": archive.pareto_width_peak,
            "peak_frontier_records": initial_frontier,
            "peak_active_archive_records": archive.active_record_peak,
            "wall_time_ns": 0,
            "ranking_time_ns": 0,
            "feature_time_ns": self._feature_time_ns,
            "target_metric_time_ns": 0,
            "symbolic_update_time_ns": 0,
            "canonicalization_time_ns": 0,
            "archive_time_ns": 0,
            "certification_time_ns": 0,
            "reporting_time_ns": 0,
            "feature_evaluation_count": self._feature_evaluation_count,
            "target_metric_evaluation_count": 0,
            "target_metric_cache_hits": 0,
            "target_metric_cache_misses": 0,
            "certification_count": 0,
        }

    def _sync_archive_metrics(self) -> None:
        assert self.frontier is not None
        archive = self.frontier.archive
        self.search_metrics["archive_size"] = archive.archive_size
        self.search_metrics["pareto_width_peak"] = archive.pareto_width_peak
        self.search_metrics["archive_record_count"] = archive.archive_record_count
        self.search_metrics["active_archive_peak"] = max(
            int(self.search_metrics["active_archive_peak"]),
            archive.active_record_peak,
        )
        self.search_metrics["peak_active_archive_records"] = self.search_metrics[
            "active_archive_peak"
        ]
        self.search_metrics["maximum_pareto_antichain_width"] = archive.pareto_width_peak

    def _record_frontier_sample(self, size: int) -> None:
        self._frontier_size_sum += int(size)
        self._frontier_sample_count += 1
        peak = max(int(self.search_metrics["frontier_peak"]), int(size))
        self.search_metrics["frontier_peak"] = peak
        self.search_metrics["peak_frontier"] = peak
        self.search_metrics["frontier_mean"] = (
            self._frontier_size_sum / self._frontier_sample_count
        )
        self.search_metrics["peak_frontier_records"] = peak

    def reward_spec(self) -> dict[str, Any]:
        """Return every coefficient that defines the configured objective."""

        common: dict[str, Any] = {
            "reward_mode": self.reward_mode,
            "discount": float(self.config.discount),
        }
        if self.reward_mode == "article_v1_expansion_potential":
            assert self.article_v1_reward is not None
            return {
                **common,
                **self.article_v1_reward.metadata(),
                "article_equation": "amended-104-plus-105-107",
            }
        if self.reward_mode in {"expansion_cost", "expansion_cost_plus_visit_bonus"}:
            return {
                **common,
                "article_equation": 24,
                "expansion_cost": 1.0,
                "terminal_success_correction": 1.0,
                "terminal_failure_return": -float(self.config.max_steps),
                "visit_bonus_coefficient": (
                    0.1
                    if self.reward_mode == "expansion_cost_plus_visit_bonus"
                    else 0.0
                ),
            }
        if self.reward_mode == "legacy_archive_shaping":
            return {
                **common,
                "cost_delta_coefficient": float(self.alpha),
                "pruned_child_coefficient": float(self.beta),
                "terminal_success_bonus": float(self.gamma_r),
                "dead_end_cost": float(self.delta),
                "trainer_visit_bonus_coefficient": 0.1,
            }
        return {
            **common,
            **asdict(self.config.target_progress_reward),
            "trainer_visit_bonus_coefficient": 0.0,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.generation_counts.clear()
        self._feature_time_ns = 0
        self._feature_evaluation_count = 0
        set_search_step = getattr(self.feature_provider, "set_search_step", None)
        if callable(set_search_step):
            set_search_step(0)
        self.solution_node = None
        state = self.problem.initial_state(self.config)
        root = SearchNode(priority=0.0, state=state)
        self.frontier = Frontier(
            canonicalizer=self.canonicalizer,
            canonicalization_enabled=getattr(
                self.config, "canonicalization_enabled", True
            ),
            pareto_dominance_enabled=getattr(
                self.config, "pareto_dominance_enabled", True
            ),
        )
        self.frontier.push(root)
        root_key = self.canonicalizer.semantic_key(root.state)
        self.generation_counts[root_key] += 1
        reset_feature_index = getattr(self.feature_provider, "reset_index", None)
        synchronize_feature_index = getattr(
            self.feature_provider, "synchronize_frontier", None
        )
        if callable(reset_feature_index):
            reset_feature_index()
        if callable(synchronize_feature_index):
            self._synchronize_feature_frontier(
                generation_count_updates={root_key: 1}
            )
        else:
            bind_generation_counts = getattr(
                self.feature_provider, "bind_generation_counts", None
            )
            if callable(bind_generation_counts):
                bind_generation_counts(self.generation_counts)
        self.current_node = root
        self._reset_search_metrics()
        # A constrained problem knows whether its root is structurally
        # eligible for target certification.  The native adapter returns true
        # here, preserving identity-target behavior exactly.
        if self.problem.is_terminal_candidate(root):
            certification_started = (
                time.perf_counter_ns() if self.instrumentation_enabled else 0
            )
            root_result = self.cert_engine.certify(root.state)
            self.search_metrics["certification_count"] += 1
            if self.instrumentation_enabled:
                self.search_metrics["certification_time_ns"] += (
                    time.perf_counter_ns() - certification_started
                )
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

        step_started = time.perf_counter_ns() if self.instrumentation_enabled else 0
        self.steps += 1
        set_search_step = getattr(self.feature_provider, "set_search_step", None)
        if callable(set_search_step):
            set_search_step(self.steps)
        nodes_before = self.current_nodes()
        article_potential_before = (
            self._article_frontier_potential(nodes_before)
            if self.article_v1_reward is not None
            else 0.0
        )
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

        if not self.frontier.remove(node):  # pragma: no cover - selection invariant
            raise AssertionError("selected frontier record was not open")
        self.current_node = node
        parent_cost = self._cost(node.state)
        # The policy selected ``node`` only.  The problem expander performs a
        # deterministic exhaustive enumeration of every legal one-gate child.
        # The native adapter delegates to the historical ``expand_node``.
        symbolic_started = time.perf_counter_ns() if self.instrumentation_enabled else 0
        children = self.problem.expand(node)
        if self.instrumentation_enabled:
            self.search_metrics["symbolic_update_time_ns"] += (
                time.perf_counter_ns() - symbolic_started
            )

        accepted = 0
        pruned = 0
        certified = 0
        certification_nonmatches = 0
        cost_delta = 0.0
        terminal_children: list[SearchNode] = []
        generated_child_potentials: list[float] = []
        generation_count_updates: dict[object, int] = {}

        self.search_metrics["expanded"] += 1
        self.search_metrics["generated"] += len(children)
        self.search_metrics["num_expanded"] += 1
        gate_attempts = (
            len(self.actions)
            if isinstance(self.problem, NativeGateSearchProblem)
            else int(getattr(self.problem, "last_gate_attempts", len(children)))
        )
        self.search_metrics["num_gate_attempts"] += gate_attempts
        self.search_metrics["num_generated"] += len(children)

        for child in children:
            generated_key = self.canonicalizer.semantic_key(child.state)
            self.generation_counts[generated_key] += 1
            generation_count_updates[generated_key] = int(
                self.generation_counts[generated_key]
            )
            if self.target_progress_reward is not None:
                generated_child_potentials.append(self._node_potential(child))
            terminal_candidate = self.problem.is_terminal_candidate(child)
            if terminal_candidate:
                self.search_metrics["terminal_candidates"] += 1
                certification_started = (
                    time.perf_counter_ns() if self.instrumentation_enabled else 0
                )
                result = self.cert_engine.certify(child.state)
                self.search_metrics["certification_count"] += 1
                if self.instrumentation_enabled:
                    self.search_metrics["certification_time_ns"] += (
                        time.perf_counter_ns() - certification_started
                    )
                if result.status == CertStatus.SUCCESS:
                    certified += 1
                    terminal_children.append(child)
                    if self.solution_node is None:
                        self.solution_node = child
                    # Local expansion remains exhaustive even when a terminal
                    # witness is discovered.
                    continue

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
                    certification_nonmatches += 1
                    self.search_metrics["certification_nonmatch"] += 1
                    continue

            # Every generated child not independently certified as the target
            # is one certification non-match.  For constrained problems a
            # structurally incomplete prefix is known nonterminal without
            # invoking the expensive final certifier.
            certification_nonmatches += 1
            self.search_metrics["certification_nonmatch"] += 1

            insertion = self.frontier.insert(child)
            if self.instrumentation_enabled:
                self.search_metrics["canonicalization_time_ns"] += int(
                    self.frontier.archive.last_canonicalization_time_ns
                )
                self.search_metrics["archive_time_ns"] += int(
                    self.frontier.archive.last_archive_time_ns
                )
            if insertion.accepted:
                accepted += 1
                self.search_metrics["accepted"] += 1
                retired = insertion.dominated_retired
                self.search_metrics["dominated_retired"] += retired
                self.search_metrics["dominated"] += retired
                self.search_metrics["num_dominance_replacements"] += retired
                if insertion.pareto_incomparable_accepted:
                    self.search_metrics["pareto_incomparable_accepted"] += 1
                    self.search_metrics["num_pareto_incomparable_acceptances"] += 1
                if insertion.reopened:
                    self.search_metrics["reopened"] += 1
                    self.search_metrics["num_reopenings"] += 1
                cost_delta += parent_cost - self._cost(child.state)
            else:
                pruned += 1
                if not insertion.duplicate_rejected:  # pragma: no cover
                    raise AssertionError("archive rejected a non-duplicate record")
                self.search_metrics["duplicate_rejected"] += 1
                self.search_metrics["canonical_pruned"] += 1
                if insertion.exact_duplicate_rejected:
                    self.search_metrics["num_exact_duplicate_rejections"] += 1
                elif insertion.dominance_rejected:
                    self.search_metrics["num_dominance_rejections"] += 1

        dead_end = int(not children)

        terminated = self.solution_node is not None or self.frontier.is_empty()
        truncated = not terminated and self.steps >= self.config.max_steps
        nodes_after = self.current_nodes()
        synchronize_feature_index = getattr(
            self.feature_provider, "synchronize_frontier", None
        )
        if callable(synchronize_feature_index):
            self._synchronize_feature_frontier(
                generation_count_updates=generation_count_updates
            )
        else:
            bind_generation_counts = getattr(
                self.feature_provider, "bind_generation_counts", None
            )
            if callable(bind_generation_counts):
                bind_generation_counts(self.generation_counts)
        self._sync_archive_metrics()
        self._record_frontier_sample(len(nodes_after))
        if not (terminated or truncated):
            self.search_metrics["frontier_sum"] += len(nodes_after)
            self.search_metrics["frontier_observation_count"] += 1
        reward_diagnostics: dict[str, float] = {}
        if self.reward_mode == "article_v1_expansion_potential":
            assert self.article_v1_reward is not None
            article_terminal = bool(terminated or truncated)
            article_potential_after = self._article_frontier_potential(
                nodes_after,
                terminal=article_terminal,
            )
            breakdown = self.article_v1_reward.transition(
                expansion_index=self.steps,
                potential_before=article_potential_before,
                potential_after=article_potential_after,
                certified_success=self.solution_node is not None,
                terminal_failure=article_terminal and self.solution_node is None,
            )
            reward = breakdown.reward
            reward_diagnostics = breakdown.info()
        elif self.reward_mode in {"expansion_cost", "expansion_cost_plus_visit_bonus"}:
            # Article baseline: one unit cost per selected-record expansion
            # and a +1 terminal correction on a successful expansion.  The
            # trainer may layer the explicitly named visit-bonus ablation on
            # top of this same environment reward.
            terminal_success_correction = float(certified > 0)
            # Equation (24) assigns every failed finite-budget episode -B,
            # even when the frontier becomes empty after k < B expansions.
            # Charge the unused expansion horizon on that terminal failure so
            # the accumulated return is exactly -B rather than -k.
            terminal_failure_correction = (
                -float(max(0, int(self.config.max_steps) - self.steps))
                if terminated and self.solution_node is None
                else 0.0
            )
            reward = (
                -1.0
                + terminal_success_correction
                + terminal_failure_correction
            )
            reward_diagnostics = {
                "reward_mode": self.reward_mode,
                "expansion_cost": 1.0,
                "terminal_success_correction": terminal_success_correction,
                "terminal_failure_correction": terminal_failure_correction,
            }
        elif self.reward_model is not None:
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
        elif self.reward_mode == "legacy_archive_shaping":
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
        reward_diagnostics.setdefault("reward_mode", self.reward_mode)
        metric_stats = getattr(self.article_target_metric, "cache_metrics", None)
        if callable(metric_stats):
            stats = dict(metric_stats())
            self.search_metrics["target_metric_evaluation_count"] = int(
                stats.get(
                    "target_metric_evaluation_count",
                    stats.get("evaluations", stats.get("evaluation_count", 0)),
                )
            )
            self.search_metrics["target_metric_cache_hits"] = int(
                stats.get(
                    "target_metric_cache_hits",
                    stats.get("hits", stats.get("cache_hits", 0)),
                )
            )
            self.search_metrics["target_metric_cache_misses"] = int(
                stats.get(
                    "target_metric_cache_misses",
                    stats.get("misses", stats.get("cache_misses", 0)),
                )
            )
            self.search_metrics["target_metric_time_ns"] = int(
                stats.get(
                    "target_metric_time_ns",
                    stats.get("time_ns", stats.get("evaluation_time_ns", 0)),
                )
            )
        if self.instrumentation_enabled:
            self.search_metrics["wall_time_ns"] += time.perf_counter_ns() - step_started
        info = self.action_info(nodes_after)
        info.update(
            {
                "cert_status": "SUCCESS" if certified else "NONE",
                "num_certified": certified,
                "num_pruned": pruned,
                "num_children": len(children),
                "num_accepted": accepted,
                "num_certification_nonmatches": certification_nonmatches,
                "terminal_certification_failures": self.search_metrics[
                    "terminal_certification_failures"
                ],
                "frontier_size": len(nodes_after),
                "selected_record_id": node.record_id,
                "action_semantics": "persistent_frontier_record",
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
