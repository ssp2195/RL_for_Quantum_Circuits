"""Versioned Article V1 ten-minute protocol primitives."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, random, subprocess, time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import numpy as np

TEN_MINUTE_CONFIG_SCHEMA = "article-v1-corpus-config-v3"
RUNTIME_PROTOCOL_SCHEMA = "article-v1-runtime-protocol-v1"
BUDGET_PROTOCOL_SCHEMA = "article-v1-budget-protocol-v1"
TRAINING_PROTOCOL_SCHEMA = "article-v1-training-protocol-v1"
SECONDARY_CPU_SCHEMA = "article-v1-secondary-cpu-v1"
TEN_MINUTE_RAW_RUN_SCHEMA = "article-v1-10min-raw-run-v1"
TEN_MINUTE_AUDIT_SCHEMA = "article-v1-10min-audit-v1"
TEN_MINUTE_CHECKPOINT_SCHEMA = "article-v1-transferable-linear-checkpoint-v5"
TEN_MINUTE_FRONTIER_ENUMERATION_SCHEMA = "article-v1-priority-then-record-id-order-v1"
PRIMARY_SCHEDULERS = ("fifo", "lifo", "uniform_cost", "seeded_random", "zero_weight_linear", "article_target_distance", "article_sarsa")
DIFFICULTIES = ("easy", "medium", "hard")

def _positive(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(float(v)) or float(v) <= 0: raise ValueError(f"{name} must be positive")
    return float(v)
def _int(v: Any, name: str, minimum: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum: raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(v)
def _bool(v: Any, name: str) -> bool:
    if type(v) is not bool: raise ValueError(f"{name} must be a boolean")
    return v

@dataclass(frozen=True)
class RuntimeProtocol:
    target_episode_cpu_seconds: float; hard_episode_cpu_limit_seconds: float; selection_quantile: float; candidate_hard_expansion_caps: tuple[int, ...]; maximum_feature_index_memory_mb: float; timeout_status: str; reference_environment_required: bool; selected_hard_expansion_cap: int | None = None
    @classmethod
    def from_mapping(cls, r: Mapping[str, Any]):
        if r.get("schema_version") != RUNTIME_PROTOCOL_SCHEMA: raise ValueError("invalid runtime protocol schema")
        if r.get("cpu_metric") != "process_time_ns": raise ValueError("runtime cpu_metric must be process_time_ns")
        if r.get("timeout_status") != "OPERABILITY_TIMEOUT": raise ValueError("runtime timeout status must be OPERABILITY_TIMEOUT")
        target, limit = _positive(r.get("target_episode_cpu_seconds"), "target CPU"), _positive(r.get("hard_episode_cpu_limit_seconds"), "hard CPU")
        if target >= limit: raise ValueError("target CPU time must be below hard CPU limit")
        q = _positive(r.get("selection_quantile"), "selection quantile")
        caps = tuple(_int(x, "candidate cap", 1) for x in r.get("candidate_hard_expansion_caps", []))
        selected_raw = r.get("selected_hard_expansion_cap")
        selected = None if selected_raw is None else _int(selected_raw, "selected hard cap", 1)
        if caps != tuple(sorted(set(caps))) or (not caps and selected is None) or q > 1: raise ValueError("invalid runtime candidate grid")
        return cls(target, limit, q, caps, _positive(r.get("maximum_feature_index_memory_mb"), "feature memory"), str(r.get("timeout_status")), _bool(r.get("reference_environment_required"), "reference environment"), selected)

@dataclass(frozen=True)
class BudgetProtocol:
    mode: str; thresholds_by_difficulty: Mapping[str, tuple[int, ...]]; require_threshold_not_above_horizon: bool; maximum_horizon_by_difficulty: Mapping[str, int]
    @classmethod
    def from_mapping(cls, r: Mapping[str, Any]):
        if r.get("schema_version") != BUDGET_PROTOCOL_SCHEMA or r.get("mode") != "fixed-max-horizon-anytime-v1": raise ValueError("invalid anytime budget protocol")
        raw_t, raw_h = r.get("thresholds_by_difficulty"), r.get("maximum_horizon_by_difficulty")
        if not isinstance(raw_t, Mapping) or not isinstance(raw_h, Mapping) or set(raw_t) != set(DIFFICULTIES) or set(raw_h) != set(DIFFICULTIES): raise ValueError("thresholds and horizons must cover all difficulties")
        ts, hs = {}, {}
        flag = _bool(r.get("require_threshold_not_above_horizon"), "threshold bound")
        for d in DIFFICULTIES:
            h = _int(raw_h[d], f"{d} horizon", 1); values = tuple(_int(x, f"{d} threshold", 1) for x in raw_t[d])
            if not values or values != tuple(sorted(set(values))) or (flag and max(values) > h): raise ValueError(f"invalid {d} thresholds")
            ts[d], hs[d] = values, h
        return cls(str(r["mode"]), ts, flag, hs)

@dataclass(frozen=True)
class TrainingProtocol:
    mode: str; eligible_splits: tuple[str, ...]; eligible_difficulties: tuple[str, ...]; target_schedule: str; episode_caps_by_difficulty: Mapping[str, int]; total_expansions_per_seed: int | None; candidate_total_expansions_per_seed: tuple[int, ...]; allow_partial_final_episode: bool; hard_targets_used_for_training: bool; convergence_stopping: bool
    @classmethod
    def from_mapping(cls, r: Mapping[str, Any]):
        if r.get("schema_version") != TRAINING_PROTOCOL_SCHEMA or r.get("mode") != "fixed-total-expansions-curriculum-v1": raise ValueError("invalid curriculum protocol")
        splits, ds = tuple(r.get("eligible_splits", [])), tuple(r.get("eligible_difficulties", []))
        if splits != ("train",) or ds != ("easy", "medium"): raise ValueError("primary training must use train easy/medium")
        if r.get("target_schedule") != "seeded-round-robin": raise ValueError("training target schedule must be seeded-round-robin")
        caps = r.get("episode_caps_by_difficulty", {})
        if set(caps) != set(ds): raise ValueError("episode caps do not cover eligible difficulties")
        parsed_caps = {d: _int(caps[d], f"{d} episode cap", 1) for d in ds}
        total = None if r.get("total_expansions_per_seed") is None else _int(r["total_expansions_per_seed"], "total expansions", 1)
        candidates = tuple(_int(x, "candidate total", 1) for x in r.get("candidate_total_expansions_per_seed", []))
        if candidates != tuple(sorted(set(candidates))) or (total is None and not candidates): raise ValueError("training total or candidate grid required")
        hard_training = _bool(r.get("hard_targets_used_for_training"), "hard training")
        convergence = _bool(r.get("convergence_stopping"), "convergence stopping")
        if hard_training: raise ValueError("hard targets must be excluded from primary training")
        if convergence: raise ValueError("ten-minute curriculum forbids convergence stopping")
        return cls(str(r["mode"]), splits, ds, str(r.get("target_schedule")), parsed_caps, total, candidates, _bool(r.get("allow_partial_final_episode"), "partial episode"), hard_training, convergence)

@dataclass(frozen=True)
class SecondaryCPUProtocol:
    enabled: bool; cpu_budget_seconds: float; schedulers: tuple[str, ...]; report_separately: bool
    @classmethod
    def from_mapping(cls, r: Mapping[str, Any]):
        if r.get("schema_version") != SECONDARY_CPU_SCHEMA: raise ValueError("invalid secondary CPU schema")
        schedulers = tuple(r.get("schedulers", []))
        if not schedulers or any(s not in PRIMARY_SCHEDULERS for s in schedulers): raise ValueError("invalid secondary schedulers")
        return cls(_bool(r.get("enabled"), "secondary enabled"), _positive(r.get("cpu_budget_seconds"), "secondary CPU"), schedulers, _bool(r.get("report_separately"), "separate report"))

@dataclass(frozen=True)
class TenMinuteConfig:
    payload: Mapping[str, Any]; runtime: RuntimeProtocol; budget: BudgetProtocol; training: TrainingProtocol; secondary_cpu: SecondaryCPUProtocol; digest: str; frozen: bool
    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, require_frozen=False):
        if raw.get("schema_version") != TEN_MINUTE_CONFIG_SCHEMA: raise ValueError(f"config schema must be {TEN_MINUTE_CONFIG_SCHEMA!r}")
        if tuple(raw.get("qubits", ())) != (2, 3): raise ValueError("V3 primary qubits must be exactly [2, 3]")
        strata = raw.get("difficulty_strata", {})
        if not isinstance(strata, Mapping) or int(strata.get("hard", {}).get("min_generator_length", 0)) < 6: raise ValueError("hard generator length must remain at least six")
        experiment = raw.get("experiment", {})
        if tuple(experiment.get("schedulers", ())) != PRIMARY_SCHEDULERS: raise ValueError("all seven primary schedulers are required")
        profile = str(raw.get("profile", ""))
        if profile == "publication":
            if len(experiment.get("training_seeds", ())) < 5: raise ValueError("publication requires at least five learner seeds")
            if len(experiment.get("random_scheduler_seeds", ())) < 10: raise ValueError("publication requires at least ten random scheduler seeds")
        runtime, budget = RuntimeProtocol.from_mapping(raw.get("runtime_protocol", {})), BudgetProtocol.from_mapping(raw.get("budget_protocol", {}))
        training, secondary = TrainingProtocol.from_mapping(raw.get("training_protocol", {})), SecondaryCPUProtocol.from_mapping(raw.get("secondary_cpu_experiment", {}))
        freeze = raw.get("publication_freeze", {}); frozen = bool(freeze.get("frozen", False)) if isinstance(freeze, Mapping) else False
        if require_frozen and not frozen: raise ValueError("publication config is not frozen")
        if frozen and (training.total_expansions_per_seed is None or training.candidate_total_expansions_per_seed or runtime.candidate_hard_expansion_caps or runtime.selected_hard_expansion_cap is None): raise ValueError("frozen config has unresolved candidate values")
        if frozen:
            if freeze.get("no_test_access") is not True or not freeze.get("source_commit"): raise ValueError("frozen config lacks provenance/no-test-access assertion")
            if not freeze.get("target_ids") or not freeze.get("seeds"): raise ValueError("frozen config must bind target IDs and seeds")
        payload = json.loads(json.dumps(raw, sort_keys=True, separators=(",", ":"))); digest = "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(payload, runtime, budget, training, secondary, digest, frozen)


@dataclass(frozen=True)
class TenMinuteCheckpoint:
    """Transferable V5 linear checkpoint bound to the curriculum protocol."""

    training_seed: int
    weights: tuple[float, ...]
    feature_schema_version: str
    feature_evaluator_schema_version: str
    frontier_enumeration_schema_version: str
    ordered_feature_names: tuple[str, ...]
    learning_rate: float
    corpus_config_digest: str
    runtime_protocol_schema_version: str
    training_protocol_schema_version: str
    total_expansion_budget: int
    total_completed_expansions: int
    eligible_splits: tuple[str, ...]
    eligible_difficulties: tuple[str, ...]
    ordered_target_ids: tuple[str, ...]
    executed_target_schedule: tuple[str, ...]
    target_schedule_seed: int
    episode_caps_by_difficulty: tuple[tuple[str, int], ...]
    effective_episode_caps: tuple[int, ...]
    episode_expansions: tuple[int, ...] = ()
    current_epsilon: float = 0.0
    policy_rng_state: Mapping[str, Any] = field(default_factory=dict)

    @property
    def weight_digest(self) -> str:
        return self.digest

    @property
    def checkpoint_family(self) -> str:
        return "standard"

    def require_evaluation_eligible(self) -> None:
        self.validate()
        if self.total_completed_expansions != self.total_expansion_budget:
            raise ValueError(
                "incomplete V5 training checkpoint is not transferable for evaluation"
            )

    @property
    def digest(self) -> str:
        payload = self.to_payload(include_weights=False)
        digest = hashlib.sha256()
        digest.update(b"article-v1-transferable-linear-checkpoint-v5\0")
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(np.asarray(self.weights, dtype="<f8").tobytes())
        return "sha256:" + digest.hexdigest()

    def validate(self) -> None:
        if self.training_protocol_schema_version != TRAINING_PROTOCOL_SCHEMA: raise ValueError("checkpoint training protocol schema mismatch")
        if self.runtime_protocol_schema_version != RUNTIME_PROTOCOL_SCHEMA: raise ValueError("checkpoint runtime protocol schema mismatch")
        if self.frontier_enumeration_schema_version != TEN_MINUTE_FRONTIER_ENUMERATION_SCHEMA: raise ValueError("checkpoint frontier enumeration schema mismatch")
        if self.eligible_splits != ("train",) or self.eligible_difficulties != ("easy", "medium"): raise ValueError("checkpoint training scope is not primary ten-minute scope")
        if not self.ordered_target_ids or len(set(self.ordered_target_ids)) != len(self.ordered_target_ids): raise ValueError("checkpoint target IDs must be nonempty and unique")
        if any(target not in self.ordered_target_ids for target in self.executed_target_schedule): raise ValueError("checkpoint schedule contains an ineligible target")
        if self.total_expansion_budget < 1 or not 0 <= self.total_completed_expansions <= self.total_expansion_budget: raise ValueError("checkpoint total expansion accounting is invalid")
        if len(self.effective_episode_caps) != len(self.executed_target_schedule) or sum(self.effective_episode_caps) < self.total_completed_expansions: raise ValueError("checkpoint effective episode caps are inconsistent")
        if self.episode_expansions:
            if len(self.episode_expansions) > len(self.executed_target_schedule): raise ValueError("checkpoint episode expansion history is too long")
            if any(value < 1 for value in self.episode_expansions): raise ValueError("checkpoint episode expansions must be positive")
            if sum(self.episode_expansions) > self.total_completed_expansions: raise ValueError("checkpoint episode expansions exceed completed total")
        if not np.isfinite(self.current_epsilon) or not 0.0 <= self.current_epsilon <= 1.0: raise ValueError("checkpoint epsilon is invalid")
        try: json.dumps(self.policy_rng_state, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error: raise ValueError("checkpoint policy RNG state is not portable JSON") from error
        values = np.asarray(self.weights, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all(): raise ValueError("checkpoint weights must be a finite vector")
        if len(self.ordered_feature_names) != len(self.weights): raise ValueError("checkpoint feature names/weights disagree")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0: raise ValueError("checkpoint learning rate must be positive")
        if not self.corpus_config_digest.startswith("sha256:"): raise ValueError("checkpoint corpus config digest is invalid")

    def to_payload(self, *, include_weights: bool = True) -> dict[str, Any]:
        payload = {"checkpoint_schema": TEN_MINUTE_CHECKPOINT_SCHEMA, "training_seed": self.training_seed, "feature_schema_version": self.feature_schema_version, "feature_evaluator_schema_version": self.feature_evaluator_schema_version, "frontier_enumeration_schema_version": self.frontier_enumeration_schema_version, "ordered_feature_names": list(self.ordered_feature_names), "learning_rate": self.learning_rate, "corpus_config_digest": self.corpus_config_digest, "runtime_protocol_schema_version": self.runtime_protocol_schema_version, "training_protocol_schema_version": self.training_protocol_schema_version, "total_expansion_budget": self.total_expansion_budget, "total_completed_expansions": self.total_completed_expansions, "eligible_splits": list(self.eligible_splits), "eligible_difficulties": list(self.eligible_difficulties), "ordered_target_ids": list(self.ordered_target_ids), "executed_target_schedule": list(self.executed_target_schedule), "target_schedule_seed": self.target_schedule_seed, "episode_caps_by_difficulty": dict(self.episode_caps_by_difficulty), "effective_episode_caps": list(self.effective_episode_caps), "episode_expansions": list(self.episode_expansions), "current_epsilon": self.current_epsilon, "policy_rng_state": json.loads(json.dumps(self.policy_rng_state))}
        if include_weights: payload["weights"] = list(self.weights)
        return payload

    def save(self, path: str | Path) -> None:
        self.validate(); payload = {**self.to_payload(), "checkpoint_digest": self.digest}; destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp"); temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"); temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "TenMinuteCheckpoint":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("checkpoint_schema") != TEN_MINUTE_CHECKPOINT_SCHEMA: raise ValueError("unsupported ten-minute checkpoint schema")
        result = cls(training_seed=int(payload["training_seed"]), weights=tuple(float(v) for v in payload["weights"]), feature_schema_version=str(payload["feature_schema_version"]), feature_evaluator_schema_version=str(payload["feature_evaluator_schema_version"]), frontier_enumeration_schema_version=str(payload["frontier_enumeration_schema_version"]), ordered_feature_names=tuple(str(v) for v in payload["ordered_feature_names"]), learning_rate=float(payload["learning_rate"]), corpus_config_digest=str(payload["corpus_config_digest"]), runtime_protocol_schema_version=str(payload["runtime_protocol_schema_version"]), training_protocol_schema_version=str(payload["training_protocol_schema_version"]), total_expansion_budget=int(payload["total_expansion_budget"]), total_completed_expansions=int(payload["total_completed_expansions"]), eligible_splits=tuple(payload["eligible_splits"]), eligible_difficulties=tuple(payload["eligible_difficulties"]), ordered_target_ids=tuple(payload["ordered_target_ids"]), executed_target_schedule=tuple(payload["executed_target_schedule"]), target_schedule_seed=int(payload["target_schedule_seed"]), episode_caps_by_difficulty=tuple(sorted((str(k), int(v)) for k, v in payload["episode_caps_by_difficulty"].items())), effective_episode_caps=tuple(int(v) for v in payload["effective_episode_caps"]), episode_expansions=tuple(int(v) for v in payload.get("episode_expansions", ())), current_epsilon=float(payload.get("current_epsilon", 0.0)), policy_rng_state=dict(payload.get("policy_rng_state", {})))
        result.validate()
        if payload.get("checkpoint_digest") != result.digest: raise ValueError("ten-minute checkpoint digest mismatch")
        return result

def load_ten_minute_config(path: str | Path, *, require_frozen=False) -> TenMinuteConfig:
    source = Path(path)
    def _load(source_path: Path) -> dict[str, Any]:
        current = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(current, dict): raise ValueError("ten-minute config must be an object")
        parent_name = current.get("inherits")
        if not parent_name: return current
        merged = _load((source_path.parent / str(parent_name)).resolve())
        for key, value in current.items():
            if key != "inherits": merged[key] = ({**merged[key], **value} if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping) else value)
        return merged
    raw = _load(source)
    training = dict(raw.get("training_protocol", {})); training["candidate_total_expansions_per_seed"] = raw.get("candidate_total_expansions_per_seed", training.get("candidate_total_expansions_per_seed", [])); raw["training_protocol"] = training
    runtime = dict(raw.get("runtime_protocol", {})); runtime["candidate_hard_expansion_caps"] = raw.get("candidate_hard_expansion_caps", runtime.get("candidate_hard_expansion_caps", [])); raw["runtime_protocol"] = runtime
    return TenMinuteConfig.from_mapping(raw, require_frozen=require_frozen)

def load_ten_minute_corpus_config(path: str | Path):
    """Load the legacy corpus portion of a validated V3 protocol config.

    V3 protocol fields are removed and the corpus schema is explicitly restored
    to V2 before calling the authoritative legacy parser.  This is a bridge,
    not a reinterpretation of V2 files.
    """
    from benchmarks.article_native_corpus import ArticleV1CorpusConfig
    config = load_ten_minute_config(path)
    payload = json.loads(json.dumps(config.payload))
    payload["schema_version"] = "article-v1-corpus-config-v2"
    for key in ("runtime_protocol", "budget_protocol", "training_protocol", "secondary_cpu_experiment", "publication_freeze", "inherits", "candidate_total_expansions_per_seed", "candidate_hard_expansion_caps"):
        payload.pop(key, None)
    return ArticleV1CorpusConfig.from_mapping(payload)

def process_cpu_seconds(start_ns: int, end_ns: int | None = None) -> float:
    end = time.process_time_ns() if end_ns is None else int(end_ns)
    if end < start_ns: raise ValueError("process CPU clock moved backwards")
    return (end - int(start_ns)) / 1e9

@dataclass(frozen=True)
class WatchdogDecision: allowed: bool; status: str | None; cpu_seconds: float
class ProcessCPUWatchdog:
    def __init__(self, hard_limit_seconds: float, *, clock_ns=time.process_time_ns): self.limit, self.clock, self.start_ns = _positive(hard_limit_seconds, "hard limit"), clock_ns, None
    def start(self): self.start_ns = int(self.clock())
    def at_safe_boundary(self):
        if self.start_ns is None: raise RuntimeError("watchdog has not been started")
        elapsed = process_cpu_seconds(self.start_ns, int(self.clock())); return WatchdogDecision(elapsed < self.limit, None if elapsed < self.limit else "OPERABILITY_TIMEOUT", elapsed)

def seeded_round_robin_schedule(target_ids: Sequence[str], *, total_expansions: int, seed: int) -> tuple[str, ...]:
    ids = tuple(sorted(str(x) for x in target_ids))
    if not ids or total_expansions < 1: raise ValueError("nonempty IDs and positive total required")
    random.Random(int(seed)).shuffle(ids := list(ids)); return tuple(ids[i % len(ids)] for i in range(total_expansions))

@dataclass
class CurriculumAccounting:
    """Deterministic fixed-total interaction coordinator.

    The coordinator owns accounting and eligibility only; the existing Trainer
    still performs every ordinary finite-horizon SARSA episode.
    """
    target_ids: tuple[str, ...]
    difficulty_by_target: Mapping[str, str]
    total_budget: int
    episode_caps_by_difficulty: Mapping[str, int]
    schedule_seed: int
    allow_partial_final_episode: bool = True

    def __post_init__(self):
        if not self.target_ids or self.total_budget < 1: raise ValueError("curriculum requires targets and a positive total budget")
        if any(self.difficulty_by_target.get(t) not in ("easy", "medium") for t in self.target_ids): raise ValueError("hard targets are excluded from primary curriculum")
        self.completed = 0
        self.expansions_by_target = {t: 0 for t in self.target_ids}
        self.episodes_by_target = {t: 0 for t in self.target_ids}
        self.expansions_by_difficulty = {"easy": 0, "medium": 0}
        self.executed_target_schedule: list[str] = []
        self._schedule = seeded_round_robin_schedule(self.target_ids, total_expansions=len(self.target_ids) * ((self.total_budget + len(self.target_ids) - 1) // len(self.target_ids)), seed=self.schedule_seed)
        self._cursor = 0

    @property
    def remaining(self) -> int: return self.total_budget - self.completed

    def next_episode(self) -> tuple[str, int] | None:
        if self.remaining <= 0: return None
        target = self._schedule[self._cursor % len(self._schedule)]; self._cursor += 1
        self.executed_target_schedule.append(target)
        cap = min(int(self.episode_caps_by_difficulty[self.difficulty_by_target[target]]), self.remaining)
        if cap < self.episode_caps_by_difficulty[self.difficulty_by_target[target]] and not self.allow_partial_final_episode: raise ValueError("final curriculum episode would be partial")
        self.episodes_by_target[target] += 1
        return target, cap

    def record_expansions(self, target_id: str, expansions: int) -> None:
        n = _int(expansions, "episode expansions")
        if target_id not in self.expansions_by_target or n > self.remaining: raise ValueError("invalid curriculum expansion accounting")
        self.expansions_by_target[target_id] += n; self.completed += n
        self.expansions_by_difficulty[self.difficulty_by_target[target_id]] += n

    def metadata(self) -> dict[str, Any]:
        return {"total_training_expansions_completed": self.completed, "total_training_expansion_budget": self.total_budget, "total_training_expansions_remaining": self.remaining, "expansions_by_target": dict(self.expansions_by_target), "expansions_by_difficulty": dict(self.expansions_by_difficulty), "episodes_by_target": dict(self.episodes_by_target), "target_schedule": list(self.executed_target_schedule), "target_schedule_seed": self.schedule_seed, "curriculum_cycle": self._cursor // len(self.target_ids), "hard_targets_used_for_training": False}

    def restore_completed_episodes(
        self,
        *,
        target_schedule: Sequence[str],
        effective_caps: Sequence[int],
        episode_expansions: Sequence[int],
    ) -> None:
        """Replay deterministic curriculum accounting without running search."""

        if not (
            len(target_schedule) == len(effective_caps) == len(episode_expansions)
        ):
            raise ValueError("curriculum recovery vectors must have equal lengths")
        for expected_target, expected_cap, expansions in zip(
            target_schedule, effective_caps, episode_expansions
        ):
            episode = self.next_episode()
            if episode != (str(expected_target), int(expected_cap)):
                raise ValueError("curriculum recovery schedule/cap mismatch")
            self.record_expansions(str(expected_target), int(expansions))

def train_fixed_interaction_curriculum(targets: Sequence[Any], *, total_expansions: int, episode_caps_by_difficulty: Mapping[str, int], target_schedule: str = "seeded-round-robin", seed: int, train_episode: Callable[[Any, int, int], Mapping[str, Any]]) -> dict[str, Any]:
    """Run a fixed-total curriculum through an existing finite-episode Trainer.

    ``train_episode(target, effective_cap, episode_index)`` is deliberately an
    adapter: the caller constructs the ordinary Article V1 environment and
    Trainer, while this coordinator owns target eligibility, cap calculation,
    total accounting, and the no-convergence rule.
    """
    if target_schedule != "seeded-round-robin": raise ValueError("only seeded-round-robin is defined for the ten-minute protocol")
    target_rows = tuple(targets)
    ids = tuple(str(getattr(t, "target_id", t["target_id"] if isinstance(t, Mapping) else t)) for t in target_rows)
    difficulty = {str(getattr(t, "target_id", t["target_id"] if isinstance(t, Mapping) else t)): str(getattr(t, "difficulty", t["difficulty"] if isinstance(t, Mapping) else "")) for t in target_rows}
    coordinator = CurriculumAccounting(ids, difficulty, total_expansions, episode_caps_by_difficulty, seed)
    by_id = {str(getattr(t, "target_id", t["target_id"] if isinstance(t, Mapping) else t)): t for t in target_rows}
    history = []
    while coordinator.remaining:
        target_id, cap = coordinator.next_episode(); result = dict(train_episode(by_id[target_id], cap, len(history)))
        expansions = _int(result.get("expansions", 0), "episode expansions")
        if expansions < 1 or expansions > cap: raise ValueError("episode returned an invalid expansion count")
        coordinator.record_expansions(target_id, expansions); result.update({"target_id": target_id, "effective_episode_cap": cap, "episode_index": len(history), "convergence_stopping": False}); history.append(result)
    return {"history": history, **coordinator.metadata()}

def anytime_success_rows(*, first_certified_hit_expansion: int | None, executed_max_horizon: int, thresholds: Iterable[int]) -> tuple[dict[str, Any], ...]:
    horizon = _int(executed_max_horizon, "horizon", 1); hit = None if first_certified_hit_expansion is None else _int(first_certified_hit_expansion, "first hit", 1); values = tuple(_int(x, "threshold", 1) for x in thresholds)
    if any(x > horizon for x in values): raise ValueError("anytime threshold exceeds executed maximum horizon")
    return tuple({"threshold": x, "success_by_threshold": bool(hit is not None and hit <= x), "first_hit_expansion": hit, "executed_max_horizon": horizon} for x in values)

def audit_ten_minute_runs(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed on incomplete/timeout evidence before aggregation."""

    rows = tuple(records)
    timeouts = []
    incomplete = []
    for index, row in enumerate(rows):
        if row.get("schema_version") != TEN_MINUTE_RAW_RUN_SCHEMA:
            raise ValueError("ten-minute audit received a non-V3 raw record")
        reason = row.get("terminal_reason")
        complete = row.get("complete")
        if type(complete) is not bool:
            raise ValueError("ten-minute raw record complete flag must be boolean")
        if reason == "OPERABILITY_TIMEOUT": timeouts.append(index)
        if complete is not True: incomplete.append(index)
        if reason == "OPERABILITY_TIMEOUT" and row.get("certified") is True:
            raise ValueError("operability timeout cannot be certified")
    passed = not timeouts and not incomplete
    return {"schema_version": TEN_MINUTE_AUDIT_SCHEMA, "passed": passed, "raw_run_count": len(rows), "operability_timeout_indices": timeouts, "incomplete_indices": incomplete, "timeouts_are_not_failures": True}

def freeze_ten_minute_config(template: str | Path, calibration: str | Path, validation: str | Path, output: str | Path, *, repo_root: str | Path | None = None) -> TenMinuteConfig:
    """Resolve a publication template only from validation-only evidence.

    The evidence files must explicitly assert that no held-out targets were
    accessed.  A dirty worktree is rejected because the resulting digest is a
    publication boundary, not an engineering estimate.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        raise ValueError("cannot freeze ten-minute protocol from a dirty worktree")
    template_cfg = load_ten_minute_config(template)
    cal_path, val_path = Path(calibration), Path(validation)
    cal = json.loads(cal_path.read_text(encoding="utf-8") if cal_path.is_file() else (cal_path / "calibration.json").read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8") if val_path.is_file() else (val_path / "validation.json").read_text(encoding="utf-8"))
    if cal.get("no_test_access") is not True or val.get("no_test_access") is not True:
        raise ValueError("calibration and validation must assert no_test_access=true")
    cap = cal.get("selected_hard_expansion_cap")
    total = val.get("selected_total_expansions_per_seed")
    if cap not in template_cfg.runtime.candidate_hard_expansion_caps:
        raise ValueError("calibration did not select a candidate hard cap")
    if total not in template_cfg.training.candidate_total_expansions_per_seed:
        raise ValueError("validation did not select a candidate training total")
    payload = json.loads(json.dumps(template_cfg.payload))
    payload["runtime_protocol"]["candidate_hard_expansion_caps"] = []
    payload["runtime_protocol"]["selected_hard_expansion_cap"] = int(cap)
    payload["budget_protocol"]["maximum_horizon_by_difficulty"]["hard"] = int(cap)
    payload["budget_protocol"]["thresholds_by_difficulty"]["hard"] = [x for x in payload["budget_protocol"]["thresholds_by_difficulty"]["hard"] if x <= int(cap)]
    payload["training_protocol"]["total_expansions_per_seed"] = int(total)
    payload["training_protocol"]["candidate_total_expansions_per_seed"] = []
    from benchmarks.article_native_corpus import build_article_v1_corpus
    corpus = build_article_v1_corpus(load_ten_minute_corpus_config(template))
    experiment = payload["experiment"]
    payload["publication_freeze"] = {
        "frozen": True,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip(),
        "no_test_access": True,
        "selected_hard_expansion_cap": int(cap),
        "selected_total_expansions_per_seed": int(total),
        "target_ids": [case.target_id for case in corpus.targets],
        "seeds": {
            "training": list(experiment["training_seeds"]),
            "random_scheduler": list(experiment["random_scheduler_seeds"]),
            "validation": list(experiment["validation_seeds"]),
        },
    }
    result = TenMinuteConfig.from_mapping(payload, require_frozen=True)
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(destination)
    return result

__all__ = ["TEN_MINUTE_CONFIG_SCHEMA", "TEN_MINUTE_RAW_RUN_SCHEMA", "TEN_MINUTE_AUDIT_SCHEMA", "TEN_MINUTE_CHECKPOINT_SCHEMA", "TEN_MINUTE_FRONTIER_ENUMERATION_SCHEMA", "TenMinuteConfig", "TenMinuteCheckpoint", "load_ten_minute_config", "load_ten_minute_corpus_config", "freeze_ten_minute_config", "RuntimeProtocol", "BudgetProtocol", "TrainingProtocol", "SecondaryCPUProtocol", "ProcessCPUWatchdog", "WatchdogDecision", "CurriculumAccounting", "train_fixed_interaction_curriculum", "process_cpu_seconds", "seeded_round_robin_schedule", "anytime_success_rows", "audit_ten_minute_runs", "PRIMARY_SCHEDULERS"]
