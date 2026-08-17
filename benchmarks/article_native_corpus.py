"""Article V1 native Clifford+T target corpus.

This module is intentionally a deterministic data layer.  It generates exact
reachable targets from the frozen native grammar, but it never constructs a
search environment and never exposes the generator witness as a solution
oracle.  Evaluation code should consume :meth:`ArticleV1TargetCase.synthesis_target`
or :meth:`ArticleV1TargetCase.evaluation_target`; the retained witness exists
only to reproduce and audit target generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from certification.simulator import SynthesisTarget, unitary_from_gates
from certification.unitary_phase_metrics import phase_frobenius_discrepancy
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType


ARTICLE_V1_CORPUS_SCHEMA = "article-v1-native-corpus-v2"
ARTICLE_V1_CONFIG_SCHEMA = "article-v1-corpus-config-v2"
ARTICLE_V1_IDENTITY_SCHEMA = "projective-identity-shared-metric-v2"
ARTICLE_V1_EVALUATION_SCHEMA = "article-v1-evaluation-target-v1"
ARTICLE_V1_CHECKPOINT_SCOPE_SCHEMA = "article-v1-checkpoint-evaluation-scope-v1"

STANDARD_CHECKPOINT_FAMILY = "standard"
OOD_LENGTH_CHECKPOINT_FAMILY = "ood_length"
CHECKPOINT_FAMILIES = (
    STANDARD_CHECKPOINT_FAMILY,
    OOD_LENGTH_CHECKPOINT_FAMILY,
)
COMPLETE_TRAINING_SCOPE = "complete_train_partition"
PARTIAL_SMOKE_TRAINING_SCOPE = "explicit_partial_smoke"
TRAINING_SCOPE_MODES = (
    COMPLETE_TRAINING_SCOPE,
    PARTIAL_SMOKE_TRAINING_SCOPE,
)
ARTICLE_V1_TRAINING_BUDGET_POLICY = "per-target-budget-with-optional-cap-v1"

SPLIT_ORDER = ("train", "validation", "test", "ood_test")
PRIMARY_SPLITS = ("train", "validation", "test")
DIFFICULTY_ORDER = ("easy", "medium", "hard")
NATIVE_GATE_NAMES = ("H", "S", "SDG", "T", "TDG", "CNOT")

_CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "pilot": _CONFIG_DIRECTORY / "article_v1_pilot.json",
        "publication": _CONFIG_DIRECTORY / "article_v1_publication.json",
    }
)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def article_delta_phi(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Article V1 phase-invariant Frobenius discrepancy.

    For equal-size unitary matrices this is

    ``sqrt(1 - abs(trace(right† @ left)) / dimension)``.

    The clipping only removes floating-point excursions outside ``[0, 1]``;
    it does not introduce a second identity tolerance.
    """

    return phase_frobenius_discrepancy(left, right)


def _phase_normalized_matrix(unitary: np.ndarray, *, decimals: int) -> np.ndarray:
    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix")
    if decimals < 8 or decimals > 15:
        raise ValueError("digest_decimals must lie in [8, 15]")
    flat = matrix.reshape(-1)
    significant = np.flatnonzero(np.abs(flat) > 10.0 ** (-(decimals - 2)))
    if significant.size == 0:
        raise ValueError("unitary has no stable global-phase anchor")
    anchor = flat[int(significant[0])]
    normalized = matrix / (anchor / abs(anchor))
    real = np.round(normalized.real, decimals=decimals)
    imag = np.round(normalized.imag, decimals=decimals)
    threshold = 10.0 ** (-decimals)
    real[np.abs(real) < threshold] = 0.0
    imag[np.abs(imag) < threshold] = 0.0
    return real + 1.0j * imag


def dense_target_digest(unitary: np.ndarray, *, decimals: int = 12) -> str:
    """Return a witness-independent, global-phase-invariant dense identity."""

    normalized = _phase_normalized_matrix(unitary, decimals=decimals)
    payload = {
        "schema": ARTICLE_V1_IDENTITY_SCHEMA,
        "shape": list(normalized.shape),
        "decimals": int(decimals),
        "entries": [
            [float(value.real), float(value.imag)]
            for value in normalized.reshape(-1)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def dense_unitary_digest(unitary: np.ndarray) -> str:
    """Return a deterministic literal dense-matrix digest.

    Corpus identity uses :func:`dense_target_digest` and therefore quotients
    global phase.  This second digest records the exact generated matrix bytes
    so a replayed manifest can independently detect representation drift.
    """

    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("unitary must contain only finite values")
    digest = hashlib.sha256()
    digest.update(b"article-v1-literal-dense-unitary-v1\0")
    digest.update(json.dumps(list(matrix.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(matrix, dtype="<c16").tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def native_gate_grammar(num_qubits: int) -> tuple[Gate, ...]:
    """Return the frozen ``5n + n(n-1)`` Article V1 native grammar."""

    width = _integer(num_qubits, name="num_qubits", minimum=1)
    gates: list[Gate] = []
    for qubit in range(width):
        gates.extend(
            Gate(gate_type, (qubit,))
            for gate_type in (
                GateType.H,
                GateType.S,
                GateType.SDG,
                GateType.T,
                GateType.TDG,
            )
        )
    gates.extend(
        Gate(GateType.CNOT, (control, target))
        for control in range(width)
        for target in range(width)
        if control != target
    )
    return tuple(gates)


@dataclass(frozen=True, slots=True)
class ArticleV1Budget:
    max_t_count: int
    max_two_qubit_count: int
    max_gates: int
    max_depth: int
    expansion_budget: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, name: str) -> "ArticleV1Budget":
        return cls(
            max_t_count=_integer(value.get("max_t_count"), name=f"{name}.max_t_count"),
            max_two_qubit_count=_integer(
                value.get("max_two_qubit_count"),
                name=f"{name}.max_two_qubit_count",
            ),
            max_gates=_integer(value.get("max_gates"), name=f"{name}.max_gates"),
            max_depth=_integer(value.get("max_depth"), name=f"{name}.max_depth"),
            expansion_budget=_integer(
                value.get("expansion_budget"),
                name=f"{name}.expansion_budget",
                minimum=1,
            ),
        )

    def resource_budget(self) -> ResourceBudget:
        return ResourceBudget(
            max_t_count=self.max_t_count,
            max_two_qubit_count=self.max_two_qubit_count,
            max_gates=self.max_gates,
            max_depth=self.max_depth,
        )

    def metadata(self) -> dict[str, int]:
        return {
            "max_t_count": self.max_t_count,
            "max_two_qubit_count": self.max_two_qubit_count,
            "max_gates": self.max_gates,
            "max_depth": self.max_depth,
            "expansion_budget": self.expansion_budget,
        }


@dataclass(frozen=True, slots=True)
class DifficultySpec:
    name: str
    min_generator_length: int
    max_generator_length: int
    budget: ArticleV1Budget

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "DifficultySpec":
        minimum = _integer(
            value.get("min_generator_length"),
            name=f"difficulty.{name}.min_generator_length",
            minimum=1,
        )
        maximum = _integer(
            value.get("max_generator_length"),
            name=f"difficulty.{name}.max_generator_length",
            minimum=minimum,
        )
        budget = ArticleV1Budget.from_mapping(
            value.get("budget", {}), name=f"difficulty.{name}.budget"
        )
        if any(
            limit < maximum
            for limit in (
                budget.max_t_count,
                budget.max_two_qubit_count,
                budget.max_gates,
                budget.max_depth,
            )
        ):
            raise ValueError(f"difficulty {name!r} budget cannot contain its witnesses")
        return cls(name, minimum, maximum, budget)

    def metadata(self) -> dict[str, object]:
        return {
            "min_generator_length": self.min_generator_length,
            "max_generator_length": self.max_generator_length,
            "budget": self.budget.metadata(),
        }


@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    seed: int
    counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "SplitSpec":
        raw_counts = value.get("counts", {})
        if set(raw_counts) != set(DIFFICULTY_ORDER):
            raise ValueError(
                f"split {name!r} counts must define exactly {DIFFICULTY_ORDER!r}"
            )
        return cls(
            name=name,
            seed=_integer(value.get("seed"), name=f"splits.{name}.seed"),
            counts=tuple(
                (
                    difficulty,
                    _integer(
                        raw_counts[difficulty],
                        name=f"splits.{name}.counts.{difficulty}",
                    ),
                )
                for difficulty in DIFFICULTY_ORDER
            ),
        )

    def count(self, difficulty: str) -> int:
        try:
            return dict(self.counts)[difficulty]
        except KeyError as exc:
            raise ValueError(f"unknown difficulty {difficulty!r}") from exc


@dataclass(frozen=True, slots=True)
class OODLengthSplit:
    training_source_split: str
    training_max_generator_length: int
    evaluation_split: str
    evaluation_min_generator_length: int
    evaluation_max_generator_length: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OODLengthSplit":
        result = cls(
            training_source_split=str(value.get("training_source_split", "train")),
            training_max_generator_length=_integer(
                value.get("training_max_generator_length"),
                name="ood_length_split.training_max_generator_length",
                minimum=1,
            ),
            evaluation_split=str(value.get("evaluation_split", "ood_test")),
            evaluation_min_generator_length=_integer(
                value.get("evaluation_min_generator_length"),
                name="ood_length_split.evaluation_min_generator_length",
                minimum=1,
            ),
            evaluation_max_generator_length=_integer(
                value.get("evaluation_max_generator_length"),
                name="ood_length_split.evaluation_max_generator_length",
                minimum=1,
            ),
        )
        if result.training_source_split not in PRIMARY_SPLITS:
            raise ValueError("OOD training source must be train, validation, or test")
        if result.evaluation_split != "ood_test":
            raise ValueError("Article V1 reserves the ood_test split for OOD evaluation")
        if (
            result.training_max_generator_length
            >= result.evaluation_min_generator_length
        ):
            raise ValueError("OOD training and evaluation length ranges must be disjoint")
        if (
            result.evaluation_max_generator_length
            < result.evaluation_min_generator_length
        ):
            raise ValueError("OOD evaluation maximum must not be below its minimum")
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "training_source_split": self.training_source_split,
            "training_max_generator_length": self.training_max_generator_length,
            "evaluation_split": self.evaluation_split,
            "evaluation_min_generator_length": self.evaluation_min_generator_length,
            "evaluation_max_generator_length": self.evaluation_max_generator_length,
        }


@dataclass(frozen=True, slots=True)
class ArticleV1CorpusConfig:
    profile: str
    qubits: tuple[int, ...]
    tau_identity: float
    digest_decimals: int
    max_generation_attempts_per_case: int
    difficulties: tuple[DifficultySpec, ...]
    splits: tuple[SplitSpec, ...]
    ood_length_split: OODLengthSplit
    experiment: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArticleV1CorpusConfig":
        if value.get("schema_version") != ARTICLE_V1_CONFIG_SCHEMA:
            raise ValueError(
                f"config schema must be {ARTICLE_V1_CONFIG_SCHEMA!r}"
            )
        qubits = tuple(
            _integer(item, name="qubits entry", minimum=1)
            for item in value.get("qubits", ())
        )
        if set(qubits) != {2, 3} or len(qubits) != 2:
            raise ValueError("Article V1 corpus qubits must be exactly [2, 3]")
        raw_difficulties = value.get("difficulty_strata", {})
        if set(raw_difficulties) != set(DIFFICULTY_ORDER):
            raise ValueError(
                f"difficulty_strata must define exactly {DIFFICULTY_ORDER!r}"
            )
        difficulties = tuple(
            DifficultySpec.from_mapping(name, raw_difficulties[name])
            for name in DIFFICULTY_ORDER
        )
        if (
            difficulties[0].min_generator_length != 2
            or difficulties[0].max_generator_length != 3
            or difficulties[1].min_generator_length != 4
            or difficulties[1].max_generator_length != 5
            or difficulties[2].min_generator_length < 6
        ):
            raise ValueError(
                "Article V1 strata must be easy=2-3, medium=4-5, hard>=6"
            )
        raw_splits = value.get("splits", {})
        if set(raw_splits) != set(SPLIT_ORDER):
            raise ValueError(f"splits must define exactly {SPLIT_ORDER!r}")
        splits = tuple(
            SplitSpec.from_mapping(name, raw_splits[name]) for name in SPLIT_ORDER
        )
        ood = OODLengthSplit.from_mapping(value.get("ood_length_split", {}))
        difficulty_by_name = {item.name: item for item in difficulties}
        ood_split = {item.name: item for item in splits}[ood.evaluation_split]
        for difficulty, count in ood_split.counts:
            if count == 0:
                continue
            spec = difficulty_by_name[difficulty]
            low = max(spec.min_generator_length, ood.evaluation_min_generator_length)
            high = min(spec.max_generator_length, ood.evaluation_max_generator_length)
            if low > high:
                raise ValueError(
                    f"OOD count for {difficulty!r} has no permitted generator length"
                )
        digest_decimals = _integer(
            value.get("digest_decimals"), name="digest_decimals", minimum=8
        )
        if digest_decimals > 15:
            raise ValueError("digest_decimals must lie in [8, 15]")
        experiment = value.get("experiment", {})
        if not isinstance(experiment, Mapping):
            raise ValueError("experiment must be a JSON object")
        required_experiment = {
            "profile_name": "article_v1_raw_metric_v2",
            "feature_schema": "article-v1-31d",
            "reward_schema": "article-v1-expansion-potential-amended",
            "target_metric_schema": "projective-unitary-metrics-v2",
            "certification_schema": "phase-frobenius-raw-v2",
        }
        for key, expected in required_experiment.items():
            if experiment.get(key) != expected:
                raise ValueError(
                    f"experiment.{key} must be {expected!r} for Article V1"
                )
        if float(experiment.get("gamma", -1.0)) != 1.0:
            raise ValueError("Article V1 requires experiment.gamma=1.0")
        beta = float(experiment.get("beta", -1.0))
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("experiment.beta must be finite and non-negative")
        _finite_positive(
            experiment.get("learning_rate"), name="experiment.learning_rate"
        )
        _finite_positive(
            experiment.get("certification_tolerance"),
            name="experiment.certification_tolerance",
        )
        if int(experiment.get("training_episodes_per_target", 0)) < 1:
            raise ValueError("training_episodes_per_target must be positive")
        seed_fields = (
            "training_seeds",
            "random_scheduler_seeds",
            "validation_seeds",
        )
        for field_name in seed_fields:
            seeds = experiment.get(field_name)
            if (
                not isinstance(seeds, list)
                or not seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise ValueError(
                    f"experiment.{field_name} must be a nonempty list of unique integers"
                )
        profile = str(value.get("profile", "unspecified"))
        if profile == "publication":
            if len(experiment["training_seeds"]) < 5:
                raise ValueError("publication requires at least five training seeds")
            if len(experiment["random_scheduler_seeds"]) < 10:
                raise ValueError("publication requires at least ten random scheduler seeds")
        epsilon = experiment.get("epsilon")
        if not isinstance(epsilon, Mapping) or set(epsilon) != {
            "start",
            "minimum",
            "decay",
        }:
            raise ValueError("experiment.epsilon must define start/minimum/decay")
        epsilon_values = {name: float(number) for name, number in epsilon.items()}
        if not (
            0.0 <= epsilon_values["minimum"] <= epsilon_values["start"] <= 1.0
            and 0.0 < epsilon_values["decay"] <= 1.0
        ):
            raise ValueError("experiment epsilon schedule is invalid")
        expected_schedulers = (
            "fifo",
            "lifo",
            "uniform_cost",
            "seeded_random",
            "zero_weight_linear",
            "article_target_distance",
            "article_sarsa",
        )
        if tuple(experiment.get("schedulers", ())) != expected_schedulers:
            raise ValueError("experiment.schedulers must be the seven Article V1 schedulers")
        multipliers = experiment.get("expansion_budget_multipliers")
        if (
            not isinstance(multipliers, list)
            or not multipliers
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not np.isfinite(float(item))
                or float(item) <= 0.0
                for item in multipliers
            )
            or list(multipliers) != sorted(set(float(item) for item in multipliers))
        ):
            raise ValueError(
                "experiment.expansion_budget_multipliers must be sorted unique positives"
            )
        for field_name in (
            "canonicalization_enabled",
            "pareto_dominance_enabled",
            "absorb_clifford_angles",
            "timing_enabled",
        ):
            if not isinstance(experiment.get(field_name), bool):
                raise ValueError(f"experiment.{field_name} must be a bool")
        if experiment.get("canonicalization_mode") != "enhanced":
            raise ValueError(
                "primary Article V1 experiment.canonicalization_mode must be 'enhanced'"
            )
        _integer(
            experiment.get("statistics_seed"),
            name="experiment.statistics_seed",
        )
        return cls(
            profile=profile,
            qubits=qubits,
            tau_identity=_finite_positive(
                value.get("tau_identity"), name="tau_identity"
            ),
            digest_decimals=digest_decimals,
            max_generation_attempts_per_case=_integer(
                value.get("max_generation_attempts_per_case"),
                name="max_generation_attempts_per_case",
                minimum=1,
            ),
            difficulties=difficulties,
            splits=splits,
            ood_length_split=ood,
            experiment=_freeze_json(experiment),
        )

    def difficulty(self, name: str) -> DifficultySpec:
        try:
            return next(item for item in self.difficulties if item.name == name)
        except StopIteration as exc:
            raise ValueError(f"unknown difficulty {name!r}") from exc

    def split(self, name: str) -> SplitSpec:
        try:
            return next(item for item in self.splits if item.name == name)
        except StopIteration as exc:
            raise ValueError(f"unknown split {name!r}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ARTICLE_V1_CONFIG_SCHEMA,
            "profile": self.profile,
            "qubits": list(self.qubits),
            "tau_identity": self.tau_identity,
            "digest_decimals": self.digest_decimals,
            "max_generation_attempts_per_case": self.max_generation_attempts_per_case,
            "difficulty_strata": {
                item.name: item.metadata() for item in self.difficulties
            },
            "splits": {
                split.name: {
                    "seed": split.seed,
                    "counts": dict(split.counts),
                }
                for split in self.splits
            },
            "ood_length_split": self.ood_length_split.metadata(),
            "experiment": _thaw_json(self.experiment),
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ArticleV1CheckpointScope:
    """Fail-closed provenance contract for one checkpoint evaluation family.

    The scope is deliberately separate from a target record.  Evaluation code
    must opt into a concrete corpus partition and feature schema before a
    learned checkpoint can be used.  This prevents a structurally valid
    ablation, OOD, or foreign-corpus checkpoint from being silently treated as
    the primary Article V1 learner.
    """

    corpus_config_digest: str
    checkpoint_family: str
    training_scope_mode: str
    expected_feature_schema_version: str
    expected_training_beta: float
    expected_certification_tolerance: float
    expected_episodes_per_target: int
    expected_learning_rate: float
    expected_epsilon_schedule: tuple[tuple[str, float], ...]
    allowed_training_seeds: tuple[int, ...]
    expected_expansion_cap: int | None
    training_budget_policy: str
    allowed_training_target_ids: tuple[str, ...]
    expected_training_expansion_budgets: tuple[tuple[str, int], ...]
    held_out_target_ids: tuple[str, ...]
    permitted_evaluation_target_ids: tuple[str, ...]
    schema_version: str = ARTICLE_V1_CHECKPOINT_SCOPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTICLE_V1_CHECKPOINT_SCOPE_SCHEMA:
            raise ValueError("unsupported Article V1 checkpoint scope schema")
        if not self.corpus_config_digest:
            raise ValueError("checkpoint scope requires a corpus config digest")
        if self.checkpoint_family not in CHECKPOINT_FAMILIES:
            raise ValueError(
                f"checkpoint family must be one of {CHECKPOINT_FAMILIES!r}"
            )
        if self.training_scope_mode not in TRAINING_SCOPE_MODES:
            raise ValueError(
                f"training scope mode must be one of {TRAINING_SCOPE_MODES!r}"
            )
        if not self.expected_feature_schema_version:
            raise ValueError("checkpoint scope requires an expected feature schema")
        if (
            not np.isfinite(float(self.expected_training_beta))
            or float(self.expected_training_beta) < 0.0
        ):
            raise ValueError("checkpoint scope training beta must be finite and non-negative")
        if (
            not np.isfinite(float(self.expected_certification_tolerance))
            or float(self.expected_certification_tolerance) <= 0.0
        ):
            raise ValueError(
                "checkpoint scope certification tolerance must be finite and positive"
            )
        if (
            isinstance(self.expected_episodes_per_target, bool)
            or not isinstance(self.expected_episodes_per_target, int)
            or self.expected_episodes_per_target < 1
        ):
            raise ValueError("checkpoint scope episodes per target must be positive")
        if (
            not np.isfinite(float(self.expected_learning_rate))
            or float(self.expected_learning_rate) <= 0.0
        ):
            raise ValueError("checkpoint scope learning rate must be finite and positive")
        expected_epsilon_names = ("decay", "minimum", "start")
        if tuple(name for name, _value in self.expected_epsilon_schedule) != (
            expected_epsilon_names
        ):
            raise ValueError(
                "checkpoint scope epsilon schedule must define decay/minimum/start"
            )
        epsilon = dict(self.expected_epsilon_schedule)
        if not (
            0.0 <= float(epsilon["minimum"]) <= float(epsilon["start"]) <= 1.0
            and 0.0 < float(epsilon["decay"]) <= 1.0
            and all(np.isfinite(float(value)) for value in epsilon.values())
        ):
            raise ValueError("checkpoint scope epsilon schedule is invalid")
        if (
            not self.allowed_training_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in self.allowed_training_seeds
            )
            or len(set(self.allowed_training_seeds))
            != len(self.allowed_training_seeds)
        ):
            raise ValueError(
                "checkpoint scope allowed training seeds must be nonempty unique integers"
            )
        if self.expected_expansion_cap is not None and (
            isinstance(self.expected_expansion_cap, bool)
            or not isinstance(self.expected_expansion_cap, int)
            or self.expected_expansion_cap < 1
        ):
            raise ValueError("checkpoint scope expansion cap must be positive or None")
        if self.training_budget_policy != ARTICLE_V1_TRAINING_BUDGET_POLICY:
            raise ValueError("unsupported Article V1 training budget policy")

        field_values = {
            "allowed training": tuple(self.allowed_training_target_ids),
            "held-out": tuple(self.held_out_target_ids),
            "permitted evaluation": tuple(self.permitted_evaluation_target_ids),
        }
        for label, values in field_values.items():
            if not values or any(not value for value in values):
                raise ValueError(f"checkpoint scope {label} target IDs must be nonempty")
            if len(set(values)) != len(values):
                raise ValueError(f"checkpoint scope {label} target IDs must be unique")

        allowed = set(self.allowed_training_target_ids)
        held_out = set(self.held_out_target_ids)
        permitted = set(self.permitted_evaluation_target_ids)
        if allowed & held_out:
            raise ValueError(
                "checkpoint scope training and held-out target IDs must be disjoint"
            )
        if not permitted <= held_out:
            raise ValueError(
                "permitted evaluation target IDs must be a subset of held-out IDs"
            )
        budget_ids = tuple(
            str(target_id)
            for target_id, _budget in self.expected_training_expansion_budgets
        )
        if budget_ids != self.allowed_training_target_ids:
            raise ValueError(
                "expected training budgets must follow the allowed training target order"
            )
        for _target_id, budget in self.expected_training_expansion_budgets:
            if isinstance(budget, bool) or int(budget) < 1:
                raise ValueError("expected training expansion budgets must be positive")
            if (
                self.expected_expansion_cap is not None
                and int(budget) > int(self.expected_expansion_cap)
            ):
                raise ValueError("expected training budget exceeds the expansion cap")

    @classmethod
    def from_partitions(
        cls,
        *,
        corpus_config_digest: str,
        checkpoint_family: str,
        training_scope_mode: str,
        expected_feature_schema_version: str,
        expected_training_beta: float,
        expected_certification_tolerance: float,
        expected_episodes_per_target: int,
        expected_learning_rate: float,
        expected_epsilon_schedule: Mapping[str, float] | Sequence[tuple[str, float]],
        allowed_training_seeds: Sequence[int],
        expected_expansion_cap: int | None,
        training_cases: Sequence[object],
        held_out_cases: Sequence[object],
        evaluation_cases: Sequence[object] | None = None,
    ) -> "ArticleV1CheckpointScope":
        """Build a scope from witness-free case metadata only."""

        training_ids = tuple(str(getattr(case, "target_id")) for case in training_cases)
        held_out_ids = tuple(str(getattr(case, "target_id")) for case in held_out_cases)
        selected = held_out_cases if evaluation_cases is None else evaluation_cases
        evaluation_ids = tuple(str(getattr(case, "target_id")) for case in selected)
        effective_budgets = tuple(
            (
                str(getattr(case, "target_id")),
                min(
                    int(getattr(getattr(case, "budget"), "expansion_budget")),
                    (
                        int(expected_expansion_cap)
                        if expected_expansion_cap is not None
                        else int(getattr(getattr(case, "budget"), "expansion_budget"))
                    ),
                ),
            )
            for case in training_cases
        )
        epsilon_items = (
            tuple(expected_epsilon_schedule.items())
            if isinstance(expected_epsilon_schedule, Mapping)
            else tuple(expected_epsilon_schedule)
        )
        return cls(
            corpus_config_digest=str(corpus_config_digest),
            checkpoint_family=str(checkpoint_family),
            training_scope_mode=str(training_scope_mode),
            expected_feature_schema_version=str(expected_feature_schema_version),
            expected_training_beta=float(expected_training_beta),
            expected_certification_tolerance=float(
                expected_certification_tolerance
            ),
            expected_episodes_per_target=int(expected_episodes_per_target),
            expected_learning_rate=float(expected_learning_rate),
            expected_epsilon_schedule=tuple(
                (str(name), float(value))
                for name, value in sorted(epsilon_items)
            ),
            allowed_training_seeds=tuple(int(seed) for seed in allowed_training_seeds),
            expected_expansion_cap=(
                None
                if expected_expansion_cap is None
                else int(expected_expansion_cap)
            ),
            training_budget_policy=ARTICLE_V1_TRAINING_BUDGET_POLICY,
            allowed_training_target_ids=training_ids,
            expected_training_expansion_budgets=effective_budgets,
            held_out_target_ids=held_out_ids,
            permitted_evaluation_target_ids=evaluation_ids,
        )

    def validate_evaluation_target(self, case: object) -> None:
        target_id = str(getattr(case, "target_id"))
        if target_id not in set(self.permitted_evaluation_target_ids):
            raise ValueError(
                f"target {target_id!r} is outside the permitted held-out evaluation scope"
            )

    def validate_evaluation_parameters(
        self,
        *,
        beta: float,
        certification_tolerance: float,
    ) -> None:
        if float(beta) != float(self.expected_training_beta):
            raise ValueError(
                "evaluation beta does not match the checkpoint training scope"
            )
        if float(certification_tolerance) != float(
            self.expected_certification_tolerance
        ):
            raise ValueError(
                "evaluation certification tolerance does not match the checkpoint "
                "training scope"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_config_digest": self.corpus_config_digest,
            "checkpoint_family": self.checkpoint_family,
            "training_scope_mode": self.training_scope_mode,
            "expected_feature_schema_version": self.expected_feature_schema_version,
            "expected_training_beta": self.expected_training_beta,
            "expected_certification_tolerance": (
                self.expected_certification_tolerance
            ),
            "expected_episodes_per_target": self.expected_episodes_per_target,
            "expected_learning_rate": self.expected_learning_rate,
            "expected_epsilon_schedule": dict(self.expected_epsilon_schedule),
            "allowed_training_seeds": list(self.allowed_training_seeds),
            "expected_expansion_cap": self.expected_expansion_cap,
            "training_budget_policy": self.training_budget_policy,
            "allowed_training_target_ids": list(self.allowed_training_target_ids),
            "expected_training_expansion_budgets": [
                {"target_id": target_id, "expansion_budget": budget}
                for target_id, budget in self.expected_training_expansion_budgets
            ],
            "held_out_target_ids": list(self.held_out_target_ids),
            "permitted_evaluation_target_ids": list(
                self.permitted_evaluation_target_ids
            ),
        }


def load_article_v1_config(
    profile_or_path: str | Path = "pilot",
) -> ArticleV1CorpusConfig:
    """Load one checked-in profile or an explicit Article V1 JSON config."""

    if isinstance(profile_or_path, str) and profile_or_path in DEFAULT_CONFIG_PATHS:
        path = DEFAULT_CONFIG_PATHS[profile_or_path]
    else:
        path = Path(profile_or_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Article V1 corpus config must contain a JSON object")
    return ArticleV1CorpusConfig.from_mapping(payload)


@dataclass(frozen=True, slots=True)
class ArticleV1EvaluationTarget:
    """Witness-free object safe to hand to a scheduler evaluation runner."""

    target_id: str
    split: str
    difficulty: str
    num_qubits: int
    generator_length: int
    budget: ArticleV1Budget
    target: SynthesisTarget
    schema_version: str = ARTICLE_V1_EVALUATION_SCHEMA
    target_specific_reachability_oracle: bool = False


@dataclass(frozen=True, slots=True)
class ArticleV1TargetCase:
    """One exact target and explicitly audit-only generator provenance."""

    target_id: str
    split: str
    difficulty: str
    split_seed: int
    generator_seed: int
    generation_attempt: int
    ordinal_within_stratum: int
    num_qubits: int
    generator_length: int
    budget: ArticleV1Budget
    unitary: np.ndarray
    generator_witness: tuple[Gate, ...]
    digest_decimals: int

    def __post_init__(self) -> None:
        if self.split not in SPLIT_ORDER:
            raise ValueError(f"unknown split {self.split!r}")
        if self.difficulty not in DIFFICULTY_ORDER:
            raise ValueError(f"unknown difficulty {self.difficulty!r}")
        if self.num_qubits not in {2, 3}:
            raise ValueError("Article V1 cases require two or three qubits")
        witness = tuple(self.generator_witness)
        if len(witness) != self.generator_length:
            raise ValueError("generator witness length does not match metadata")
        if any(gate.gate_type.name not in NATIVE_GATE_NAMES for gate in witness):
            raise ValueError("generator witness contains a non-native gate")
        target = SynthesisTarget(self.unitary)
        if target.num_qubits != self.num_qubits:
            raise ValueError("target dimension and num_qubits disagree")
        matrix = np.array(target.unitary, dtype=np.complex128, copy=True)
        matrix.setflags(write=False)
        if dense_target_digest(matrix, decimals=self.digest_decimals) != self.target_id:
            raise ValueError("target_id is not the dense target digest")
        replay = unitary_from_gates(self.num_qubits, witness)
        if not np.allclose(replay, matrix, atol=1e-9, rtol=1e-9):
            raise ValueError("audit witness does not replay to the stored dense target")
        object.__setattr__(self, "unitary", matrix)
        object.__setattr__(self, "generator_witness", witness)

    def synthesis_target(self) -> SynthesisTarget:
        """Return the dense target; the audit witness is intentionally omitted."""

        return SynthesisTarget(self.unitary)

    def evaluation_target(self) -> ArticleV1EvaluationTarget:
        """Return a witness-free evaluation record with fixed budgets."""

        return ArticleV1EvaluationTarget(
            target_id=self.target_id,
            split=self.split,
            difficulty=self.difficulty,
            num_qubits=self.num_qubits,
            generator_length=self.generator_length,
            budget=self.budget,
            target=self.synthesis_target(),
        )

    def metadata(self) -> dict[str, object]:
        resource_budget = self.budget.metadata().copy()
        resource_budget.pop("expansion_budget")
        return {
            "schema_version": ARTICLE_V1_CORPUS_SCHEMA,
            "generation_schema_version": ARTICLE_V1_CORPUS_SCHEMA,
            "identity_schema": ARTICLE_V1_IDENTITY_SCHEMA,
            "target_id": self.target_id,
            "target_digest": self.target_id,
            "target_unitary_digest": dense_unitary_digest(self.unitary),
            "target_identity_digest": self.target_id,
            "target_matrix_shape": list(self.unitary.shape),
            "split": self.split,
            "stratum": self.difficulty,
            "difficulty": self.difficulty,
            "split_seed": self.split_seed,
            "generator_seed": self.generator_seed,
            "generation_attempt": self.generation_attempt,
            "ordinal_within_stratum": self.ordinal_within_stratum,
            "num_qubits": self.num_qubits,
            "generator_length": self.generator_length,
            "budget": self.budget.metadata(),
            "resource_budget": resource_budget,
            "generator": {
                "sampling": "uniform-over-native-grammar-with-replacement",
                "grammar": list(NATIVE_GATE_NAMES),
                "grammar_cardinality": 5 * self.num_qubits
                + self.num_qubits * (self.num_qubits - 1),
                "witness_operations": [
                    {"gate": gate.gate_type.name, "qubits": list(gate.qubits)}
                    for gate in self.generator_witness
                ],
            },
            "generator_witness": [
                {"gate": gate.gate_type.name, "qubits": list(gate.qubits)}
                for gate in self.generator_witness
            ],
            "generator_witness_provenance": "reachability-and-replay-audit-only",
            "generator_witness_evaluation_prohibited": True,
            "target_specific_reachability_oracle": False,
        }


@dataclass(frozen=True, slots=True)
class ArticleV1Corpus:
    config: ArticleV1CorpusConfig
    targets: tuple[ArticleV1TargetCase, ...]
    rejection_counts: tuple[tuple[str, int], ...]

    def cases(
        self,
        *,
        split: str | None = None,
        difficulty: str | None = None,
    ) -> tuple[ArticleV1TargetCase, ...]:
        if split is not None and split not in SPLIT_ORDER:
            raise ValueError(f"unknown split {split!r}")
        if difficulty is not None and difficulty not in DIFFICULTY_ORDER:
            raise ValueError(f"unknown difficulty {difficulty!r}")
        return tuple(
            case
            for case in self.targets
            if (split is None or case.split == split)
            and (difficulty is None or case.difficulty == difficulty)
        )

    def evaluation_targets(
        self,
        *,
        split: str,
        difficulty: str | None = None,
    ) -> tuple[ArticleV1EvaluationTarget, ...]:
        return tuple(
            case.evaluation_target()
            for case in self.cases(split=split, difficulty=difficulty)
        )

    def checkpoint_scope(
        self,
        *,
        checkpoint_family: str = STANDARD_CHECKPOINT_FAMILY,
        expected_feature_schema_version: str = "article-v1-31d",
        training_scope_mode: str = COMPLETE_TRAINING_SCOPE,
        training_target_ids: Sequence[str] | None = None,
        expected_training_beta: float | None = None,
        expected_certification_tolerance: float | None = None,
        expected_episodes_per_target: int | None = None,
        expected_learning_rate: float | None = None,
        expected_epsilon_schedule: Mapping[str, float] | None = None,
        allowed_training_seeds: Sequence[int] | None = None,
        expected_expansion_cap: int | None = None,
    ) -> ArticleV1CheckpointScope:
        """Return the preregistered standard or length-OOD evaluation scope."""

        held_out = tuple(
            case.evaluation_target()
            for case in self.targets
            if case.split != "train"
        )
        if checkpoint_family == STANDARD_CHECKPOINT_FAMILY:
            training = self.evaluation_targets(split="train")
            evaluation = tuple(
                case for case in held_out if case.split in {"validation", "test"}
            )
        elif checkpoint_family == OOD_LENGTH_CHECKPOINT_FAMILY:
            definition = self.config.ood_length_split
            training = tuple(
                case
                for case in self.evaluation_targets(
                    split=definition.training_source_split
                )
                if case.generator_length <= definition.training_max_generator_length
            )
            evaluation = tuple(
                case
                for case in held_out
                if case.split == definition.evaluation_split
                or (
                    case.split == "validation"
                    and case.generator_length
                    <= definition.training_max_generator_length
                )
            )
        else:
            raise ValueError(
                f"checkpoint family must be one of {CHECKPOINT_FAMILIES!r}"
            )
        if training_scope_mode == COMPLETE_TRAINING_SCOPE:
            if training_target_ids is not None:
                raise ValueError(
                    "complete training scopes do not accept a target-ID override"
                )
            if allowed_training_seeds is not None:
                raise ValueError(
                    "complete training scopes do not accept a seed override"
                )
        elif training_scope_mode == PARTIAL_SMOKE_TRAINING_SCOPE:
            if not training_target_ids:
                raise ValueError(
                    "partial smoke scopes require explicit training target IDs"
                )
            by_id = {case.target_id: case for case in training}
            requested = tuple(str(target_id) for target_id in training_target_ids)
            if len(set(requested)) != len(requested) or any(
                target_id not in by_id for target_id in requested
            ):
                raise ValueError(
                    "partial smoke training IDs must be unique members of the family partition"
                )
            training = tuple(by_id[target_id] for target_id in requested)
        else:
            raise ValueError(
                f"training scope mode must be one of {TRAINING_SCOPE_MODES!r}"
            )
        experiment = self.config.experiment
        resolved_training_seeds = (
            tuple(int(seed) for seed in experiment["training_seeds"])
            if allowed_training_seeds is None
            else tuple(int(seed) for seed in allowed_training_seeds)
        )
        return ArticleV1CheckpointScope.from_partitions(
            corpus_config_digest=self.config.digest,
            checkpoint_family=checkpoint_family,
            training_scope_mode=training_scope_mode,
            expected_feature_schema_version=expected_feature_schema_version,
            expected_training_beta=(
                float(experiment["beta"])
                if expected_training_beta is None
                else float(expected_training_beta)
            ),
            expected_certification_tolerance=(
                float(experiment["certification_tolerance"])
                if expected_certification_tolerance is None
                else float(expected_certification_tolerance)
            ),
            expected_episodes_per_target=(
                int(experiment["training_episodes_per_target"])
                if expected_episodes_per_target is None
                else int(expected_episodes_per_target)
            ),
            expected_learning_rate=(
                float(experiment["learning_rate"])
                if expected_learning_rate is None
                else float(expected_learning_rate)
            ),
            expected_epsilon_schedule=(
                experiment["epsilon"]
                if expected_epsilon_schedule is None
                else expected_epsilon_schedule
            ),
            allowed_training_seeds=resolved_training_seeds,
            expected_expansion_cap=expected_expansion_cap,
            training_cases=training,
            held_out_cases=held_out,
            evaluation_cases=evaluation,
        )

    def manifest(self) -> dict[str, object]:
        counts = {
            split: {
                difficulty: len(self.cases(split=split, difficulty=difficulty))
                for difficulty in DIFFICULTY_ORDER
            }
            for split in SPLIT_ORDER
        }
        ood = self.config.ood_length_split
        ood_training = tuple(
            case
            for case in self.cases(split=ood.training_source_split)
            if case.generator_length <= ood.training_max_generator_length
        )
        ood_evaluation = self.cases(split=ood.evaluation_split)
        return {
            "schema_version": ARTICLE_V1_CORPUS_SCHEMA,
            "config_profile": self.config.profile,
            "config_digest": self.config.digest,
            "identity_schema": ARTICLE_V1_IDENTITY_SCHEMA,
            "tau_identity": self.config.tau_identity,
            "identity_tolerance": self.config.tau_identity,
            "digest_decimals": self.config.digest_decimals,
            "qubits": list(self.config.qubits),
            "native_gate_grammar": list(NATIVE_GATE_NAMES),
            "native_gate_cardinality": {
                str(width): len(native_gate_grammar(width))
                for width in self.config.qubits
            },
            "split_order": list(SPLIT_ORDER),
            "split_seeds": {
                split.name: split.seed for split in self.config.splits
            },
            "experiment": _thaw_json(self.config.experiment),
            "difficulty_order": list(DIFFICULTY_ORDER),
            "difficulty_strata": {
                item.name: item.metadata() for item in self.config.difficulties
            },
            "counts": counts,
            "target_count": len(self.targets),
            "target_ids_are_globally_unique": len(
                {case.target_id for case in self.targets}
            )
            == len(self.targets),
            "projective_identity_targets_rejected": True,
            "semantic_duplicates_rejected_with_delta_phi": True,
            "rejection_counts": dict(self.rejection_counts),
            "ood_length_split": {
                **ood.metadata(),
                "training_target_ids": [case.target_id for case in ood_training],
                "evaluation_target_ids": [case.target_id for case in ood_evaluation],
                "semantic_overlap": bool(
                    {case.target_id for case in ood_training}
                    & {case.target_id for case in ood_evaluation}
                ),
            },
            "target_specific_reachability_oracle": False,
            "generator_witness_policy": (
                "manifest provenance and replay audit only; prohibited for "
                "training, scheduler selection, evaluation, and solution substitution"
            ),
            "cases": [
                {
                    **case.metadata(),
                    "expansion_budget_grid": sorted(
                        {
                            max(
                                1,
                                int(
                                    round(
                                        case.budget.expansion_budget
                                        * float(multiplier)
                                    )
                                ),
                            )
                            for multiplier in self.config.experiment[
                                "expansion_budget_multipliers"
                            ]
                        }
                    ),
                }
                for case in self.targets
            ],
        }


def _stream_seed(split_seed: int, difficulty: str) -> int:
    payload = f"{ARTICLE_V1_CORPUS_SCHEMA}:{split_seed}:{difficulty}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _witness(
    *, num_qubits: int, generator_length: int, generator_seed: int
) -> tuple[Gate, ...]:
    grammar = native_gate_grammar(num_qubits)
    rng = random.Random(generator_seed)
    return tuple(rng.choice(grammar) for _ in range(generator_length))


def _is_semantic_duplicate(
    unitary: np.ndarray,
    accepted: Sequence[ArticleV1TargetCase],
    *,
    tau_identity: float,
) -> bool:
    return any(
        case.unitary.shape == unitary.shape
        and article_delta_phi(unitary, case.unitary) <= tau_identity
        for case in accepted
    )


def build_article_v1_corpus(
    config: ArticleV1CorpusConfig | str | Path = "pilot",
) -> ArticleV1Corpus:
    """Build the deterministic Article V1 corpus described by ``config``."""

    resolved = (
        config
        if isinstance(config, ArticleV1CorpusConfig)
        else load_article_v1_config(config)
    )
    accepted: list[ArticleV1TargetCase] = []
    target_ids: set[str] = set()
    rejected = {"projective_identity": 0, "semantic_duplicate": 0, "digest_collision": 0}
    ood = resolved.ood_length_split

    for split_name in SPLIT_ORDER:
        split = resolved.split(split_name)
        for difficulty_index, difficulty_name in enumerate(DIFFICULTY_ORDER):
            difficulty = resolved.difficulty(difficulty_name)
            count = split.count(difficulty_name)
            if count == 0:
                continue
            low = difficulty.min_generator_length
            high = difficulty.max_generator_length
            if split_name == ood.evaluation_split:
                low = max(low, ood.evaluation_min_generator_length)
                high = min(high, ood.evaluation_max_generator_length)
            stream = random.Random(_stream_seed(split.seed, difficulty_name))
            for ordinal in range(count):
                num_qubits = resolved.qubits[
                    (ordinal + difficulty_index) % len(resolved.qubits)
                ]
                generator_length = low + (ordinal % (high - low + 1))
                for attempt in range(1, resolved.max_generation_attempts_per_case + 1):
                    generator_seed = stream.getrandbits(63)
                    witness = _witness(
                        num_qubits=num_qubits,
                        generator_length=generator_length,
                        generator_seed=generator_seed,
                    )
                    unitary = unitary_from_gates(num_qubits, witness)
                    identity = np.eye(1 << num_qubits, dtype=np.complex128)
                    if article_delta_phi(unitary, identity) <= resolved.tau_identity:
                        rejected["projective_identity"] += 1
                        continue
                    if _is_semantic_duplicate(
                        unitary, accepted, tau_identity=resolved.tau_identity
                    ):
                        rejected["semantic_duplicate"] += 1
                        continue
                    target_id = dense_target_digest(
                        unitary, decimals=resolved.digest_decimals
                    )
                    if target_id in target_ids:
                        rejected["digest_collision"] += 1
                        continue
                    case = ArticleV1TargetCase(
                        target_id=target_id,
                        split=split_name,
                        difficulty=difficulty_name,
                        split_seed=split.seed,
                        generator_seed=generator_seed,
                        generation_attempt=attempt,
                        ordinal_within_stratum=ordinal,
                        num_qubits=num_qubits,
                        generator_length=generator_length,
                        budget=difficulty.budget,
                        unitary=unitary,
                        generator_witness=witness,
                        digest_decimals=resolved.digest_decimals,
                    )
                    accepted.append(case)
                    target_ids.add(target_id)
                    break
                else:
                    raise RuntimeError(
                        "could not generate a unique non-identity target for "
                        f"{split_name}/{difficulty_name}/{ordinal}"
                    )

    return ArticleV1Corpus(
        config=resolved,
        targets=tuple(accepted),
        rejection_counts=tuple(sorted(rejected.items())),
    )


__all__ = [
    "ARTICLE_V1_CHECKPOINT_SCOPE_SCHEMA",
    "ARTICLE_V1_CONFIG_SCHEMA",
    "ARTICLE_V1_CORPUS_SCHEMA",
    "ARTICLE_V1_EVALUATION_SCHEMA",
    "ARTICLE_V1_IDENTITY_SCHEMA",
    "ARTICLE_V1_TRAINING_BUDGET_POLICY",
    "CHECKPOINT_FAMILIES",
    "COMPLETE_TRAINING_SCOPE",
    "DEFAULT_CONFIG_PATHS",
    "DIFFICULTY_ORDER",
    "NATIVE_GATE_NAMES",
    "OOD_LENGTH_CHECKPOINT_FAMILY",
    "PARTIAL_SMOKE_TRAINING_SCOPE",
    "PRIMARY_SPLITS",
    "SPLIT_ORDER",
    "STANDARD_CHECKPOINT_FAMILY",
    "TRAINING_SCOPE_MODES",
    "ArticleV1Budget",
    "ArticleV1Corpus",
    "ArticleV1CorpusConfig",
    "ArticleV1CheckpointScope",
    "ArticleV1EvaluationTarget",
    "ArticleV1TargetCase",
    "DifficultySpec",
    "OODLengthSplit",
    "SplitSpec",
    "article_delta_phi",
    "build_article_v1_corpus",
    "dense_target_digest",
    "dense_unitary_digest",
    "load_article_v1_config",
    "native_gate_grammar",
]
