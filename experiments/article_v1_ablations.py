"""Versioned, validation-only Article V1 ablation protocol.

This module deliberately sits beside, rather than inside, the publication
runner.  It defines the ablations before outcomes are observed, fails closed
on any non-validation evaluation target, and keeps the richer historical modes
as explicitly supplementary case studies.

Calling :func:`run_article_v1_ablations` launches training/evaluation work.  No
campaign runs at import time or as part of the unit-test suite.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from benchmarks.article_native_corpus import (
    COMPLETE_TRAINING_SCOPE,
    DIFFICULTY_ORDER,
    STANDARD_CHECKPOINT_FAMILY,
    ArticleV1CheckpointScope,
    ArticleV1Corpus,
    ArticleV1EvaluationTarget,
)
from experiments.profiles import (
    ARTICLE_V1_PROFILE,
    COMPOSITE_TARGET_PROGRESS_PROFILE,
    EXTENDED_TARGET_AWARE_PROFILE,
    GHZ3_DIRECT_PROFILE,
    TOFFOLI_PARITY_PROFILE,
)
from rl.article_features import (
    ARTICLE_V1_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION,
    EXTENDED_ARTICLE_FEATURE_SCHEMA_VERSION,
)


ARTICLE_V1_ABLATION_PROTOCOL_SCHEMA_VERSION = "article-v1-ablation-protocol-v1"
ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION = "article-v1-ablation-record-v1"
ARTICLE_V1_VALIDATION_SUBSET_SCHEMA_VERSION = (
    "article-v1-preregistered-validation-subset-v1"
)
ARTICLE_V1_ABLATION_CSV_SCHEMA_VERSION = "article-v1-ablations-csv-v1"

FULL_VALIDATION_SCOPE = "all_validation_targets"
SUBSET_VALIDATION_SCOPE = "preregistered_validation_subset"
SUPPLEMENTARY_SCOPE = "supplementary_external_case_study"

REQUIRED_ABLATION_IDS = (
    "no_target_feature",
    "no_frontier_context",
    "no_reward_shaping",
    "direct_target_distance",
    "pareto_pruning_off",
    "enhanced_pauli_canonicalization_off",
)
SUPPLEMENTARY_METHOD_IDS = (
    "extended_target_aware_37d",
    "composite_target_progress",
    "ghz3_direct_protocol",
    "toffoli_parity_protocol",
)


@dataclass(frozen=True, slots=True)
class ArticleV1AblationConfig:
    """One frozen ablation configuration or supplementary registry entry."""

    ablation_id: str
    config_schema_version: str
    label: str
    role: str
    profile_name: str
    feature_schema_version: str
    feature_dimension: int | None
    reward_schema_version: str
    beta: float | None
    scheduler: str
    checkpoint_mode: str
    evaluation_scope: str
    pareto_dominance_enabled: bool
    absorb_clifford_angles: bool
    canonicalization_mode: str = "enhanced"
    direct_target_distance_primary_baseline: bool = False
    enabled_in_article_v1_protocol: bool = True

    def __post_init__(self) -> None:
        if not self.ablation_id or not self.config_schema_version:
            raise ValueError("ablation ID and configuration schema are required")
        if self.feature_dimension is not None and self.feature_dimension < 0:
            raise ValueError("feature dimension must be non-negative or None")
        if self.beta is not None and self.beta < 0.0:
            raise ValueError("reward beta must be non-negative or None")
        if self.enabled_in_article_v1_protocol:
            if self.evaluation_scope not in {
                FULL_VALIDATION_SCOPE,
                SUBSET_VALIDATION_SCOPE,
            }:
                raise ValueError("Article V1 ablations must use a validation scope")
            if self.profile_name != ARTICLE_V1_PROFILE.name:
                raise ValueError("required Article V1 ablations must retain the profile")
        elif self.evaluation_scope != SUPPLEMENTARY_SCOPE:
            raise ValueError("supplementary methods must use the supplementary scope")
        if self.canonicalization_mode not in {"enhanced", "raw_witness"}:
            raise ValueError("unsupported ablation canonicalization mode")

    def metadata(self) -> dict[str, object]:
        return {
            "protocol_schema_version": ARTICLE_V1_ABLATION_PROTOCOL_SCHEMA_VERSION,
            **asdict(self),
        }


def _article_config(
    ablation_id: str,
    *,
    label: str,
    feature_schema: str,
    feature_dimension: int,
    beta: float | None,
    scheduler: str = "article_sarsa",
    checkpoint_mode: str = "train_variant",
    scope: str = FULL_VALIDATION_SCOPE,
    pareto: bool = True,
    absorb: bool = True,
    canonicalization_mode: str = "enhanced",
    role: str = "required_ablation",
    direct_baseline: bool = False,
) -> ArticleV1AblationConfig:
    return ArticleV1AblationConfig(
        ablation_id=ablation_id,
        config_schema_version=f"article-v1-ablation-{ablation_id.replace('_', '-')}-v1",
        label=label,
        role=role,
        profile_name=ARTICLE_V1_PROFILE.name,
        feature_schema_version=feature_schema,
        feature_dimension=feature_dimension,
        reward_schema_version=ARTICLE_V1_PROFILE.reward_schema,
        beta=beta,
        scheduler=scheduler,
        checkpoint_mode=checkpoint_mode,
        evaluation_scope=scope,
        pareto_dominance_enabled=pareto,
        absorb_clifford_angles=absorb,
        canonicalization_mode=canonicalization_mode,
        direct_target_distance_primary_baseline=direct_baseline,
    )


def _supplementary_config(
    ablation_id: str,
    *,
    label: str,
    profile: object,
    feature_dimension: int | None,
) -> ArticleV1AblationConfig:
    return ArticleV1AblationConfig(
        ablation_id=ablation_id,
        config_schema_version=f"supplementary-{ablation_id.replace('_', '-')}-v1",
        label=label,
        role="supplementary_case_study",
        profile_name=str(getattr(profile, "name")),
        feature_schema_version=str(getattr(profile, "feature_schema")),
        feature_dimension=feature_dimension,
        reward_schema_version=str(getattr(profile, "reward_schema")),
        beta=None,
        scheduler="external_profile_runner",
        checkpoint_mode="external_profile",
        evaluation_scope=SUPPLEMENTARY_SCOPE,
        pareto_dominance_enabled=True,
        absorb_clifford_angles=True,
        canonicalization_mode="enhanced",
        enabled_in_article_v1_protocol=False,
    )


_REGISTRY_ENTRIES = (
    _article_config(
        "no_target_feature",
        label="Remove target process-infidelity coordinate family",
        feature_schema=ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION,
        feature_dimension=28,
        beta=1.0,
    ),
    _article_config(
        "no_frontier_context",
        label="Remove frontier z-score block",
        feature_schema=ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION,
        feature_dimension=21,
        beta=1.0,
    ),
    _article_config(
        "no_reward_shaping",
        label="Exact amended base reward with beta=0",
        feature_schema=ARTICLE_V1_FEATURE_SCHEMA_VERSION,
        feature_dimension=31,
        beta=0.0,
    ),
    _article_config(
        "direct_target_distance",
        label="Direct process-infidelity scheduling without learning",
        feature_schema="not-applicable-direct-target-distance",
        feature_dimension=0,
        beta=None,
        scheduler="article_target_distance",
        checkpoint_mode="none",
        role="primary_baseline",
        direct_baseline=True,
    ),
    _article_config(
        "pareto_pruning_off",
        label="Disable Pareto dominance pruning",
        feature_schema=ARTICLE_V1_FEATURE_SCHEMA_VERSION,
        feature_dimension=31,
        beta=1.0,
        checkpoint_mode="reuse_primary",
        scope=SUBSET_VALIDATION_SCOPE,
        pareto=False,
    ),
    _article_config(
        "enhanced_pauli_canonicalization_off",
        label="Disable enhanced Pauli canonicalization via raw DAG witness keys",
        feature_schema=ARTICLE_V1_FEATURE_SCHEMA_VERSION,
        feature_dimension=31,
        beta=1.0,
        checkpoint_mode="reuse_primary",
        scope=SUBSET_VALIDATION_SCOPE,
        absorb=False,
        canonicalization_mode="raw_witness",
    ),
    _supplementary_config(
        "extended_target_aware_37d",
        label="Extended target-aware 37-dimensional feature provider",
        profile=EXTENDED_TARGET_AWARE_PROFILE,
        feature_dimension=37,
    ),
    _supplementary_config(
        "composite_target_progress",
        label="Composite process/support/entanglement potential",
        profile=COMPOSITE_TARGET_PROGRESS_PROFILE,
        feature_dimension=None,
    ),
    _supplementary_config(
        "ghz3_direct_protocol",
        label="GHZ-3-specific direct protocol",
        profile=GHZ3_DIRECT_PROFILE,
        feature_dimension=None,
    ),
    _supplementary_config(
        "toffoli_parity_protocol",
        label="Toffoli parity-network protocol",
        profile=TOFFOLI_PARITY_PROFILE,
        feature_dimension=None,
    ),
)

ARTICLE_V1_ABLATION_REGISTRY: Mapping[str, ArticleV1AblationConfig] = MappingProxyType(
    {entry.ablation_id: entry for entry in _REGISTRY_ENTRIES}
)

if tuple(ARTICLE_V1_ABLATION_REGISTRY) != REQUIRED_ABLATION_IDS + SUPPLEMENTARY_METHOD_IDS:
    raise AssertionError("Article V1 ablation registry order drifted")
if ARTICLE_V1_ABLATION_REGISTRY["extended_target_aware_37d"].feature_schema_version != (
    EXTENDED_ARTICLE_FEATURE_SCHEMA_VERSION
):
    raise AssertionError("extended 37D supplementary schema drifted")


@dataclass(frozen=True, slots=True)
class PreregisteredValidationSubset:
    """Outcome-independent target selection committed before ablation runs."""

    schema_version: str
    corpus_config_digest: str
    selection_rule: str
    per_difficulty: int
    target_ids: tuple[str, ...]
    selection_digest: str

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_config_digest": self.corpus_config_digest,
            "selection_rule": self.selection_rule,
            "per_difficulty": self.per_difficulty,
            "target_ids": list(self.target_ids),
            "selection_digest": self.selection_digest,
            "outcomes_consulted": False,
            "permitted_split": "validation",
        }


def _require_cases_from_split(
    cases: Iterable[object],
    *,
    expected_split: str,
) -> tuple[object, ...]:
    selected = tuple(cases)
    if not selected:
        raise ValueError(f"{expected_split} cases must be nonempty")
    target_ids: set[str] = set()
    for case in selected:
        split = str(getattr(case, "split", ""))
        if split != expected_split:
            raise ValueError(
                f"{expected_split}-only protocol rejected {split!r} target; "
                "test/OOD leakage is prohibited"
            )
        target_id = str(getattr(case, "target_id", ""))
        if not target_id:
            raise ValueError("every case must expose a stable target_id")
        if target_id in target_ids:
            raise ValueError(f"duplicate target ID {target_id!r}")
        target_ids.add(target_id)
    return selected


def preregister_validation_subset(
    cases: Iterable[ArticleV1EvaluationTarget],
    *,
    corpus_config_digest: str,
    per_difficulty: int = 1,
) -> PreregisteredValidationSubset:
    """Select a frozen balanced subset using metadata-only SHA-256 ordering.

    Exactly ``per_difficulty`` targets are chosen from each Article V1
    difficulty.  The ordering depends only on the versioned rule, corpus
    configuration digest, difficulty label, and target ID; no scheduler result
    or generator witness is inspected.
    """

    validation = _require_cases_from_split(cases, expected_split="validation")
    if isinstance(per_difficulty, bool) or not isinstance(per_difficulty, int):
        raise TypeError("per_difficulty must be an integer")
    if per_difficulty < 1:
        raise ValueError("per_difficulty must be positive")
    if not corpus_config_digest:
        raise ValueError("corpus_config_digest is required")

    selected: list[object] = []
    for difficulty in DIFFICULTY_ORDER:
        group = [
            case for case in validation if getattr(case, "difficulty", None) == difficulty
        ]
        if len(group) < per_difficulty:
            raise ValueError(
                f"validation split has {len(group)} {difficulty!r} cases; "
                f"requires {per_difficulty}"
            )

        def selection_key(case: object) -> tuple[str, str]:
            target_id = str(getattr(case, "target_id"))
            payload = "\0".join(
                (
                    ARTICLE_V1_VALIDATION_SUBSET_SCHEMA_VERSION,
                    corpus_config_digest,
                    difficulty,
                    target_id,
                )
            ).encode("utf-8")
            return sha256(payload).hexdigest(), target_id

        selected.extend(sorted(group, key=selection_key)[:per_difficulty])

    target_ids = tuple(str(getattr(case, "target_id")) for case in selected)
    digest_payload = json.dumps(
        {
            "schema_version": ARTICLE_V1_VALIDATION_SUBSET_SCHEMA_VERSION,
            "corpus_config_digest": corpus_config_digest,
            "per_difficulty": per_difficulty,
            "target_ids": target_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return PreregisteredValidationSubset(
        schema_version=ARTICLE_V1_VALIDATION_SUBSET_SCHEMA_VERSION,
        corpus_config_digest=corpus_config_digest,
        selection_rule=(
            "lowest SHA-256(schema, corpus digest, difficulty, target ID); "
            "one balanced quota per difficulty"
        ),
        per_difficulty=per_difficulty,
        target_ids=target_ids,
        selection_digest=f"sha256:{sha256(digest_payload).hexdigest()}",
    )


def select_preregistered_validation_cases(
    cases: Iterable[ArticleV1EvaluationTarget],
    subset: PreregisteredValidationSubset,
) -> tuple[ArticleV1EvaluationTarget, ...]:
    validation = _require_cases_from_split(cases, expected_split="validation")
    by_id = {str(getattr(case, "target_id")): case for case in validation}
    missing = [target_id for target_id in subset.target_ids if target_id not in by_id]
    if missing:
        raise ValueError(f"preregistered validation targets are missing: {missing!r}")
    return tuple(by_id[target_id] for target_id in subset.target_ids)


def ablation_registry_records() -> list[dict[str, object]]:
    """Return JSON-ready registry records in their frozen protocol order."""

    return [entry.metadata() for entry in ARTICLE_V1_ABLATION_REGISTRY.values()]


def _validate_ablation_record(record: Mapping[str, object]) -> None:
    if record.get("schema_version") != ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported Article V1 ablation record schema")
    if record.get("split") != "validation":
        raise ValueError("ablation output accepts validation records only; test leakage rejected")
    if record.get("test_targets_observed") is not False:
        raise ValueError("ablation record must attest that no test targets were observed")
    ablation_id = str(record.get("ablation_id", ""))
    if ablation_id not in REQUIRED_ABLATION_IDS:
        raise ValueError("only required Article V1 runs belong in ablation output")


def _ablation_record(
    config: ArticleV1AblationConfig,
    raw_run: Mapping[str, object],
    *,
    checkpoint: object | None,
    subset: PreregisteredValidationSubset,
) -> dict[str, object]:
    if raw_run.get("split") != "validation":
        raise ValueError("runner returned a non-validation record; refusing ablation output")
    if str(raw_run.get("target_id", "")) == "":
        raise ValueError("runner record has no target_id")
    checkpoint_digest = (
        "none" if checkpoint is None else str(getattr(checkpoint, "weight_digest"))
    )
    selection_digest = (
        subset.selection_digest
        if config.evaluation_scope == SUBSET_VALIDATION_SCOPE
        else f"all-validation:{subset.corpus_config_digest}"
    )
    record: dict[str, object] = {
        "schema_version": ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION,
        "protocol_schema_version": ARTICLE_V1_ABLATION_PROTOCOL_SCHEMA_VERSION,
        "ablation_id": config.ablation_id,
        "ablation_config_schema_version": config.config_schema_version,
        "role": config.role,
        "target_id": str(raw_run["target_id"]),
        "split": "validation",
        "difficulty": str(raw_run.get("difficulty", "")),
        "evaluation_scope": config.evaluation_scope,
        "selection_schema_version": subset.schema_version,
        "selection_digest": selection_digest,
        "scheduler": config.scheduler,
        "feature_schema_version": config.feature_schema_version,
        "feature_dimension": config.feature_dimension,
        "reward_schema_version": config.reward_schema_version,
        "beta": config.beta,
        "exact_base_reward_only": config.beta == 0.0,
        "pareto_dominance_enabled": config.pareto_dominance_enabled,
        "absorb_clifford_angles": config.absorb_clifford_angles,
        "canonicalization_mode": config.canonicalization_mode,
        "direct_target_distance_primary_baseline": (
            config.direct_target_distance_primary_baseline
        ),
        "checkpoint_digest": checkpoint_digest,
        "training_seed": (
            None if checkpoint is None else int(getattr(checkpoint, "training_seed"))
        ),
        "evaluation_seed": int(raw_run.get("evaluation_seed", 0)),
        "certified": bool(raw_run.get("certified", False)),
        "expansions": int(raw_run.get("expansions", 0)),
        "runtime_seconds": float(raw_run.get("runtime_seconds", 0.0)),
        "test_targets_observed": False,
        "raw_run": _json_ready(raw_run),
    }
    _validate_ablation_record(record)
    return record


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_ready(item())
    return str(value)


ABLATION_CSV_COLUMNS = (
    "csv_schema_version",
    "schema_version",
    "protocol_schema_version",
    "ablation_id",
    "ablation_config_schema_version",
    "role",
    "target_id",
    "split",
    "difficulty",
    "evaluation_scope",
    "selection_schema_version",
    "selection_digest",
    "scheduler",
    "feature_schema_version",
    "feature_dimension",
    "reward_schema_version",
    "beta",
    "exact_base_reward_only",
    "pareto_dominance_enabled",
    "absorb_clifford_angles",
    "canonicalization_mode",
    "direct_target_distance_primary_baseline",
    "checkpoint_digest",
    "training_seed",
    "evaluation_seed",
    "certified",
    "expansions",
    "runtime_seconds",
    "test_targets_observed",
    "raw_run_json",
)


def write_ablations_csv(
    path: str | Path,
    records: Iterable[Mapping[str, object]],
) -> Path:
    """Write deterministic, validation-only ``ablations.csv`` rows."""

    checked = tuple(dict(record) for record in records)
    for record in checked:
        _validate_ablation_record(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_CSV_COLUMNS)
        writer.writeheader()
        for record in checked:
            row = {
                name: record.get(name)
                for name in ABLATION_CSV_COLUMNS
                if name not in {"csv_schema_version", "raw_run_json"}
            }
            row["csv_schema_version"] = ARTICLE_V1_ABLATION_CSV_SCHEMA_VERSION
            row["raw_run_json"] = json.dumps(
                _json_ready(record.get("raw_run", {})),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            writer.writerow(row)
    os.replace(temporary, destination)
    return destination


def run_article_v1_ablations(
    corpus: ArticleV1Corpus,
    *,
    output_csv: str | Path | None = None,
    checkpoint_output_dir: str | Path | None = None,
    expansion_cap: int | None = None,
    subset_per_difficulty: int = 1,
    training_seeds: Sequence[int] | None = None,
    primary_checkpoints: Sequence[object] = (),
    trainer: Callable[..., object] | None = None,
    evaluator: Callable[..., Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Execute required ablations on train/validation data only.

    ``primary_checkpoints`` may supply the already-trained 31D checkpoints used
    for the two expensive search-toggle ablations.  Missing primary seeds are
    trained from the train split.  The function is explicit because a full
    publication campaign is intentionally not a unit-test or import side effect.
    """

    if trainer is None or evaluator is None:
        from experiments.article_v1_runner import (
            evaluate_article_v1_run,
            train_article_v1_checkpoint,
        )

        trainer = train_article_v1_checkpoint if trainer is None else trainer
        evaluator = evaluate_article_v1_run if evaluator is None else evaluator
    assert trainer is not None
    assert evaluator is not None

    if expansion_cap is not None and (
        isinstance(expansion_cap, bool)
        or not isinstance(expansion_cap, int)
        or expansion_cap < 1
    ):
        raise ValueError("expansion_cap must be a positive integer or None")

    training_cases = _require_cases_from_split(
        corpus.evaluation_targets(split="train"), expected_split="train"
    )
    validation_cases = _require_cases_from_split(
        corpus.evaluation_targets(split="validation"), expected_split="validation"
    )
    subset = preregister_validation_subset(
        validation_cases,
        corpus_config_digest=corpus.config.digest,
        per_difficulty=subset_per_difficulty,
    )
    subset_cases = select_preregistered_validation_cases(validation_cases, subset)

    experiment = corpus.config.experiment
    seeds = tuple(
        int(seed)
        for seed in (
            experiment["training_seeds"] if training_seeds is None else training_seeds
        )
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be nonempty and unique")
    primary_by_seed = {
        int(getattr(checkpoint, "training_seed")): checkpoint
        for checkpoint in primary_checkpoints
    }
    saved_checkpoint_paths: list[str] = []

    def train_config(config: ArticleV1AblationConfig, seed: int) -> object:
        checkpoint = trainer(
            training_cases,
            corpus_config_digest=corpus.config.digest,
            training_seed=seed,
            episodes_per_target=int(experiment["training_episodes_per_target"]),
            learning_rate=float(experiment["learning_rate"]),
            epsilon_start=float(experiment["epsilon"]["start"]),
            epsilon_minimum=float(experiment["epsilon"]["minimum"]),
            epsilon_decay=float(experiment["epsilon"]["decay"]),
            beta=float(config.beta),
            feature_schema=config.feature_schema_version,
            checkpoint_family=STANDARD_CHECKPOINT_FAMILY,
            training_scope_mode=COMPLETE_TRAINING_SCOPE,
            expansion_cap=expansion_cap,
            certification_tolerance=float(experiment["certification_tolerance"]),
        )
        if checkpoint_output_dir is not None:
            save = getattr(checkpoint, "save", None)
            if not callable(save):
                raise TypeError("ablation checkpoint does not support save(path)")
            path = (
                Path(checkpoint_output_dir)
                / f"{config.ablation_id}-seed-{int(seed)}.json"
            )
            save(path)
            saved_checkpoint_paths.append(str(path))
        return checkpoint

    records: list[dict[str, object]] = []
    for ablation_id in REQUIRED_ABLATION_IDS:
        config = ARTICLE_V1_ABLATION_REGISTRY[ablation_id]
        cases = (
            subset_cases
            if config.evaluation_scope == SUBSET_VALIDATION_SCOPE
            else validation_cases
        )
        evaluation_beta = (
            float(experiment["beta"])
            if config.beta is None
            else float(config.beta)
        )
        checkpoint_scope = ArticleV1CheckpointScope.from_partitions(
            corpus_config_digest=corpus.config.digest,
            checkpoint_family=STANDARD_CHECKPOINT_FAMILY,
            training_scope_mode=COMPLETE_TRAINING_SCOPE,
            expected_feature_schema_version=config.feature_schema_version,
            expected_training_beta=evaluation_beta,
            expected_certification_tolerance=float(
                experiment["certification_tolerance"]
            ),
            expected_episodes_per_target=int(
                experiment["training_episodes_per_target"]
            ),
            expected_learning_rate=float(experiment["learning_rate"]),
            expected_epsilon_schedule=experiment["epsilon"],
            allowed_training_seeds=seeds,
            expected_expansion_cap=expansion_cap,
            training_cases=training_cases,
            held_out_cases=validation_cases,
            evaluation_cases=cases,
        )
        if config.checkpoint_mode == "none":
            checkpoints: tuple[object | None, ...] = (None,)
        elif config.checkpoint_mode == "reuse_primary":
            resolved: list[object] = []
            primary_config = _article_config(
                "primary_reference",
                label="Primary Article V1 checkpoint",
                feature_schema=ARTICLE_V1_FEATURE_SCHEMA_VERSION,
                feature_dimension=31,
                beta=float(experiment["beta"]),
            )
            for seed in seeds:
                checkpoint = primary_by_seed.get(seed)
                if checkpoint is None:
                    checkpoint = train_config(primary_config, seed)
                    primary_by_seed[seed] = checkpoint
                resolved.append(checkpoint)
            checkpoints = tuple(resolved)
        elif config.checkpoint_mode == "train_variant":
            checkpoints = tuple(train_config(config, seed) for seed in seeds)
        else:  # pragma: no cover - registry construction invariant
            raise AssertionError(f"unexpected checkpoint mode {config.checkpoint_mode!r}")

        for checkpoint in checkpoints:
            for case in cases:
                expansion_budget = int(case.budget.expansion_budget)
                if expansion_cap is not None:
                    expansion_budget = min(expansion_budget, expansion_cap)
                raw_run = evaluator(
                    case,
                    scheduler=config.scheduler,
                    expansion_budget=expansion_budget,
                    evaluation_seed=0,
                    checkpoint=checkpoint,
                    checkpoint_scope=checkpoint_scope,
                    beta=evaluation_beta,
                    certification_tolerance=float(
                        experiment["certification_tolerance"]
                    ),
                    canonicalization_enabled=True,
                    pareto_dominance_enabled=config.pareto_dominance_enabled,
                    absorb_clifford_angles=config.absorb_clifford_angles,
                    canonicalization_mode=config.canonicalization_mode,
                )
                records.append(
                    _ablation_record(
                        config,
                        raw_run,
                        checkpoint=checkpoint,
                        subset=subset,
                    )
                )

    if output_csv is not None:
        write_ablations_csv(output_csv, records)
    return {
        "schema_version": ARTICLE_V1_ABLATION_PROTOCOL_SCHEMA_VERSION,
        "record_schema_version": ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION,
        "corpus_config_digest": corpus.config.digest,
        "evaluation_split": "validation",
        "test_targets_observed": False,
        "validation_subset": subset.metadata(),
        "configurations": ablation_registry_records(),
        "records": records,
        "checkpoint_paths": saved_checkpoint_paths,
        "output_csv": None if output_csv is None else str(Path(output_csv)),
    }


__all__ = [
    "ABLATION_CSV_COLUMNS",
    "ARTICLE_V1_ABLATION_CSV_SCHEMA_VERSION",
    "ARTICLE_V1_ABLATION_PROTOCOL_SCHEMA_VERSION",
    "ARTICLE_V1_ABLATION_RECORD_SCHEMA_VERSION",
    "ARTICLE_V1_ABLATION_REGISTRY",
    "ARTICLE_V1_VALIDATION_SUBSET_SCHEMA_VERSION",
    "FULL_VALIDATION_SCOPE",
    "REQUIRED_ABLATION_IDS",
    "SUBSET_VALIDATION_SCOPE",
    "SUPPLEMENTARY_METHOD_IDS",
    "SUPPLEMENTARY_SCOPE",
    "ArticleV1AblationConfig",
    "PreregisteredValidationSubset",
    "ablation_registry_records",
    "preregister_validation_subset",
    "run_article_v1_ablations",
    "select_preregistered_validation_cases",
    "write_ablations_csv",
]
