from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.article_native_corpus import load_article_v1_config
from experiments.article_v1_ten_minute_protocol import (
    TenMinuteConfig,
    load_ten_minute_config,
)


ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "configs/article_v1_10min_pilot.json"
TEMPLATE = ROOT / "configs/article_v1_10min_publication_template.json"


def _payload(path=PILOT):
    return deepcopy(dict(load_ten_minute_config(path).payload))


def test_v2_and_v3_use_explicit_separate_loaders():
    assert load_article_v1_config("pilot").profile == "pilot"
    with pytest.raises(ValueError, match="article-v1-corpus-config-v3"):
        load_ten_minute_config(ROOT / "configs/article_v1_pilot.json")


def test_threshold_cannot_exceed_horizon():
    payload = _payload()
    payload["budget_protocol"]["thresholds_by_difficulty"]["hard"].append(4096)
    with pytest.raises(ValueError, match="hard thresholds"):
        TenMinuteConfig.from_mapping(payload)


def test_publication_seed_counts_are_enforced():
    payload = _payload(TEMPLATE)
    payload["experiment"]["training_seeds"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="five learner seeds"):
        TenMinuteConfig.from_mapping(payload)
    payload = _payload(TEMPLATE)
    payload["experiment"]["random_scheduler_seeds"] = list(range(9))
    with pytest.raises(ValueError, match="ten random"):
        TenMinuteConfig.from_mapping(payload)


def test_hard_training_and_convergence_stopping_fail_closed():
    payload = _payload()
    payload["training_protocol"]["hard_targets_used_for_training"] = True
    with pytest.raises(ValueError, match="hard targets"):
        TenMinuteConfig.from_mapping(payload)
    payload = _payload()
    payload["training_protocol"]["convergence_stopping"] = True
    with pytest.raises(ValueError, match="convergence"):
        TenMinuteConfig.from_mapping(payload)


def test_frozen_config_has_no_candidate_grids_and_binds_provenance():
    payload = _payload(TEMPLATE)
    payload["publication_freeze"] = {
        "frozen": True,
        "source_commit": "a" * 40,
        "no_test_access": True,
        "target_ids": ["target"],
        "seeds": {"training": [1], "random_scheduler": [2]},
    }
    payload["training_protocol"]["total_expansions_per_seed"] = 20000
    payload["training_protocol"]["candidate_total_expansions_per_seed"] = []
    payload["runtime_protocol"]["candidate_hard_expansion_caps"] = []
    payload["runtime_protocol"]["selected_hard_expansion_cap"] = 1792
    assert TenMinuteConfig.from_mapping(payload, require_frozen=True).frozen

    payload["runtime_protocol"]["candidate_hard_expansion_caps"] = [1792]
    with pytest.raises(ValueError, match="unresolved candidate"):
        TenMinuteConfig.from_mapping(payload, require_frozen=True)


def test_digest_changes_with_scientific_field():
    original = TenMinuteConfig.from_mapping(_payload())
    changed = _payload()
    changed["training_protocol"]["total_expansions_per_seed"] += 1
    assert TenMinuteConfig.from_mapping(changed).digest != original.digest
