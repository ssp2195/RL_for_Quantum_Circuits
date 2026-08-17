from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import reporting.article_v1_campaign_projection as projection_module
from reporting.article_v1 import AppendOnlyJSONLRunStore
from reporting.article_v1_campaign_projection import (
    ARTICLE_V1_COST_PROJECTION_SCHEMA,
    IDEALIZED_PARALLEL_MODE,
    REQUIRED_CAMPAIGN_AUDIT_INTEGRITY_CHECKS,
    main as projection_main,
    project_pilot_cost,
)


def _raw_run(
    *, target_id: str, scheduler: str, expansions: int, runtime_seconds: float
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "split": "test",
        "difficulty": "easy",
        "scheduler": scheduler,
        "resource_budget": {
            "max_t_count": 4,
            "max_two_qubit_count": 4,
            "max_gates": 4,
            "max_depth": 4,
        },
        "expansion_budget": 64,
        "checkpoint_digest": "none",
        "training_seed": None,
        "evaluation_seed": 0,
        "feature_schema_version": "article-v1-31d",
        "reward_schema_version": "article-v1-expansion-potential-amended",
        "target_metric_schema_version": "projective-unitary-metrics-v2",
        "certification_schema_version": "phase-frobenius-raw-v2",
        "code_version": "pilot-commit",
        "source_worktree_digest": "sha256:pilot-worktree",
        "certified": False,
        "terminated": False,
        "truncated": True,
        "expansions": expansions,
        "runtime_seconds": runtime_seconds,
    }


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _audited_pilot(tmp_path: Path) -> tuple[Path, Path]:
    run_directory = tmp_path / "pilot"
    raw_path = run_directory / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(raw_path)
    store.append(
        _raw_run(
            target_id="pilot-a",
            scheduler="fifo",
            expansions=10,
            runtime_seconds=2.0,
        )
    )
    store.append(
        _raw_run(
            target_id="pilot-b",
            scheduler="lifo",
            expansions=20,
            runtime_seconds=2.0,
        )
    )
    store.append(
        _raw_run(
            target_id="pilot-c",
            scheduler="uniform_cost",
            expansions=40,
            runtime_seconds=2.0,
        )
    )
    checkpoints = run_directory / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "seed-19.json").write_bytes(b"{" + b"x" * 198 + b"}")
    (checkpoints / "ood-seed-19.json").write_bytes(b"{" + b"y" * 298 + b"}")

    audit = {
        "schema_version": "article-v1-campaign-audit-v1",
        "passed": True,
        "config_profile": "pilot",
        "config_digest": "sha256:pilot-config",
        "code_version": "pilot-commit",
        "source_worktree_digest": "sha256:pilot-worktree",
        "expected_run_count": 3,
        "observed_run_count": 3,
        "expected_by_split": {"test": 3, "ood_test": 0},
        "observed_by_split": {"test": 3, "ood_test": 0},
        "missing_run_keys": [],
        "unexpected_run_keys": [],
        "duplicate_run_keys": [],
        "independently_certified_success_count": 0,
        "integrity_checks": {
            name: True for name in REQUIRED_CAMPAIGN_AUDIT_INTEGRITY_CHECKS
        },
        "raw_ledger_path": "raw_runs.jsonl",
        "raw_ledger_sha256": _sha256(raw_path),
    }
    (run_directory / "campaign_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_directory, raw_path


def test_projection_is_deterministic_and_scales_the_audited_distribution(
    tmp_path: Path,
) -> None:
    run_directory, raw_path = _audited_pilot(tmp_path)
    output_path = run_directory / "pilot_cost_projection.json"

    first = project_pilot_cost(
        run_directory,
        "publication",
        observed_pilot_peak_rss_bytes=1_500_000_000,
    )
    first_bytes = output_path.read_bytes()
    second = project_pilot_cost(
        run_directory,
        "publication",
        observed_pilot_peak_rss_bytes=1_500_000_000,
    )

    assert first == second
    assert first_bytes == output_path.read_bytes()
    assert json.loads(first_bytes) == first
    assert first["schema_version"] == ARTICLE_V1_COST_PROJECTION_SCHEMA
    assert first["executes_search"] is False
    assert first["source"]["raw_ledger_sha256"] == _sha256(raw_path)
    assert first["observed_pilot"]["raw_record_count"] == 3
    assert first["observed_pilot"]["maximum_expansion_throughput"] == 20.0
    assert first["observed_pilot"]["seconds_per_expansion"]["median"] == 0.1
    assert first["observed_pilot"][
        "runtime_seconds_by_scheduler_and_difficulty"
    ]["fifo"]["easy"]["p95"] == 2.0
    assert first["projected_raw_record_count"] == 10_000
    assert first["projected_checkpoint_count"] == 25
    assert first["projected_checkpoint_count_breakdown"] == {
        "standard": 5,
        "ood_length": 5,
        "trained_ablation_variants": 15,
        "trained_ablation_ids": [
            "no_target_feature",
            "no_frontier_context",
            "no_reward_shaping",
        ],
        "total": 25,
        "reused_primary_ablations_create_new_files": False,
    }
    expected_cpu_hours = (
        first["publication_cardinalities"]["worst_case_expansion_count"]
        * 0.1
        / 3600.0
    )
    assert first["projected_cpu_hours"]["central"] == pytest.approx(
        expected_cpu_hours
    )
    assert first["projected_wall_time"]["hours"]["central"] == pytest.approx(
        expected_cpu_hours
    )
    assert first["projected_disk_use"]["bytes"]["central"] > 0
    assert first["projected_maximum_per_process_ram"]["bytes"] == 1_500_000_000


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda audit: audit.__setitem__("passed", False), "passed=true"),
        (
            lambda audit: audit["integrity_checks"].__setitem__(
                "raw_ledger_complete", False
            ),
            "integrity check",
        ),
    ),
)
def test_projection_rejects_nonpassing_audit_without_overwriting_output(
    tmp_path: Path, mutation, message: str
) -> None:
    run_directory, _raw_path = _audited_pilot(tmp_path)
    audit_path = run_directory / "campaign_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    mutation(audit)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    output = run_directory / "pilot_cost_projection.json"
    output.write_bytes(b"preserve-existing-projection")

    with pytest.raises(ValueError, match=message):
        project_pilot_cost(
            run_directory,
            "publication",
            observed_pilot_peak_rss_bytes=1,
        )
    assert output.read_bytes() == b"preserve-existing-projection"


def test_projection_rejects_ledger_changed_after_passing_audit(tmp_path: Path) -> None:
    run_directory, raw_path = _audited_pilot(tmp_path)
    with raw_path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="SHA-256"):
        project_pilot_cost(
            run_directory,
            "publication",
            observed_pilot_peak_rss_bytes=1,
        )


def test_projection_rejects_ledger_mutation_during_record_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_directory, _raw_path = _audited_pilot(tmp_path)
    original_load = AppendOnlyJSONLRunStore.load_records

    def mutating_load(self, *, repair_partial=True):
        assert repair_partial is False
        records = original_load(self, repair_partial=repair_partial)
        with self.path.open("ab") as handle:
            handle.write(b"\n")
        return records

    monkeypatch.setattr(AppendOnlyJSONLRunStore, "load_records", mutating_load)
    with pytest.raises(ValueError, match="changed during audited record loading"):
        project_pilot_cost(
            run_directory,
            "publication",
            observed_pilot_peak_rss_bytes=1,
        )
    assert not (run_directory / "pilot_cost_projection.json").exists()


def test_parallel_workers_require_an_explicit_estimate_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="idealized-parallel-estimate"):
        project_pilot_cost(
            tmp_path / "not-read",
            "publication",
            observed_pilot_peak_rss_bytes=1,
            worker_count=2,
        )


def test_module_cli_forwards_explicit_parallel_estimate_and_prints_json(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_project(run_directory, publication_config, **kwargs):
        captured.update(
            {
                "run_directory": run_directory,
                "publication_config": publication_config,
                **kwargs,
            }
        )
        return {"schema_version": ARTICLE_V1_COST_PROJECTION_SCHEMA}

    monkeypatch.setattr(projection_module, "project_pilot_cost", fake_project)
    assert projection_main(
        [
            "--pilot-run-dir",
            str(tmp_path / "pilot"),
            "--publication-config",
            "configs/article_v1_publication.json",
            "--workers",
            "4",
            "--execution-mode",
            IDEALIZED_PARALLEL_MODE,
            "--observed-pilot-peak-rss-bytes",
            "123456",
        ]
    ) == 0

    assert json.loads(capsys.readouterr().out) == {
        "schema_version": ARTICLE_V1_COST_PROJECTION_SCHEMA
    }
    assert captured["worker_count"] == 4
    assert captured["execution_mode"] == IDEALIZED_PARALLEL_MODE
    assert captured["observed_pilot_peak_rss_bytes"] == 123456
