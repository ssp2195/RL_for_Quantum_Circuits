"""Run the article-aligned held-out native Clifford+T experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

from benchmarks.native_corpus import build_native_target_corpus
from experiments.article_benchmark import (
    DEFAULT_SCHEDULERS,
    evaluate_native_corpus,
    run_tiny_ablations,
    select_article_policy,
)


DEFAULT_TRAINING_SEED = 20_260_815
DEFAULT_EVALUATION_SEEDS = (11, 23, 37)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Article-aligned native Clifford+T benchmark",
        "",
        f"Overall acceptance: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "The learned action selects one persistent frontier record. The native",
        "engine enumerates all legal one-gate children, and the dense certifier",
        "checks only the returned concrete DAG witness.",
        "",
        "## Held-out results",
        "",
        "| Scheduler | Successes | Runs | Success rate | Mean successful expansions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["heldout_schedulers"].items():
        mean = result["successful_expansions_mean"]
        mean_text = "n/a" if mean is None else f"{mean:.3f}"
        lines.append(
            f"| {name} | {result['successes']} | {result['runs']} | "
            f"{result['success_rate']:.3f} | {mean_text} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a held-out, bounded, no-ancilla native Clifford+T search",
            "benchmark. It is separate from the target-specific Toffoli parity",
            "normal form and from the QFT reference/AQFT capability benchmark.",
            "Generator witnesses are retained only in the corpus manifest for",
            "replay; they are never passed to search or substituted on failure.",
            "",
        ]
    )
    return "\n".join(lines)


def run_article_benchmark(
    output_dir: str | Path,
    *,
    training_seed: int = DEFAULT_TRAINING_SEED,
    evaluation_seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    episodes_per_target: int = 3,
    training_max_steps: int = 32,
    evaluation_max_steps: int = 64,
    learner: str = "sarsa",
    learning_rates: Sequence[float] = (1e-3, 5e-4),
) -> dict[str, Any]:
    """Train on the fixed train split and certify the untouched test split."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    corpus = build_native_target_corpus()
    train_cases = corpus.cases(suite="bounded_synthesis", split="train")
    validation_cases = corpus.cases(
        suite="bounded_synthesis", split="validation"
    )
    test_cases = corpus.cases(suite="bounded_synthesis", split="test")

    trained, model_selection = select_article_policy(
        train_cases,
        validation_cases,
        seed=training_seed,
        episodes_per_target=episodes_per_target,
        training_max_steps=training_max_steps,
        validation_max_steps=evaluation_max_steps,
        learners=(learner,),
        learning_rates=tuple(float(value) for value in learning_rates),
    )
    selection_path = destination / "model_selection.json"
    _write_json(selection_path, model_selection)
    heldout_path = destination / "heldout_evaluation.json"
    heldout = evaluate_native_corpus(
        test_cases,
        seeds=tuple(int(seed) for seed in evaluation_seeds),
        max_steps=evaluation_max_steps,
        schedulers=DEFAULT_SCHEDULERS,
        trained_policy=trained,
        output_path=heldout_path,
    )
    ablation_case = next(case for case in validation_cases if case.num_qubits == 2)
    ablations_path = destination / "tiny_ablations.json"
    run_tiny_ablations(
        ablation_case,
        seed=training_seed,
        max_steps=min(32, evaluation_max_steps),
        output_path=ablations_path,
    )

    manifest = corpus.manifest()
    manifest["cases"] = [case.metadata() for case in corpus.all_cases]
    manifest_path = destination / "corpus_manifest.json"
    _write_json(manifest_path, manifest)

    schedulers = heldout["schedulers"]
    fully_successful = [
        name for name, result in schedulers.items() if result["success_rate"] == 1.0
    ]
    checks = {
        "semantic_split_ids_are_unique": bool(
            manifest["target_ids_are_globally_unique"]
        ),
        "generic_search_has_no_target_specific_oracle": not bool(
            manifest["target_specific_reachability_oracle"]
        ),
        "multiple_schedulers_certify_every_heldout_run": len(fully_successful) >= 2,
        "learned_policy_certifies_every_heldout_run": schedulers["learned"][
            "success_rate"
        ]
        == 1.0,
        "reference_witness_not_used_for_search": bool(
            heldout["no_reference_witness_used_for_search"]
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "article-alignment-benchmark-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "training_seed": int(training_seed),
        "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
        "episodes_per_training_target": int(episodes_per_target),
        "training_max_steps": int(training_max_steps),
        "evaluation_max_steps": int(evaluation_max_steps),
        "learner": learner,
        "learning_rate_candidates": [float(value) for value in learning_rates],
        "heldout_schedulers": schedulers,
        "fully_successful_schedulers": fully_successful,
        "policy": trained.metadata(),
        "artifacts": {
            "heldout_evaluation": str(heldout_path.resolve()),
            "tiny_ablations": str(ablations_path.resolve()),
            "corpus_manifest": str(manifest_path.resolve()),
            "model_selection": str(selection_path.resolve()),
            "summary_json": str((destination / "summary.json").resolve()),
            "summary_markdown": str((destination / "summary.md").resolve()),
        },
        "scope": (
            "Held-out bounded native Clifford+T frontier scheduling; not "
            "unrestricted optimal synthesis, Toffoli-normal-form evidence, or QFT synthesis."
        ),
    }
    _write_json(destination / "summary.json", report)
    (destination / "summary.md").write_text(
        _summary_markdown(report), encoding="utf-8"
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    article_v1_commands = {
        "pilot",
        "generate-corpus",
        "train",
        "evaluate",
        "audit",
        "aggregate",
        "ablations",
        "mini-ci",
        "calibrate-certifier",
        "benchmark-features",
        "capture-replay-checkpoint",
        "measure-replay-timing",
        "plan",
        "validate-10min",
        "freeze-10min-protocol",
        "train-10min",
        "evaluate-10min",
        "audit-10min",
        "calibrate-10min-horizon",
        "plan-10min",
        "evaluate-cpu-budget",
        "report-10min",
    }
    if arguments and arguments[0] in article_v1_commands:
        from experiments.article_v1_runner import main as article_v1_main

        return article_v1_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs") / "article-native-heldout",
    )
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=DEFAULT_EVALUATION_SEEDS
    )
    parser.add_argument("--episodes-per-target", type=int, default=3)
    parser.add_argument("--training-max-steps", type=int, default=32)
    parser.add_argument("--evaluation-max-steps", type=int, default=64)
    parser.add_argument(
        "--learner",
        choices=("sarsa", "expected_sarsa", "contextual_bandit"),
        default="sarsa",
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=(1e-3, 5e-4)
    )
    args = parser.parse_args(arguments)
    report = run_article_benchmark(
        args.artifacts_dir,
        training_seed=args.training_seed,
        evaluation_seeds=tuple(args.evaluation_seeds),
        episodes_per_target=args.episodes_per_target,
        training_max_steps=args.training_max_steps,
        evaluation_max_steps=args.evaluation_max_steps,
        learner=args.learner,
        learning_rates=tuple(args.learning_rates),
    )
    console_summary = {
        "passed": report["passed"],
        "checks": report["checks"],
        "fully_successful_schedulers": report["fully_successful_schedulers"],
        "policy": report["policy"],
        "artifacts": report["artifacts"],
    }
    print(json.dumps(console_summary, indent=2, sort_keys=True, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
