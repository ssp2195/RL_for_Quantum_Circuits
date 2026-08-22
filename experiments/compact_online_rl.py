"""Thirty-CPU-minute compact online-RL frontier-ranking experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rl.compact_sarsa import FEATURE_NAMES, train_compact_online
from search.compact_parity import (
    CompactParityProblem,
    materialize_authoritative_state,
    toffoli_compact_target,
)


def _certify(solution_node: object) -> tuple[bool, dict[str, object], object]:
    from benchmarks.toffoli import toffoli_reference_unitary
    from certification.base import CertStatus
    from certification.simulator import SimulatorCertificationEngine, SynthesisTarget

    target = toffoli_compact_target()
    authoritative = materialize_authoritative_state(solution_node, target)
    certifier = SimulatorCertificationEngine(
        SynthesisTarget(toffoli_reference_unitary(), quotient_global_phase=True)
    )
    result = certifier.certify(authoritative)
    return (
        result.status is CertStatus.SUCCESS,
        {
            "status": result.status.name,
            "score": float(result.score),
            "info": dict(result.info),
            "full_gate_count": int(authoritative.num_gates),
            "full_t_count": int(authoritative.t_count),
            "full_cnot_count": int(authoritative.two_qubit_count),
            "full_depth": int(authoritative.depth),
        },
        authoritative,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Train a linear SARSA policy online to rank frontier records in "
            "the exact compact Toffoli/CCZ parity-network core."
        )
    )
    result.add_argument("--episodes", type=int, default=64)
    result.add_argument("--training-max-expansions", type=int, default=256)
    result.add_argument("--evaluation-max-expansions", type=int, default=3_000)
    result.add_argument("--checkpoint-interval", type=int, default=4)
    result.add_argument("--cpu-seconds", type=float, default=1_800.0)
    result.add_argument("--seed", type=int, default=23)
    result.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    target = toffoli_compact_target()
    problem = CompactParityProblem(target, max_cnot=6)
    training = train_compact_online(
        problem,
        episodes=args.episodes,
        training_max_expansions=args.training_max_expansions,
        evaluation_max_expansions=args.evaluation_max_expansions,
        checkpoint_interval=args.checkpoint_interval,
        cpu_limit_seconds=args.cpu_seconds,
        seed=args.seed,
    )

    certified = False
    certificate: dict[str, object] = {"status": "NOT_RUN"}
    witness: list[dict[str, object]] = []
    if training.final_evaluation.solution_node is not None:
        certified, certificate, _ = _certify(training.final_evaluation.solution_node)
        witness = [
            {
                "gate": operation.name,
                "qubits": list(operation.qubits),
                "term_index": operation.term_index,
            }
            for operation in training.final_evaluation.core_witness
        ]

    payload = {
        "schema": "compact-online-rl-report-v1",
        "problem": problem.metadata(),
        "algorithm": "linear semi-gradient SARSA",
        "action_semantics": "persistent frontier-record selection",
        "offline_dataset_used": False,
        "feature_names": list(FEATURE_NAMES),
        "completed": training.completed,
        "cpu_seconds": training.cpu_seconds,
        "episodes_requested": training.episodes_requested,
        "episodes_completed": training.episodes_completed,
        "training_successes": training.training_successes,
        "initial_greedy": {
            "success": training.initial_evaluation.success,
            "expansions": training.initial_evaluation.expansions,
        },
        "learned_greedy": {
            "success": training.final_evaluation.success,
            "expansions": training.final_evaluation.expansions,
            "generated": training.final_evaluation.generated,
            "peak_frontier": training.final_evaluation.peak_frontier,
        },
        "checkpoints": [
            {
                "episode": checkpoint.episode,
                "success": checkpoint.success,
                "expansions": checkpoint.expansions,
                "weight_norm": checkpoint.weight_norm,
            }
            for checkpoint in training.checkpoint_evaluations
        ],
        "weights": list(training.weights),
        "core_witness": witness,
        "independent_full_pipeline_certificate": certificate,
        "claim_boundary": (
            "Exact signed unit-coefficient CNOT+T/TDG phase-polynomial core "
            "with a fixed Hadamard shell; not arbitrary interleaved-H "
            "Clifford+T synthesis and not a scalability claim."
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    success = training.completed and training.final_evaluation.success and certified
    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
