"""Deterministic evaluator for small dense-certified synthesis instances.

Example:

``python evaluate.py --qubits 2 --target H:0,T:1,CNOT:0-1 --max-steps 100``
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Sequence

from certification.simulator import (
    SimulatorCertificationEngine,
    SynthesisTarget,
    unitary_from_gates,
)
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import Config
from enums import GateType
from env.rl_env import CircuitSynthesisEnv
from rl.policy import LinearQPolicy


def parse_target(specification: str, num_qubits: int) -> list[Gate]:
    """Parse a compact ``GATE:q[,q]`` target witness specification."""
    if not specification.strip():
        return []
    gates: list[Gate] = []
    for token in specification.split(","):
        try:
            name, operands = token.strip().upper().split(":", 1)
            gate_type = GateType[name]
            qubits = tuple(int(value) for value in operands.split("-"))
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"invalid target token {token!r}; use e.g. H:0 or CNOT:0-1"
            ) from error
        expected_arity = 2 if gate_type is GateType.CNOT else 1
        if len(qubits) != expected_arity or len(set(qubits)) != len(qubits):
            raise ValueError(f"invalid operands for {gate_type.name}: {qubits!r}")
        if any(qubit < 0 or qubit >= num_qubits for qubit in qubits):
            raise ValueError(f"target operands out of range: {qubits!r}")
        gates.append(Gate(gate_type, qubits))
    return gates


def evaluate(
    *,
    num_qubits: int,
    target_gates: Sequence[Gate],
    budget: ResourceBudget,
    max_steps: int = 100,
    seed: int | None = 0,
    scheduler: str = "fifo",
    collect_trace: bool = False,
) -> dict:
    """Run a deterministic scheduling baseline and return a JSON-ready report.

    ``collect_trace`` records compact per-expansion metrics for optional
    post-processing.  It is off by default so the original CLI output stays
    concise for ordinary evaluations.
    """
    target = SynthesisTarget(unitary_from_gates(num_qubits, target_gates))
    config = Config(
        num_qubits=num_qubits,
        budget=budget,
        max_steps=max_steps,
        # This is only the Gym adapter cap; the core archive stays dynamic.
        max_frontier=max(1, 64),
        seed=seed,
    )
    env = CircuitSynthesisEnv(config, SimulatorCertificationEngine(target))
    policy = LinearQPolicy(env.feature_dim, seed=seed)
    env.reset(seed=seed)
    terminated = env.solution_node is not None
    truncated = False
    trace: list[dict] = []

    while not (terminated or truncated):
        nodes = env.current_nodes()
        if not nodes:
            break
        if scheduler == "fifo":
            node = min(nodes, key=lambda candidate: int(candidate.record_id or 0))
        elif scheduler == "greedy":
            node = policy.select_node(nodes, epsilon=0.0)
        elif scheduler == "random":
            node = policy.select_node(nodes, epsilon=1.0)
        else:
            raise ValueError("scheduler must be one of: fifo, greedy, random")
        index = next(index for index, candidate in enumerate(nodes) if candidate is node)
        _, reward, terminated, truncated, info = env.step(index)
        if collect_trace:
            trace.append(
                {
                    "expansion": env.steps,
                    "selected_record_id": info.get("selected_record_id"),
                    "frontier_size": int(info.get("frontier_size", 0)),
                    "num_children": int(info.get("num_children", 0)),
                    "num_accepted": int(info.get("num_accepted", 0)),
                    "num_pruned": int(info.get("num_pruned", 0)),
                    "reward": float(reward),
                }
            )

    witness_actions = []
    if env.solution_node is not None:
        witness_actions = env.solution_node.reconstruct_actions()
    report = {
        "certified": env.solution_node is not None,
        "terminated": terminated,
        "truncated": truncated,
        "expansions": env.steps,
        "frontier_size": len(env.current_nodes()),
        "witness": [repr(action) for action in witness_actions],
        "witness_operations": [
            {
                "gate": action.gate_type.name,
                "qubits": list(action.qubits),
            }
            for action in witness_actions
        ],
        "scheduler": scheduler,
    }
    if collect_trace:
        report["trace"] = trace
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--target", default="", help="comma-separated GATE:operands witness")
    parser.add_argument("--max-t", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-gates", type=int, default=12)
    parser.add_argument("--max-two-qubit", type=int)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler", choices=("fifo", "greedy", "random"), default="fifo")
    args = parser.parse_args(argv)

    target_gates = parse_target(args.target, args.qubits)
    report = evaluate(
        num_qubits=args.qubits,
        target_gates=target_gates,
        budget=ResourceBudget(
            args.max_t,
            args.max_depth,
            args.max_gates,
            args.max_two_qubit,
        ),
        max_steps=args.max_steps,
        seed=args.seed,
        scheduler=args.scheduler,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["certified"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
