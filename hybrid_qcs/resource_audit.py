"""Policy-independent bounded audits and checkable finite closed-cover proofs.

An infeasibility certificate is a resource-labelled inductive over-approximation:
root coverage, absence of target labels, and coverage of every legal successor.
The checker does not trust the audit's queue, scheduler, visitation counters,
learned scores, or an alleged timeout. Its transition and cost replay is separate
from the learned engine. A certificate proves ONLY its complete domain manifest.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable

from .model import Budget
from .oracle_synthesis import BooleanOracleSpec
from .resource_domain import (
    BoundedProblem, ExactImage, Resources, canonical_digest, certify_witness,
    dominates, nonnegative_integer,
)
from .resource_search import SearchResult, WorkLimits, WorkMeter

CERTIFICATE_SCHEMA = "qcs-resource-closed-cover-v1"


def reference_successor(problem, image, resources, token):
    """Scalar reference replay, independent of transition()/Resources.append()."""
    op = problem.operations[token]
    out, phases = [], []
    for column in range(len(image.mapping)):
        state, phase = image.mapping[column], image.phases[column]
        wires = op.qubits
        if op.family in {"T", "TDG", "S", "SDG"}:
            exponent = {"T": 1, "TDG": 7, "S": 2, "SDG": 6}[op.family]
            if state & (1 << wires[0]):
                phase = (phase + exponent) % 8
        else:
            target = wires[-1]
            enabled = all(state & (1 << q) for q in wires[:-1])
            if enabled:
                state ^= 1 << target
        out.append(state)
        phases.append(phase)
    depths = list(resources.wire_depths)
    t_count, cnot_count, gate_count = resources.t_count, resources.cnot_count, resources.gate_count
    for gate in op.native:
        next_layer = max(depths[q] for q in gate.qubits) + 1
        for q in gate.qubits:
            depths[q] = next_layer
        t_count += gate.name in {"T", "TDG"}
        cnot_count += gate.name == "CNOT"
        gate_count += 1
    result = Resources(t_count, cnot_count, gate_count, tuple(depths), resources.operations + 1)
    return ExactImage(tuple(out), tuple(phases)), result


def _encoded_label(image, resources):
    return {"mapping": list(image.mapping), "phases": list(image.phases),
            "resources": asdict(resources)}


def _certificate(problem, groups, labels):
    cover = [_encoded_label(labels[rid][0], labels[rid][1])
             for image in sorted(groups, key=lambda x: (x.mapping, x.phases))
             for rid in sorted(groups[image], key=lambda i: labels[i][1].vector())]
    payload = {"schema": CERTIFICATE_SCHEMA, "claim": "infeasible",
               "problem": problem.manifest(), "problem_digest": problem.digest, "cover": cover}
    return {**payload, "payload_sha256": canonical_digest(payload)}


def audit_bound(problem: BoundedProblem, *, limits: WorkLimits = WorkLimits(),
                cancel: Callable[[], bool] | None = None) -> SearchResult:
    """Complete finite exploration when limits allow; otherwise returns unknown."""
    meter = WorkMeter(limits, cancel)
    zero = Resources.zero(problem.width)
    # Keep persistent parents even when a label is dominated subsequently.
    labels = [(problem.root, zero, None, None)]
    groups = {problem.root: [0]}
    pending = deque([0])

    def finish(status, reason, witness=None, proof=None):
        return SearchResult(status, reason, problem.digest, witness, meter.edges,
                            len(labels), meter.wall, meter.cpu, proof=proof)

    def witness(rid):
        path = []
        while labels[rid][2] is not None:
            path.append(labels[rid][3])
            rid = labels[rid][2]
        return tuple(reversed(path))

    while pending:
        rid = pending.popleft()
        image, resources, _, _ = labels[rid]
        if rid not in groups[image]:
            continue
        if image == problem.goal:
            cert = certify_witness(problem, witness(rid))
            return finish("feasible" if cert["success"] else "unknown",
                          "independent_witness" if cert["success"] else "native_certification_failed", cert)
        reason = meter.reason()
        if reason:
            return finish("unknown", reason)
        if resources.operations >= problem.max_operations:
            continue
        for token in range(len(problem.operations)):
            reason = meter.reason()
            if reason:
                return finish("unknown", reason)
            successor, cost = reference_successor(problem, image, resources, token)
            # Count every reference edge/cap check, including infeasible ones.
            meter.edges += 1
            if not cost.within(problem):
                continue
            group = groups.get(successor, [])
            if any(dominates(labels[i][1], cost) for i in group):
                continue
            if len(labels) >= limits.max_records:
                return finish("unknown", "record_limit")
            survivors = [i for i in group if not dominates(cost, labels[i][1])]
            child = len(labels)
            labels.append((successor, cost, rid, token))
            groups[successor] = [*survivors, child]
            pending.append(child)
            if successor == problem.goal:
                cert = certify_witness(problem, witness(child))
                return finish("feasible" if cert["success"] else "unknown",
                              "independent_witness" if cert["success"] else "native_certification_failed", cert)
    # A checker, not this return flag, is required before accepting the proof.
    return finish("infeasible", "complete_closed_cover", proof=_certificate(problem, groups, labels))


def problem_from_manifest(manifest: dict) -> BoundedProblem:
    """Rebuild only the supported fixed grammar; do not execute supplied code."""
    if not isinstance(manifest, dict):
        raise ValueError("domain manifest must be an object")
    n = nonnegative_integer(manifest["num_inputs"], "num_inputs")
    if not 1 <= n <= 3:
        raise ValueError("supported truth tables have 1-3 inputs")
    truth = manifest["truth_table"]
    if not isinstance(truth, list) or len(truth) != 1 << n or any(type(x) is not int or x not in (0, 1) for x in truth):
        raise ValueError("invalid exact truth table")
    problem = BoundedProblem(BooleanOracleSpec("certificate-target", n, tuple(truth)),
                             Budget(**manifest["budget"]), manifest["max_operations"],
                             manifest["clean_workspace"], manifest["mode"])
    if problem.manifest() != manifest:
        raise ValueError("unsupported or modified domain/grammar manifest")
    return problem


def _decode_label(label, problem):
    if set(label) != {"mapping", "phases", "resources"}:
        raise ValueError("invalid label fields")
    mapping, phases = label["mapping"], label["phases"]
    size = 1 << problem.spec.num_inputs
    if (len(mapping) != size or len(phases) != size or len(set(mapping)) != size or
            any(type(x) is not int or not 0 <= x < 1 << problem.width for x in mapping) or
            any(type(x) is not int or not 0 <= x < 8 for x in phases)):
        raise ValueError("invalid exact state label")
    data = label["resources"]
    if set(data) != {"t_count", "cnot_count", "gate_count", "wire_depths", "operations"}:
        raise ValueError("invalid resource fields")
    for name in ("t_count", "cnot_count", "gate_count", "operations"):
        nonnegative_integer(data[name], name)
    depths = data["wire_depths"]
    if len(depths) != problem.width:
        raise ValueError("wrong wire-depth dimension")
    for x in depths:
        nonnegative_integer(x, "wire depth")
    cost = Resources(data["t_count"], data["cnot_count"], data["gate_count"], tuple(depths), data["operations"])
    if not cost.within(problem):
        raise ValueError("certificate label exceeds its declared resource bounds")
    return ExactImage(tuple(mapping), tuple(phases)), cost


def verify_closed_cover(certificate: dict, expected_problem: BoundedProblem | None = None, *,
                        limits: WorkLimits = WorkLimits(1_000_000, 100_000, 60., 60.),
                        cancel: Callable[[], bool] | None = None) -> dict:
    """Check the inductive invariant; failure/interruption never proves absence."""
    meter = WorkMeter(limits, cancel)

    def outcome(valid, reason, digest=None):
        return {"valid": valid, "reason": reason, "problem_digest": digest,
                "edges_checked": meter.edges, "wall_seconds": meter.wall, "cpu_seconds": meter.cpu}

    try:
        if certificate.get("schema") != CERTIFICATE_SCHEMA or certificate.get("claim") != "infeasible":
            return outcome(False, "not_an_infeasibility_certificate")
        payload = {k: v for k, v in certificate.items() if k != "payload_sha256"}
        if canonical_digest(payload) != certificate.get("payload_sha256"):
            return outcome(False, "payload_digest_mismatch")
        problem = problem_from_manifest(certificate["problem"])
        if certificate["problem_digest"] != problem.digest:
            return outcome(False, "domain_digest_mismatch")
        if expected_problem is not None and expected_problem.digest != problem.digest:
            return outcome(False, "wrong_problem_scope", problem.digest)
        cover = certificate["cover"]
        if not isinstance(cover, list) or not cover or len(cover) > limits.max_records:
            return outcome(False, "invalid_or_oversized_cover", problem.digest)
        groups = {}
        for item in cover:
            if meter.reason():
                return outcome(False, meter.reason(), problem.digest)
            image, cost = _decode_label(item, problem)
            if image == problem.goal:
                return outcome(False, "target_in_cover", problem.digest)
            groups.setdefault(image, []).append(cost)
        zero = Resources.zero(problem.width)
        if not any(dominates(cost, zero) for cost in groups.get(problem.root, ())):
            return outcome(False, "root_not_covered", problem.digest)
        for image, costs in groups.items():
            for cost in costs:
                if cost.operations >= problem.max_operations:
                    continue
                for token in range(len(problem.operations)):
                    reason = meter.reason()
                    if reason:
                        return outcome(False, reason, problem.digest)
                    child, resource = reference_successor(problem, image, cost, token)
                    meter.edges += 1
                    if not resource.within(problem):
                        continue
                    if not any(dominates(old, resource) for old in groups.get(child, ())):
                        return outcome(False, "successor_not_covered", problem.digest)
        return outcome(True, "inductive_closed_cover_verified", problem.digest)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        return outcome(False, f"invalid_certificate: {exc}")


def write_certificate(path: Path, certificate: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(certificate, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
