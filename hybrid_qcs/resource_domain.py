"""Exact finite synthesis domains for budget-conditioned optimization.

The evaluator domain preserves the existing NCT grammar and native lowering.
The direct-phase domain is explicitly H-free at the operation level: affine
X/CNOT permutations and Z-axis pi/4 phases. Neither domain is unrestricted
Clifford+T synthesis. Clean workspace is fixed for each problem; no borrowed
workspace or cross-width semantic merging is permitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import cached_property
import hashlib
import json
from typing import Iterable

import numpy as np

from .certify import unitary_from_gates
from .model import Budget, Gate, HybridState
from .oracle_synthesis import (
    BooleanOracleSpec, OracleLayout, oracle_macro_library, x_native,
)

OBJECTIVES = ("t_count", "cnot_count", "depth", "gate_count")
CAP_FIELDS = dict(zip(OBJECTIVES, ("max_t_count", "max_cnot_count", "max_depth", "max_gates")))
PHASE_TURNS = {"T": 1, "TDG": -1, "S": 2, "SDG": -2}
SCHEMA = "qcs-resource-domain-v1"


def nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExactImage:
    """Promised-input basis images and exact eighth-root phase exponents."""
    mapping: tuple[int, ...]
    phases: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Operation:
    family: str
    qubits: tuple[int, ...]
    native: tuple[Gate, ...]

    @property
    def name(self) -> str:
        return f"{self.family}({','.join(map(str, self.qubits))})"


@dataclass(frozen=True, slots=True)
class Resources:
    t_count: int
    cnot_count: int
    gate_count: int
    wire_depths: tuple[int, ...]
    operations: int

    @classmethod
    def zero(cls, width: int) -> Resources:
        return cls(0, 0, 0, (0,) * width, 0)

    @property
    def depth(self) -> int:
        return max(self.wire_depths, default=0)

    def append(self, operation: Operation) -> Resources:
        depths = list(self.wire_depths)
        t = cnot = 0
        for gate in operation.native:
            layer = 1 + max(depths[q] for q in gate.qubits)
            for q in gate.qubits:
                depths[q] = layer
            t += int(gate.is_non_clifford)
            cnot += int(gate.is_two_qubit)
        return Resources(self.t_count + t, self.cnot_count + cnot,
                         self.gate_count + len(operation.native), tuple(depths),
                         self.operations + 1)

    def vector(self) -> tuple[int, ...]:
        # All wire depths, not just maximum depth, are continuation-relevant.
        return (self.t_count, self.cnot_count, self.gate_count,
                self.operations, *self.wire_depths)

    def within(self, problem: BoundedProblem) -> bool:
        b = problem.budget
        return (self.t_count <= b.max_t_count and self.cnot_count <= b.max_cnot_count
                and self.gate_count <= b.max_gates and self.depth <= b.max_depth
                and self.operations <= problem.max_operations)

    def to_dict(self) -> dict:
        return {**asdict(self), "depth": self.depth}


def dominates(left: Resources, right: Resources) -> bool:
    return all(a <= b for a, b in zip(left.vector(), right.vector(), strict=True))


@dataclass(frozen=True)
class BoundedProblem:
    spec: BooleanOracleSpec
    budget: Budget
    max_operations: int
    clean_workspace: int = 0
    mode: str = "evaluator"

    def __post_init__(self) -> None:
        nonnegative_integer(self.max_operations, "max_operations")
        nonnegative_integer(self.clean_workspace, "clean_workspace")
        if self.mode not in {"evaluator", "direct_phase"}:
            raise ValueError("mode must be evaluator or direct_phase")
        if self.clean_workspace > 2:
            raise ValueError("this exact small-width implementation permits 0-2 clean work wires")
        if not isinstance(self.budget, Budget):
            raise TypeError("budget must be a Budget")

    @property
    def width(self) -> int:
        return self.spec.num_inputs + int(self.mode == "evaluator") + self.clean_workspace

    @property
    def work_wires(self) -> tuple[int, ...]:
        start = self.spec.num_inputs + int(self.mode == "evaluator")
        return tuple(range(start, self.width))

    @cached_property
    def operations(self) -> tuple[Operation, ...]:
        if self.mode == "evaluator":
            layout = OracleLayout.standard(self.spec.num_inputs, self.clean_workspace)
            return tuple(Operation(m.family, m.qubits, m.native_gates)
                         for m in oracle_macro_library(layout))
        actions = []
        for q in range(self.width):
            for family in ("T", "TDG", "S", "SDG"):
                actions.append(Operation(family, (q,), (Gate(family, (q,)),)))
            actions.append(Operation("X", (q,), x_native(q)))
        for c in range(self.width):
            for t in range(self.width):
                if c != t:
                    actions.append(Operation("CNOT", (c, t), (Gate("CNOT", (c, t)),)))
        return tuple(actions)

    @cached_property
    def root(self) -> ExactImage:
        size = 1 << self.spec.num_inputs
        return ExactImage(tuple(range(size)), (0,) * size)

    @cached_property
    def goal(self) -> ExactImage:
        if self.mode == "evaluator":
            flag = 1 << self.spec.num_inputs
            mapping = tuple(x | (flag if self.spec.value(x) else 0) for x in self.root.mapping)
            return ExactImage(mapping, self.root.phases)
        return ExactImage(self.root.mapping, tuple(4 * bit for bit in self.spec.truth_table))

    def capped(self, objective: str, cap: int) -> BoundedProblem:
        if objective not in CAP_FIELDS:
            raise ValueError(f"unsupported objective {objective!r}")
        nonnegative_integer(cap, "cap")
        return replace(self, budget=replace(self.budget, **{CAP_FIELDS[objective]: cap}))

    def manifest(self) -> dict:
        return {
            "schema": SCHEMA, "mode": self.mode, "num_inputs": self.spec.num_inputs,
            "truth_table": list(self.spec.truth_table), "clean_workspace": self.clean_workspace,
            "physical_width": self.width, "phase_mode": "exact", "borrowed_ancillas": 0,
            "connectivity": "all-to-all", "budget": asdict(self.budget),
            "max_operations": self.max_operations,
            "grammar": [{"name": a.name, "native": [g.label() for g in a.native]}
                        for a in self.operations],
            "scope": ("fixed-lowering NCT evaluator; data/flag/work roles preserved"
                      if self.mode == "evaluator" else
                      "affine X/CNOT and pi/4 phase networks; not unrestricted Clifford+T"),
        }

    @cached_property
    def digest(self) -> str:
        return canonical_digest(self.manifest())


def transition(image: ExactImage, operation: Operation) -> ExactImage:
    """Exact transition; no heuristic or floating-point semantic hashing."""
    f, qs = operation.family, operation.qubits
    if f in PHASE_TURNS:
        turns = PHASE_TURNS[f]
        return ExactImage(image.mapping, tuple((p + turns * ((x >> qs[0]) & 1)) % 8
                                               for x, p in zip(image.mapping, image.phases, strict=True)))
    if f == "X":
        mapping = tuple(x ^ (1 << qs[0]) for x in image.mapping)
    elif f == "CNOT":
        mapping = tuple(x ^ ((((x >> qs[0]) & 1)) << qs[1]) for x in image.mapping)
    elif f == "TOFFOLI":
        mapping = tuple(x ^ ((((x >> qs[0]) & 1) & ((x >> qs[1]) & 1)) << qs[2])
                        for x in image.mapping)
    else:
        raise ValueError(f"unsupported exact operation {f!r}")
    return ExactImage(mapping, image.phases)


def clean_mask(problem: BoundedProblem, image: ExactImage) -> int:
    """Certifiably zero on every promised input, not merely on one sample."""
    return sum(1 << q for q in problem.work_wires
               if all(not (x & (1 << q)) for x in image.mapping))


def discrepancy(problem: BoundedProblem, image: ExactImage) -> float:
    bits = sum((a ^ b).bit_count() for a, b in zip(image.mapping, problem.goal.mapping, strict=True))
    phase = sum(min((a - b) % 8, (b - a) % 8) / 4
                for a, b in zip(image.phases, problem.goal.phases, strict=True))
    return (bits + phase) / (len(image.mapping) * (problem.width + 1))


def tokens_from_names(problem: BoundedProblem, names: Iterable[str]) -> tuple[int, ...]:
    by_name = {op.name: i for i, op in enumerate(problem.operations)}
    try:
        return tuple(by_name[name] for name in names)
    except KeyError as exc:
        raise ValueError(f"witness operation is outside this grammar: {exc.args[0]}") from exc


def certify_witness(problem: BoundedProblem, tokens: Iterable[int], *, tolerance: float = 1e-9) -> dict:
    """Independently replay the native unitary and persistent DAG of a witness.

    The evaluator is checked for BOTH flag inputs, not merely y=0. The search
    can use y=0 because the inherited grammar never controls on the flag.
    Circuit counts are literal emitted native counts, never counts over search.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError("tolerance must be finite and in (0, 1e-6]")
    witness = tuple(tokens)
    if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < len(problem.operations)
           for t in witness):
        raise ValueError("witness contains an invalid operation token")
    resources = Resources.zero(problem.width)
    image = problem.root
    native = []
    used = set()
    for token in witness:
        op = problem.operations[token]
        image = transition(image, op)
        resources = resources.append(op)
        native.extend(op.native)
        for gate in op.native:
            used.update(gate.qubits)
    exact_match = image == problem.goal
    within = resources.within(problem)
    if not within:
        return {"success": False, "reason": "resource_bound", "problem_digest": problem.digest,
                "resources": resources.to_dict()}
    full = unitary_from_gates(problem.width, native)
    logical_width = problem.spec.num_inputs + int(problem.mode == "evaluator")
    columns = 1 << logical_width
    expected = np.zeros((1 << problem.width, columns), dtype=np.complex128)
    for source in range(columns):
        x = source & ((1 << problem.spec.num_inputs) - 1)
        output = source
        coefficient = 1.0
        if problem.mode == "evaluator":
            output ^= problem.spec.value(x) << problem.spec.num_inputs
        else:
            coefficient = (-1.0) ** problem.spec.value(x)
        expected[output, source] = coefficient
    actual = full[:, :columns]
    error = float(np.max(np.abs(actual - expected)))
    dirty_rows = [row for row in range(1 << problem.width)
                  if any(row & (1 << q) for q in problem.work_wires)]
    leakage = float(np.sum(np.abs(actual[dirty_rows, :]) ** 2) / columns) if dirty_rows else 0.0
    state = HybridState.identity(problem.width, problem.budget)
    for gate in native:
        state = state.apply(gate, partial_order_reduction=False)
        if state is None:
            raise AssertionError("native replay rejected a resource-feasible witness")
    state.validate()
    dag = state.materialize_dag()
    if (state.t_count, state.cnot_count, state.gate_count, state.wire_depths) != (
            resources.t_count, resources.cnot_count, resources.gate_count, resources.wire_depths):
        raise AssertionError("macro resource accounting disagrees with native DAG replay")
    used_work = len(used.intersection(problem.work_wires))
    return {
        "success": bool(exact_match and error <= tolerance and leakage <= tolerance),
        "problem_digest": problem.digest, "exact_discrete_semantics": exact_match,
        "native_isometry_error": error, "workspace_leakage": leakage,
        "numeric_tolerance": tolerance, "dag_validated": True, "dag_nodes": len(dag.nodes),
        "resources": resources.to_dict(), "tokens": list(witness),
        "operation_witness": [problem.operations[t].name for t in witness],
        "native_witness": [g.label() for g in native],
        "available_clean_workspace": problem.clean_workspace,
        "used_clean_workspace": used_work,
        "required_logical_output_bits": int(problem.mode == "evaluator"),
        "additional_decomposition_scratch": 0,
        "auxiliary_qubits_for_this_contract": used_work,
        "auxiliary_qubits_if_wrapped_as_phase_oracle": used_work + int(problem.mode == "evaluator"),
        "scope": problem.manifest()["scope"],
    }
