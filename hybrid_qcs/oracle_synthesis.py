r"""Hierarchical exact synthesis of small Boolean verification phase oracles.

The learned search does not receive a target-specific circuit decomposition.  It
searches for a reversible predicate evaluator over a generic NOT/CNOT/Toffoli
macro grammar.  Outer linear SARSA selects a persistent frontier record and a
disjoint linear LinUCB policy ranks that record's still-pending macro
continuations.  Macro semantics and contract-relative canonicalization are exact.

Once an evaluator ``U_g`` has been independently certified, the phase oracle is
constructed by the universal compute-phase-uncompute identity

    O_g = U_g^\dagger Z_f U_g,

where ``f`` is the predicate flag.  Every macro is then lowered to the unchanged
native Clifford+T grammar and passed through the repository's strengthened
Clifford-tableau/Pauli-rotation canonicalizer before independent clean-ancilla
isometry certification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from itertools import combinations
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from .ancilla_benchmarks import toffoli_decomposition
from .ancilla_certify import AncillaCertificationResult, certify_ancilla_state
from .ancilla_contract import AncillaContract, AncillaSynthesisTarget, PhaseMode
from .certify import unitary_from_gates
from .model import Budget, Gate, HybridState, INVERSE_GATE


MACRO_FAMILIES = ("X", "CNOT", "TOFFOLI")


@dataclass(frozen=True, slots=True)
class BooleanOracleSpec:
    """One exact Boolean predicate supplied by its complete truth table.

    Qubit ``q`` stores input bit ``x_{q+1}``; consequently q0 is the least
    significant computational-basis bit even though human-readable bit strings
    are written as ``x1 x2 ... xn``.
    """

    name: str
    num_inputs: int
    truth_table: tuple[int, ...]
    marked_bitstrings: tuple[str, ...] = ()
    application: str = "Boolean phase oracle"

    def __post_init__(self) -> None:
        if isinstance(self.num_inputs, bool) or not isinstance(self.num_inputs, int):
            raise TypeError("num_inputs must be an integer")
        if self.num_inputs <= 0 or self.num_inputs > 3:
            raise ValueError("the current exact oracle experiment supports 1-3 inputs")
        table = tuple(int(value) for value in self.truth_table)
        if len(table) != 1 << self.num_inputs:
            raise ValueError("truth table has the wrong length")
        if any(value not in (0, 1) for value in table):
            raise ValueError("truth table entries must be Boolean")
        object.__setattr__(self, "truth_table", table)
        object.__setattr__(self, "marked_bitstrings", tuple(self.marked_bitstrings))

    @classmethod
    def from_marked_bitstrings(
        cls,
        name: str,
        num_inputs: int,
        marked_bitstrings: Iterable[str],
        *,
        application: str = "marked-state phase oracle",
    ) -> "BooleanOracleSpec":
        marked = tuple(str(bits) for bits in marked_bitstrings)
        table = [0] * (1 << num_inputs)
        for bitstring in marked:
            if len(bitstring) != num_inputs or set(bitstring) - {"0", "1"}:
                raise ValueError("marked bit strings must have exactly num_inputs bits")
            # Human order is x1,x2,...; q0 stores x1 and is the basis LSB.
            basis = sum((bit == "1") << position for position, bit in enumerate(bitstring))
            table[basis] = 1
        return cls(
            name=name,
            num_inputs=num_inputs,
            truth_table=tuple(table),
            marked_bitstrings=marked,
            application=application,
        )

    @property
    def marked_inputs(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.truth_table) if value)

    @property
    def truth_mask(self) -> int:
        return sum(value << index for index, value in enumerate(self.truth_table))

    @property
    def unique_marked_input(self) -> int | None:
        return self.marked_inputs[0] if len(self.marked_inputs) == 1 else None

    def value(self, basis: int) -> int:
        return self.truth_table[int(basis)]

    def phase_unitary(self) -> np.ndarray:
        diagonal = np.asarray(
            [(-1.0 if value else 1.0) for value in self.truth_table],
            dtype=np.complex128,
        )
        return np.diag(diagonal)


@dataclass(frozen=True, slots=True)
class OracleLayout:
    """Read-only data, one predicate flag, and fixed clean work qubits."""

    data_qubits: tuple[int, ...]
    flag_qubit: int
    work_qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        data = tuple(self.data_qubits)
        work = tuple(self.work_qubits)
        object.__setattr__(self, "data_qubits", data)
        object.__setattr__(self, "work_qubits", work)
        flat = (*data, self.flag_qubit, *work)
        if len(flat) != len(set(flat)):
            raise ValueError("oracle wire roles must be disjoint")
        if set(flat) != set(range(len(flat))):
            raise ValueError("oracle layouts must use contiguous physical wire indices")

    @classmethod
    def standard(cls, num_inputs: int, num_work: int) -> "OracleLayout":
        return cls(
            data_qubits=tuple(range(num_inputs)),
            flag_qubit=num_inputs,
            work_qubits=tuple(range(num_inputs + 1, num_inputs + 1 + num_work)),
        )

    @property
    def total_qubits(self) -> int:
        return len(self.data_qubits) + 1 + len(self.work_qubits)

    def data_basis_to_physical(self, basis: int) -> int:
        physical = 0
        for position, qubit in enumerate(self.data_qubits):
            physical |= ((basis >> position) & 1) << qubit
        return physical


@dataclass(frozen=True, slots=True)
class OracleMacro:
    family: str
    qubits: tuple[int, ...]
    native_gates: tuple[Gate, ...]

    def __post_init__(self) -> None:
        if self.family not in MACRO_FAMILIES:
            raise ValueError(f"unsupported oracle macro family {self.family!r}")
        object.__setattr__(self, "qubits", tuple(self.qubits))
        object.__setattr__(self, "native_gates", tuple(self.native_gates))

    @property
    def name(self) -> str:
        return f"{self.family}({','.join(str(q) for q in self.qubits)})"

    @property
    def t_count(self) -> int:
        return sum(gate.is_non_clifford for gate in self.native_gates)

    @property
    def cnot_count(self) -> int:
        return sum(gate.is_two_qubit for gate in self.native_gates)

    @property
    def gate_count(self) -> int:
        return len(self.native_gates)

    def sort_key(self) -> tuple[int, tuple[int, ...]]:
        return MACRO_FAMILIES.index(self.family), self.qubits


def x_native(qubit: int) -> tuple[Gate, ...]:
    """Exact X = H S^2 H in the frozen native grammar."""

    return (
        Gate("H", (qubit,)),
        Gate("S", (qubit,)),
        Gate("S", (qubit,)),
        Gate("H", (qubit,)),
    )


def inverse_native_gates(gates: Sequence[Gate]) -> tuple[Gate, ...]:
    return tuple(
        Gate(INVERSE_GATE[gate.name], gate.qubits)
        for gate in reversed(tuple(gates))
    )


@lru_cache(maxsize=None)
def oracle_macro_library(layout: OracleLayout) -> tuple[OracleMacro, ...]:
    """Generic read-only-input NCT grammar for one-output Boolean evaluators.

    Data lines may be temporarily complemented, but CNOT and Toffoli targets are
    restricted to the flag or work register.  This is target-independent and is
    sufficient for arbitrary three-input predicates with the declared workspace
    budget, although finite search remains exponential.
    """

    actions: list[OracleMacro] = []
    # NOT on data supports negative literals; NOT on the flag supports constants.
    for qubit in (*layout.data_qubits, layout.flag_qubit):
        actions.append(OracleMacro("X", (qubit,), x_native(qubit)))

    controls = (*layout.data_qubits, *layout.work_qubits)
    targets = (layout.flag_qubit, *layout.work_qubits)
    for control in controls:
        for target in targets:
            if control == target:
                continue
            actions.append(
                OracleMacro(
                    "CNOT",
                    (control, target),
                    (Gate("CNOT", (control, target)),),
                )
            )

    for target in targets:
        controls_for_target = tuple(control for control in controls if control != target)
        for control0, control1 in combinations(controls_for_target, 2):
            actions.append(
                OracleMacro(
                    "TOFFOLI",
                    (control0, control1, target),
                    toffoli_decomposition(control0, control1, target),
                )
            )
    actions.sort(key=OracleMacro.sort_key)
    return tuple(actions)


def apply_macro_to_basis(basis: int, macro: OracleMacro) -> int:
    if macro.family == "X":
        return int(basis) ^ (1 << macro.qubits[0])
    if macro.family == "CNOT":
        control, target = macro.qubits
        return int(basis) ^ (
            (1 << target) if ((int(basis) >> control) & 1) else 0
        )
    control0, control1, target = macro.qubits
    enabled = ((int(basis) >> control0) & 1) and ((int(basis) >> control1) & 1)
    return int(basis) ^ ((1 << target) if enabled else 0)


def apply_macro_to_mapping(mapping: Sequence[int], macro: OracleMacro) -> tuple[int, ...]:
    return tuple(apply_macro_to_basis(basis, macro) for basis in mapping)


def update_wire_depths(
    wire_depths: Sequence[int],
    native_gates: Sequence[Gate],
) -> tuple[tuple[int, ...], int]:
    depths = list(int(value) for value in wire_depths)
    for gate in native_gates:
        layer = 1 + max(depths[qubit] for qubit in gate.qubits)
        for qubit in gate.qubits:
            depths[qubit] = layer
    return tuple(depths), max(depths, default=0)


@dataclass(frozen=True)
class OracleEvaluatorTarget:
    spec: BooleanOracleSpec
    layout: OracleLayout
    budget: Budget
    max_macros: int
    family: str = "generic-reversible-boolean-evaluator"

    @cached_property
    def root_mapping(self) -> tuple[int, ...]:
        return tuple(
            self.layout.data_basis_to_physical(basis)
            for basis in range(1 << self.spec.num_inputs)
        )

    @cached_property
    def target_mapping(self) -> tuple[int, ...]:
        flag = 1 << self.layout.flag_qubit
        return tuple(
            physical | (flag if self.spec.value(basis) else 0)
            for basis, physical in enumerate(self.root_mapping)
        )

    @cached_property
    def target_truth_mask(self) -> int:
        return self.spec.truth_mask

    def mismatch_components(self, mapping: Sequence[int]) -> tuple[int, int, int, int]:
        """Return total bit mismatch, flag mismatch, data corruption, dirty work."""

        total = flag = data = work = 0
        data_mask = sum(1 << qubit for qubit in self.layout.data_qubits)
        work_mask = sum(1 << qubit for qubit in self.layout.work_qubits)
        flag_mask = 1 << self.layout.flag_qubit
        for current, expected in zip(mapping, self.target_mapping, strict=True):
            delta = int(current) ^ int(expected)
            total += delta.bit_count()
            flag += int(bool(delta & flag_mask))
            data += (delta & data_mask).bit_count()
            work += (int(current) & work_mask).bit_count()
        return total, flag, data, work

    def distance(self, mapping: Sequence[int]) -> float:
        total, flag, data, work = self.mismatch_components(mapping)
        samples = 1 << self.spec.num_inputs
        normalizer = max(1, samples * self.layout.total_qubits)
        # Exact target-relative potential; all coefficients are fixed globally.
        return (
            4.0 * flag + 2.0 * data + 2.0 * work + float(total)
        ) / (9.0 * normalizer)


@dataclass(frozen=True, slots=True)
class OracleResourceState:
    t_count: int
    cnot_count: int
    gate_count: int
    wire_depths: tuple[int, ...]
    macro_count: int

    @classmethod
    def zero(cls, total_qubits: int) -> "OracleResourceState":
        return cls(0, 0, 0, tuple(0 for _ in range(total_qubits)), 0)

    @property
    def depth(self) -> int:
        return max(self.wire_depths, default=0)

    def append(self, macro: OracleMacro) -> "OracleResourceState":
        depths, _ = update_wire_depths(self.wire_depths, macro.native_gates)
        return OracleResourceState(
            t_count=self.t_count + macro.t_count,
            cnot_count=self.cnot_count + macro.cnot_count,
            gate_count=self.gate_count + macro.gate_count,
            wire_depths=depths,
            macro_count=self.macro_count + 1,
        )

    def within(self, target: OracleEvaluatorTarget) -> bool:
        budget = target.budget
        return bool(
            self.t_count <= budget.max_t_count
            and self.cnot_count <= budget.max_cnot_count
            and self.gate_count <= budget.max_gates
            and self.depth <= budget.max_depth
            and self.macro_count <= target.max_macros
        )

    def vector(self) -> tuple[int, ...]:
        return (
            self.t_count,
            self.cnot_count,
            self.gate_count,
            *self.wire_depths,
            self.macro_count,
        )


def _weakly_dominates(left: Sequence[int], right: Sequence[int]) -> bool:
    return all(int(a) <= int(b) for a, b in zip(left, right, strict=True))


def _strictly_dominates(left: Sequence[int], right: Sequence[int]) -> bool:
    return _weakly_dominates(left, right) and any(
        int(a) < int(b) for a, b in zip(left, right, strict=True)
    )


@dataclass(slots=True)
class OracleRecord:
    record_id: int
    mapping: tuple[int, ...]
    resources: OracleResourceState
    macro_witness: tuple[int, ...]
    pending_mask: int
    distance: float
    mismatch_total: int
    flag_mismatch: int
    data_mismatch: int
    workspace_dirty: int
    allocations: int = 0
    context_cache: dict[int, np.ndarray] = field(default_factory=dict)
    outer_feature_cache: np.ndarray | None = None


@dataclass(slots=True)
class OracleSearchProfile:
    attempted_edges: int = 0
    accepted_records: int = 0
    duplicate_or_dominated: int = 0
    dominance_comparisons: int = 0
    outer_rows_scored: int = 0
    inner_rows_scored: int = 0
    context_rows_built: int = 0
    frontier_peak: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "attempted_edges": self.attempted_edges,
            "accepted_records": self.accepted_records,
            "duplicate_or_dominated": self.duplicate_or_dominated,
            "dominance_comparisons": self.dominance_comparisons,
            "outer_rows_scored": self.outer_rows_scored,
            "inner_rows_scored": self.inner_rows_scored,
            "context_rows_built": self.context_rows_built,
            "frontier_peak": self.frontier_peak,
        }


@dataclass(frozen=True, slots=True)
class OracleStep:
    selected_record_id: int
    tokens: tuple[int, ...]
    reward: float
    attempted_edges: int
    accepted: int
    terminated: bool
    truncated: bool
    solution_record_id: int | None


class OracleMacroDeferredSearch:
    """Fair deferred exact search over a target-independent NCT macro grammar."""

    def __init__(
        self,
        target: OracleEvaluatorTarget,
        *,
        max_allocations: int = 4_096,
        max_edges: int = 16_384,
        batch_size: int = 2,
        fairness_start_k: int = 32,
        success_bonus: float = 20.0,
        failure_penalty: float = 20.0,
        shaping_weight: float = 4.0,
        edge_cost_weight: float = 0.015,
    ) -> None:
        self.target = target
        self.actions = oracle_macro_library(target.layout)
        self.max_allocations = int(max_allocations)
        self.max_edges = int(max_edges)
        self.batch_size = int(batch_size)
        self.fairness_start_k = int(fairness_start_k)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.shaping_weight = float(shaping_weight)
        self.edge_cost_weight = float(edge_cost_weight)
        self.records: dict[int, OracleRecord] = {}
        self.frontier: dict[int, OracleRecord] = {}
        self.pareto: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
        self.next_record_id = 0
        self.allocations = 0
        self.edge_attempts = 0
        self.solution_record_id: int | None = None
        self.profile = OracleSearchProfile()
        self.reset()

    def reset(self) -> None:
        self.records.clear()
        self.frontier.clear()
        self.pareto.clear()
        self.next_record_id = 0
        self.allocations = 0
        self.edge_attempts = 0
        self.solution_record_id = None
        self.profile = OracleSearchProfile()
        resources = OracleResourceState.zero(self.target.layout.total_qubits)
        root = self._new_record(self.target.root_mapping, resources, ())
        self.frontier[root.record_id] = root
        self.pareto[root.mapping] = [(root.resources.vector(), root.record_id)]
        self.profile.frontier_peak = 1
        if root.mapping == self.target.target_mapping:
            self.solution_record_id = root.record_id

    def _legal_mask(self, record: OracleRecord) -> int:
        if record.resources.macro_count >= self.target.max_macros:
            return 0
        mask = 0
        last_token = record.macro_witness[-1] if record.macro_witness else None
        last_macro = None if last_token is None else self.actions[last_token]
        for token, macro in enumerate(self.actions):
            if last_token == token:
                # Every macro in the declared grammar is self-inverse.
                continue
            if last_macro is not None:
                if set(last_macro.qubits).isdisjoint(macro.qubits) and macro.sort_key() < last_macro.sort_key():
                    continue
            if record.resources.append(macro).within(self.target):
                mask |= 1 << token
        return mask

    def _new_record(
        self,
        mapping: Sequence[int],
        resources: OracleResourceState,
        witness: Sequence[int],
    ) -> OracleRecord:
        mapping_tuple = tuple(int(value) for value in mapping)
        total, flag, data, work = self.target.mismatch_components(mapping_tuple)
        record = OracleRecord(
            record_id=self.next_record_id,
            mapping=mapping_tuple,
            resources=resources,
            macro_witness=tuple(int(token) for token in witness),
            pending_mask=0,
            distance=self.target.distance(mapping_tuple),
            mismatch_total=total,
            flag_mismatch=flag,
            data_mismatch=data,
            workspace_dirty=work,
        )
        self.next_record_id += 1
        record.pending_mask = self._legal_mask(record)
        self.records[record.record_id] = record
        return record

    def open_records(self) -> tuple[OracleRecord, ...]:
        return tuple(self.frontier.values())

    def pending_tokens(self, record: OracleRecord) -> tuple[int, ...]:
        return tuple(
            token
            for token in range(len(self.actions))
            if record.pending_mask & (1 << token)
        )

    def frontier_potential(self) -> float:
        if self.solution_record_id is not None:
            return 0.0
        if not self.frontier:
            return -2.0
        return -min(record.distance for record in self.frontier.values())

    def _insert(
        self,
        mapping: Sequence[int],
        resources: OracleResourceState,
        witness: Sequence[int],
    ) -> OracleRecord | None:
        key = tuple(mapping)
        vector = resources.vector()
        group = self.pareto.setdefault(key, [])
        for existing, _ in group:
            self.profile.dominance_comparisons += 1
            if _weakly_dominates(existing, vector):
                self.profile.duplicate_or_dominated += 1
                return None
        survivors: list[tuple[tuple[int, ...], int]] = []
        for existing, record_id in group:
            self.profile.dominance_comparisons += 1
            if _strictly_dominates(vector, existing):
                self.frontier.pop(record_id, None)
            else:
                survivors.append((existing, record_id))
        record = self._new_record(mapping, resources, witness)
        survivors.append((vector, record.record_id))
        self.pareto[key] = survivors
        if record.pending_mask:
            self.frontier[record.record_id] = record
        self.profile.accepted_records += 1
        self.profile.frontier_peak = max(self.profile.frontier_peak, len(self.frontier))
        return record

    def _forced_fair_edge(self) -> tuple[int, int] | None:
        next_allocation = self.allocations + 1
        root = math.isqrt(next_allocation)
        if root < self.fairness_start_k or root * root != next_allocation:
            return None
        for record_id in sorted(self.frontier):
            tokens = self.pending_tokens(self.frontier[record_id])
            if tokens:
                return record_id, tokens[0]
        return None

    def process_batch(
        self,
        record_id: int,
        tokens: Sequence[int],
        *,
        allow_fairness_override: bool = True,
    ) -> OracleStep:
        if record_id not in self.frontier:
            raise KeyError("selected record is not active")
        if self.solution_record_id is not None:
            raise RuntimeError("cannot expand after a solution has been found")
        forced = self._forced_fair_edge() if allow_fairness_override else None
        if forced is not None:
            record_id, forced_token = forced
            tokens = (forced_token,)
        selected = self.frontier[record_id]
        before = self.frontier_potential()
        attempted = accepted = 0
        executed: list[int] = []
        total_native_cost = 0
        for token_value in tuple(tokens)[: self.batch_size]:
            if self.edge_attempts >= self.max_edges:
                break
            token = int(token_value)
            bit = 1 << token
            if not selected.pending_mask & bit:
                continue
            selected.pending_mask &= ~bit
            selected.context_cache.pop(token, None)
            macro = self.actions[token]
            attempted += 1
            self.edge_attempts += 1
            self.profile.attempted_edges += 1
            total_native_cost += macro.gate_count
            mapping = apply_macro_to_mapping(selected.mapping, macro)
            resources = selected.resources.append(macro)
            if not resources.within(self.target):
                continue
            child = self._insert(
                mapping,
                resources,
                (*selected.macro_witness, token),
            )
            executed.append(token)
            if child is not None:
                accepted += 1
                if child.mapping == self.target.target_mapping:
                    self.solution_record_id = child.record_id
                    break
        selected.allocations += 1
        self.allocations += 1
        if selected.pending_mask == 0:
            self.frontier.pop(selected.record_id, None)
        after = self.frontier_potential()
        reward = self.shaping_weight * (after - before)
        reward -= self.edge_cost_weight * total_native_cost / max(1, self.target.budget.max_gates)
        terminated = self.solution_record_id is not None or not self.frontier
        truncated = self.allocations >= self.max_allocations or self.edge_attempts >= self.max_edges
        if self.solution_record_id is not None:
            reward += self.success_bonus
        elif terminated or truncated:
            reward -= self.failure_penalty
        return OracleStep(
            selected_record_id=record_id,
            tokens=tuple(executed),
            reward=float(reward),
            attempted_edges=attempted,
            accepted=accepted,
            terminated=terminated,
            truncated=truncated,
            solution_record_id=self.solution_record_id,
        )


ORACLE_OUTER_FEATURE_NAMES = (
    "bias",
    "distance",
    "flag_mismatch_fraction",
    "data_mismatch_fraction",
    "workspace_dirty_fraction",
    "t_fraction",
    "cnot_fraction",
    "gate_fraction",
    "depth_fraction",
    "macro_fraction",
    "pending_fraction",
    "last_X",
    "last_CNOT",
    "last_TOFFOLI",
    "zero_literal_prepared_fraction",
)
ORACLE_OUTER_FEATURE_DIM = len(ORACLE_OUTER_FEATURE_NAMES)


def _zero_literal_prepared_fraction(record: OracleRecord, target: OracleEvaluatorTarget) -> float:
    marked = target.spec.unique_marked_input
    if marked is None:
        return 0.0
    zero_positions = [
        position
        for position in range(target.spec.num_inputs)
        if ((marked >> position) & 1) == 0
    ]
    if not zero_positions:
        return 1.0
    prepared = 0
    for position in zero_positions:
        qubit = target.layout.data_qubits[position]
        if all(
            ((output >> qubit) & 1) == (1 ^ ((basis >> position) & 1))
            for basis, output in enumerate(record.mapping)
        ):
            prepared += 1
    return prepared / len(zero_positions)


def oracle_outer_features(
    record: OracleRecord,
    target: OracleEvaluatorTarget,
) -> np.ndarray:
    if record.outer_feature_cache is None:
        budget = target.budget
        samples = 1 << target.spec.num_inputs
        workspace_slots = max(1, samples * max(1, len(target.layout.work_qubits)))
        last_family = (
            None
            if not record.macro_witness
            else oracle_macro_library(target.layout)[record.macro_witness[-1]].family
        )
        record.outer_feature_cache = np.asarray(
            [
                1.0,
                min(2.0, record.distance),
                record.flag_mismatch / max(1, samples),
                record.data_mismatch / max(1, samples * target.spec.num_inputs),
                record.workspace_dirty / workspace_slots,
                record.resources.t_count / max(1, budget.max_t_count),
                record.resources.cnot_count / max(1, budget.max_cnot_count),
                record.resources.gate_count / max(1, budget.max_gates),
                record.resources.depth / max(1, budget.max_depth),
                record.resources.macro_count / max(1, target.max_macros),
                0.0,  # pending fraction is the only mutable coordinate
                float(last_family == "X"),
                float(last_family == "CNOT"),
                float(last_family == "TOFFOLI"),
                _zero_literal_prepared_fraction(record, target),
            ],
            dtype=np.float64,
        )
    values = record.outer_feature_cache.copy()
    values[10] = record.pending_mask.bit_count() / max(
        1, len(oracle_macro_library(target.layout))
    )
    if values.shape != (ORACLE_OUTER_FEATURE_DIM,):
        raise AssertionError("oracle outer feature schema mismatch")
    return values


@dataclass(slots=True)
class LinearOracleOuterSarsa:
    learning_rate: float = 0.01
    gamma: float = 1.0
    seed: int = 11
    clip: float = 50.0
    weights: np.ndarray = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    updates: int = field(init=False, default=0)
    rows_scored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.weights = np.zeros(ORACLE_OUTER_FEATURE_DIM, dtype=np.float64)
        # A mathematically transparent warm start; SARSA remains free to update it.
        self.weights[1] = -1.0
        self.weights[2] = -0.5
        self.weights[3] = -0.5
        self.weights[4] = -0.5
        self.weights[7] = -0.05
        self.weights[8] = -0.05
        self.rng = np.random.default_rng(self.seed)

    def choose(
        self,
        records: Sequence[OracleRecord],
        target: OracleEvaluatorTarget,
        epsilon: float,
        *,
        profile: OracleSearchProfile | None = None,
    ) -> tuple[int, np.ndarray, float]:
        if not records:
            raise RuntimeError("cannot choose from an empty oracle frontier")
        features = np.vstack([oracle_outer_features(record, target) for record in records])
        scores = features @ self.weights
        self.rows_scored += len(records)
        if profile is not None:
            profile.outer_rows_scored += len(records)
        if self.rng.random() < epsilon:
            index = int(self.rng.integers(len(records)))
        else:
            maximum = float(np.max(scores))
            candidates = np.flatnonzero(np.isclose(scores, maximum, atol=1e-12, rtol=0.0))
            index = min(candidates, key=lambda i: records[int(i)].record_id)
            index = int(index)
        return records[index].record_id, features[index].copy(), float(scores[index])

    def frontier_value(
        self,
        records: Sequence[OracleRecord],
        target: OracleEvaluatorTarget,
    ) -> float:
        if not records:
            return -20.0
        return max(float(oracle_outer_features(record, target) @ self.weights) for record in records)

    def update(
        self,
        features: np.ndarray,
        q_value: float,
        reward: float,
        next_q: float | None,
        *,
        duration: int,
    ) -> None:
        bootstrap = 0.0 if next_q is None else (self.gamma ** max(1, duration)) * next_q
        delta = float(reward) + bootstrap - float(q_value)
        self.weights += self.learning_rate * delta * np.asarray(features, dtype=np.float64)
        np.clip(self.weights, -self.clip, self.clip, out=self.weights)
        self.updates += 1


ORACLE_INNER_CONTEXT_NAMES = (
    "bias",
    "current_distance",
    "projected_distance",
    "distance_reduction",
    "projected_flag_mismatch_fraction",
    "projected_data_mismatch_fraction",
    "projected_workspace_dirty_fraction",
    "native_gate_cost_fraction",
    "native_t_cost_fraction",
    "native_cnot_cost_fraction",
    "target_is_flag",
    "target_is_workspace",
    "x_on_required_zero_literal",
    "controls_active_on_unique_marked_input",
    "operand_overlap_with_last_macro",
)
ORACLE_INNER_CONTEXT_DIM = len(ORACLE_INNER_CONTEXT_NAMES)


def oracle_inner_context(
    record: OracleRecord,
    macro: OracleMacro,
    target: OracleEvaluatorTarget,
) -> np.ndarray:
    projected = apply_macro_to_mapping(record.mapping, macro)
    total, flag, data, work = target.mismatch_components(projected)
    projected_distance = target.distance(projected)
    samples = 1 << target.spec.num_inputs
    workspace_slots = max(1, samples * max(1, len(target.layout.work_qubits)))
    macro_target = macro.qubits[-1]
    marked = target.spec.unique_marked_input
    x_on_zero = 0.0
    active_controls = 0.0
    if macro.family == "X" and macro.qubits[0] in target.layout.data_qubits and marked is not None:
        position = target.layout.data_qubits.index(macro.qubits[0])
        x_on_zero = float(((marked >> position) & 1) == 0)
    if macro.family in {"CNOT", "TOFFOLI"} and marked is not None:
        controls = macro.qubits[:-1]
        values: list[int] = []
        root_basis = target.layout.data_basis_to_physical(marked)
        current_marked_output = record.mapping[marked]
        for control in controls:
            # Evaluate controls in the current image of the marked input.  This
            # naturally accounts for temporary literal complementation.
            values.append((current_marked_output >> control) & 1)
        active_controls = float(all(values))
    last_macro = (
        None
        if not record.macro_witness
        else oracle_macro_library(target.layout)[record.macro_witness[-1]]
    )
    overlap = (
        0.0
        if last_macro is None
        else len(set(last_macro.qubits).intersection(macro.qubits))
        / max(1, len(set(macro.qubits)))
    )
    values = np.asarray(
        [
            1.0,
            record.distance,
            projected_distance,
            record.distance - projected_distance,
            flag / max(1, samples),
            data / max(1, samples * target.spec.num_inputs),
            work / workspace_slots,
            macro.gate_count / max(1, target.budget.max_gates),
            macro.t_count / max(1, target.budget.max_t_count),
            macro.cnot_count / max(1, target.budget.max_cnot_count),
            float(macro_target == target.layout.flag_qubit),
            float(macro_target in target.layout.work_qubits),
            x_on_zero,
            active_controls,
            overlap,
        ],
        dtype=np.float64,
    )
    if values.shape != (ORACLE_INNER_CONTEXT_DIM,):
        raise AssertionError("oracle inner context schema mismatch")
    return values


@dataclass(slots=True)
class DisjointOracleLinUCB:
    alpha: float = 0.6
    regularization: float = 1.0
    a_inverse: dict[str, np.ndarray] = field(init=False, repr=False)
    b_vector: dict[str, np.ndarray] = field(init=False, repr=False)
    updates: int = field(init=False, default=0)
    rows_scored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        self.a_inverse = {}
        self.b_vector = {}

    def _ensure(self, family: str) -> None:
        if family not in self.a_inverse:
            self.a_inverse[family] = np.eye(ORACLE_INNER_CONTEXT_DIM) / self.regularization
            self.b_vector[family] = np.zeros(ORACLE_INNER_CONTEXT_DIM, dtype=np.float64)

    def posterior_mean(self, family: str) -> np.ndarray:
        self._ensure(family)
        return self.a_inverse[family] @ self.b_vector[family]

    def choose(
        self,
        record: OracleRecord,
        tokens: Sequence[int],
        actions: Sequence[OracleMacro],
        target: OracleEvaluatorTarget,
        *,
        explore: bool,
        profile: OracleSearchProfile | None = None,
    ) -> tuple[int, np.ndarray, float, str]:
        best: tuple[float, int, int, np.ndarray, str] | None = None
        for token_value in tokens:
            token = int(token_value)
            macro = actions[token]
            context = record.context_cache.get(token)
            if context is None:
                context = oracle_inner_context(record, macro, target)
                record.context_cache[token] = context
                if profile is not None:
                    profile.context_rows_built += 1
            family = macro.family
            mean = float(context @ self.posterior_mean(family))
            bonus = 0.0
            if explore:
                inverse = self.a_inverse[family]
                bonus = self.alpha * math.sqrt(max(0.0, float(context @ inverse @ context)))
            candidate = (mean + bonus, -token, token, context, family)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("cannot choose from an empty macro continuation set")
        self.rows_scored += len(tokens)
        if profile is not None:
            profile.inner_rows_scored += len(tokens)
        return best[2], best[3].copy(), best[0], best[4]

    def rank(
        self,
        record: OracleRecord,
        tokens: Sequence[int],
        actions: Sequence[OracleMacro],
        target: OracleEvaluatorTarget,
        *,
        limit: int,
        profile: OracleSearchProfile | None = None,
    ) -> tuple[int, ...]:
        scored: list[tuple[float, int]] = []
        for token_value in tokens:
            token = int(token_value)
            macro = actions[token]
            context = record.context_cache.get(token)
            if context is None:
                context = oracle_inner_context(record, macro, target)
                record.context_cache[token] = context
                if profile is not None:
                    profile.context_rows_built += 1
            scored.append((float(context @ self.posterior_mean(macro.family)), token))
        scored.sort(key=lambda item: (-item[0], item[1]))
        self.rows_scored += len(scored)
        if profile is not None:
            profile.inner_rows_scored += len(scored)
        return tuple(token for _, token in scored[: max(0, int(limit))])

    def update(self, family: str, context: np.ndarray, reward: float) -> None:
        self._ensure(family)
        vector = np.asarray(context, dtype=np.float64)
        inverse = self.a_inverse[family]
        projected = inverse @ vector
        denominator = 1.0 + float(vector @ projected)
        self.a_inverse[family] = inverse - np.outer(projected, projected) / denominator
        self.b_vector[family] += float(reward) * vector
        self.updates += 1


def _epsilon(episode: int, episodes: int) -> float:
    fraction = episode / max(1, episodes - 1)
    return 0.35 + fraction * (0.04 - 0.35)


def _deterministic_training_tokens(
    environment: OracleMacroDeferredSearch,
    record: OracleRecord,
    *,
    limit: int,
) -> tuple[int, ...]:
    scored = [
        (
            oracle_inner_context(record, environment.actions[token], environment.target)[2],
            token,
        )
        for token in environment.pending_tokens(record)
    ]
    # Lower projected distance first, then deterministic token order.
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(token for _, token in scored[:limit])


def train_oracle_outer_sarsa(
    targets: Sequence[OracleEvaluatorTarget],
    *,
    episodes: int = 80,
    seed: int = 11,
    max_allocations: int = 1_024,
    batch_size: int = 2,
    policy: LinearOracleOuterSarsa | None = None,
    bandit: DisjointOracleLinUCB | None = None,
) -> LinearOracleOuterSarsa:
    if not targets:
        raise ValueError("outer training requires at least one oracle target")
    policy = LinearOracleOuterSarsa(seed=seed) if policy is None else policy
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = OracleMacroDeferredSearch(
            target,
            max_allocations=max_allocations,
            max_edges=max_allocations * batch_size,
            batch_size=batch_size,
            fairness_start_k=10_000,
        )
        epsilon = _epsilon(episode, episodes)
        if environment.solution_record_id is not None:
            continue
        record_id, features, q_value = policy.choose(
            environment.open_records(), target, epsilon, profile=environment.profile
        )
        while True:
            record = environment.frontier[record_id]
            if bandit is None:
                tokens = _deterministic_training_tokens(environment, record, limit=batch_size)
            else:
                tokens = bandit.rank(
                    record,
                    environment.pending_tokens(record),
                    environment.actions,
                    target,
                    limit=batch_size,
                    profile=environment.profile,
                )
            step = environment.process_batch(record_id, tokens, allow_fairness_override=False)
            done = step.terminated or step.truncated
            if done:
                policy.update(features, q_value, step.reward, None, duration=max(1, step.attempted_edges))
                break
            next_id, next_features, next_q = policy.choose(
                environment.open_records(), target, epsilon, profile=environment.profile
            )
            policy.update(
                features,
                q_value,
                step.reward,
                next_q,
                duration=max(1, step.attempted_edges),
            )
            record_id, features, q_value = next_id, next_features, next_q
    return policy


def train_oracle_inner_bandit(
    outer: LinearOracleOuterSarsa,
    targets: Sequence[OracleEvaluatorTarget],
    *,
    episodes: int = 120,
    alpha: float = 0.6,
    max_allocations: int = 1_024,
    bandit: DisjointOracleLinUCB | None = None,
) -> DisjointOracleLinUCB:
    if not targets:
        raise ValueError("inner training requires at least one oracle target")
    bandit = DisjointOracleLinUCB(alpha=alpha) if bandit is None else bandit
    bandit.alpha = float(alpha)
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = OracleMacroDeferredSearch(
            target,
            max_allocations=max_allocations,
            max_edges=max_allocations,
            batch_size=1,
            fairness_start_k=10_000,
        )
        while environment.frontier and environment.solution_record_id is None:
            record_id, _, _ = outer.choose(
                environment.open_records(), target, 0.0, profile=environment.profile
            )
            record = environment.frontier[record_id]
            tokens = environment.pending_tokens(record)
            if not tokens:
                environment.frontier.pop(record.record_id, None)
                continue
            token, context, _, family = bandit.choose(
                record,
                tokens,
                environment.actions,
                target,
                explore=True,
                profile=environment.profile,
            )
            before_value = outer.frontier_value(environment.open_records(), target)
            before_distance = record.distance
            step = environment.process_batch(record_id, (token,), allow_fairness_override=False)
            after_value = (
                20.0
                if environment.solution_record_id is not None
                else outer.frontier_value(environment.open_records(), target)
            )
            child_witness = (*record.macro_witness, token)
            child_distance = min(
                (
                    candidate.distance
                    for candidate in environment.records.values()
                    if candidate.macro_witness == child_witness
                ),
                default=before_distance,
            )
            immediate_progress = before_distance - child_distance
            macro = environment.actions[token]
            reward = immediate_progress + 0.05 * (after_value - before_value)
            reward -= 0.01 * macro.gate_count / max(1, target.budget.max_gates)
            if step.accepted == 0:
                reward -= 0.03
            if step.solution_record_id is not None:
                reward += 2.0
            bandit.update(family, context, reward)
            if step.terminated or step.truncated:
                break
    return bandit


@dataclass(frozen=True, slots=True)
class EvaluatorCertification:
    success: bool
    mapping_match: bool
    native_replay_match: bool
    projective_error: float
    workspace_leakage: float
    native_gate_count: int
    t_count: int
    cnot_count: int
    depth: int
    native_witness: tuple[str, ...]


def lower_macro_witness(
    target: OracleEvaluatorTarget,
    macro_tokens: Sequence[int],
) -> HybridState:
    state = HybridState.identity(target.layout.total_qubits, target.budget)
    actions = oracle_macro_library(target.layout)
    for token in macro_tokens:
        for gate in actions[int(token)].native_gates:
            child = state.apply(gate, partial_order_reduction=False)
            if child is None:
                raise RuntimeError("a certified macro witness exceeded the native budget")
            state = child
    return state


def evaluator_target_isometry(target: OracleEvaluatorTarget) -> np.ndarray:
    matrix = np.zeros(
        (1 << target.layout.total_qubits, 1 << target.spec.num_inputs),
        dtype=np.complex128,
    )
    for column, output in enumerate(target.target_mapping):
        matrix[output, column] = 1.0
    return matrix


def certify_evaluator_witness(
    target: OracleEvaluatorTarget,
    macro_tokens: Sequence[int],
    *,
    tolerance: float = 1e-9,
) -> tuple[HybridState, EvaluatorCertification]:
    state = lower_macro_witness(target, macro_tokens)
    gates = state.reconstruct_gates()
    full = unitary_from_gates(target.layout.total_qubits, gates)
    input_matrix = np.zeros(
        (1 << target.layout.total_qubits, 1 << target.spec.num_inputs),
        dtype=np.complex128,
    )
    for column, basis in enumerate(target.root_mapping):
        input_matrix[basis, column] = 1.0
    actual = full @ input_matrix
    expected = evaluator_target_isometry(target)
    overlap = np.vdot(expected.ravel(), actual.ravel())
    phase = 1.0 + 0.0j if abs(overlap) < tolerance else overlap / abs(overlap)
    projective_error = float(np.max(np.abs(actual - phase * expected)))
    valid_rows = set(target.target_mapping)
    leakage = float(
        sum(
            abs(actual[row, column]) ** 2
            for column in range(actual.shape[1])
            for row in range(actual.shape[0])
            if row not in valid_rows
        )
        / actual.shape[1]
    )
    mapping_match = projective_error <= tolerance
    # Rebuild the same native path to ensure persistent-DAG replay stability.
    replay = HybridState.identity(target.layout.total_qubits, target.budget)
    for gate in gates:
        child = replay.apply(gate, partial_order_reduction=False)
        if child is None:
            raise AssertionError("oracle evaluator native replay failed")
        replay = child
    replay_match = (
        replay.canonical_key == state.canonical_key
        and replay.resource_vector() == state.resource_vector()
    )
    return state, EvaluatorCertification(
        success=bool(mapping_match and replay_match and leakage <= tolerance),
        mapping_match=bool(mapping_match),
        native_replay_match=bool(replay_match),
        projective_error=projective_error,
        workspace_leakage=leakage,
        native_gate_count=state.gate_count,
        t_count=state.t_count,
        cnot_count=state.cnot_count,
        depth=state.depth,
        native_witness=tuple(gate.label() for gate in gates),
    )


def _phase_target_from_spec(
    target: OracleEvaluatorTarget,
    phase_gates: Sequence[Gate],
) -> AncillaSynthesisTarget:
    contract = AncillaContract(
        total_qubits=target.layout.total_qubits,
        logical_qubits=target.layout.data_qubits,
        clean_ancillas=(target.layout.flag_qubit, *target.layout.work_qubits),
        borrowed_ancillas=(),
        phase_mode=PhaseMode.EXACT,
    )
    logical = target.spec.phase_unitary()
    reference = unitary_from_gates(target.layout.total_qubits, phase_gates)
    evaluator_budget = target.budget
    phase_budget = Budget(
        max_t_count=2 * evaluator_budget.max_t_count,
        max_cnot_count=2 * evaluator_budget.max_cnot_count,
        max_gates=2 * evaluator_budget.max_gates + 2,
        max_depth=2 * evaluator_budget.max_depth + 2,
    )
    state = HybridState.identity(target.layout.total_qubits, phase_budget)
    for gate in phase_gates:
        child = state.apply(gate, partial_order_reduction=False)
        if child is None:
            raise RuntimeError("phase-oracle construction exceeded the native budget")
        state = child
    target_isometry = contract.target_isometry(logical)
    digest = json.dumps(
        {
            "name": target.spec.name,
            "truth_table": target.spec.truth_table,
            "layout": {
                "data": target.layout.data_qubits,
                "flag": target.layout.flag_qubit,
                "work": target.layout.work_qubits,
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    import hashlib

    return AncillaSynthesisTarget(
        name=f"phase-{target.spec.name}",
        split="test",
        contract=contract,
        budget=phase_budget,
        logical_unitary=logical,
        target_isometry=target_isometry,
        canonical_key=state.canonical_key,
        tableau_payload=state.tableau.canonical_payload(),
        rotation_payloads=tuple(rotation.canonical_payload() for rotation in state.rotations),
        reference_unitary=reference,
        generator_length=len(tuple(phase_gates)),
        target_digest=hashlib.sha256(digest).hexdigest(),
        family="verification-phase-oracle",
        convention=(
            "q0 stores x1 as the least-significant input bit; the predicate flag "
            "and all work qubits are clean and restored"
        ),
    )


def assemble_phase_oracle(
    target: OracleEvaluatorTarget,
    evaluator_state: HybridState,
    *,
    tolerance: float = 1e-9,
) -> tuple[HybridState, AncillaSynthesisTarget, AncillaCertificationResult]:
    evaluator_gates = evaluator_state.reconstruct_gates()
    phase_gates = (
        *evaluator_gates,
        Gate("S", (target.layout.flag_qubit,)),
        Gate("S", (target.layout.flag_qubit,)),
        *inverse_native_gates(evaluator_gates),
    )
    phase_target = _phase_target_from_spec(target, phase_gates)
    state = HybridState.identity(target.layout.total_qubits, phase_target.budget)
    for gate in phase_gates:
        child = state.apply(gate, partial_order_reduction=False)
        if child is None:
            raise RuntimeError("assembled phase oracle exceeded its native budget")
        state = child
    certification = certify_ancilla_state(phase_target, state, tolerance=tolerance)
    return state, phase_target, certification


@dataclass(frozen=True, slots=True)
class OracleSynthesisResult:
    success: bool
    stop_reason: str
    target: str
    wall_seconds: float
    cpu_seconds: float
    training_seconds: float
    allocations: int
    attempted_macro_edges: int
    frontier_peak: int
    outer_updates: int
    inner_updates: int
    macro_witness: tuple[str, ...]
    evaluator_native_gate_count: int
    evaluator_t_count: int
    evaluator_cnot_count: int
    evaluator_depth: int
    phase_oracle_native_gate_count: int
    phase_oracle_t_count: int
    phase_oracle_cnot_count: int
    phase_oracle_depth: int
    phase_oracle_projective_error: float | None
    phase_oracle_exact_error: float | None
    phase_oracle_leakage: float | None
    phase_oracle_witness: tuple[str, ...]
    profile: dict[str, int]
    trace: tuple[dict[str, float | int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "stop_reason": self.stop_reason,
            "target": self.target,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "training_seconds": self.training_seconds,
            "allocations": self.allocations,
            "attempted_macro_edges": self.attempted_macro_edges,
            "frontier_peak": self.frontier_peak,
            "outer_updates": self.outer_updates,
            "inner_updates": self.inner_updates,
            "macro_witness": list(self.macro_witness),
            "evaluator": {
                "native_gate_count": self.evaluator_native_gate_count,
                "t_count": self.evaluator_t_count,
                "cnot_count": self.evaluator_cnot_count,
                "depth": self.evaluator_depth,
            },
            "phase_oracle": {
                "native_gate_count": self.phase_oracle_native_gate_count,
                "t_count": self.phase_oracle_t_count,
                "cnot_count": self.phase_oracle_cnot_count,
                "depth": self.phase_oracle_depth,
                "projective_error": self.phase_oracle_projective_error,
                "exact_error": self.phase_oracle_exact_error,
                "ancilla_leakage": self.phase_oracle_leakage,
                "witness": list(self.phase_oracle_witness),
            },
            "profile": dict(self.profile),
            "trace": list(self.trace),
        }


def evaluate_oracle_hierarchy(
    outer: LinearOracleOuterSarsa,
    bandit: DisjointOracleLinUCB,
    target: OracleEvaluatorTarget,
    *,
    max_allocations: int = 8_192,
    max_edges: int = 16_384,
    batch_size: int = 2,
    wall_limit: float = 30.0,
    training_seconds: float = 0.0,
) -> OracleSynthesisResult:
    environment = OracleMacroDeferredSearch(
        target,
        max_allocations=max_allocations,
        max_edges=max_edges,
        batch_size=batch_size,
        fairness_start_k=32,
    )
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    trace: list[dict[str, float | int]] = []
    stop_reason = "frontier_exhausted"
    while environment.frontier and environment.solution_record_id is None:
        if environment.allocations >= max_allocations:
            stop_reason = "allocation_cap"
            break
        if environment.edge_attempts >= max_edges:
            stop_reason = "edge_cap"
            break
        if time.perf_counter() - wall_start >= wall_limit:
            stop_reason = "wall_limit"
            break
        record_id, _, _ = outer.choose(
            environment.open_records(), target, 0.0, profile=environment.profile
        )
        record = environment.frontier[record_id]
        tokens = bandit.rank(
            record,
            environment.pending_tokens(record),
            environment.actions,
            target,
            limit=batch_size,
            profile=environment.profile,
        )
        step = environment.process_batch(record_id, tokens)
        trace.append(
            {
                "allocation": environment.allocations,
                "attempted_edges": environment.edge_attempts,
                "frontier_size": len(environment.frontier),
                "best_distance": -environment.frontier_potential(),
                "selected_distance": record.distance,
            }
        )
        if step.terminated or step.truncated:
            break

    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    if environment.solution_record_id is None:
        return OracleSynthesisResult(
            success=False,
            stop_reason=stop_reason,
            target=target.spec.name,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            training_seconds=training_seconds,
            allocations=environment.allocations,
            attempted_macro_edges=environment.edge_attempts,
            frontier_peak=environment.profile.frontier_peak,
            outer_updates=outer.updates,
            inner_updates=bandit.updates,
            macro_witness=(),
            evaluator_native_gate_count=0,
            evaluator_t_count=0,
            evaluator_cnot_count=0,
            evaluator_depth=0,
            phase_oracle_native_gate_count=0,
            phase_oracle_t_count=0,
            phase_oracle_cnot_count=0,
            phase_oracle_depth=0,
            phase_oracle_projective_error=None,
            phase_oracle_exact_error=None,
            phase_oracle_leakage=None,
            phase_oracle_witness=(),
            profile=environment.profile.to_dict(),
            trace=tuple(trace),
        )

    solution = environment.records[environment.solution_record_id]
    evaluator_state, evaluator_cert = certify_evaluator_witness(
        target, solution.macro_witness
    )
    phase_state, _, phase_cert = assemble_phase_oracle(target, evaluator_state)
    success = bool(evaluator_cert.success and phase_cert.success)
    return OracleSynthesisResult(
        success=success,
        stop_reason="certified" if success else "certification_failed",
        target=target.spec.name,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        training_seconds=training_seconds,
        allocations=environment.allocations,
        attempted_macro_edges=environment.edge_attempts,
        frontier_peak=environment.profile.frontier_peak,
        outer_updates=outer.updates,
        inner_updates=bandit.updates,
        macro_witness=tuple(environment.actions[token].name for token in solution.macro_witness),
        evaluator_native_gate_count=evaluator_cert.native_gate_count,
        evaluator_t_count=evaluator_cert.t_count,
        evaluator_cnot_count=evaluator_cert.cnot_count,
        evaluator_depth=evaluator_cert.depth,
        phase_oracle_native_gate_count=phase_state.gate_count,
        phase_oracle_t_count=phase_state.t_count,
        phase_oracle_cnot_count=phase_state.cnot_count,
        phase_oracle_depth=phase_state.depth,
        phase_oracle_projective_error=phase_cert.projective_isometry_error,
        phase_oracle_exact_error=phase_cert.exact_isometry_error,
        phase_oracle_leakage=phase_cert.ancilla_leakage,
        phase_oracle_witness=phase_cert.witness,
        profile=environment.profile.to_dict(),
        trace=tuple(trace),
    )


def save_oracle_policies(
    path: str | Path,
    outer: LinearOracleOuterSarsa,
    bandit: DisjointOracleLinUCB,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"outer_weights": outer.weights}
    for family in sorted(bandit.a_inverse):
        arrays[f"a_inverse__{family}"] = bandit.a_inverse[family]
        arrays[f"b_vector__{family}"] = bandit.b_vector[family]
    np.savez_compressed(output, **arrays)


__all__ = [
    "BooleanOracleSpec",
    "DisjointOracleLinUCB",
    "EvaluatorCertification",
    "LinearOracleOuterSarsa",
    "OracleEvaluatorTarget",
    "OracleLayout",
    "OracleMacro",
    "OracleMacroDeferredSearch",
    "OracleResourceState",
    "OracleSynthesisResult",
    "apply_macro_to_basis",
    "apply_macro_to_mapping",
    "assemble_phase_oracle",
    "certify_evaluator_witness",
    "evaluate_oracle_hierarchy",
    "inverse_native_gates",
    "lower_macro_witness",
    "oracle_inner_context",
    "oracle_macro_library",
    "oracle_outer_features",
    "save_oracle_policies",
    "train_oracle_inner_bandit",
    "train_oracle_outer_sarsa",
    "x_native",
]
