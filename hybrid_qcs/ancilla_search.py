"""Hierarchical exact search for fixed-pool ancilla-assisted Clifford+T circuits.

Outer semi-gradient SARSA allocates a frontier record.  A disjoint linear
LinUCB policy ranks that record's still-pending native continuations.  All gate
semantics, canonicalization, Pareto pruning, ancilla-contract checks, and final
certification remain deterministic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import math
import time
from typing import Sequence

import numpy as np

from .ancilla_certify import AncillaCertificationResult, certify_ancilla_state
from .ancilla_contract import (
    AncillaSynthesisTarget,
    contract_distance_and_leakage,
)
from .model import Gate, HybridState, INVERSE_GATE, TransitionProfile, generate_gates
from .pauli import Pauli, conjugate_by_gate
from .search import _strictly_dominates, _weakly_dominates, symbolic_distance_components


OUTER_FEATURE_NAMES = (
    "bias",
    "t_fraction",
    "cnot_fraction",
    "gate_fraction",
    "depth_fraction",
    "logical_depth_mean",
    "workspace_depth_mean",
    "rotation_fraction",
    "anticommuting_pair_fraction",
    "mean_pauli_weight_fraction",
    "tableau_mismatch_fraction",
    "rotation_sequence_mismatch_fraction",
    "rotation_multiset_mismatch_fraction",
    "symbolic_distance_fraction",
    "contract_distance",
    "ancilla_leakage",
    "logical_rotation_support_fraction",
    "workspace_rotation_support_fraction",
    "ancilla_touch_fraction",
    "logical_workspace_cnot_fraction",
    "pending_fraction",
    "last_H",
    "last_S",
    "last_SDG",
    "last_T",
    "last_TDG",
    "last_CNOT",
    "logical_register_fraction",
    "workspace_fraction",
)
OUTER_FEATURE_DIM = len(OUTER_FEATURE_NAMES)

INNER_CONTEXT_NAMES = (
    "bias",
    "contract_distance",
    "ancilla_leakage",
    "symbolic_distance_fraction",
    "tableau_reduction_fraction",
    "signed_rotation_match",
    "axis_rotation_match",
    "candidate_axis_logical_support",
    "candidate_axis_workspace_support",
    "target_tableau_operand_mismatch_fraction",
    "remaining_t_slack",
    "remaining_cnot_slack",
    "remaining_gate_slack",
    "remaining_depth_slack",
    "last_gate_operand_overlap",
    "workspace_depth_fraction",
)
INNER_CONTEXT_DIM = len(INNER_CONTEXT_NAMES)


@dataclass(slots=True)
class AncillaRecord:
    record_id: int
    state: HybridState
    isometry: np.ndarray
    symbolic_distance: int
    contract_distance: float
    leakage: float
    pending_mask: int
    ancilla_touch_count: int
    logical_workspace_cnot_count: int
    allocations: int = 0
    context_cache: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AncillaStep:
    selected_record_id: int
    tokens: tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    attempted_edges: int
    generated: int
    accepted: int
    rejected: int
    frontier_size: int
    solution_record_id: int | None


@dataclass(slots=True)
class AncillaSearchProfile:
    transitions: TransitionProfile = field(default_factory=TransitionProfile)
    archive_lookups: int = 0
    dominance_comparisons: int = 0
    certification_calls: int = 0
    outer_rows_scored: int = 0
    inner_rows_scored: int = 0
    context_rows_built: int = 0
    isometry_updates: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            **self.transitions.to_dict(),
            "archive_lookups": self.archive_lookups,
            "dominance_comparisons": self.dominance_comparisons,
            "certification_calls": self.certification_calls,
            "outer_rows_scored": self.outer_rows_scored,
            "inner_rows_scored": self.inner_rows_scored,
            "context_rows_built": self.context_rows_built,
            "isometry_updates": self.isometry_updates,
        }


@lru_cache(maxsize=None)
def _single_qubit_local_matrix(name: str) -> np.ndarray:
    if name == "H":
        result = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    elif name == "S":
        result = np.diag([1.0, 1j]).astype(np.complex128)
    elif name == "SDG":
        result = np.diag([1.0, -1j]).astype(np.complex128)
    elif name == "T":
        result = np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(np.complex128)
    elif name == "TDG":
        result = np.diag([1.0, np.exp(-1j * np.pi / 4)]).astype(np.complex128)
    else:
        raise ValueError(f"unsupported single-qubit gate {name!r}")
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _cnot_row_permutation(num_qubits: int, control: int, target: int) -> np.ndarray:
    mapping = np.empty(1 << num_qubits, dtype=np.int64)
    for source in range(1 << num_qubits):
        mapping[source] = source ^ (
            (1 << target) if ((source >> control) & 1) else 0
        )
    mapping.setflags(write=False)
    return mapping


def apply_gate_to_isometry(isometry: np.ndarray, gate: Gate) -> np.ndarray:
    """Left-apply one native gate to a matrix of state-vector columns."""

    columns = np.asarray(isometry, dtype=np.complex128)
    dimension = columns.shape[0]
    num_qubits = dimension.bit_length() - 1
    if 1 << num_qubits != dimension:
        raise ValueError("isometry row count must be a power of two")
    if gate.name == "CNOT":
        control, target = gate.qubits
        permutation = _cnot_row_permutation(num_qubits, control, target)
        result = np.empty_like(columns)
        result[permutation, :] = columns
        return result

    matrix = _single_qubit_local_matrix(gate.name)
    (qubit,) = gate.qubits
    bit = 1 << qubit
    result = columns.copy()
    for low in range(dimension):
        if low & bit:
            continue
        high = low | bit
        zero = columns[low, :].copy()
        one = columns[high, :].copy()
        result[low, :] = matrix[0, 0] * zero + matrix[0, 1] * one
        result[high, :] = matrix[1, 0] * zero + matrix[1, 1] * one
    return result


def _cheap_legal_continuation(state: HybridState, gate: Gate) -> bool:
    wire_parents = {state.wire_tails[q] for q in gate.qubits}
    if len(wire_parents) == 1:
        previous = next(iter(wire_parents))
        if (
            previous is not None
            and INVERSE_GATE[previous.gate.name] == gate.name
            and previous.gate.qubits == gate.qubits
        ):
            return False
    if state.tail is not None:
        last = state.tail.gate
        if set(last.qubits).isdisjoint(gate.qubits) and gate.sort_key() < last.sort_key():
            return False

    layer = 1 + max(state.wire_depths[q] for q in gate.qubits)
    return not (
        state.gate_count + 1 > state.budget.max_gates
        or layer > state.budget.max_depth
        or state.t_count + int(gate.is_non_clifford) > state.budget.max_t_count
        or state.cnot_count + int(gate.is_two_qubit) > state.budget.max_cnot_count
    )


class AncillaDeferredSearch:
    """Deferred exact search under a fixed clean/borrowed ancilla contract."""

    def __init__(
        self,
        target: AncillaSynthesisTarget,
        *,
        max_allocations: int = 1_024,
        max_edges: int = 32_768,
        batch_size: int = 4,
        shaping_weight: float = 0.5,
        success_bonus: float = 20.0,
        failure_penalty: float = 20.0,
        fairness_start_k: int = 8,
        tolerance: float = 1e-9,
    ) -> None:
        if max_allocations <= 0 or max_edges <= 0 or batch_size <= 0:
            raise ValueError("search limits and batch size must be positive")
        self.target = target
        self.actions = generate_gates(target.num_qubits)
        self.max_allocations = int(max_allocations)
        self.max_edges = int(max_edges)
        self.batch_size = int(batch_size)
        self.shaping_weight = float(shaping_weight)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.fairness_start_k = int(fairness_start_k)
        self.tolerance = float(tolerance)
        self.records: dict[int, AncillaRecord] = {}
        self.frontier: dict[int, AncillaRecord] = {}
        self.pareto: dict[tuple[object, ...], list[tuple[tuple[int, ...], int]]] = {}
        self.next_record_id = 0
        self.allocations = 0
        self.edge_attempts = 0
        self.generated = 0
        self.accepted = 0
        self.rejected = 0
        self.solution_record_id: int | None = None
        self.frontier_peak = 0
        self.profile = AncillaSearchProfile()
        self.reset()

    def _legal_mask(self, state: HybridState) -> int:
        mask = 0
        for token, gate in enumerate(self.actions):
            if _cheap_legal_continuation(state, gate):
                mask |= 1 << token
        return mask

    def _gate_resource_increments(self, gate: Gate) -> tuple[int, int]:
        contract = self.target.contract
        touches_workspace = int(
            any(q in contract.clean_ancillas or q in contract.borrowed_ancillas for q in gate.qubits)
        )
        roles = tuple(contract.operand_role(q) for q in gate.qubits)
        cross_cnot = int(
            gate.name == "CNOT"
            and len(set(roles)) > 1
            and "logical" in roles
        )
        return touches_workspace, cross_cnot

    def reset(self) -> None:
        self.records.clear()
        self.frontier.clear()
        self.pareto.clear()
        self.next_record_id = 0
        self.allocations = 0
        self.edge_attempts = 0
        self.generated = 0
        self.accepted = 0
        self.rejected = 0
        self.solution_record_id = None
        self.frontier_peak = 0
        self.profile = AncillaSearchProfile()
        root_state = HybridState.identity(self.target.num_qubits, self.target.budget)
        root = self._new_record(
            root_state,
            np.array(self.target.contract.input_embedding, copy=True),
            ancilla_touch_count=0,
            logical_workspace_cnot_count=0,
        )
        self.frontier[root.record_id] = root
        self.pareto[self._archive_key(root_state)] = [
            (self._resource_vector(root), root.record_id)
        ]
        self.frontier_peak = 1
        if root.contract_distance <= self.tolerance and root.leakage <= self.tolerance:
            result = certify_ancilla_state(self.target, root_state, tolerance=self.tolerance)
            if result.success:
                self.solution_record_id = root.record_id

    def _new_record(
        self,
        state: HybridState,
        isometry: np.ndarray,
        *,
        ancilla_touch_count: int,
        logical_workspace_cnot_count: int,
    ) -> AncillaRecord:
        distance, leakage = contract_distance_and_leakage(isometry, self.target)
        symbolic = symbolic_distance_components(state, self.target)[0]
        record = AncillaRecord(
            record_id=self.next_record_id,
            state=state,
            isometry=np.asarray(isometry, dtype=np.complex128),
            symbolic_distance=symbolic,
            contract_distance=distance,
            leakage=leakage,
            pending_mask=self._legal_mask(state),
            ancilla_touch_count=ancilla_touch_count,
            logical_workspace_cnot_count=logical_workspace_cnot_count,
        )
        self.next_record_id += 1
        self.records[record.record_id] = record
        return record

    def _resource_vector(self, record: AncillaRecord) -> tuple[int, ...]:
        return (
            *record.state.resource_vector(),
            record.ancilla_touch_count,
            record.logical_workspace_cnot_count,
        )

    def _archive_key(self, state: HybridState) -> tuple[object, ...]:
        if self.target.contract.phase_mode.value == "projective":
            return state.canonical_key
        # The current Clifford tableau is intentionally projective and does not
        # retain every scalar phase generated by Clifford multiplication.  In
        # exact-phase mode, semantic merging is therefore disabled rather than
        # risk pruning phase-distinct witnesses.  Exact certification remains
        # available; a future exact-phase canonicalizer can replace this
        # conservative witness key.
        return (
            "exact-phase-witness-v1",
            tuple(gate.label() for gate in state.reconstruct_gates()),
        )

    def open_records(self) -> tuple[AncillaRecord, ...]:
        return tuple(self.frontier.values())

    def pending_tokens(self, record: AncillaRecord) -> tuple[int, ...]:
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
        return -min(
            record.contract_distance + min(1.0, record.leakage)
            for record in self.frontier.values()
        )

    def _insert(
        self,
        state: HybridState,
        isometry: np.ndarray,
        *,
        ancilla_touch_count: int,
        logical_workspace_cnot_count: int,
    ) -> AncillaRecord | None:
        self.profile.archive_lookups += 1
        key = self._archive_key(state)
        probe = AncillaRecord(
            record_id=-1,
            state=state,
            isometry=isometry,
            symbolic_distance=0,
            contract_distance=0.0,
            leakage=0.0,
            pending_mask=0,
            ancilla_touch_count=ancilla_touch_count,
            logical_workspace_cnot_count=logical_workspace_cnot_count,
        )
        resources = self._resource_vector(probe)
        group = self.pareto.setdefault(key, [])
        for existing, _ in group:
            self.profile.dominance_comparisons += 1
            if _weakly_dominates(existing, resources):
                self.rejected += 1
                return None

        survivors: list[tuple[tuple[int, ...], int]] = []
        for existing, record_id in group:
            self.profile.dominance_comparisons += 1
            if _strictly_dominates(resources, existing):
                self.frontier.pop(record_id, None)
            else:
                survivors.append((existing, record_id))

        record = self._new_record(
            state,
            isometry,
            ancilla_touch_count=ancilla_touch_count,
            logical_workspace_cnot_count=logical_workspace_cnot_count,
        )
        survivors.append((resources, record.record_id))
        self.pareto[key] = survivors
        if record.pending_mask:
            self.frontier[record.record_id] = record
        self.accepted += 1
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
    ) -> AncillaStep:
        if self.solution_record_id is not None:
            raise RuntimeError("the search has already certified a solution")
        try:
            selected = self.frontier[int(record_id)]
        except KeyError as exc:
            raise KeyError(f"record {record_id} is not active") from exc

        requested = [int(token) for token in tokens]
        if allow_fairness_override:
            forced = self._forced_fair_edge()
            if forced is not None:
                forced_record, forced_token = forced
                selected = self.frontier[forced_record]
                requested = [forced_token]

        before = self.frontier_potential()
        selected.allocations += 1
        self.allocations += 1
        generated_now = accepted_now = attempted_now = 0
        processed: list[int] = []

        for token in requested[: self.batch_size]:
            if self.edge_attempts >= self.max_edges:
                break
            if token < 0 or token >= len(self.actions):
                raise IndexError(f"continuation token {token} is out of range")
            bit = 1 << token
            if not selected.pending_mask & bit:
                continue
            selected.pending_mask &= ~bit
            processed.append(token)
            self.edge_attempts += 1
            attempted_now += 1
            gate = self.actions[token]
            child_state = selected.state.apply(
                gate,
                partial_order_reduction=True,
                profile=self.profile.transitions,
            )
            if child_state is None:
                raise AssertionError("cheap legality mask admitted an invalid transition")
            child_isometry = apply_gate_to_isometry(selected.isometry, gate)
            self.profile.isometry_updates += 1
            self.generated += 1
            generated_now += 1
            touch_increment, cross_increment = self._gate_resource_increments(gate)
            child = self._insert(
                child_state,
                child_isometry,
                ancilla_touch_count=selected.ancilla_touch_count + touch_increment,
                logical_workspace_cnot_count=(
                    selected.logical_workspace_cnot_count + cross_increment
                ),
            )
            if child is None:
                continue
            accepted_now += 1
            if (
                child.contract_distance <= self.tolerance
                and child.leakage <= self.tolerance
            ):
                self.profile.certification_calls += 1
                certification = certify_ancilla_state(
                    self.target, child_state, tolerance=self.tolerance
                )
                if certification.success:
                    self.solution_record_id = child.record_id
                    break

        if selected.pending_mask == 0:
            self.frontier.pop(selected.record_id, None)

        self.frontier_peak = max(self.frontier_peak, len(self.frontier))
        exhausted = not self.frontier and self.solution_record_id is None
        truncated = (
            self.solution_record_id is None
            and (
                self.allocations >= self.max_allocations
                or self.edge_attempts >= self.max_edges
            )
        )
        terminated = self.solution_record_id is not None or exhausted
        after = self.frontier_potential()
        reward = -float(attempted_now) + self.shaping_weight * (after - before)
        if self.solution_record_id is not None:
            reward += self.success_bonus
        elif exhausted or truncated:
            reward -= self.failure_penalty

        return AncillaStep(
            selected_record_id=selected.record_id,
            tokens=tuple(processed),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            attempted_edges=attempted_now,
            generated=generated_now,
            accepted=accepted_now,
            rejected=generated_now - accepted_now,
            frontier_size=len(self.frontier),
            solution_record_id=self.solution_record_id,
        )

    def solution_state(self) -> HybridState | None:
        if self.solution_record_id is None:
            return None
        return self.records[self.solution_record_id].state

    def solution_certification(self) -> AncillaCertificationResult | None:
        state = self.solution_state()
        return None if state is None else certify_ancilla_state(self.target, state)

    def metrics(self) -> dict[str, object]:
        return {
            "success": self.solution_record_id is not None,
            "allocations": self.allocations,
            "edge_attempts": self.edge_attempts,
            "generated": self.generated,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "frontier_peak": self.frontier_peak,
            "records": len(self.records),
            "profile": self.profile.to_dict(),
        }


def _identity_tableau_payload(n: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        [(1 << q, 0, 1) for q in range(n)]
        + [(0, 1 << q, 1) for q in range(n)]
    )


def _rotation_support_fractions(record: AncillaRecord, target: AncillaSynthesisTarget) -> tuple[float, float]:
    logical_mask = sum(1 << q for q in target.contract.logical_qubits)
    workspace_mask = sum(
        1 << q
        for q in (*target.contract.clean_ancillas, *target.contract.borrowed_ancillas)
    )
    if not record.state.rotations:
        return 0.0, 0.0
    logical = workspace = 0
    for rotation in record.state.rotations:
        support = rotation.axis.x_mask | rotation.axis.z_mask
        logical += (support & logical_mask).bit_count()
        workspace += (support & workspace_mask).bit_count()
    denominator = max(1, len(record.state.rotations))
    return (
        logical / max(1, denominator * len(target.contract.logical_qubits)),
        workspace
        / max(
            1,
            denominator
            * (
                len(target.contract.clean_ancillas)
                + len(target.contract.borrowed_ancillas)
            ),
        ),
    )


def ancilla_outer_features(
    record: AncillaRecord,
    target: AncillaSynthesisTarget,
) -> np.ndarray:
    state = record.state
    total, tableau_mismatch, sequence_mismatch, multiset_mismatch = (
        symbolic_distance_components(state, target)
    )
    n = state.num_qubits
    maximum_pairs = max(1, math.comb(max(2, state.budget.max_t_count), 2))
    target_nonidentity = sum(
        left != right
        for left, right in zip(
            target.tableau_payload,
            _identity_tableau_payload(n),
            strict=True,
        )
    )
    del target_nonidentity  # represented implicitly by target-relative discrepancies
    logical_depths = [state.wire_depths[q] for q in target.contract.logical_qubits]
    workspace_qubits = (*target.contract.clean_ancillas, *target.contract.borrowed_ancillas)
    workspace_depths = [state.wire_depths[q] for q in workspace_qubits]
    logical_support, workspace_support = _rotation_support_fractions(record, target)
    last_name = None if state.last_gate is None else state.last_gate.name
    values = np.asarray(
        [
            1.0,
            state.t_count / max(1, state.budget.max_t_count),
            state.cnot_count / max(1, state.budget.max_cnot_count),
            state.gate_count / max(1, state.budget.max_gates),
            state.depth / max(1, state.budget.max_depth),
            sum(logical_depths) / max(1, len(logical_depths) * state.budget.max_depth),
            sum(workspace_depths) / max(1, len(workspace_depths) * state.budget.max_depth),
            len(state.rotations) / max(1, state.budget.max_t_count),
            state.anticommuting_pairs / maximum_pairs,
            state.mean_pauli_weight / max(1, n),
            tableau_mismatch / max(1, 2 * n),
            sequence_mismatch
            / max(1, state.budget.max_t_count + len(target.rotation_payloads)),
            multiset_mismatch
            / max(1, state.budget.max_t_count + len(target.rotation_payloads)),
            total / max(1, 4 * n + 2 * state.budget.max_t_count),
            min(1.0, record.contract_distance),
            min(1.0, record.leakage),
            logical_support,
            workspace_support,
            record.ancilla_touch_count / max(1, state.budget.max_gates),
            record.logical_workspace_cnot_count / max(1, state.budget.max_cnot_count),
            record.pending_mask.bit_count() / max(1, len(generate_gates(n))),
            *(1.0 if last_name == name else 0.0 for name in ("H", "S", "SDG", "T", "TDG", "CNOT")),
            target.contract.num_logical_qubits / 6.0,
            (
                target.contract.num_clean_ancillas
                + target.contract.num_borrowed_ancillas
            )
            / 3.0,
        ],
        dtype=np.float64,
    )
    if values.shape != (OUTER_FEATURE_DIM,):
        raise AssertionError(
            f"ancilla outer feature schema produced {values.shape}, expected {(OUTER_FEATURE_DIM,)}"
        )
    return values


@dataclass(slots=True)
class LinearAncillaOuterSarsa:
    learning_rate: float = 0.003
    gamma: float = 1.0
    seed: int = 0
    theta: np.ndarray = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    rows_scored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.theta = np.zeros(OUTER_FEATURE_DIM, dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def choose(
        self,
        records: Sequence[AncillaRecord],
        target: AncillaSynthesisTarget,
        epsilon: float,
    ) -> tuple[int, np.ndarray, float]:
        nodes = tuple(records)
        if not nodes:
            raise RuntimeError("cannot choose from an empty frontier")
        matrix = np.stack([ancilla_outer_features(record, target) for record in nodes])
        self.rows_scored += len(nodes)
        scores = matrix @ self.theta
        if self.rng.random() < epsilon:
            index = int(self.rng.integers(len(nodes)))
        else:
            best = float(np.max(scores))
            tied = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=1e-12))
            index = min(
                (int(value) for value in tied),
                key=lambda value: nodes[value].record_id,
            )
        return nodes[index].record_id, matrix[index].copy(), float(scores[index])

    def frontier_value(
        self,
        records: Sequence[AncillaRecord],
        target: AncillaSynthesisTarget,
    ) -> float:
        nodes = tuple(records)
        if not nodes:
            return -20.0
        matrix = np.stack([ancilla_outer_features(record, target) for record in nodes])
        return float(np.max(matrix @ self.theta))

    def update(
        self,
        features: np.ndarray,
        q_value: float,
        reward: float,
        next_q_value: float | None,
        *,
        duration: int = 1,
    ) -> float:
        target_value = float(reward)
        if next_q_value is not None:
            target_value += (self.gamma ** max(1, int(duration))) * next_q_value
        td_error = target_value - float(q_value)
        self.theta += self.learning_rate * td_error * features
        np.clip(self.theta, -50.0, 50.0, out=self.theta)
        return float(td_error)


def _projected_tableau_mismatch(
    state: HybridState,
    gate: Gate,
    target: AncillaSynthesisTarget,
) -> int:
    if not gate.is_clifford:
        return sum(
            left != right
            for left, right in zip(
                state.tableau.canonical_payload(), target.tableau_payload, strict=True
            )
        )
    current_images = state.tableau.forward_x + state.tableau.forward_z
    projected = (
        conjugate_by_gate(axis, gate.name, gate.qubits).axis_payload()
        for axis in current_images
    )
    return sum(
        left != right
        for left, right in zip(projected, target.tableau_payload, strict=True)
    )


def _candidate_rotation_payload(state: HybridState, gate: Gate) -> tuple[int, int, int]:
    axis = state.tableau.inverse_z[gate.qubits[0]]
    turns = 1 if gate.name == "T" else -1
    if axis.sign < 0:
        axis = axis.positive_axis()
        turns = -turns
    return axis.x_mask, axis.z_mask, turns


def _operand_tableau_mismatch(
    state: HybridState,
    gate: Gate,
    target: AncillaSynthesisTarget,
) -> int:
    operand_mask = sum(1 << q for q in gate.qubits)
    mismatch = 0
    for current, desired in zip(
        state.tableau.canonical_payload(), target.tableau_payload, strict=True
    ):
        if current == desired:
            continue
        if ((current[0] | current[1] | desired[0] | desired[1]) & operand_mask) != 0:
            mismatch += 1
    return mismatch


def ancilla_inner_context(
    record: AncillaRecord,
    gate: Gate,
    target: AncillaSynthesisTarget,
) -> np.ndarray:
    state = record.state
    n = state.num_qubits
    total, tableau_before, _, _ = symbolic_distance_components(state, target)
    tableau_after = _projected_tableau_mismatch(state, gate, target)
    tableau_reduction = tableau_before - tableau_after

    current_payloads = tuple(rotation.canonical_payload() for rotation in state.rotations)
    remaining = Counter(target.rotation_payloads)
    remaining.subtract(Counter(current_payloads))
    signed_match = axis_match = 0.0
    logical_axis_support = workspace_axis_support = 0.0
    if gate.is_non_clifford:
        candidate = _candidate_rotation_payload(state, gate)
        signed_match = 1.0 if remaining[candidate] > 0 else 0.0
        axis_match = 1.0 if any(
            count > 0 and token[:2] == candidate[:2]
            for token, count in remaining.items()
        ) else 0.0
        support = candidate[0] | candidate[1]
        logical_mask = sum(1 << q for q in target.contract.logical_qubits)
        workspace_mask = sum(
            1 << q
            for q in (*target.contract.clean_ancillas, *target.contract.borrowed_ancillas)
        )
        logical_axis_support = (support & logical_mask).bit_count() / max(
            1, target.contract.num_logical_qubits
        )
        workspace_axis_support = (support & workspace_mask).bit_count() / max(
            1,
            target.contract.num_clean_ancillas
            + target.contract.num_borrowed_ancillas,
        )

    next_t = state.t_count + int(gate.is_non_clifford)
    next_cnot = state.cnot_count + int(gate.is_two_qubit)
    next_gate = state.gate_count + 1
    next_layer = 1 + max(state.wire_depths[q] for q in gate.qubits)
    last = state.last_gate
    overlap = (
        0.0
        if last is None
        else len(set(last.qubits).intersection(gate.qubits)) / max(1, len(gate.qubits))
    )
    workspace_qubits = (*target.contract.clean_ancillas, *target.contract.borrowed_ancillas)
    workspace_depth = max((state.wire_depths[q] for q in workspace_qubits), default=0)
    values = np.asarray(
        [
            1.0,
            min(1.0, record.contract_distance),
            min(1.0, record.leakage),
            total / max(1, 4 * n + 2 * state.budget.max_t_count),
            tableau_reduction / max(1, 2 * n),
            signed_match,
            axis_match,
            logical_axis_support,
            workspace_axis_support,
            _operand_tableau_mismatch(state, gate, target) / max(1, 2 * n),
            (state.budget.max_t_count - next_t) / max(1, state.budget.max_t_count),
            (state.budget.max_cnot_count - next_cnot)
            / max(1, state.budget.max_cnot_count),
            (state.budget.max_gates - next_gate) / max(1, state.budget.max_gates),
            (state.budget.max_depth - next_layer) / max(1, state.budget.max_depth),
            overlap,
            workspace_depth / max(1, state.budget.max_depth),
        ],
        dtype=np.float64,
    )
    if values.shape != (INNER_CONTEXT_DIM,):
        raise AssertionError("ancilla inner context schema has the wrong dimension")
    return values


@dataclass(slots=True)
class DisjointAncillaLinUCB:
    alpha: float = 0.5
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
            self.a_inverse[family] = (
                np.eye(INNER_CONTEXT_DIM, dtype=np.float64) / self.regularization
            )
            self.b_vector[family] = np.zeros(INNER_CONTEXT_DIM, dtype=np.float64)

    def posterior_mean(self, family: str) -> np.ndarray:
        self._ensure(family)
        return self.a_inverse[family] @ self.b_vector[family]

    def choose(
        self,
        record: AncillaRecord,
        tokens: Sequence[int],
        actions: Sequence[Gate],
        target: AncillaSynthesisTarget,
        *,
        explore: bool,
        profile: AncillaSearchProfile | None = None,
    ) -> tuple[int, np.ndarray, float, str]:
        if not tokens:
            raise RuntimeError("cannot choose from an empty continuation set")
        best: tuple[float, int, int, np.ndarray, str] | None = None
        for token_value in tokens:
            token = int(token_value)
            gate = actions[token]
            family = target.contract.gate_role_class(gate)
            context = record.context_cache.get(token)
            if context is None:
                context = ancilla_inner_context(record, gate, target)
                record.context_cache[token] = context
                if profile is not None:
                    profile.context_rows_built += 1
            mean = float(context @ self.posterior_mean(family))
            bonus = 0.0
            if explore:
                inverse = self.a_inverse[family]
                bonus = self.alpha * math.sqrt(
                    max(0.0, float(context @ inverse @ context))
                )
            candidate = (mean + bonus, -token, token, context, family)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        self.rows_scored += len(tokens)
        if profile is not None:
            profile.inner_rows_scored += len(tokens)
        return best[2], best[3].copy(), best[0], best[4]

    def rank(
        self,
        record: AncillaRecord,
        tokens: Sequence[int],
        actions: Sequence[Gate],
        target: AncillaSynthesisTarget,
        *,
        limit: int,
        profile: AncillaSearchProfile | None = None,
    ) -> tuple[int, ...]:
        if limit <= 0:
            return ()
        scored: list[tuple[float, int]] = []
        for token_value in tokens:
            token = int(token_value)
            gate = actions[token]
            family = target.contract.gate_role_class(gate)
            context = record.context_cache.get(token)
            if context is None:
                context = ancilla_inner_context(record, gate, target)
                record.context_cache[token] = context
                if profile is not None:
                    profile.context_rows_built += 1
            scored.append((float(context @ self.posterior_mean(family)), token))
        self.rows_scored += len(scored)
        if profile is not None:
            profile.inner_rows_scored += len(scored)
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(token for _, token in scored[:limit])

    def update(self, family: str, context: np.ndarray, reward: float) -> None:
        self._ensure(family)
        vector = np.asarray(context, dtype=np.float64)
        inverse = self.a_inverse[family]
        projected = inverse @ vector
        denominator = 1.0 + float(vector @ projected)
        self.a_inverse[family] = inverse - np.outer(projected, projected) / denominator
        self.b_vector[family] += float(reward) * vector
        self.updates += 1


def initialize_ancilla_outer_from_mixed(
    mixed_policy: object,
    *,
    seed: int = 11,
) -> LinearAncillaOuterSarsa:
    """Warm-start shared linear coordinates from the mixed-gate outer policy."""

    from .mixed_crossover import OUTER_FEATURE_NAMES as MIXED_NAMES

    source = np.asarray(getattr(mixed_policy, "theta"), dtype=np.float64)
    if source.shape != (len(MIXED_NAMES),):
        raise ValueError("mixed outer policy has an unexpected parameter shape")
    result = LinearAncillaOuterSarsa(seed=seed)
    source_by_name = dict(zip(MIXED_NAMES, source, strict=True))
    direct = {
        "bias", "t_fraction", "cnot_fraction", "gate_fraction",
        "depth_fraction", "rotation_fraction",
        "anticommuting_pair_fraction", "mean_pauli_weight_fraction",
        "tableau_mismatch_fraction", "rotation_sequence_mismatch_fraction",
        "rotation_multiset_mismatch_fraction", "symbolic_distance_fraction",
        "last_H", "last_S", "last_SDG", "last_T", "last_TDG", "last_CNOT",
    }
    for index, name in enumerate(OUTER_FEATURE_NAMES):
        if name in direct and name in source_by_name:
            result.theta[index] = source_by_name[name]
    mean_depth = source_by_name.get("mean_wire_depth_fraction", 0.0)
    result.theta[OUTER_FEATURE_NAMES.index("logical_depth_mean")] = mean_depth
    # The mixed register-size coefficient is transferred to the logical-width
    # coordinate; new contract-specific coordinates start at zero.
    result.theta[OUTER_FEATURE_NAMES.index("logical_register_fraction")] = (
        source_by_name.get("register_fraction", 0.0)
    )
    return result


def initialize_ancilla_bandit_from_mixed(
    mixed_bandit: object,
    targets: Sequence[AncillaSynthesisTarget],
    *,
    alpha: float = 0.5,
) -> DisjointAncillaLinUCB:
    """Replicate gate-family posterior means across logical/workspace roles."""

    from .mixed_crossover import INNER_CONTEXT_NAMES as MIXED_CONTEXT_NAMES

    result = DisjointAncillaLinUCB(alpha=alpha)
    shared = {
        "bias",
        "symbolic_distance_fraction",
        "tableau_reduction_fraction",
        "signed_rotation_match",
        "axis_rotation_match",
        "target_tableau_operand_mismatch_fraction",
        "remaining_t_slack",
        "remaining_cnot_slack",
        "remaining_gate_slack",
        "remaining_depth_slack",
        "last_gate_operand_overlap",
    }
    families: set[str] = set()
    for target in targets:
        for gate in generate_gates(target.num_qubits):
            families.add(target.contract.gate_role_class(gate))
    for family in sorted(families):
        gate_name = family.split(":", 1)[0]
        source_mean = np.asarray(
            mixed_bandit.posterior_mean(gate_name), dtype=np.float64
        )
        source_by_name = dict(
            zip(MIXED_CONTEXT_NAMES, source_mean, strict=True)
        )
        mapped = np.zeros(INNER_CONTEXT_DIM, dtype=np.float64)
        for index, name in enumerate(INNER_CONTEXT_NAMES):
            if name in shared and name in source_by_name:
                mapped[index] = source_by_name[name]
        result._ensure(family)
        result.b_vector[family] = result.regularization * mapped
    return result


def _epsilon(episode: int, episodes: int) -> float:
    fraction = episode / max(1, episodes - 1)
    return 0.30 + fraction * (0.03 - 0.30)


def train_ancilla_outer_sarsa(
    targets: Sequence[AncillaSynthesisTarget],
    *,
    episodes: int = 24,
    seed: int = 11,
    max_allocations: int = 96,
    batch_size: int = 4,
    policy: LinearAncillaOuterSarsa | None = None,
) -> LinearAncillaOuterSarsa:
    """Train outer SARSA with a deterministic native continuation order."""

    if not targets:
        raise ValueError("outer training requires targets")
    policy = LinearAncillaOuterSarsa(seed=seed) if policy is None else policy
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = AncillaDeferredSearch(
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
            environment.open_records(), target, epsilon
        )
        while True:
            record = environment.frontier[record_id]
            tokens = environment.pending_tokens(record)[:batch_size]
            step = environment.process_batch(
                record_id, tokens, allow_fairness_override=False
            )
            done = step.terminated or step.truncated
            if done:
                policy.update(
                    features,
                    q_value,
                    step.reward,
                    None,
                    duration=max(1, step.attempted_edges),
                )
                break
            next_id, next_features, next_q = policy.choose(
                environment.open_records(), target, epsilon
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


def train_ancilla_inner_bandit(
    outer: LinearAncillaOuterSarsa,
    targets: Sequence[AncillaSynthesisTarget],
    *,
    episodes: int = 36,
    alpha: float = 0.5,
    max_allocations: int = 96,
    bandit: DisjointAncillaLinUCB | None = None,
) -> DisjointAncillaLinUCB:
    """Train LinUCB against a frozen outer value function, one edge at a time."""

    if not targets:
        raise ValueError("inner training requires targets")
    bandit = DisjointAncillaLinUCB(alpha=alpha) if bandit is None else bandit
    bandit.alpha = float(alpha)
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = AncillaDeferredSearch(
            target,
            max_allocations=max_allocations,
            max_edges=max_allocations,
            batch_size=1,
            fairness_start_k=10_000,
        )
        while environment.frontier and environment.solution_record_id is None:
            record_id, _, _ = outer.choose(environment.open_records(), target, 0.0)
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
            step = environment.process_batch(
                record.record_id,
                (token,),
                allow_fairness_override=False,
            )
            after_value = (
                20.0
                if environment.solution_record_id is not None
                else outer.frontier_value(environment.open_records(), target)
            )
            reward = 0.05 * (after_value - before_value) - 0.01
            if step.accepted == 0:
                reward -= 0.05
            if step.solution_record_id is not None:
                reward += 2.0
            bandit.update(family, context, reward)
            if step.terminated or step.truncated:
                break
    return bandit


@dataclass(frozen=True)
class AncillaEvaluationResult:
    method: str
    target: str
    success: bool
    certified: bool
    stop_reason: str
    wall_seconds: float
    cpu_seconds: float
    allocations: int
    attempted_edges: int
    frontier_peak: int
    projective_isometry_error: float | None
    ancilla_leakage: float | None
    witness: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target": self.target,
            "success": self.success,
            "certified": self.certified,
            "stop_reason": self.stop_reason,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "allocations": self.allocations,
            "attempted_edges": self.attempted_edges,
            "frontier_peak": self.frontier_peak,
            "projective_isometry_error": self.projective_isometry_error,
            "ancilla_leakage": self.ancilla_leakage,
            "witness": "; ".join(self.witness),
        }


def evaluate_ancilla_hierarchy(
    outer: LinearAncillaOuterSarsa,
    bandit: DisjointAncillaLinUCB,
    target: AncillaSynthesisTarget,
    *,
    max_allocations: int = 512,
    batch_size: int = 4,
    wall_limit: float = 5.0,
) -> AncillaEvaluationResult:
    environment = AncillaDeferredSearch(
        target,
        max_allocations=max_allocations,
        max_edges=max_allocations * batch_size,
        batch_size=batch_size,
    )
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    stop_reason = "frontier_exhausted"
    while environment.frontier and environment.solution_record_id is None:
        if environment.allocations >= max_allocations:
            stop_reason = "allocation_cap"
            break
        if time.perf_counter() - wall_start >= wall_limit:
            stop_reason = "wall_limit"
            break
        record_id, _, _ = outer.choose(environment.open_records(), target, 0.0)
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
        if step.terminated:
            break
        if step.truncated:
            stop_reason = "allocation_cap"
            break
    certification = environment.solution_certification()
    success = bool(certification and certification.success)
    return AncillaEvaluationResult(
        method="ancilla_deferred_outer_sarsa_inner_linucb",
        target=target.name,
        success=success,
        certified=success,
        stop_reason="certified" if success else stop_reason,
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        allocations=environment.allocations,
        attempted_edges=environment.edge_attempts,
        frontier_peak=environment.frontier_peak,
        projective_isometry_error=(
            None if certification is None else certification.projective_isometry_error
        ),
        ancilla_leakage=(
            None if certification is None else certification.ancilla_leakage
        ),
        witness=() if certification is None else certification.witness,
    )


__all__ = [
    "AncillaDeferredSearch",
    "AncillaEvaluationResult",
    "AncillaRecord",
    "AncillaSearchProfile",
    "AncillaStep",
    "DisjointAncillaLinUCB",
    "INNER_CONTEXT_DIM",
    "INNER_CONTEXT_NAMES",
    "LinearAncillaOuterSarsa",
    "OUTER_FEATURE_DIM",
    "OUTER_FEATURE_NAMES",
    "ancilla_inner_context",
    "ancilla_outer_features",
    "apply_gate_to_isometry",
    "evaluate_ancilla_hierarchy",
    "initialize_ancilla_bandit_from_mixed",
    "initialize_ancilla_outer_from_mixed",
    "train_ancilla_inner_bandit",
    "train_ancilla_outer_sarsa",
]
