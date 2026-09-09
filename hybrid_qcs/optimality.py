"""Proof-producing resource audits for small exact quantum circuits.

The learning controllers in :mod:`hybrid_qcs` remain linear search schedulers.
This module is deliberately deterministic: a learned search may provide an
incumbent circuit, while exhaustive finite-state procedures establish lower
bounds in explicitly declared synthesis domains.

The principal exact class consists of three-input, single-violation Boolean
phase oracles. These are the phase-marking primitives that occur in the small
BNN formal-verification example. A marked string is mapped to ``111`` by
Clifford X gates, a CCZ phase polynomial is synthesized, and the X gates are
undone. Exhaustive phase-polynomial enumeration proves the minimum T-count,
while a finite parity-network BFS proves the minimum CNOT count and CNOT depth
for that normal-form class.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from heapq import heappop, heappush
from itertools import combinations
from pathlib import Path
from typing import Sequence
import json
import math

import numpy as np


class PhaseMode(str, Enum):
    """Terminal phase convention."""

    PROJECTIVE = "projective"
    EXACT = "exact"


class OptimalityStatus(str, Enum):
    """Strength of a resource statement."""

    EXACT = "exact"
    CONDITIONAL = "conditional"
    UPPER_BOUND = "upper_bound_only"


@dataclass(frozen=True)
class OptimizationContract:
    """A fully qualified finite synthesis domain."""

    name: str
    gate_library: tuple[str, ...]
    logical_qubits: int
    clean_ancillas: int = 0
    borrowed_ancillas: int = 0
    connectivity: str = "all_to_all"
    phase_mode: PhaseMode = PhaseMode.EXACT
    measurements_allowed: bool = False
    objective: tuple[str, ...] = ("t_count", "cnot_count")
    proof_class: str = "finite_exact_search"

    def __post_init__(self) -> None:
        if self.logical_qubits < 1:
            raise ValueError("logical_qubits must be positive")
        if self.clean_ancillas < 0 or self.borrowed_ancillas < 0:
            raise ValueError("ancilla counts must be non-negative")
        if not self.gate_library:
            raise ValueError("gate_library must be non-empty")
        if not self.objective:
            raise ValueError("objective must be non-empty")


@dataclass(frozen=True, order=True)
class AdditiveResources:
    """Additive resources used by the exact macro-level optimizer."""

    macro_count: int = 0
    t_count: int = 0
    cnot_count: int = 0
    native_gate_count: int = 0

    def plus(self, other: "AdditiveResources") -> "AdditiveResources":
        return AdditiveResources(
            self.macro_count + other.macro_count,
            self.t_count + other.t_count,
            self.cnot_count + other.cnot_count,
            self.native_gate_count + other.native_gate_count,
        )


@dataclass(frozen=True)
class PhasePolynomialProof:
    masks: tuple[int, ...]
    coefficients_mod8: tuple[int, ...]
    target_exponents_mod8: tuple[int, ...]
    minimum_t_count: int
    assignments_examined: int
    exact: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParityNetworkProof:
    minimum_cnot_count: int
    minimum_cnot_depth: int
    cnot_sequence: tuple[tuple[int, int], ...]
    layers: tuple[tuple[tuple[int, int], ...], ...]
    parity_host_events: tuple[tuple[int, int], ...]
    states_explored_count: int
    states_explored_depth: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MarkedStateOracleCertificate:
    marked_bitstring: str
    contract: OptimizationContract
    status: OptimalityStatus
    minimum_t_count: int
    minimum_cnot_count: int
    minimum_cnot_depth: int
    native_gate_count_upper_bound: int
    total_depth_upper_bound: int
    phase_coefficients_mod8: tuple[int, ...]
    cnot_sequence: tuple[tuple[int, int], ...]
    native_witness: tuple[str, ...]
    exact_matrix_error: float
    proof_scope: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contract"]["phase_mode"] = self.contract.phase_mode.value
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class LinearReversibleCertificate:
    name: str
    qubits: int
    output_from_input: tuple[int, ...]
    minimum_cnot_count: int
    minimum_cnot_depth: int
    count_witness: tuple[tuple[int, int], ...]
    depth_layers: tuple[tuple[tuple[int, int], ...], ...]
    states_explored_count: int
    states_explored_depth: int
    status: OptimalityStatus = OptimalityStatus.EXACT

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class Macro:
    kind: str
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"X", "CNOT", "TOFFOLI"}:
            raise ValueError(f"unsupported macro {self.kind}")
        expected = {"X": 1, "CNOT": 2, "TOFFOLI": 3}[self.kind]
        if len(self.qubits) != expected or len(set(self.qubits)) != expected:
            raise ValueError("invalid macro operands")

    @property
    def resources(self) -> AdditiveResources:
        # Fixed exact native lowerings used by the oracle branch.
        if self.kind == "X":
            return AdditiveResources(1, 0, 0, 4)  # H S S H
        if self.kind == "CNOT":
            return AdditiveResources(1, 0, 1, 1)
        return AdditiveResources(1, 7, 6, 15)

    def text(self) -> str:
        return f"{self.kind}({','.join(str(q) for q in self.qubits)})"


@dataclass(frozen=True)
class EvaluatorOptimalityCertificate:
    marked_bitstring: str
    contract: OptimizationContract
    status: OptimalityStatus
    optimum: AdditiveResources
    witness: tuple[str, ...]
    incumbent_matches_optimum: bool
    states_settled: int
    edges_generated: int
    proof_scope: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contract"]["phase_mode"] = self.contract.phase_mode.value
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Phase-polynomial T-count proof
# ---------------------------------------------------------------------------


def _parity(mask: int, value: int) -> int:
    return (mask & value).bit_count() & 1


def phase_exponents(coefficients_mod8: Sequence[int], n: int = 3) -> tuple[int, ...]:
    masks = tuple(range(1, 1 << n))
    if len(coefficients_mod8) != len(masks):
        raise ValueError("coefficient count does not match nonzero parities")
    return tuple(
        sum(int(a) * _parity(mask, x) for a, mask in zip(coefficients_mod8, masks)) % 8
        for x in range(1 << n)
    )


@lru_cache(maxsize=1)
def prove_ccz_phase_polynomial_t_optimum() -> PhasePolynomialProof:
    """Exhaust all 8^7 three-variable linear phase polynomials.

    Coefficient parity is the T-count after even coefficients are absorbed
    into the Clifford frame. The target exponent is four only on ``111``.
    """

    n = 3
    masks = np.arange(1, 1 << n, dtype=np.int16)
    xs = np.arange(1, 1 << n, dtype=np.int16)
    incidence = np.empty((len(xs), len(masks)), dtype=np.int16)
    for i, x in enumerate(xs.tolist()):
        for j, mask in enumerate(masks.tolist()):
            incidence[i, j] = _parity(mask, x)
    target = np.array([4 if x == 7 else 0 for x in xs.tolist()], dtype=np.int16)

    total = 8 ** len(masks)
    best = len(masks) + 1
    best_coeff: tuple[int, ...] | None = None
    best_rank: tuple[int, int, tuple[int, ...]] | None = None
    phase_lengths = np.array([0, 1, 1, 2, 2, 2, 1, 1], dtype=np.int16)
    chunk = 1 << 16
    shifts = np.arange(len(masks), dtype=np.uint64) * 3
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        numbers = np.arange(start, stop, dtype=np.uint64)[:, None]
        coeff = ((numbers >> shifts[None, :]) & 7).astype(np.int16)
        values = (coeff @ incidence.T) & 7
        good = np.all(values == target[None, :], axis=1)
        if not np.any(good):
            continue
        candidates = coeff[good]
        odd = np.sum(candidates & 1, axis=1)
        local = int(np.min(odd))
        # T-optimal coefficients need not be +/-1. Choose a reproducible
        # minimum-native-phase-cost witness among all T-optimal solutions.
        if local <= best:
            for row in candidates[odd == local]:
                coeff_tuple = tuple(int(v) for v in row.tolist())
                rank = (local, int(np.sum(phase_lengths[row])), coeff_tuple)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best = local
                    best_coeff = coeff_tuple

    if best_coeff is None:
        raise AssertionError("CCZ phase polynomial was not found")
    target_full = tuple(4 if x == 7 else 0 for x in range(1 << n))
    exact = phase_exponents(best_coeff, n) == target_full
    return PhasePolynomialProof(
        masks=tuple(int(v) for v in masks.tolist()),
        coefficients_mod8=best_coeff,
        target_exponents_mod8=target_full,
        minimum_t_count=best,
        assignments_examined=total,
        exact=exact,
    )


# ---------------------------------------------------------------------------
# Parity-network CNOT count/depth proof
# ---------------------------------------------------------------------------


def _identity_rows(n: int) -> tuple[int, ...]:
    return tuple(1 << i for i in range(n))


def _apply_cnot_rows(rows: tuple[int, ...], control: int, target: int) -> tuple[int, ...]:
    out = list(rows)
    out[target] ^= out[control]
    return tuple(out)


def _required_mask(rows: tuple[int, ...], required: dict[int, int]) -> int:
    mask = 0
    for row in rows:
        bit = required.get(row)
        if bit is not None:
            mask |= 1 << bit
    return mask


def _all_disjoint_cnot_layers(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    gates = tuple((c, t) for c in range(n) for t in range(n) if c != t)
    layers: set[tuple[tuple[int, int], ...]] = {(g,) for g in gates}
    for r in range(2, n // 2 + 1):
        for subset in combinations(gates, r):
            used: set[int] = set()
            valid = True
            for c, t in subset:
                if c in used or t in used:
                    valid = False
                    break
                used.add(c)
                used.add(t)
            if valid:
                layers.add(tuple(sorted(subset)))
    return tuple(sorted(layers, key=lambda x: (len(x), x)))


@lru_cache(maxsize=1)
def prove_ccz_parity_network_optimum() -> ParityNetworkProof:
    n = 3
    required_masks = tuple(range(1, 1 << n))
    required = {mask: i for i, mask in enumerate(required_masks)}
    full = (1 << len(required_masks)) - 1
    identity = _identity_rows(n)
    start_seen = _required_mask(identity, required)
    start = (identity, start_seen)
    gates = tuple((c, t) for c in range(n) for t in range(n) if c != t)

    queue = deque([start])
    parent: dict[tuple[tuple[int, ...], int], tuple[tuple[tuple[int, ...], int], tuple[int, int]] | None] = {start: None}
    goal = None
    while queue:
        state = queue.popleft()
        rows, seen = state
        if rows == identity and seen == full:
            goal = state
            break
        for gate in gates:
            new_rows = _apply_cnot_rows(rows, *gate)
            new_seen = seen | _required_mask(new_rows, required)
            nxt = (new_rows, new_seen)
            if nxt not in parent:
                parent[nxt] = (state, gate)
                queue.append(nxt)
    if goal is None:
        raise AssertionError("no parity network found")

    sequence: list[tuple[int, int]] = []
    cur = goal
    while parent[cur] is not None:
        prev, gate = parent[cur]
        sequence.append(gate)
        cur = prev
    sequence.reverse()

    rows = identity
    seen_parities: set[int] = set(rows)
    events: list[tuple[int, int]] = [(i, rows[i]) for i in range(n)]
    for c, t in sequence:
        rows = _apply_cnot_rows(rows, c, t)
        for wire, row in enumerate(rows):
            if row in required and row not in seen_parities:
                seen_parities.add(row)
                events.append((wire, row))
    if set(seen_parities) != set(required_masks) or rows != identity:
        raise AssertionError("invalid reconstructed parity network")

    layers_all = _all_disjoint_cnot_layers(n)
    q2 = deque([start])
    parent2: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[int, ...], int], tuple[tuple[int, int], ...]] | None,
    ] = {start: None}
    goal2 = None
    while q2:
        state = q2.popleft()
        rows, seen = state
        if rows == identity and seen == full:
            goal2 = state
            break
        for layer in layers_all:
            new_rows = rows
            for gate in layer:
                new_rows = _apply_cnot_rows(new_rows, *gate)
            new_seen = seen | _required_mask(new_rows, required)
            nxt = (new_rows, new_seen)
            if nxt not in parent2:
                parent2[nxt] = (state, layer)
                q2.append(nxt)
    if goal2 is None:
        raise AssertionError("no parity-depth network found")
    layers: list[tuple[tuple[int, int], ...]] = []
    cur2 = goal2
    while parent2[cur2] is not None:
        prev, layer = parent2[cur2]
        layers.append(layer)
        cur2 = prev
    layers.reverse()

    return ParityNetworkProof(
        minimum_cnot_count=len(sequence),
        minimum_cnot_depth=len(layers),
        cnot_sequence=tuple(sequence),
        layers=tuple(layers),
        parity_host_events=tuple(events),
        states_explored_count=len(parent),
        states_explored_depth=len(parent2),
    )


# ---------------------------------------------------------------------------
# Native marked-state phase circuits and dense verification
# ---------------------------------------------------------------------------


def _single_qubit_matrix(name: str) -> np.ndarray:
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if name == "H":
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    if name == "S":
        return np.diag([1, 1j]).astype(np.complex128)
    if name == "SDG":
        return np.diag([1, -1j]).astype(np.complex128)
    if name == "T":
        return np.diag([1, np.exp(1j * math.pi / 4)]).astype(np.complex128)
    if name == "TDG":
        return np.diag([1, np.exp(-1j * math.pi / 4)]).astype(np.complex128)
    raise ValueError(name)


def _apply_native_to_vector(vec: np.ndarray, n: int, gate: tuple[str, tuple[int, ...]]) -> np.ndarray:
    name, qs = gate
    out = np.zeros_like(vec)
    if name in {"X", "H", "S", "SDG", "T", "TDG"}:
        q = qs[0]
        mat = _single_qubit_matrix(name)
        for basis, amp in enumerate(vec):
            bit = (basis >> q) & 1
            base = basis & ~(1 << q)
            out[base] += mat[0, bit] * amp
            out[base | (1 << q)] += mat[1, bit] * amp
        return out
    if name == "CNOT":
        c, t = qs
        for basis, amp in enumerate(vec):
            dst = basis ^ (1 << t) if ((basis >> c) & 1) else basis
            out[dst] += amp
        return out
    raise ValueError(name)


def native_unitary(n: int, gates: Sequence[tuple[str, tuple[int, ...]]]) -> np.ndarray:
    dim = 1 << n
    unitary = np.zeros((dim, dim), dtype=np.complex128)
    for col in range(dim):
        vec = np.zeros(dim, dtype=np.complex128)
        vec[col] = 1
        for gate in gates:
            vec = _apply_native_to_vector(vec, n, gate)
        unitary[:, col] = vec
    return unitary


def _x_native(q: int) -> list[tuple[str, tuple[int, ...]]]:
    return [("H", (q,)), ("S", (q,)), ("S", (q,)), ("H", (q,))]


def _phase_gates_for_coefficient(coeff: int, q: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Lower every residue in Z8; all odd residues cost exactly one T gate."""
    names = {0: (), 1: ("T",), 2: ("S",), 3: ("T", "S"),
             4: ("S", "S"), 5: ("TDG", "SDG"), 6: ("SDG",), 7: ("TDG",)}
    return tuple((name, (q,)) for name in names[coeff % 8])


def synthesize_ccz_from_proofs() -> tuple[tuple[str, tuple[int, ...]], ...]:
    phase = prove_ccz_phase_polynomial_t_optimum()
    parity = prove_ccz_parity_network_optimum()
    coefficient = {mask: c for mask, c in zip(phase.masks, phase.coefficients_mod8)}
    rows = _identity_rows(3)
    used: set[int] = set()
    gates: list[tuple[str, tuple[int, ...]]] = []

    def emit_available() -> None:
        for wire, mask in enumerate(rows):
            if mask in coefficient and mask not in used:
                gates.extend(_phase_gates_for_coefficient(coefficient[mask], wire))
                used.add(mask)

    emit_available()
    for c, t in parity.cnot_sequence:
        gates.append(("CNOT", (c, t)))
        rows = _apply_cnot_rows(rows, c, t)
        emit_available()
    if rows != _identity_rows(3) or used != set(phase.masks):
        raise AssertionError("phase-network lowering did not close")
    return tuple(gates)


def synthesize_marked_state_phase_oracle(marked_bitstring: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if len(marked_bitstring) != 3 or set(marked_bitstring) - {"0", "1"}:
        raise ValueError("marked_bitstring must contain three binary digits")
    zero_qubits = [q for q, bit in enumerate(marked_bitstring) if bit == "0"]
    gates: list[tuple[str, tuple[int, ...]]] = []
    for q in zero_qubits:
        gates.extend(_x_native(q))
    gates.extend(synthesize_ccz_from_proofs())
    for q in reversed(zero_qubits):
        gates.extend(_x_native(q))
    return tuple(gates)


def _native_depth(gates: Sequence[tuple[str, tuple[int, ...]]], n: int) -> int:
    wire_depth = [0] * n
    for _, qs in gates:
        level = max(wire_depth[q] for q in qs) + 1
        for q in qs:
            wire_depth[q] = level
    return max(wire_depth, default=0)


def marked_state_target(marked_bitstring: str) -> np.ndarray:
    if len(marked_bitstring) != 3:
        raise ValueError(marked_bitstring)
    index = sum((bit == "1") << q for q, bit in enumerate(marked_bitstring))
    target = np.eye(8, dtype=np.complex128)
    target[index, index] = -1
    return target


def certify_marked_state_oracle(marked_bitstring: str) -> MarkedStateOracleCertificate:
    phase = prove_ccz_phase_polynomial_t_optimum()
    parity = prove_ccz_parity_network_optimum()
    witness = synthesize_marked_state_phase_oracle(marked_bitstring)
    actual = native_unitary(3, witness)
    target = marked_state_target(marked_bitstring)
    error = float(np.linalg.norm(actual - target, ord="fro") / math.sqrt(16))
    contract = OptimizationContract(
        name=f"single-violation-phase-{marked_bitstring}",
        gate_library=("H", "S", "SDG", "T", "TDG", "CNOT"),
        logical_qubits=3,
        phase_mode=PhaseMode.EXACT,
        objective=("t_count", "cnot_count", "cnot_depth"),
        proof_class="ancilla-free affine-input phase-polynomial normal form",
    )
    t_count = sum(name in {"T", "TDG"} for name, _ in witness)
    cnot_count = sum(name == "CNOT" for name, _ in witness)
    if t_count != phase.minimum_t_count or cnot_count != parity.minimum_cnot_count:
        raise AssertionError("resource mismatch")
    return MarkedStateOracleCertificate(
        marked_bitstring=marked_bitstring,
        contract=contract,
        status=OptimalityStatus.CONDITIONAL,
        minimum_t_count=phase.minimum_t_count,
        minimum_cnot_count=parity.minimum_cnot_count,
        minimum_cnot_depth=parity.minimum_cnot_depth,
        native_gate_count_upper_bound=len(witness),
        total_depth_upper_bound=_native_depth(witness, 3),
        phase_coefficients_mod8=phase.coefficients_mod8,
        cnot_sequence=parity.cnot_sequence,
        native_witness=tuple(f"{name}({','.join(map(str, qs))})" for name, qs in witness),
        exact_matrix_error=error,
        proof_scope=(
            "Exact minimum T count and minimum CNOT count/depth within the "
            "ancilla-free affine-input CNOT+T phase-polynomial normal form; "
            "outer X conjugations are Clifford and preserve these resources."
        ),
    )


# ---------------------------------------------------------------------------
# Exact macro-evaluator optimizer for BNN-like single-violation predicates
# ---------------------------------------------------------------------------


def oracle_macro_grammar() -> tuple[Macro, ...]:
    """Exactly the synthesis engine's role-aware five-wire NCT grammar.

    The flag is a target only. Allowing it to control workspace, as the old
    auditor did, proves an optimum in a different domain from learned search.
    Construct independently, and regression-test equality with the engine.
    """
    data, flag, work = (0, 1, 2), 3, (4,)
    controls = (*data, *work)
    writable = (flag, *work)
    actions = [Macro("X", (q,)) for q in (*data, flag)]
    for c in controls:
        for t in writable:
            if c != t:
                actions.append(Macro("CNOT", (c, t)))
    for t in writable:
        for c0, c1 in combinations(tuple(q for q in controls if q != t), 2):
            actions.append(Macro("TOFFOLI", (c0, c1, t)))
    family_order = {"X": 0, "CNOT": 1, "TOFFOLI": 2}
    return tuple(sorted(actions, key=lambda a: (family_order[a.kind], a.qubits)))


def _apply_macro_basis(value: int, macro: Macro) -> int:
    if macro.kind == "X":
        return value ^ (1 << macro.qubits[0])
    if macro.kind == "CNOT":
        c, t = macro.qubits
        return value ^ (1 << t) if ((value >> c) & 1) else value
    c0, c1, t = macro.qubits
    return value ^ (1 << t) if ((value >> c0) & 1 and (value >> c1) & 1) else value


def _apply_macro_mapping(mapping: tuple[int, ...], macro: Macro) -> tuple[int, ...]:
    return tuple(_apply_macro_basis(v, macro) for v in mapping)


def _marked_index(marked_bitstring: str) -> int:
    if len(marked_bitstring) != 3 or set(marked_bitstring) - {"0", "1"}:
        raise ValueError(marked_bitstring)
    return sum((bit == "1") << q for q, bit in enumerate(marked_bitstring))


def promised_input_mapping() -> tuple[int, ...]:
    return tuple(range(8))


def evaluator_target_mapping(marked_bitstring: str) -> tuple[int, ...]:
    marked = _marked_index(marked_bitstring)
    return tuple(value ^ (1 << 3) if value == marked else value for value in range(8))


@lru_cache(maxsize=8)
def prove_marked_evaluator_lexicographic_optimum(marked_bitstring: str) -> EvaluatorOptimalityCertificate:
    """Lexicographic Dijkstra proof in the declared five-wire NCT grammar."""

    start = promised_input_mapping()
    target = evaluator_target_mapping(marked_bitstring)
    actions = oracle_macro_grammar()
    zero = AdditiveResources()
    heap: list[tuple[AdditiveResources, int, tuple[int, ...]]] = [(zero, 0, start)]
    best: dict[tuple[int, ...], AdditiveResources] = {start: zero}
    parent: dict[tuple[int, ...], tuple[tuple[int, ...], Macro] | None] = {start: None}
    serial = 1
    settled = 0
    generated = 0
    goal: tuple[int, ...] | None = None
    while heap:
        cost, _, mapping = heappop(heap)
        if best.get(mapping) != cost:
            continue
        settled += 1
        if mapping == target:
            goal = mapping
            break
        for action in actions:
            generated += 1
            nxt = _apply_macro_mapping(mapping, action)
            new_cost = cost.plus(action.resources)
            old = best.get(nxt)
            if old is None or new_cost < old:
                best[nxt] = new_cost
                parent[nxt] = (mapping, action)
                heappush(heap, (new_cost, serial, nxt))
                serial += 1
    if goal is None:
        raise AssertionError("evaluator target not reachable")
    witness: list[Macro] = []
    cur = goal
    while parent[cur] is not None:
        prev, action = parent[cur]
        witness.append(action)
        cur = prev
    witness.reverse()
    optimum = best[goal]
    contract = OptimizationContract(
        name=f"promised-single-violation-evaluator-{marked_bitstring}",
        gate_library=("X", "CNOT", "TOFFOLI"),
        logical_qubits=3,
        clean_ancillas=2,
        phase_mode=PhaseMode.EXACT,
        objective=("macro_count", "t_count", "cnot_count", "native_gate_count"),
        proof_class="exact promised-input NCT mapping graph with fixed native macro costs",
    )
    return EvaluatorOptimalityCertificate(
        marked_bitstring=marked_bitstring,
        contract=contract,
        status=OptimalityStatus.EXACT,
        optimum=optimum,
        witness=tuple(m.text() for m in witness),
        incumbent_matches_optimum=(marked_bitstring == "100" and optimum == AdditiveResources(6, 21, 21, 48)),
        states_settled=settled,
        edges_generated=generated,
        proof_scope=(
            "Exact lexicographic optimum on the eight promised inputs with flag "
            "and work initialized to zero, under the target-independent five-wire "
            "X/CNOT/Toffoli grammar and fixed exact native macro lowerings."
        ),
    )


# ---------------------------------------------------------------------------
# Exact linear reversible CNOT count/depth proofs on up to four qubits
# ---------------------------------------------------------------------------


def permutation_rows(output_from_input: Sequence[int]) -> tuple[int, ...]:
    n = len(output_from_input)
    if sorted(output_from_input) != list(range(n)):
        raise ValueError("not a permutation")
    return tuple(1 << int(src) for src in output_from_input)


def _reconstruct_count_path(target, parent):
    path = []
    cur = target
    while parent[cur] is not None:
        prev, gate = parent[cur]
        path.append(gate)
        cur = prev
    path.reverse()
    return tuple(path)


def _reconstruct_layer_path(target, parent):
    path = []
    cur = target
    while parent[cur] is not None:
        prev, layer = parent[cur]
        path.append(layer)
        cur = prev
    path.reverse()
    return tuple(path)


@lru_cache(maxsize=32)
def prove_linear_reversible_optimum(name: str, output_from_input: tuple[int, ...]) -> LinearReversibleCertificate:
    n = len(output_from_input)
    if not 1 <= n <= 4:
        raise ValueError("exact auditor is bounded to one through four qubits")
    start = _identity_rows(n)
    target = permutation_rows(output_from_input)
    gates = tuple((c, t) for c in range(n) for t in range(n) if c != t)

    q = deque([start])
    parent = {start: None}
    while q:
        rows = q.popleft()
        if rows == target:
            break
        for gate in gates:
            nxt = _apply_cnot_rows(rows, *gate)
            if nxt not in parent:
                parent[nxt] = (rows, gate)
                q.append(nxt)
    if target not in parent:
        raise AssertionError("target permutation unreachable")
    count_path = _reconstruct_count_path(target, parent)

    layers = _all_disjoint_cnot_layers(n)
    qd = deque([start])
    parent_d = {start: None}
    while qd:
        rows = qd.popleft()
        if rows == target:
            break
        for layer in layers:
            nxt = rows
            for gate in layer:
                nxt = _apply_cnot_rows(nxt, *gate)
            if nxt not in parent_d:
                parent_d[nxt] = (rows, layer)
                qd.append(nxt)
    if target not in parent_d:
        raise AssertionError("target permutation unreachable by layers")
    depth_path = _reconstruct_layer_path(target, parent_d)
    return LinearReversibleCertificate(
        name=name,
        qubits=n,
        output_from_input=tuple(output_from_input),
        minimum_cnot_count=len(count_path),
        minimum_cnot_depth=len(depth_path),
        count_witness=count_path,
        depth_layers=depth_path,
        states_explored_count=len(parent),
        states_explored_depth=len(parent_d),
    )


def four_qubit_linear_audits() -> tuple[LinearReversibleCertificate, ...]:
    return (
        prove_linear_reversible_optimum("register-half-swap", (2, 3, 0, 1)),
        prove_linear_reversible_optimum("wire-reversal", (3, 2, 1, 0)),
    )


# ---------------------------------------------------------------------------
# Toffoli audit and serialization helpers
# ---------------------------------------------------------------------------


def toffoli_from_ccz_witness() -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (("H", (2,)),) + synthesize_ccz_from_proofs() + (("H", (2,)),)


def toffoli_target_matrix() -> np.ndarray:
    target = np.zeros((8, 8), dtype=np.complex128)
    for x in range(8):
        y = x ^ (1 << 2) if ((x & 1) and (x & 2)) else x
        target[y, x] = 1
    return target


def certify_toffoli_normal_form() -> dict:
    witness = toffoli_from_ccz_witness()
    actual = native_unitary(3, witness)
    error = float(np.linalg.norm(actual - toffoli_target_matrix(), ord="fro") / math.sqrt(16))
    return {
        "name": "three-qubit-toffoli",
        "status": OptimalityStatus.CONDITIONAL.value,
        "t_count": sum(g[0] in {"T", "TDG"} for g in witness),
        "cnot_count": sum(g[0] == "CNOT" for g in witness),
        "cnot_depth": prove_ccz_parity_network_optimum().minimum_cnot_depth,
        "native_gate_count": len(witness),
        "total_depth_upper_bound": _native_depth(witness, 3),
        "exact_matrix_error": error,
        "proof_scope": (
            "H-conjugated ancilla-free phase-polynomial normal form. The "
            "exhaustive proof establishes 7 T gates and a 6-CNOT parity network "
            "inside this declared class; no broader native-depth claim is made."
        ),
        "witness": [f"{name}({','.join(map(str, qs))})" for name, qs in witness],
    }


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "AdditiveResources", "EvaluatorOptimalityCertificate",
    "LinearReversibleCertificate", "Macro", "MarkedStateOracleCertificate",
    "OptimizationContract", "OptimalityStatus", "ParityNetworkProof",
    "PhaseMode", "PhasePolynomialProof", "certify_marked_state_oracle",
    "certify_toffoli_normal_form", "evaluator_target_mapping",
    "four_qubit_linear_audits", "marked_state_target", "native_unitary",
    "oracle_macro_grammar", "phase_exponents", "permutation_rows",
    "prove_ccz_parity_network_optimum", "prove_ccz_phase_polynomial_t_optimum",
    "prove_linear_reversible_optimum", "prove_marked_evaluator_lexicographic_optimum",
    "promised_input_mapping", "synthesize_ccz_from_proofs",
    "synthesize_marked_state_phase_oracle", "toffoli_from_ccz_witness",
    "write_json",
]
