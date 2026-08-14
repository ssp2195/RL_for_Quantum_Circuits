"""Exact three-qubit Toffoli search in a CCZ parity-network normal form.

The only deterministic gates outside the diagonal CCZ core are the two
Hadamards on q2.  Inside the core this module enumerates every one-gate
continuation admitted by the seven-term phase-polynomial identity; it never
reads, stores, or replays the Stage 2 known Toffoli witness.  A learned policy
can therefore rank persistent frontier records, but cannot select a gate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from types import MappingProxyType
from typing import Hashable, Mapping

from canonical.canonicalizer import Canonicalizer
from canonical.hash import stable_hash
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from search.action import Action
from search.node import SearchNode


REQUIRED_PHASE_TERMS: Mapping[int, int] = MappingProxyType(
    {
        0b001: +1,
        0b010: +1,
        0b100: +1,
        0b011: -1,
        0b101: -1,
        0b110: -1,
        0b111: +1,
    }
)
"""The exact signed parities in ``4*x0*x1*x2 (mod 8)``."""

PHASE_TERM_ORDER = tuple(sorted(REQUIRED_PHASE_TERMS))
"""Stable ordinal order used by :attr:`ToffoliParityProgress.emitted_terms`."""

_TERM_BIT = {mask: index for index, mask in enumerate(PHASE_TERM_ORDER)}
_FULL_EMITTED_TERMS = (1 << len(PHASE_TERM_ORDER)) - 1
_UNREACHABLE_CNOT_COST = 1_000_000
IDENTITY_BASIS_ROWS = (0b001, 0b010, 0b100)
CORE_CNOT_LIMIT = 6
CORE_PHASE_LIMIT = 7
NORMAL_FORM_CONTRACT = (
    "toffoli-ccz-seven-term-parity-network",
    "outer-H-q2",
    "phase-terms-once",
    "return-linear-basis-to-identity",
    "all-to-all-directed-cnot",
)


class ToffoliStage(str, Enum):
    """The only continuation stages in the declared normal form."""

    PRE_H = "PRE_H"
    CORE = "CORE"
    POST_H = "POST_H"
    DONE = "DONE"


@dataclass(frozen=True, slots=True)
class ToffoliParityProgress:
    """Immutable continuation information derived from an authoritative DAG.

    ``emitted_terms`` is an ordinal seven-bit set: bit ``i`` corresponds to
    ``PHASE_TERM_ORDER[i]``.  It deliberately does *not* use ``1 << mask``;
    parities are labels, not a compact bitset encoding themselves.
    """

    stage: ToffoliStage
    basis_rows: tuple[int, int, int]
    emitted_terms: int

    def canonical_payload(self) -> tuple[object, ...]:
        return (
            self.stage.value,
            tuple(int(row) for row in self.basis_rows),
            int(self.emitted_terms),
            NORMAL_FORM_CONTRACT,
        )

    @property
    def emitted_count(self) -> int:
        return int(self.emitted_terms).bit_count()

    @property
    def all_terms_emitted(self) -> bool:
        return self.emitted_terms == _FULL_EMITTED_TERMS


class InvalidToffoliParityPrefix(ValueError):
    """Raised when a public DAG is not a prefix of this normal form."""


def _parity(mask: int, assignment: int) -> int:
    return (int(mask) & int(assignment)).bit_count() & 1


def phase_identity_rows() -> tuple[dict[str, int], ...]:
    """Return the exhaustive seven-term CCZ phase identity table.

    Each row proves ``4*x0*x1*x2 == sum(c*p(x)) (mod 8)`` for one Boolean
    input assignment.  The table is used for reporting and tests only; it is
    not a hidden route planner for the search.
    """

    rows: list[dict[str, int]] = []
    for assignment in range(8):
        lhs = 4 * ((assignment & 1) != 0) * ((assignment & 2) != 0) * (
            (assignment & 4) != 0
        )
        rhs = sum(
            coefficient * _parity(mask, assignment)
            for mask, coefficient in REQUIRED_PHASE_TERMS.items()
        )
        rows.append(
            {
                "assignment": assignment,
                "lhs_mod_8": int(lhs % 8),
                "rhs_mod_8": int(rhs % 8),
                "matches": int(lhs % 8 == rhs % 8),
            }
        )
    return tuple(rows)


def phase_identity_holds() -> bool:
    """Return whether the seven-term identity holds for all eight inputs."""

    return all(bool(row["matches"]) for row in phase_identity_rows())


def _apply_cnot_to_rows(
    rows: tuple[int, int, int], control: int, target: int
) -> tuple[int, int, int]:
    if control == target or control not in range(3) or target not in range(3):
        raise ValueError("a Toffoli parity CNOT needs distinct q0..q2 operands")
    result = list(rows)
    result[target] ^= result[control]
    return tuple(result)  # type: ignore[return-value]


def _cnot_basis_distances() -> Mapping[tuple[int, int, int], int]:
    """Compute exact CNOT distance to identity on the 3x3 invertible graph."""

    distances: dict[tuple[int, int, int], int] = {IDENTITY_BASIS_ROWS: 0}
    queue: deque[tuple[int, int, int]] = deque((IDENTITY_BASIS_ROWS,))
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for control in range(3):
            for target in range(3):
                if control == target:
                    continue
                child = _apply_cnot_to_rows(current, control, target)
                if child not in distances:
                    distances[child] = next_distance
                    queue.append(child)
    # GL(3, F_2) has (7 * 6 * 4) = 168 elements.  This makes accidental
    # traversal of a non-invertible or incomplete row graph visible at import.
    if len(distances) != 168:  # pragma: no cover - construction invariant
        raise AssertionError("three-wire CNOT basis graph must have 168 vertices")
    return MappingProxyType(distances)


CNOT_BASIS_DISTANCE_TO_IDENTITY = _cnot_basis_distances()
"""Exact minimum CNOT distances from each reachable basis to identity."""


@lru_cache(maxsize=262_144)
def _core_can_reach_terminal(
    basis_rows: tuple[int, int, int],
    emitted_terms: int,
    cnot_count: int,
) -> bool:
    """Return exact compact-graph reachability to a structural CCZ core.

    Every compact transition consumes either one previously un-emitted phase
    term or one CNOT.  The graph is therefore acyclic under
    ``popcount(emitted_terms) + cnot_count`` and this memoized reverse-style
    reachability check is exact rather than heuristic.  It is a safe pruning
    oracle only: its Boolean result is never exposed to the policy features.
    """

    if emitted_terms == _FULL_EMITTED_TERMS and basis_rows == IDENTITY_BASIS_ROWS:
        return True
    if cnot_count > CORE_CNOT_LIMIT:
        return False

    for mask in basis_rows:
        term_index = _TERM_BIT.get(mask)
        if term_index is None:
            continue
        bit = 1 << term_index
        if emitted_terms & bit:
            continue
        if _core_can_reach_terminal(basis_rows, emitted_terms | bit, cnot_count):
            return True

    if cnot_count >= CORE_CNOT_LIMIT:
        return False

    for control in range(3):
        for target in range(3):
            if control == target:
                continue
            child_rows = _apply_cnot_to_rows(basis_rows, control, target)
            next_count = cnot_count + 1
            # This is the mandatory distance rule; include it in the compact
            # graph itself so the table proves reachability after every
            # pruning rule applied by the expander.
            if CNOT_BASIS_DISTANCE_TO_IDENTITY[child_rows] > CORE_CNOT_LIMIT - next_count:
                continue
            if _core_can_reach_terminal(child_rows, emitted_terms, next_count):
                return True
    return False


@lru_cache(maxsize=262_144)
def _minimum_additional_core_cnot(
    basis_rows: tuple[int, int, int], emitted_terms: int, cnot_count: int
) -> int:
    """Return the exact minimum further CNOT count in the compact graph.

    The value is an admissible resource lower bound, not a policy feature or
    hidden witness distance.  It is computed solely from the public phase
    term specification and all six CNOT row transitions.
    """

    if emitted_terms == _FULL_EMITTED_TERMS and basis_rows == IDENTITY_BASIS_ROWS:
        return 0
    if cnot_count > CORE_CNOT_LIMIT:
        return _UNREACHABLE_CNOT_COST

    best = _UNREACHABLE_CNOT_COST
    for mask in basis_rows:
        term_index = _TERM_BIT.get(mask)
        if term_index is None:
            continue
        bit = 1 << term_index
        if emitted_terms & bit:
            continue
        best = min(
            best,
            _minimum_additional_core_cnot(
                basis_rows, emitted_terms | bit, cnot_count
            ),
        )

    if cnot_count >= CORE_CNOT_LIMIT:
        return best
    for control in range(3):
        for target in range(3):
            if control == target:
                continue
            child_rows = _apply_cnot_to_rows(basis_rows, control, target)
            next_count = cnot_count + 1
            if CNOT_BASIS_DISTANCE_TO_IDENTITY[child_rows] > CORE_CNOT_LIMIT - next_count:
                continue
            suffix = _minimum_additional_core_cnot(
                child_rows, emitted_terms, next_count
            )
            if suffix < _UNREACHABLE_CNOT_COST:
                best = min(best, 1 + suffix)
    return best


def _operation_tuple(state: CircuitState) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (gate.gate_type.name, tuple(int(qubit) for qubit in gate.qubits))
        for gate in state.dag.gates
    )


@lru_cache(maxsize=131_072)
def _analyze_operations(
    operations: tuple[tuple[str, tuple[int, ...]], ...]
) -> ToffoliParityProgress:
    """Analyze immutable operation tuples without object-identity caching."""

    if not operations:
        return ToffoliParityProgress(ToffoliStage.PRE_H, IDENTITY_BASIS_ROWS, 0)

    if operations[0] != ("H", (2,)):
        raise InvalidToffoliParityPrefix("the normal form must begin with H(2)")

    rows = IDENTITY_BASIS_ROWS
    emitted = 0
    cnot_count = 0
    phase_count = 0

    for index, (name, qubits) in enumerate(operations[1:], start=1):
        final_operation = index == len(operations) - 1
        if name == "H":
            if qubits != (2,) or not final_operation:
                raise InvalidToffoliParityPrefix(
                    "H(2) is allowed only as the final normal-form operation"
                )
            if emitted != _FULL_EMITTED_TERMS or rows != IDENTITY_BASIS_ROWS:
                raise InvalidToffoliParityPrefix(
                    "the final H(2) requires all phases and identity basis"
                )
            return ToffoliParityProgress(ToffoliStage.DONE, rows, emitted)

        if name == "CNOT":
            if len(qubits) != 2:
                raise InvalidToffoliParityPrefix("CNOT needs a control and target")
            control, target = qubits
            rows = _apply_cnot_to_rows(rows, control, target)
            cnot_count += 1
            if cnot_count > CORE_CNOT_LIMIT:
                raise InvalidToffoliParityPrefix("the parity core permits at most six CNOTs")
            continue

        if name not in {"T", "TDG"} or len(qubits) != 1:
            raise InvalidToffoliParityPrefix(
                "the parity core permits only CNOT, T, and TDG operations"
            )
        qubit = qubits[0]
        if qubit not in range(3):
            raise InvalidToffoliParityPrefix("phase operation qubit is outside q0..q2")
        mask = rows[qubit]
        expected_sign = REQUIRED_PHASE_TERMS.get(mask)
        if expected_sign is None:
            raise InvalidToffoliParityPrefix("phase operation is on a non-required parity")
        expected_name = "T" if expected_sign > 0 else "TDG"
        if name != expected_name:
            raise InvalidToffoliParityPrefix("phase operation has the wrong required sign")
        bit = 1 << _TERM_BIT[mask]
        if emitted & bit:
            raise InvalidToffoliParityPrefix("required phase parity was emitted twice")
        emitted |= bit
        phase_count += 1
        if phase_count > CORE_PHASE_LIMIT:
            raise InvalidToffoliParityPrefix("the parity core permits exactly seven phases")

    stage = (
        ToffoliStage.POST_H
        if emitted == _FULL_EMITTED_TERMS and rows == IDENTITY_BASIS_ROWS
        else ToffoliStage.CORE
    )
    return ToffoliParityProgress(stage, rows, emitted)


def analyze_toffoli_prefix(state: CircuitState) -> ToffoliParityProgress:
    """Derive constrained-progress information from the authoritative DAG.

    No record ID, frontier position, cached digest, or stored reference path
    enters this calculation.  The cache key is the immutable complete DAG gate
    operation tuple so copied witnesses get identical analysis safely.
    """

    if not isinstance(state, CircuitState):
        raise TypeError("Toffoli parity analysis requires a CircuitState")
    if state.dag.num_qubits != 3:
        raise ValueError("Toffoli parity search supports exactly three qubits")
    # Public ``CircuitState`` construction and every normal-form transition
    # already maintain the DAG invariants.  Do not re-walk the full mutable
    # graph for each frontier feature/potential query: the analyzer's cache is
    # intentionally keyed by the complete immutable operation tuple below,
    # and final dense certification still validates the authoritative DAG.
    return _analyze_operations(_operation_tuple(state))


class ToffoliProblemCanonicalizer:
    """Continuation-safe archive identity for the constrained normal form."""

    schema_version = "toffoli-parity-network-v1"

    def __init__(self, *, phase_sensitive: bool = False) -> None:
        self.phase_sensitive = bool(phase_sensitive)
        self.base_canonicalizer = Canonicalizer(phase_sensitive=self.phase_sensitive)

    def semantic_key(self, state: CircuitState) -> tuple[object, ...]:
        progress = analyze_toffoli_prefix(state)
        return (
            self.schema_version,
            self.base_canonicalizer.semantic_key(state),
            progress.canonical_payload(),
            NORMAL_FORM_CONTRACT,
        )

    identity_key = semantic_key

    def identity_hash(self, state: CircuitState) -> str:
        return stable_hash(repr(self.semantic_key(state)))

    def resource_hash(self, state: CircuitState) -> str:
        return self.base_canonicalizer.resource_hash(state)

    def combined_hash(self, state: CircuitState) -> str:
        return stable_hash(repr((self.semantic_key(state), self.resource_hash(state))))


class ToffoliParityNetworkProblem:
    """Enumerate an exact seven-phase CCZ parity-network normal form.

    This deliberately solves a narrowly declared synthesis problem, not
    arbitrary depth-15 Clifford+T search.  Every child is created through the
    public :meth:`CircuitState.apply_gate` transition and independently
    analyzed from its resulting DAG before being returned to the frontier.
    """

    name = "toffoli-parity-network"
    schema_version = "toffoli-parity-network-v1"
    qubit_convention = "q0 is LSB; controls q0,q1; target q2"
    REQUIRED_PHASE_TERMS = REQUIRED_PHASE_TERMS
    CNOT_BASIS_DISTANCE_TO_IDENTITY = CNOT_BASIS_DISTANCE_TO_IDENTITY
    reject_failed_terminal = True

    @staticmethod
    def _phase_term_digest() -> str:
        payload = repr(tuple(REQUIRED_PHASE_TERMS.items())).encode("ascii")
        return f"sha256:{sha256(payload).hexdigest()}"

    @property
    def target_fingerprint(self) -> str:
        # This is a problem-contract fingerprint, not a witness.  Runners
        # replace it with their analytical CCX matrix fingerprint for policy
        # checkpoint metadata.
        return f"toffoli-parity-terms:{self._phase_term_digest().split(':', 1)[1]}"

    def initial_state(self, config: object) -> CircuitState:
        num_qubits = getattr(config, "num_qubits", None)
        if num_qubits != 3:
            raise ValueError("Toffoli parity network requires Config(num_qubits=3)")
        budget = getattr(config, "budget", None)
        if not isinstance(budget, ResourceBudget):
            raise TypeError("Toffoli parity network requires a ResourceBudget")
        return self.default_initial_state(budget=budget)

    @staticmethod
    def default_initial_state(*, budget: ResourceBudget) -> CircuitState:
        if not isinstance(budget, ResourceBudget):
            raise TypeError("budget must be a ResourceBudget")
        return CircuitState(CircuitDAG(3), budget)

    def analyze(self, state: CircuitState) -> ToffoliParityProgress:
        return analyze_toffoli_prefix(state)

    def canonicalizer(self, *, phase_sensitive: bool = False) -> ToffoliProblemCanonicalizer:
        return ToffoliProblemCanonicalizer(phase_sensitive=phase_sensitive)

    @staticmethod
    def _complete_core(progress: ToffoliParityProgress) -> bool:
        return (
            progress.emitted_terms == _FULL_EMITTED_TERMS
            and progress.basis_rows == IDENTITY_BASIS_ROWS
        )

    @staticmethod
    def _remaining_outer_h(progress: ToffoliParityProgress) -> int:
        if progress.stage is ToffoliStage.PRE_H:
            return 2
        if progress.stage in {ToffoliStage.CORE, ToffoliStage.POST_H}:
            return 1
        return 0

    @classmethod
    def _resource_feasible(
        cls, state: CircuitState, progress: ToffoliParityProgress
    ) -> bool:
        """Apply exact normal-form resource lower bounds before expansion.

        Seven distinct phase terms are compulsory.  The compact graph supplies
        an exact lower bound for remaining CNOTs, and the two outer Hadamards
        are fixed by the declared normal form.  Thus rejecting an infeasible
        prefix cannot remove a suffix accepted by this problem.
        """

        remaining_phases = len(PHASE_TERM_ORDER) - progress.emitted_count
        if state.t_count + remaining_phases > state.budget.max_t_count:
            return False

        if progress.stage is ToffoliStage.PRE_H:
            minimum_cnot = _minimum_additional_core_cnot(
                IDENTITY_BASIS_ROWS, 0, 0
            )
        elif progress.stage is ToffoliStage.DONE:
            minimum_cnot = 0
        else:
            minimum_cnot = _minimum_additional_core_cnot(
                progress.basis_rows,
                progress.emitted_terms,
                state.two_qubit_count,
            )
        if minimum_cnot >= _UNREACHABLE_CNOT_COST:
            return False

        maximum_cnot = state.budget.max_two_qubit_count
        if maximum_cnot is not None and state.two_qubit_count + minimum_cnot > maximum_cnot:
            return False

        minimum_future_gates = (
            remaining_phases + minimum_cnot + cls._remaining_outer_h(progress)
        )
        return state.num_gates + minimum_future_gates <= state.budget.max_gates

    def is_terminal_candidate(self, value: SearchNode | CircuitState) -> bool:
        state = value.state if isinstance(value, SearchNode) else value
        return self.analyze(state).stage is ToffoliStage.DONE

    @staticmethod
    def _child(node: SearchNode, gate: Gate) -> SearchNode | None:
        state = node.state.copy()
        if not state.apply_gate(gate):
            return None
        # ``Action`` remains the public witness/reconstruction representation.
        action = Action(gate.gate_type, tuple(gate.qubits))
        return SearchNode(priority=0.0, state=state, parent=node, action=action)

    def _append_if_valid(
        self,
        children: list[SearchNode],
        node: SearchNode,
        gate: Gate,
        *,
        cnot_distance_prune: bool = False,
    ) -> None:
        child = self._child(node, gate)
        if child is None:
            return
        progress = self.analyze(child.state)
        if not self._resource_feasible(child.state, progress):
            return
        if cnot_distance_prune:
            remaining = CORE_CNOT_LIMIT - child.state.two_qubit_count
            distance = CNOT_BASIS_DISTANCE_TO_IDENTITY[progress.basis_rows]
            if distance > remaining:
                return
        if progress.stage is ToffoliStage.CORE and not _core_can_reach_terminal(
            progress.basis_rows,
            progress.emitted_terms,
            child.state.two_qubit_count,
        ):
            return
        children.append(child)

    def expand(self, node: SearchNode) -> list[SearchNode]:
        """Return every legal one-gate normal-form continuation of ``node``."""

        if not isinstance(node, SearchNode):
            raise TypeError("Toffoli parity expansion requires a SearchNode")
        progress = self.analyze(node.state)
        children: list[SearchNode] = []

        if not self._resource_feasible(node.state, progress):
            return children

        if progress.stage is ToffoliStage.PRE_H:
            self._append_if_valid(children, node, Gate(GateType.H, (2,)))
            return children

        if progress.stage is ToffoliStage.POST_H:
            self._append_if_valid(children, node, Gate(GateType.H, (2,)))
            return children

        if progress.stage is ToffoliStage.DONE:
            return children

        # Phase terms are determined solely by the currently exposed parity
        # rows and the required signed polynomial.  No stored next gate is
        # consulted, and no invalid phase alternative is generated.
        for qubit, mask in enumerate(progress.basis_rows):
            sign = REQUIRED_PHASE_TERMS.get(mask)
            term_index = _TERM_BIT.get(mask)
            if sign is None or term_index is None:
                continue
            bit = 1 << term_index
            if progress.emitted_terms & bit:
                continue
            phase_gate = GateType.T if sign > 0 else GateType.TDG
            self._append_if_valid(children, node, Gate(phase_gate, (qubit,)))

        # CNOT candidates are all directed pairs in lexicographic order.  The
        # exact distance table only rejects states that cannot return to the
        # identity linear basis within the remaining six-CNOT core budget.
        if node.state.two_qubit_count < CORE_CNOT_LIMIT:
            for control in range(3):
                for target in range(3):
                    if control == target:
                        continue
                    self._append_if_valid(
                        children,
                        node,
                        Gate(GateType.CNOT, (control, target)),
                        cnot_distance_prune=True,
                    )
        return children

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "qubit_convention": self.qubit_convention,
            "normal_form": (
                "Exact Toffoli synthesis within the fixed seven-term CCZ parity-network "
                "normal form; not a proof of general unconstrained Clifford+T synthesis."
            ),
            "required_phase_terms": {
                f"{mask:03b}": coefficient
                for mask, coefficient in REQUIRED_PHASE_TERMS.items()
            },
            "phase_term_order": list(PHASE_TERM_ORDER),
            "phase_term_digest": self._phase_term_digest(),
            "core_cnot_limit": CORE_CNOT_LIMIT,
            "core_phase_limit": CORE_PHASE_LIMIT,
            "basis_graph_vertices": len(CNOT_BASIS_DISTANCE_TO_IDENTITY),
            "normal_form_contract": NORMAL_FORM_CONTRACT,
        }


__all__ = [
    "CNOT_BASIS_DISTANCE_TO_IDENTITY",
    "CORE_CNOT_LIMIT",
    "CORE_PHASE_LIMIT",
    "IDENTITY_BASIS_ROWS",
    "InvalidToffoliParityPrefix",
    "NORMAL_FORM_CONTRACT",
    "PHASE_TERM_ORDER",
    "REQUIRED_PHASE_TERMS",
    "ToffoliParityNetworkProblem",
    "ToffoliParityProgress",
    "ToffoliProblemCanonicalizer",
    "ToffoliStage",
    "analyze_toffoli_prefix",
    "phase_identity_holds",
    "phase_identity_rows",
]
