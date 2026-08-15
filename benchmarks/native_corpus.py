"""Seeded, replayable targets from the unrestricted native gate grammar.

The corpus is deliberately a data layer, not a search problem.  A benchmark
consumer receives a dense target and public provenance; it must not use the
generator witness as a reachability oracle or as a substitute solution.  The
witness is retained only so a failed seed can be replayed and audited.

Dataset partitions are separated by a phase-normalized digest of the dense
unitary, rather than by generator-circuit syntax.  This prevents two different
native witnesses for the same target from leaking across partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from types import MappingProxyType
from typing import Mapping

import numpy as np

from canonical.canonicalizer import Canonicalizer
from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from search.action_space import generate_actions


CORPUS_SCHEMA = "native-clifford-t-corpus-v1"
SEMANTIC_IDENTITY_SCHEMA = "phase-normalized-dense-sha256-v1"
NATIVE_GATE_NAMES = ("H", "S", "SDG", "T", "TDG", "CNOT")
SPLIT_ORDER = ("train", "validation", "test")
DEFAULT_SPLIT_SEEDS: Mapping[str, int] = MappingProxyType(
    {"train": 1_729, "validation": 2_753, "test": 3_769}
)


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _phase_normalized_matrix(unitary: np.ndarray, *, decimals: int) -> np.ndarray:
    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix")
    if decimals < 8 or decimals > 15:
        raise ValueError("decimals must lie in [8, 15]")

    flat = matrix.reshape(-1)
    # Use the first numerically significant entry, rather than the largest
    # entry, because equal-magnitude ties can drift across different replay
    # paths for one semantic target.
    significant = np.flatnonzero(np.abs(flat) > 10.0 ** (-(decimals - 2)))
    if significant.size == 0:
        raise ValueError("unitary has no stable non-zero phase anchor")
    anchor = flat[int(significant[0])]
    phase = anchor / abs(anchor)
    normalized = matrix / phase
    real = np.round(normalized.real, decimals=decimals)
    imag = np.round(normalized.imag, decimals=decimals)
    real[np.abs(real) < 10.0 ** (-decimals)] = 0.0
    imag[np.abs(imag) < 10.0 ** (-decimals)] = 0.0
    return real + 1.0j * imag


def semantic_target_digest(unitary: np.ndarray, *, decimals: int = 12) -> str:
    """Return a deterministic global-phase-invariant dense target identity."""

    normalized = _phase_normalized_matrix(unitary, decimals=decimals)
    payload = {
        "schema": SEMANTIC_IDENTITY_SCHEMA,
        "shape": list(normalized.shape),
        "decimals": decimals,
        "entries": [
            [float(value.real), float(value.imag)]
            for value in normalized.reshape(-1)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _canonical_diagnostic_digest(num_qubits: int, witness: tuple[Gate, ...]) -> str:
    """Hash the symbolic key for diagnostics, never for target certification."""

    diagnostic_budget = ResourceBudget(
        max_t_count=64,
        max_two_qubit_count=64,
        max_depth=64,
        max_gates=64,
    )
    state = CircuitState(CircuitDAG(num_qubits), diagnostic_budget)
    for gate in witness:
        if not state.apply_gate(gate):  # pragma: no cover - fixed ample budget
            raise AssertionError("diagnostic replay unexpectedly exceeded its budget")
    payload = repr(Canonicalizer().semantic_key(state)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeTargetCase:
    """One exact reachable target with immutable replay provenance."""

    target_id: str
    split: str
    suite: str
    split_seed: int
    generator_seed: int
    num_qubits: int
    witness: tuple[Gate, ...]
    unitary: np.ndarray
    canonical_diagnostic_digest: str

    def __post_init__(self) -> None:
        if self.split not in SPLIT_ORDER:
            raise ValueError(f"unknown dataset split {self.split!r}")
        if self.suite not in {"semantic", "bounded_synthesis"}:
            raise ValueError(f"unknown corpus suite {self.suite!r}")
        if self.num_qubits not in {1, 2, 3}:
            raise ValueError("native corpus cases must use 1, 2, or 3 qubits")
        if not self.witness:
            raise ValueError("a corpus target must retain a non-empty replay witness")
        if any(gate.gate_type.name not in NATIVE_GATE_NAMES for gate in self.witness):
            raise ValueError("corpus witness contains a non-native gate")

        target = SynthesisTarget(self.unitary)
        if target.num_qubits != self.num_qubits:
            raise ValueError("target dimension and num_qubits disagree")
        matrix = np.array(target.unitary, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "unitary", matrix)
        object.__setattr__(self, "witness", tuple(self.witness))

        observed_id = semantic_target_digest(matrix)
        if observed_id != self.target_id:
            raise ValueError("target_id does not match the dense target semantics")

    def synthesis_target(self) -> SynthesisTarget:
        """Return the only target object a generic search run should consume."""

        return SynthesisTarget(self.unitary)

    def metadata(self) -> dict[str, object]:
        """Return JSON-ready replay metadata with an explicit anti-shortcut flag."""

        return {
            "schema_version": CORPUS_SCHEMA,
            "semantic_identity_schema": SEMANTIC_IDENTITY_SCHEMA,
            "target_id": self.target_id,
            "split": self.split,
            "suite": self.suite,
            "split_seed": self.split_seed,
            "generator_seed": self.generator_seed,
            "num_qubits": self.num_qubits,
            "generator_length": len(self.witness),
            "generator_profile": (
                "uniform-native"
                if self.suite == "semantic" or self.num_qubits == 1
                else "uniform-native-conditioned-on-at-least-one-directed-cnot"
            ),
            "generator_witness": [
                {"gate": gate.gate_type.name, "qubits": list(gate.qubits)}
                for gate in self.witness
            ],
            "canonical_diagnostic_digest": self.canonical_diagnostic_digest,
            "target_specific_reachability_oracle": False,
            "witness_use": "replay-and-audit-only",
        }


@dataclass(frozen=True, slots=True)
class NativeTargetCorpus:
    """A deterministic pair of property and bounded-synthesis suites."""

    semantic_cases: tuple[NativeTargetCase, ...]
    bounded_synthesis_cases: tuple[NativeTargetCase, ...]
    split_seeds: tuple[tuple[str, int], ...]

    @property
    def all_cases(self) -> tuple[NativeTargetCase, ...]:
        return self.semantic_cases + self.bounded_synthesis_cases

    def cases(self, *, suite: str, split: str) -> tuple[NativeTargetCase, ...]:
        if suite not in {"semantic", "bounded_synthesis"}:
            raise ValueError(f"unknown corpus suite {suite!r}")
        if split not in SPLIT_ORDER:
            raise ValueError(f"unknown dataset split {split!r}")
        source = (
            self.semantic_cases
            if suite == "semantic"
            else self.bounded_synthesis_cases
        )
        return tuple(case for case in source if case.split == split)

    def manifest(self) -> dict[str, object]:
        identities = [case.target_id for case in self.all_cases]
        return {
            "schema_version": CORPUS_SCHEMA,
            "semantic_identity_schema": SEMANTIC_IDENTITY_SCHEMA,
            "split_seeds": dict(self.split_seeds),
            "semantic_case_count": len(self.semantic_cases),
            "bounded_synthesis_case_count": len(self.bounded_synthesis_cases),
            "target_ids_are_globally_unique": len(identities) == len(set(identities)),
            "native_gate_names": list(NATIVE_GATE_NAMES),
            "qubit_endian": "little; q0 is the least-significant basis bit",
            "target_specific_reachability_oracle": False,
        }


def _sample_witness(
    *,
    num_qubits: int,
    length: int,
    generator_seed: int,
) -> tuple[Gate, ...]:
    actions = tuple(generate_actions(num_qubits))
    rng = random.Random(generator_seed)
    return tuple(
        Gate(action.gate_type, tuple(action.qubits))
        for action in (rng.choice(actions) for _ in range(length))
    )


def _build_suite(
    *,
    suite: str,
    per_split: int,
    split_seeds: Mapping[str, int],
    seen_target_ids: set[str],
) -> tuple[NativeTargetCase, ...]:
    cases: list[NativeTargetCase] = []
    max_attempts_per_case = 1_000
    for split in SPLIT_ORDER:
        split_seed = split_seeds[split]
        split_rng = random.Random((split_seed << 1) ^ (0 if suite == "semantic" else 1))
        for index in range(per_split):
            num_qubits = 1 + (index % 3)
            # The property suite stresses longer symbolic words.  The synthesis
            # suite intentionally stays shallow enough for multiple schedulers.
            if suite == "semantic":
                length = 3 + (index % 5)
            else:
                length = 1 if num_qubits == 1 else 2

            for _ in range(max_attempts_per_case):
                generator_seed = split_rng.getrandbits(63)
                witness = _sample_witness(
                    num_qubits=num_qubits,
                    length=length,
                    generator_seed=generator_seed,
                )
                if (
                    suite == "bounded_synthesis"
                    and num_qubits > 1
                    and not any(gate.gate_type is GateType.CNOT for gate in witness)
                ):
                    continue
                # This independent dense replay is the target oracle.  The
                # canonical form below remains diagnostics-only.
                unitary = unitary_from_gates(num_qubits, witness)
                target_id = semantic_target_digest(unitary)
                if target_id in seen_target_ids:
                    continue
                seen_target_ids.add(target_id)
                cases.append(
                    NativeTargetCase(
                        target_id=target_id,
                        split=split,
                        suite=suite,
                        split_seed=split_seed,
                        generator_seed=generator_seed,
                        num_qubits=num_qubits,
                        witness=witness,
                        unitary=unitary,
                        canonical_diagnostic_digest=_canonical_diagnostic_digest(
                            num_qubits, witness
                        ),
                    )
                )
                break
            else:  # pragma: no cover - generous bound and tiny target counts
                raise RuntimeError(
                    f"could not generate a new semantic target for {suite}/{split}"
                )
    return tuple(cases)


def build_native_target_corpus(
    *,
    semantic_per_split: int = 6,
    bounded_synthesis_per_split: int = 3,
    split_seeds: Mapping[str, int] | None = None,
) -> NativeTargetCorpus:
    """Generate deterministic semantic and held-out bounded-search datasets.

    ``bounded_synthesis_per_split`` defaults to three so each partition has one
    1-, 2-, and 3-qubit target.  Increasing it cycles through the same widths.
    """

    semantic_per_split = _validate_positive_int(
        semantic_per_split, name="semantic_per_split"
    )
    bounded_synthesis_per_split = _validate_positive_int(
        bounded_synthesis_per_split, name="bounded_synthesis_per_split"
    )
    normalized_seeds = dict(DEFAULT_SPLIT_SEEDS if split_seeds is None else split_seeds)
    if set(normalized_seeds) != set(SPLIT_ORDER):
        raise ValueError(f"split_seeds must define exactly {SPLIT_ORDER!r}")
    for split, seed in normalized_seeds.items():
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed for {split!r} must be a non-negative integer")

    seen_target_ids: set[str] = set()
    # Reserve the deliberately tiny bounded instances first.  This avoids a
    # larger property corpus accidentally consuming every one-gate identity
    # available to a later held-out partition.
    bounded_cases = _build_suite(
        suite="bounded_synthesis",
        per_split=bounded_synthesis_per_split,
        split_seeds=normalized_seeds,
        seen_target_ids=seen_target_ids,
    )
    semantic_cases = _build_suite(
        suite="semantic",
        per_split=semantic_per_split,
        split_seeds=normalized_seeds,
        seen_target_ids=seen_target_ids,
    )
    return NativeTargetCorpus(
        semantic_cases=semantic_cases,
        bounded_synthesis_cases=bounded_cases,
        split_seeds=tuple((split, normalized_seeds[split]) for split in SPLIT_ORDER),
    )


def ccz_reference_unitary() -> np.ndarray:
    """Return analytical CCZ(0, 1, 2) under the repository LSB convention."""

    unitary = np.eye(8, dtype=np.complex128)
    unitary[7, 7] = -1.0
    unitary.setflags(write=False)
    return unitary


def known_ccz_native_witness() -> tuple[Gate, ...]:
    """Return the fixed diagonal phase-network inside the known CCX witness.

    This is a reference certification witness, not a reachability oracle for
    generic search.  The relationship is CCZ = H(2) CCX H(2).
    """

    from benchmarks.toffoli import KNOWN_TOFFOLI_GATES

    if (
        KNOWN_TOFFOLI_GATES[0] != Gate(GateType.H, (2,))
        or KNOWN_TOFFOLI_GATES[-1] != Gate(GateType.H, (2,))
    ):  # pragma: no cover - catches accidental benchmark drift
        raise AssertionError("known Toffoli witness no longer has the declared outer H gates")
    return tuple(KNOWN_TOFFOLI_GATES[1:-1])


@dataclass(frozen=True, slots=True)
class CCZReferenceBenchmark:
    """A separately scoped analytical CCZ target and known native witness."""

    unitary: np.ndarray
    witness: tuple[Gate, ...]

    def __post_init__(self) -> None:
        target = SynthesisTarget(self.unitary)
        if target.num_qubits != 3:
            raise ValueError("CCZ reference must be a three-qubit unitary")
        matrix = np.array(target.unitary, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "unitary", matrix)
        object.__setattr__(self, "witness", tuple(self.witness))

    def synthesis_target(self) -> SynthesisTarget:
        return SynthesisTarget(self.unitary)

    def metadata(self) -> dict[str, object]:
        return {
            "benchmark": "CCZ-3",
            "scope": "known-native-witness-certification-reference",
            "relationship": "CCZ = H(2) CCX H(2)",
            "num_qubits": 3,
            "qubit_order": "little; q0 is the least-significant basis bit",
            "target_id": semantic_target_digest(self.unitary),
            "witness_gate_count": len(self.witness),
            "witness_t_count": sum(gate.is_non_clifford() for gate in self.witness),
            "witness_two_qubit_count": sum(gate.is_two_qubit() for gate in self.witness),
            "target_specific_reachability_oracle": False,
            "witness_use": "reference-certification-only",
        }


def ccz_reference_benchmark() -> CCZReferenceBenchmark:
    """Build the immutable, independently target-defined CCZ benchmark."""

    return CCZReferenceBenchmark(
        unitary=ccz_reference_unitary(),
        witness=known_ccz_native_witness(),
    )


__all__ = [
    "CORPUS_SCHEMA",
    "CCZReferenceBenchmark",
    "DEFAULT_SPLIT_SEEDS",
    "NATIVE_GATE_NAMES",
    "NativeTargetCase",
    "NativeTargetCorpus",
    "SEMANTIC_IDENTITY_SCHEMA",
    "SPLIT_ORDER",
    "build_native_target_corpus",
    "ccz_reference_benchmark",
    "ccz_reference_unitary",
    "known_ccz_native_witness",
    "semantic_target_digest",
]
