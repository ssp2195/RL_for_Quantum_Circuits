"""Persistent quantum/classical protocol DAG and executable native lowering.

Large product banks never allocate a dense global isometry. Each coherent
primitive carries the unchanged local (G, tableau, rotations, phase, resources)
HybridState and an explicit physical-wire embedding. Symbolic interface proofs
justify composition on clean targets. No global unitary is claimed for a Kraus
measurement/reset event.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Iterable

from ..model import Budget, Gate, HybridState
from ..ancilla_benchmarks import toffoli_decomposition
from .contract import CleanupProblem, Layout, digest

INVERSE = {'H': 'H', 'S': 'SDG', 'SDG': 'S', 'T': 'TDG', 'TDG': 'T', 'CNOT': 'CNOT'}
X_WORD = (('H', (0,)), ('S', (0,)), ('S', (0,)), ('H', (0,)))
Z_WORD = (('S', (0,)), ('S', (0,)))
CZ_WORD = (('H', (1,)), ('CNOT', (0, 1)), ('H', (1,)))
# A phase-correct clean-target AND. The 9-gate relative-phase word gives i on
# |110>; the final S-dagger on the target removes precisely that phase. The
# independent exact verifier checks all four clean-input columns. NOT a 4-T
# Toffoli on arbitrary target inputs. See Gidney (2018) for the established
# four-T temporary-AND resource mechanism.
AND4 = (('H', (2,)), ('T', (2,)), ('CNOT', (1, 2)), ('TDG', (2,)),
        ('CNOT', (0, 2)), ('T', (2,)), ('CNOT', (1, 2)), ('TDG', (2,)),
        ('H', (2,)), ('SDG', (2,)))


@lru_cache(None)
def local_word(kind: str, primitive: str = 'and4') -> tuple:
    if primitive not in ('and4', 'ccx7'):
        raise ValueError('unsupported AND implementation')
    if kind in ('AND', 'UNAND'):
        word = AND4 if primitive == 'and4' else tuple((g.name, g.qubits) for g in toffoli_decomposition(0, 1, 2))
        return word if kind == 'AND' else tuple((INVERSE[n], q) for n, q in reversed(word))
    if kind == 'Z': return Z_WORD
    if kind == 'CZ': return CZ_WORD
    if kind == 'X': return X_WORD
    if kind == 'H': return (('H', (0,)),)
    if kind == 'CX': return (('CNOT', (0, 1)),)
    if kind == 'MINUS': return X_WORD + Z_WORD + X_WORD + Z_WORD
    raise ValueError(f'unknown coherent primitive {kind}')


@lru_cache(None)
def local_hybrid(kind: str, primitive: str = 'and4') -> HybridState:
    """The existing authoritative representation, on <=3 active wires."""
    word = local_word(kind, primitive)
    width = 1 + max(q for _, qs in word for q in qs)
    state = HybridState.identity(width, Budget(1000, 1000, 1000, 1000))
    for name, qs in word:
        state = state.apply(Gate(name, qs), partial_order_reduction=False)
        if state is None:
            raise AssertionError('primitive lowering exceeded its validation budget')
    state.materialize_dag().validate()
    return state


@dataclass(frozen=True)
class Instruction:
    kind: str
    wires: tuple[int, ...]
    stage: str
    guard: int | None = None
    outcome: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'wires', tuple(self.wires))
        arity = {'AND': 3, 'UNAND': 3, 'Z': 1, 'CZ': 2, 'MX': 1, 'MINUS': 1, 'CX': 2}
        if self.kind not in arity or len(self.wires) != arity[self.kind] or len(set(self.wires)) != len(self.wires):
            raise ValueError('invalid instruction kind or arity')
        if any(type(q) is not int or q < 0 for q in self.wires):
            raise ValueError('invalid quantum wire')
        if type(self.stage) is not str or not self.stage:
            raise ValueError('stage required')
        for v in (self.guard, self.outcome):
            if v is not None and (type(v) is not int or v < 0):
                raise ValueError('invalid classical bit')
        if self.kind == 'MX':
            if self.outcome is None or self.guard is not None:
                raise ValueError('MX requires a fresh outcome and no guard')
        elif self.outcome is not None or (self.guard is not None and self.kind not in ('Z', 'CZ')):
            raise ValueError('only Clifford phases may be classically conditioned')

    def payload(self) -> dict:
        return {'kind': self.kind, 'wires': list(self.wires), 'stage': self.stage,
                'guard': self.guard, 'outcome': self.outcome}

    @classmethod
    def from_payload(cls, d: dict) -> 'Instruction':
        op = cls(**d)
        if op.payload() != d:
            raise ValueError('noncanonical instruction')
        return op

    def hybrid_blocks(self, primitive: str) -> tuple[HybridState, ...]:
        if self.kind == 'MX':
            return local_hybrid('H'), local_hybrid('X')
        return (local_hybrid(self.kind, primitive),)


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    previous: 'ProtocolEvent | None'
    op: Instruction
    parents: tuple[int, ...]
    classical_parents: tuple[int, ...]
    index: int
    coherent_blocks: tuple[HybridState, ...]


def append_event(tail: ProtocolEvent | None, op: Instruction, primitive: str) -> ProtocolEvent:
    """Structural sharing; a branch copies only one event and dependency links."""
    qneed = set(op.wires); quantum = set(); classical = set(); p = tail
    # This walk is bounded by the protocol size, not by Hilbert dimension. The
    # immutable DAG is materialized only at final verification/export.
    stage_boundary = tail is not None and tail.op.stage != op.stage
    while p is not None:
        hit = qneed.intersection(p.op.wires)
        if hit:
            quantum.add(p.index); qneed.difference_update(hit)
        if op.guard is not None and p.op.outcome == op.guard:
            classical.add(p.index)
        if stage_boundary:
            quantum.add(p.index)  # explicit stage barrier, all preceding events
        p = p.previous
    if op.guard is not None and not classical:
        raise ValueError('use of an unavailable measurement result')
    return ProtocolEvent(tail, op, tuple(sorted(quantum)), tuple(sorted(classical)),
                         0 if tail is None else tail.index+1, op.hybrid_blocks(primitive))


def instructions(tail: ProtocolEvent | None) -> tuple[Instruction, ...]:
    result = []
    while tail is not None:
        result.append(tail.op); tail = tail.previous
    return tuple(reversed(result))


@dataclass(frozen=True)
class Protocol:
    problem_digest: str
    layout: Layout
    primitive: str
    cleanup_mode: str
    ops: tuple[Instruction, ...]
    provenance: str = 'structured_frontier_discovery'

    def __post_init__(self) -> None:
        if self.primitive not in ('and4', 'ccx7') or self.cleanup_mode not in ('measured', 'coherent'):
            raise ValueError('unsupported protocol grammar')
        if any(max(op.wires) >= self.layout.width for op in self.ops):
            raise ValueError('instruction outside physical register')

    def payload(self) -> dict:
        return {'schema': 'cleanup-protocol-v1', 'problem_digest': self.problem_digest,
                'layout': {'r': self.layout.r, 'm': self.layout.m, 'side': self.layout.side},
                'primitive': self.primitive, 'cleanup_mode': self.cleanup_mode,
                'ops': [op.payload() for op in self.ops], 'provenance': self.provenance}

    @property
    def digest(self) -> str:
        return digest(self.payload())

    @classmethod
    def from_payload(cls, d: dict) -> 'Protocol':
        if d.get('schema') != 'cleanup-protocol-v1':
            raise ValueError('not a cleanup protocol')
        p = cls(d['problem_digest'], Layout(**d['layout']), d['primitive'], d['cleanup_mode'],
                tuple(Instruction.from_payload(op) for op in d['ops']), d['provenance'])
        if p.payload() != d:
            raise ValueError('unrecognized protocol fields')
        return p

    def qasm3(self) -> str:
        """Fully expanded native operations, measurement and conditional reset.

        Stage barriers make the two measurement batches explicit. No custom
        AND/CZ unitary, uncosted reset, or postselected branch is exported.
        """
        c = 1 + max((op.outcome for op in self.ops if op.outcome is not None), default=-1)
        lines = ['OPENQASM 3.0;', 'include "stdgates.inc";',
                 f'qubit[{self.layout.width}] q;']
        if c: lines.append(f'bit[{c}] c;')
        stage = None
        def emit(word, wires, guard=None):
            text = []
            names = {'CNOT': 'cx', 'SDG': 'sdg', 'TDG': 'tdg'}
            for name, qs in word:
                text.append(f'{names.get(name, name.lower())} ' + ', '.join(f'q[{wires[j]}]' for j in qs) + ';')
            if guard is None: lines.extend(text)
            else: lines.extend([f'if (c[{guard}] == 1) {{'] + ['  '+s for s in text] + ['}'])
        for op in self.ops:
            if op.stage != stage:
                if stage is not None: lines.append('barrier q;')
                lines.append('// '+op.stage); stage = op.stage
            if op.kind == 'MX':
                emit(local_word('H'), op.wires)
                lines.append(f'c[{op.outcome}] = measure q[{op.wires[0]}];')
                emit(local_word('X'), op.wires, op.outcome)
            else:
                emit(local_word(op.kind, self.primitive), op.wires, op.guard)
        return '\n'.join(lines)+'\n'


@dataclass(frozen=True)
class ResourceState:
    """Worst-case native schedule, expected counts and live occupancy.

    Expected gate counts are exact ONLY after branch uniformity is certified.
    Timings are conservative gate-level ASAP values in the declared tick model.
    No claim of optimal conditional scheduling is made.
    """
    depths: tuple[int, ...]
    tdepths: tuple[int, ...]
    ticks: tuple[int, ...]
    classical: tuple[tuple[int, int], ...] = ()
    occupied: frozenset[int] = frozenset()
    peak_aux: int = 0
    gates: int = 0
    cnot: int = 0
    t_count: int = 0
    expected_twice: int = 0
    measurements: int = 0
    rounds: tuple[str, ...] = ()
    stage: str = ''
    forward_and: int = 0
    cleanup_t: int = 0
    conditional_cz: int = 0

    @classmethod
    def zero(cls, width: int) -> 'ResourceState':
        return cls((0,)*width, (0,)*width, (0,)*width)

    def append(self, op: Instruction, primitive: str, problem: CleanupProblem) -> 'ResourceState':
        if max(op.wires) >= len(self.depths): raise ValueError('wire outside resource ledger')
        d, td, clock = list(self.depths), list(self.tdepths), list(self.ticks)
        classical = dict(self.classical); occupied = set(self.occupied)
        if self.stage and self.stage != op.stage:
            d = [max(d)]*len(d); td = [max(td)]*len(td); clock = [max(clock)]*len(clock)
        g, cx, nt, twice, cleanup_t = self.gates, self.cnot, self.t_count, self.expected_twice, self.cleanup_t
        def native(word, wires, guard=None):
            nonlocal g, cx, nt, twice, cleanup_t
            if guard is not None and guard not in classical: raise ValueError('missing classical producer')
            for name, qs0 in word:
                qs = tuple(wires[j] for j in qs0)
                level = 1+max(d[q] for q in qs)
                tlevel = int(name in ('T','TDG'))+max(td[q] for q in qs)
                tick = max(max(clock[q] for q in qs), classical.get(guard, 0)) + problem.hardware.gate_ticks
                for q in qs: d[q] = level; td[q] = tlevel; clock[q] = tick
                g += 1; cx += name == 'CNOT'; nt += name in ('T','TDG')
                twice += 2 if guard is None else 1
                if op.stage not in ('prepare','consumer'): cleanup_t += name in ('T','TDG')
        rounds = self.rounds; measures = self.measurements
        if op.kind == 'MX':
            if op.outcome in classical: raise ValueError('outcome bit overwritten')
            native(local_word('H'), op.wires)
            q = op.wires[0]
            clock[q] += problem.hardware.measurement_ticks
            classical[op.outcome] = clock[q]+problem.hardware.feedback_ticks
            native(local_word('X'), op.wires, op.outcome)
            occupied.discard(q); measures += 1
            if op.stage not in rounds: rounds += (op.stage,)
        else:
            native(local_word(op.kind, primitive), op.wires, op.guard)
            if op.kind == 'AND': occupied.add(op.wires[2])
            if op.kind == 'UNAND': occupied.discard(op.wires[2])
        return ResourceState(tuple(d), tuple(td), tuple(clock), tuple(sorted(classical.items())),
                             frozenset(occupied), max(self.peak_aux, len(occupied)),
                             g, cx, nt, twice, measures, rounds, op.stage,
                             self.forward_and + (op.kind == 'AND'), cleanup_t,
                             self.conditional_cz + (op.kind == 'CZ' and op.guard is not None))

    def report(self) -> dict:
        return {'peak_aux': self.peak_aux, 'physical_qubits': len(self.depths),
                'native_gates': self.gates, 'cnot': self.cnot, 't_count': self.t_count,
                'native_depth': max(self.depths, default=0), 't_depth': max(self.tdepths, default=0),
                'worst_case_ticks': max(self.ticks, default=0),
                'expected_native_gates': str(Fraction(self.expected_twice, 2)),
                'measurements': self.measurements, 'measurement_rounds': len(self.rounds),
                'retained_classical_bits': len(self.classical),
                'forward_and': self.forward_and, 'cleanup_t': self.cleanup_t,
                'worst_case_correction_cz': self.conditional_cz,
                'aux_at_end': len(self.occupied),
                'timing_scope': 'all-to-all, explicit stage barriers; abstract ticks, not hardware measurements',
                'native_count_scope': 'all branches: worst case; includes measurement H and conditional-X reset lowering'}


def resources(protocol: Protocol, problem: CleanupProblem) -> ResourceState:
    result = ResourceState.zero(protocol.layout.width)
    for op in protocol.ops: result = result.append(op, protocol.primitive, problem)
    return result
