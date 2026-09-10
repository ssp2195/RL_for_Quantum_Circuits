"""Finite phase-obligation domain with exact clean-input semantics.

This is an explicit restricted normal-form domain, not unrestricted Clifford+T.
The target is a phase polynomial, never a hidden circuit. Learning only orders
CNOT and required-phase continuations. Phase gates may be delayed until their required parity is available.
An optional eager-emission restriction is retained as an explicit ablation. All final data parities and zero work rows are restored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import cached_property
from itertools import combinations
from typing import Iterable

import numpy as np

from .model import Budget, Gate, HybridState
from .resource_domain import canonical_digest, nonnegative_integer

PHASE_NATIVE = {0: (), 1: ('T',), 2: ('S',), 3: ('S', 'T'),
                4: ('S', 'S'), 5: ('SDG', 'TDG'), 6: ('SDG',), 7: ('TDG',)}
SCHEMA = 'scheduled-phase-obligation-v1'


@dataclass(frozen=True)
class PhaseProblem:
    n: int
    coefficients: tuple[tuple[int, int], ...]
    ancillas: int = 0
    max_cnot: int = 24
    max_depth: int = 64
    max_gates: int = 128
    constant: int = 0
    name: str = 'phase-polynomial'
    emission: str = 'deferred'
    max_t_depth: int = 64

    def __post_init__(self):
        for field in ('n', 'ancillas', 'max_cnot', 'max_depth', 'max_gates', 'constant', 'max_t_depth'):
            nonnegative_integer(getattr(self, field), field)
        if not 1 <= self.n <= 6 or self.ancillas > 2:
            raise ValueError('supported domain: 1-6 data wires and 0-2 clean ancillas')
        if self.emission not in ('eager', 'deferred'):
            raise ValueError('emission must be eager or deferred')
        if self.constant not in (0, 4):
            raise ValueError('constant is 0 or 4 modulo eight')
        merged = {}
        for mask, coefficient in self.coefficients:
            if type(mask) is not int or not 1 <= mask < 1 << self.n or type(coefficient) is not int:
                raise ValueError('invalid exact parity/coefficient')
            merged[mask] = (merged.get(mask, 0) + coefficient) % 8
        object.__setattr__(self, 'coefficients', tuple(sorted((m, c) for m, c in merged.items() if c)))

    @property
    def width(self):
        return self.n + self.ancillas

    @cached_property
    def identity_rows(self):
        return tuple(1 << q for q in range(self.n)) + (0,) * self.ancillas

    @cached_property
    def actions(self):
        cnots = tuple((c, t) for c in range(self.width) for t in range(self.width) if c != t)
        return cnots + (tuple((q, q) for q in range(self.width)) if self.emission == 'deferred' else ())

    @cached_property
    def phase_table(self):
        return dict(self.coefficients)

    @cached_property
    def phase_bits(self):
        return {m: 1 << i for i, (m, _) in enumerate(self.coefficients)}

    @property
    def t_count(self):
        return sum(c % 2 for _, c in self.coefficients)

    def manifest(self):
        d = asdict(self)
        d.pop('name')  # A name is metadata, not part of the target.
        d['coefficients'] = [list(p) for p in self.coefficients]
        return {'schema': SCHEMA, **d, 'phase_mode': 'exact',
                'connectivity': 'all-to-all', 'lowering': {str(k): list(v) for k, v in PHASE_NATIVE.items()},
                'phase_emission': self.emission,
                'scope': 'fixed polynomial, scheduled CNOT/required-phase continuation; no measurement'}

    @cached_property
    def digest(self):
        return canonical_digest(self.manifest())

    def cap(self, objective, value):
        if objective not in ('cnot', 'depth', 'gates', 't_depth'):
            raise ValueError('objective must be cnot, depth, t_depth or gates')
        return replace(self, **{'max_' + objective: value})

    def target_exponents(self):
        return tuple((self.constant + sum(c * ((x & m).bit_count() % 2)
                                         for m, c in self.coefficients)) % 8 for x in range(1 << self.n))


@dataclass(frozen=True, slots=True)
class PhaseState:
    rows: tuple[int, ...]
    remaining: int
    depths: tuple[int, ...]
    cnot: int = 0
    gates: int = 0
    t_depths: tuple[int, ...] = ()

    @property
    def key(self):
        return self.rows, self.remaining

    @property
    def depth(self):
        return max(self.depths, default=0)

    @property
    def costs(self):
        return self.cnot, self.gates, *self.depths, *self.t_depths

    @property
    def t_depth(self):
        return max(self.t_depths, default=0)

    def within(self, p):
        return self.cnot <= p.max_cnot and self.depth <= p.max_depth and self.gates <= p.max_gates and self.t_depth <= p.max_t_depth

    def terminal(self, p):
        return self.remaining == 0 and self.rows == p.identity_rows


def _append(depths, native):
    d = list(depths)
    for name, qs in native:
        layer = 1 + max(d[q] for q in qs)
        for q in qs:
            d[q] = layer
    return tuple(d)


def _append_t_depth(depths, native):
    d = list(depths)
    for name, qs in native:
        layer = max(d[q] for q in qs) + int(name in ('T', 'TDG'))
        for q in qs:
            d[q] = layer
    return tuple(d)


def root_state(p):
    remaining = (1 << len(p.coefficients)) - 1
    native = []
    if p.constant == 4:
        # Chronological X Z X Z = -I; no uncounted scalar phase.
        native.extend((g, (0,)) for g in ('H', 'S', 'S', 'H', 'S', 'S') * 2)
    for q, mask in enumerate(p.identity_rows):
        bit = p.phase_bits.get(mask, 0)
        if p.emission == 'eager' and remaining & bit:
            native.extend((g, (q,)) for g in PHASE_NATIVE[p.phase_table[mask]])
            remaining ^= bit
    native = tuple(native)
    return PhaseState(p.identity_rows, remaining, _append((0,) * p.width, native), 0, len(native), _append_t_depth((0,) * p.width, native)), native


def successor(p, state, token):
    c, t = p.actions[token]
    rows = list(state.rows)
    if c != t:
        rows[t] ^= rows[c]
    native = [] if c == t else [('CNOT', (c, t))]
    remaining = state.remaining
    bit = p.phase_bits.get(rows[t], 0)
    if c == t and not remaining & bit:
        raise ValueError('phase is not available on the selected wire')
    if (p.emission == 'eager' or c == t) and remaining & bit:
        native.extend((g, (t,)) for g in PHASE_NATIVE[p.phase_table[rows[t]]])
        remaining ^= bit
    native = tuple(native)
    return PhaseState(tuple(rows), remaining, _append(state.depths, native),
                      state.cnot + int(c != t), state.gates + len(native), _append_t_depth(state.t_depths, native)), native


def covers(a, b):
    return a.key == b.key and all(x <= y for x, y in zip(a.costs, b.costs, strict=True))


def completion_lower_bounds(p, s):
    """Admissible suffix bounds, independent of every learned score.

    A CNOT changes only one row and creates at most one unavailable target
    parity. Every outstanding phase block must be emitted exactly once.
    """
    absent = sum(bool(s.remaining & (1 << i)) and m not in s.rows
                 for i, (m, _) in enumerate(p.coefficients))
    repairs = sum(a != b for a, b in zip(s.rows, p.identity_rows))
    cnots = max(absent, repairs)
    phases = sum(len(PHASE_NATIVE[c]) for i, (_, c) in enumerate(p.coefficients)
                 if s.remaining & (1 << i))
    return cnots, cnots + phases


def completion_possible(p, s):
    cnots, gates = completion_lower_bounds(p, s)
    odd_remaining = sum(c % 2 for i, (_, c) in enumerate(p.coefficients) if s.remaining & (1 << i))
    capacity = sum(p.max_t_depth - d for d in s.t_depths)
    return s.within(p) and s.cnot + cnots <= p.max_cnot and s.gates + gates <= p.max_gates and odd_remaining <= capacity


def witness_native(p, tokens):
    state, prefix = root_state(p)
    native = list(prefix)
    for token in tokens:
        if type(token) is not int or not 0 <= token < len(p.actions):
            raise ValueError('invalid continuation token')
        state, added = successor(p, state, token)
        native.extend(added)
    return state, tuple(native)


def certify_phase(p: PhaseProblem, tokens: Iterable[int], *, dag=True):
    """Independent scalar basis replay + vectorized native isometry + DAG replay.

    Exact basis/phase replay evaluates ALL promised inputs, not sampled states.
    H gates in the fixed -I prefix are checked by dense replay and the algebraic
    XZXZ identity; the emitted CNOT/S/T word is replayed exactly independently.
    """
    tokens = tuple(tokens)
    state, native = witness_native(p, tokens)
    size, rows = 1 << p.n, 1 << p.width
    mapping = list(range(size))
    phases = [p.constant] * size
    start = 12 if p.constant else 0
    exact_depths = [0] * p.width
    exact_t_depths = [0] * p.width
    count_cnot = count_t = 0
    actual = np.eye(rows, size, dtype=np.complex128)
    all_rows = np.arange(rows)
    for i, (name, qs) in enumerate(native):
        layer = 1 + max(exact_depths[q] for q in qs)
        for q in qs:
            exact_depths[q] = layer
        tlayer = max(exact_t_depths[q] for q in qs) + int(name in ('T', 'TDG'))
        for q in qs:
            exact_t_depths[q] = tlayer
        count_cnot += name == 'CNOT'
        count_t += name in ('T', 'TDG')
        if name == 'CNOT':
            c, t = qs
            permutation = all_rows ^ (((all_rows >> c) & 1) << t)
            actual = actual[permutation, :]
            if i >= start:
                for x in range(size):
                    if mapping[x] & (1 << c):
                        mapping[x] ^= 1 << t
        elif name == 'H':
            q = qs[0]
            low = all_rows[(all_rows & (1 << q)) == 0]
            high = low | (1 << q)
            l, h = actual[low].copy(), actual[high].copy()
            actual[low], actual[high] = (l + h) / np.sqrt(2), (l - h) / np.sqrt(2)
        else:
            turns = {'T': 1, 'TDG': -1, 'S': 2, 'SDG': -2}[name]
            actual[(all_rows & (1 << qs[0])) != 0] *= np.exp(1j * np.pi * turns / 4)
            if i >= start:
                for x in range(size):
                    if mapping[x] & (1 << qs[0]):
                        phases[x] = (phases[x] + turns) % 8
    target = p.target_exponents()
    expected = np.zeros_like(actual)
    expected[np.arange(size), np.arange(size)] = np.exp(1j * np.pi * np.asarray(target) / 4)
    error = float(np.max(np.abs(actual - expected)))
    leakage = float(np.sum(np.abs(actual[size:]) ** 2) / size)
    exact = mapping == list(range(size)) and tuple(phases) == target
    cost_match = (count_cnot, len(native), tuple(exact_depths)) == (state.cnot, state.gates, state.depths) and tuple(exact_t_depths) == state.t_depths
    dag_valid = None
    if dag and exact and state.within(p):
        h = HybridState.identity(p.width, Budget(count_t, count_cnot, len(native), max(exact_depths)))
        for name, qs in native:
            h = h.apply(Gate(name, qs), partial_order_reduction=False)
            if h is None:
                raise AssertionError('native symbolic replay rejected a feasible gate')
        h.validate()
        d = h.materialize_dag()
        dag_valid = len(d.gates) == len(native) and h.wire_depths == tuple(exact_depths)
    used = {q for _, qs in native for q in qs if q >= p.n}
    return {'success': bool(exact and cost_match and state.terminal(p) and state.within(p)
                            and error < 1e-9 and (dag_valid is not False)),
            'problem_digest': p.digest, 'tokens': list(tokens),
            'native': [[g, list(qs)] for g, qs in native], 'exact_basis_phase_match': exact,
            'native_isometry_error': error, 'workspace_leakage': leakage, 'dag_validated': dag_valid,
            'resources': {'t_count': count_t, 'cnot': count_cnot, 'depth': max(exact_depths),
                          'gates': len(native), 't_depth': max(exact_t_depths), 't_wire_depths': exact_t_depths, 'wire_depths': exact_depths,
                          'available_ancillas': p.ancillas, 'used_ancillas': len(used)},
            'scope': 'correct exact phase oracle; no optimality conclusion from this witness alone'}


def boolean_phase_coefficients(truth):
    """Exact ANF-to-phase identity for degree <=3; rejects unsupported degree."""
    truth = tuple(truth)
    n = (len(truth) - 1).bit_length()
    if len(truth) != 1 << n or not 1 <= n <= 6 or any(type(v) is not int or v not in (0, 1) for v in truth):
        raise ValueError('expected a Boolean truth table of length 2**n')
    anf = list(truth)
    for q in range(n):
        for mask in range(1 << n):
            if mask & (1 << q):
                anf[mask] ^= anf[mask ^ (1 << q)]
    coefficients = {}
    for mask in range(1, 1 << n):
        if not anf[mask]:
            continue
        degree = mask.bit_count()
        if degree > 3:
            raise ValueError('H-free pi/4 phase domain requires Boolean degree <=3')
        subset = mask
        while subset:
            coefficient = (1 << (3 - degree)) * (1 if subset.bit_count() % 2 else -1)
            coefficients[subset] = (coefficients.get(subset, 0) + coefficient) % 8
            subset = (subset - 1) & mask
    return n, tuple(sorted((m, c) for m, c in coefficients.items() if c)), 4 * anf[0]


def problem_from_manifest(d):
    if d.get('schema') != SCHEMA:
        raise ValueError('unknown phase domain schema')
    p = PhaseProblem(d['n'], tuple(tuple(pair) for pair in d['coefficients']), d['ancillas'],
                     d['max_cnot'], d['max_depth'], d['max_gates'], d['constant'], emission=d['emission'], max_t_depth=d['max_t_depth'])
    if p.manifest() != d:
        raise ValueError('modified/unsupported phase domain manifest')
    return p
