"""Explicit input, lifecycle, resource and hardware contracts for Theorem 2.

Source: H - Prove Cleanup Theorem(1).pdf, pp. 59--60 (not its earlier
heralded-probability theorem). All raw inputs range independently over the full
Boolean cube. Auxiliary optimum is relative to full-bank materialization.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Iterable

SCHEMA = 'outer-product-cleanup-v1'
THEOREM = 'lifecycle-full-bank-inactive-clean-helpers-v1'
BASE_COMMIT = '6c2f15ffc2e0ce524c6492f40f234a2862dba51a'
# The verifier treats the mathematical converse, explicitly identified here,
# as a trusted theorem; finite tests are NOT a proof of that converse.
ASSUMPTIONS = (
    'full-independent-raw-input-cube', 'F_ij=a*b_i*x_j',
    'helpers=a*(P*b+Q*x+c)', 'all-r*m-products-coexist',
    'quadratic-diagonal-sign-consumer',
    'one-main-garbage-only-instrument-then-Clifford-feedback',
    'helpers-preserved-through-main-feedback',
    'one-final-direct-X-helper-measurement-and-Clifford-feedback',
    'no-postselection', 'no-extra-unaccounted-workspace',
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def positive(value: int, name: str, maximum: int = 4096) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [1,{maximum}]')


# Exact square-free ANF: an integer bit mask denotes a monomial; xor is addition.
Polynomial = frozenset[int]


def plus(*args: Polynomial) -> Polynomial:
    result: set[int] = set()
    for arg in args:
        result.symmetric_difference_update(arg)
    return frozenset(result)


def times(a: Polynomial, b: Polynomial) -> Polynomial:
    result: set[int] = set()
    for x in a:
        for y in b:
            z = x | y
            if z in result:
                result.remove(z)
            else:
                result.add(z)
    return frozenset(result)


def evaluate(p: Polynomial, x: int) -> int:
    return sum((x & m) == m for m in p) & 1


@dataclass(frozen=True)
class Consumer:
    """q(a,b,x,F), a quadratic Boolean sign phase on displayed live wires.

    A term () is constant one, (wire,) is Z and (u,v) is CZ. Duplicates
    cancel in F2. No helper variable or non-diagonal operation is implicit.
    """
    terms: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        out: set[tuple[str, ...]] = set()
        for term in self.terms:
            if not isinstance(term, (tuple, list)) or len(term) > 2 or any(type(s) is not str for s in term):
                raise ValueError('consumer must consist of constant, linear and quadratic wire terms')
            t = tuple(sorted(set(term)))  # z*z = z on the Boolean cube.
            if t in out:
                out.remove(t)
            else:
                out.add(t)
        object.__setattr__(self, 'terms', tuple(sorted(out)))


@dataclass(frozen=True)
class Hardware:
    """Illustrative timing model, not hardware measurements.

    Every coherent native gate costs gate_ticks. Measurement is in Z after an
    explicit H. Reset is an explicit classical-X = H S S H. Feedback readiness
    costs feedback_ticks. All-to-all native connectivity is declared.
    """
    gate_ticks: int = 1
    measurement_ticks: int = 5
    feedback_ticks: int = 2

    def __post_init__(self) -> None:
        positive(self.gate_ticks, 'gate_ticks')
        positive(self.measurement_ticks, 'measurement_ticks')
        if type(self.feedback_ticks) is not int or self.feedback_ticks < 0:
            raise ValueError('feedback_ticks must be nonnegative')


@dataclass(frozen=True)
class Limits:
    max_aux: int | None = None
    max_t: int | None = None
    max_cnot: int | None = None
    max_gates: int | None = None
    max_depth: int | None = None
    max_t_depth: int | None = None
    max_rounds: int | None = None
    max_ticks: int | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{name} must be nonnegative or None')

    def accepts(self, r: dict) -> bool:
        keys = {'max_aux': 'peak_aux', 'max_t': 't_count', 'max_cnot': 'cnot',
                'max_gates': 'native_gates', 'max_depth': 'native_depth',
                'max_t_depth': 't_depth', 'max_rounds': 'measurement_rounds',
                'max_ticks': 'worst_case_ticks'}
        return all(v is None or r[keys[k]] <= v for k, v in asdict(self).items())


@dataclass(frozen=True)
class CleanupProblem:
    name: str
    r: int
    m: int
    consumer: Consumer
    limits: Limits = field(default_factory=Limits)
    hardware: Hardware = field(default_factory=Hardware)
    split: str = 'test'

    def __post_init__(self) -> None:
        positive(self.r, 'r', 64); positive(self.m, 'm', 64)
        if type(self.name) is not str or not self.name:
            raise ValueError('nonempty name required')
        if not isinstance(self.consumer, Consumer) or not isinstance(self.limits, Limits) or not isinstance(self.hardware, Hardware):
            raise TypeError('explicit consumer, limits and hardware required')
        valid = {'a'} | {f'b{i}' for i in range(self.r)} | {f'x{j}' for j in range(self.m)} | {
            f'f{i}_{j}' for i in range(self.r) for j in range(self.m)}
        if any(s not in valid for term in self.consumer.terms for s in term):
            raise ValueError('consumer references an unknown or helper wire')

    @property
    def n(self) -> int:
        return 1 + self.r + self.m

    @property
    def workspace_lower_bound(self) -> int:
        return self.r * self.m + min(self.r, self.m)

    @property
    def target_polynomial(self) -> Polynomial:
        symbols = {'a': frozenset({1})}
        symbols.update({f'b{i}': frozenset({1 << (1+i)}) for i in range(self.r)})
        symbols.update({f'x{j}': frozenset({1 << (1+self.r+j)}) for j in range(self.m)})
        symbols.update({f'f{i}_{j}': times(symbols['a'], times(symbols[f'b{i}'], symbols[f'x{j}']))
                        for i in range(self.r) for j in range(self.m)})
        result = frozenset()
        for term in self.consumer.terms:
            p = frozenset({0})
            for s in term:
                p = times(p, symbols[s])
            result = plus(result, p)
        return result

    @property
    def oracle_digest(self) -> str:
        return digest({'n': self.n, 'target_anf': sorted(self.target_polynomial)})

    def manifest(self) -> dict:
        return {'schema': SCHEMA, 'name': self.name, 'r': self.r, 'm': self.m,
                'consumer': [list(t) for t in self.consumer.terms],
                'limits': asdict(self.limits), 'hardware': asdict(self.hardware),
                'split': self.split, 'assumptions': list(ASSUMPTIONS),
                'phase_mode': 'exact', 'input_domain': 'full_cube'}

    @property
    def digest(self) -> str:
        return digest(self.manifest())

    def with_aux_cap(self, cap: int) -> 'CleanupProblem':
        return replace(self, limits=replace(self.limits, max_aux=cap))

    @classmethod
    def from_manifest(cls, d: dict) -> 'CleanupProblem':
        if (d.get('schema') != SCHEMA or tuple(d.get('assumptions', ())) != ASSUMPTIONS
                or d.get('phase_mode') != 'exact' or d.get('input_domain') != 'full_cube'):
            raise ValueError('unsupported theorem premises')
        p = cls(d['name'], d['r'], d['m'], Consumer(tuple(tuple(t) for t in d['consumer'])),
                Limits(**d['limits']), Hardware(**d['hardware']), d['split'])
        if p.manifest() != d:
            raise ValueError('noncanonical or unrecognized contract fields')
        return p


@dataclass(frozen=True)
class Layout:
    r: int
    m: int
    side: str

    def __post_init__(self) -> None:
        positive(self.r, 'r', 64); positive(self.m, 'm', 64)
        if self.side not in ('row', 'column'):
            raise ValueError('helper side must be row or column')

    @property
    def n(self) -> int:
        return 1 + self.r + self.m

    @property
    def k(self) -> int:
        return self.r if self.side == 'row' else self.m

    @property
    def width(self) -> int:
        return self.n + self.r * self.m + self.k

    @property
    def aux(self) -> tuple[int, ...]:
        return tuple(range(self.n, self.width))

    @property
    def products(self) -> tuple[int, ...]:
        return tuple(range(self.n, self.n+self.r*self.m))

    @property
    def helpers(self) -> tuple[int, ...]:
        return tuple(range(self.n+self.r*self.m, self.width))

    def helper_operands(self, h: int) -> tuple[int, int, int]:
        if not 0 <= h < self.k:
            raise ValueError('helper index outside layout')
        return 0, 1+h if self.side == 'row' else 1+self.r+h, self.helpers[h]

    def product_operands(self, k: int) -> tuple[int, int, int]:
        if not 0 <= k < self.r*self.m:
            raise ValueError('product index outside layout')
        i, j = divmod(k, self.m)
        return ((self.helpers[i], 1+self.r+j, self.products[k]) if self.side == 'row'
                else (1+i, self.helpers[j], self.products[k]))

    def helper_for(self, k: int) -> int:
        i, j = divmod(k, self.m)
        return i if self.side == 'row' else j

    def wire(self, symbol: str) -> int:
        if symbol == 'a': return 0
        if symbol.startswith('b'): return 1+int(symbol[1:])
        if symbol.startswith('x'): return 1+self.r+int(symbol[1:])
        if symbol.startswith('f'):
            i, j = map(int, symbol[1:].split('_'))
            return self.products[i*self.m+j]
        raise ValueError('unknown wire symbol')
