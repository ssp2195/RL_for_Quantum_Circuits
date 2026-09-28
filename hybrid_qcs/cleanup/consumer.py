"""Exact consumer reduction: Boolean substitution, binary rank, and live features.

Theorems 2--4 in docs/consumer_cleanup/PROOFS.md specify the admitted class.
No fitted score participates in a mathematical equality or lower bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from .contract import CleanupProblem, Polynomial, plus, times


class Interrupted(RuntimeError):
    """A cooperative cancellation is not a mathematical infeasibility result."""


def check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise Interrupted('cancelled')


def bits(mask: int) -> tuple[int, ...]:
    if type(mask) is not int or mask < 0:
        raise ValueError('a bit mask must be a nonnegative integer')
    return tuple(i for i in range(mask.bit_length()) if mask >> i & 1)


def rank_factors(rows: tuple[int, ...], columns: int) -> tuple[tuple[int, int], ...]:
    """Return M = XOR u v^T by exact binary elimination; no numerical rank."""
    if type(columns) is not int or columns < 0 or any(
            type(row) is not int or not 0 <= row < (1 << columns) for row in rows):
        raise ValueError('invalid binary matrix')
    rest = list(rows)
    answer = []
    while any(rest):
        v = next(row for row in rest if row)
        pivot = v & -v
        u = sum(1 << i for i, row in enumerate(rest) if row & pivot)
        answer.append((u, v))
        for i in bits(u):
            rest[i] ^= v
    return tuple(answer)


@dataclass(frozen=True)
class Factorization:
    quadratic: Polynomial
    b_features: tuple[int, ...]
    x_features: tuple[int, ...]
    rows: tuple[int, ...]
    factors: tuple[tuple[int, int], ...]

    @property
    def rank(self) -> int:
        return len(self.factors)

    @property
    def supports(self) -> tuple[frozenset[int], ...]:
        """Raw-variable masks of nonlinear features needed by each factor."""
        return tuple(frozenset(
            f for f in ([self.b_features[i] for i in bits(u)] +
                        [self.x_features[j] for j in bits(v)]) if f.bit_count() == 2
        ) for u, v in self.factors)

    def polynomial(self) -> Polynomial:
        result = self.quadratic
        for u, v in self.factors:
            left = frozenset(self.b_features[i] for i in bits(u))
            right = frozenset(self.x_features[j] for j in bits(v))
            result = plus(result, times(frozenset({1}), times(left, right)))
        return result

    def payload(self) -> dict:
        return {'quadratic': sorted(self.quadratic), 'b_features': list(self.b_features),
                'x_features': list(self.x_features), 'rows': list(self.rows),
                'factors': [list(pair) for pair in self.factors], 'rank': self.rank}


def factor_consumer(p: CleanupProblem) -> Factorization:
    """Separate a's higher-degree phase into B(b)^T M X(x), after cancellation."""
    bmask = ((1 << p.r) - 1) << 1
    xmask = ((1 << p.m) - 1) << (1 + p.r)
    phase = p.target_polynomial
    quadratic = frozenset(mask for mask in phase if mask.bit_count() <= 2)
    pairs = []
    for mask in sorted(phase - quadratic):
        b, x = mask & bmask, mask & xmask
        if not mask & 1 or not b or not x or (b | x | 1) != mask or max(b.bit_count(), x.bit_count()) > 2:
            raise ValueError('phase outside the degree-two separated-feature class')
        pairs.append((b, x))
    bf = tuple(sorted({b for b, _ in pairs}))
    xf = tuple(sorted({x for _, x in pairs}))
    rows = tuple(sum(1 << j for j, x in enumerate(xf) if (b, x) in pairs) for b in bf)
    result = Factorization(quadratic, bf, xf, rows, rank_factors(rows, len(xf)))
    if result.polynomial() != phase:
        raise AssertionError('exact factorization failed reconstruction')
    return result


def boundary(supports: tuple[frozenset[int], ...], completed: int) -> frozenset[int]:
    past: set[int] = set()
    future: set[int] = set()
    for j, support in enumerate(supports):
        (past if completed >> j & 1 else future).update(support)
    return frozenset(past & future)


def order_peak(supports: tuple[frozenset[int], ...], order: tuple[int, ...]) -> int:
    if sorted(order) != list(range(len(supports))):
        raise ValueError('factor order is not a permutation')
    completed = peak = 0
    for j in order:
        peak = max(peak, 1 + len(boundary(supports, completed) | supports[j]))
        completed |= 1 << j
    return peak


def minimum_live_order(supports: tuple[frozenset[int], ...], *, max_factors: int = 16,
                       cancel: Callable[[], bool] | None = None) -> dict:
    """Exact subset minimax recurrence. Complexity O(d 2^d) set operations.

    The optimum is only over orders of the supplied factors without feature
    recomputation. A cap/cancellation is explicit, never a false proof.
    """
    d = len(supports)
    if type(max_factors) is not int or not 0 <= max_factors <= 16:
        raise ValueError('exact factor limit must be in [0,16]')
    if d > max_factors:
        raise ValueError(f'{d} factors exceed exact subset limit {max_factors}')
    all_done = (1 << d) - 1
    choice: dict[int, int] = {}

    @lru_cache(None)
    def value(a: int) -> int:
        check_cancel(cancel)
        if a == all_done:
            return 0
        live = boundary(supports, a)
        cost, j = min((max(1 + len(live | supports[j]), value(a | (1 << j))), j)
                      for j in range(d) if not a >> j & 1)
        choice[a] = j
        return cost

    optimum = value(0)
    a = 0
    order = []
    while a != all_done:
        j = choice[a]
        order.append(j)
        a |= 1 << j
    return {'order': order, 'peak': optimum, 'states': value.cache_info().currsize,
            'supports': [sorted(s) for s in supports],
            'values': [value(a) for a in range(1 << d)],
            'scope': 'fixed rank factors, atomic factors, no feature recomputation',
            'global_oracle_optimality': False}


def verify_live_certificate(receipt: dict) -> bool:
    """Check every recurrence edge without running the schedule generator."""
    try:
        supports = tuple(frozenset(s) for s in receipt['supports'])
        d = len(supports)
        if d > 16 or receipt['scope'] != 'fixed rank factors, atomic factors, no feature recomputation':
            return False
        if receipt['global_oracle_optimality'] is not False:
            return False
        values = receipt['values']
        if len(values) != 1 << d or any(type(v) is not int or v < 0 for v in values) or values[-1] != 0:
            return False
        if receipt['states'] != 1 << d:
            return False
        for a in range((1 << d) - 1):
            past = set().union(*(s for j, s in enumerate(supports) if a >> j & 1))
            future = set().union(*(s for j, s in enumerate(supports) if not a >> j & 1))
            needed = [max(1 + len((past & future) | s), values[a | (1 << j)])
                      for j, s in enumerate(supports) if not a >> j & 1]
            if values[a] != min(needed):
                return False
        return receipt['peak'] == values[0] == order_peak(supports, tuple(receipt['order']))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


@dataclass(frozen=True)
class QuadraticForm:
    linear: int
    products: tuple[tuple[int, int], ...]

    def polynomial(self) -> Polynomial:
        result = frozenset(1 << i for i in bits(self.linear))
        for u, v in self.products:
            result = plus(result, times(frozenset(1 << i for i in bits(u)),
                                        frozenset(1 << j for j in bits(v))))
        return result

    def payload(self):
        return {'linear': self.linear, 'products': [list(pair) for pair in self.products]}


def quadratic_form(poly: Polynomial, *, compress: bool = True) -> QuadraticForm:
    """Exact quadratic elimination with fixed pivot order; see Corollary 3a.

    Product pivots occur in just one of the two linear forms. This gives
    in-place CNOT preparation of their parities without extra ancillas.
    """
    if any(mask.bit_count() not in (1, 2) for mask in poly):
        raise ValueError('a factor must be a nonconstant quadratic Boolean function')
    rest = poly
    products = []
    while any(mask.bit_count() == 2 for mask in rest):
        pair = min(mask for mask in rest if mask.bit_count() == 2)
        i, j = bits(pair)
        u, v = 1 << i, 1 << j
        if compress:
            for mask in rest:
                if mask.bit_count() != 2 or mask == pair:
                    continue
                if mask >> j & 1:
                    u ^= mask ^ (1 << j)
                if mask >> i & 1:
                    v ^= mask ^ (1 << i)
        products.append(tuple(sorted((u, v))))
        uv = times(frozenset(1 << k for k in bits(u)), frozenset(1 << k for k in bits(v)))
        rest = plus(rest, uv)
    form = QuadraticForm(sum(rest), tuple(products))
    if form.polynomial() != poly:
        raise AssertionError('quadratic elimination changed a function')
    return form


def factor_forms(f: Factorization, *, compress: bool = True):
    """Lower rank factors to shared products of linear forms and linear terms."""
    forms = tuple((quadratic_form(frozenset(f.b_features[i] for i in bits(u)), compress=compress),
                   quadratic_form(frozenset(f.x_features[j] for j in bits(v)), compress=compress))
                  for u, v in f.factors)
    features = tuple(sorted({pair for left, right in forms for form in (left, right) for pair in form.products}))
    index = {feature: j for j, feature in enumerate(features)}
    supports = tuple(frozenset(index[pair] for form in pair_forms for pair in form.products) for pair_forms in forms)
    return forms, features, supports
