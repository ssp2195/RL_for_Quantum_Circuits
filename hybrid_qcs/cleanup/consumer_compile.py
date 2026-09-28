"""Construct complete phase oracles from proved consumer identities.

Small exact binary elimination and subset dynamic programming replace guesses.
The native words, measurement/reset, conditional correction and stage barriers
are retained in every result. Each coherent operation uses the existing hybrid
representation. No whole-register dense matrix is allocated.
"""
from __future__ import annotations
from typing import Callable

from .contract import CleanupProblem
from .consumer import bits, factor_consumer, minimum_live_order, check_cancel, order_peak, factor_forms, QuadraticForm
from .ir import Instruction
from .phase_protocol import PhaseProtocol, verify_phase


def compile_factors(p: CleanupProblem, *, storage: str = 'stream',
                    order: tuple[int, ...] | None = None, max_factors: int = 16,
                    compress_quadratics: bool = True,
                    cancel: Callable[[], bool] | None = None) -> tuple[PhaseProtocol, dict]:
    """Three proved constructions: store all features, stream, or recompute.

    Streaming uses an exact no-recomputation peak optimum for the *fixed*
    factorization. Explicit orders are upper bounds. Beyond max_factors, the
    canonical order is used with an explicit unproved-optimum status.
    """
    if type(max_factors) is not int or not 0 <= max_factors <= 16:
        raise ValueError('exact factor limit must be in [0,16]')
    if storage not in ('all', 'stream', 'recompute'):
        raise ValueError('storage must be all, stream or recompute')
    check_cancel(cancel)
    f = factor_consumer(p)
    forms, feature_table, supports = factor_forms(f, compress=compress_quadratics)
    all_features = frozenset().union(*supports)
    schedule = None
    schedule_status = 'fixed_order_upper_bound'
    if order is None:
        order = tuple(range(f.rank))
        if storage == 'stream':
            if f.rank <= max_factors:
                schedule = minimum_live_order(supports, max_factors=max_factors, cancel=cancel)
                order = tuple(schedule['order'])
                schedule_status = 'proved_fixed_factor_no_recomputation_peak'
            else:
                schedule_status = 'exact_order_limit; canonical_order_upper_bound'
    if sorted(order) != list(range(f.rank)):
        raise ValueError('order must contain every factor exactly once')

    capacity = (0 if not f.rank else 1 + len(all_features) if storage == 'all' else
                1 + max(map(len, supports), default=0) if storage == 'recompute' else
                order_peak(supports, order))
    free = list(range(p.n + 1, p.n + capacity))
    live: dict[int, int] = {}
    ops: list[Instruction] = []
    bit = 0
    batch = 0
    computed = 0
    helper = p.n

    def linear_pair(u, v):
        if not u & ~v or not v & ~u:
            raise ValueError('product has no independent in-place pivots')
        i, j = bits(u & ~v)[0], bits(v & ~u)[0]
        changes = [(q, i) for q in bits(u) if q != i] + [(q, j) for q in bits(v) if q != j]
        return i, j, changes

    feature_index = {feature: j for j, feature in enumerate(feature_table)}

    def prepare(features):
        nonlocal computed
        for feature in sorted(features):
            if feature not in live:
                if not free:
                    raise AssertionError('proved live width is insufficient')
                target = free.pop(0)
                u, v = feature_table[feature]
                i, j, changes = linear_pair(u, v)
                ops.extend(Instruction('CX', pair, 'prepare') for pair in changes)
                live[feature] = target
                ops.append(Instruction('AND', (i, j, target), 'prepare'))
                ops.extend(Instruction('CX', pair, 'prepare') for pair in reversed(changes))
                computed += 1

    def erase(features):
        nonlocal bit, batch
        todo = sorted(features)
        if not todo:
            return
        stage = f'measure_features_{batch}'
        corrections = []
        for feature in todo:
            target = live.pop(feature)
            ops.append(Instruction('MX', (target,), stage, outcome=bit))
            u, v = feature_table[feature]
            i, j, changes = linear_pair(u, v)
            corrections.extend(Instruction('CX', pair, f'correct_features_{batch}') for pair in changes)
            corrections.append(Instruction('CZ', (i, j), f'correct_features_{batch}', guard=bit))
            corrections.extend(Instruction('CX', pair, f'correct_features_{batch}') for pair in reversed(changes))
            free.append(target)
            bit += 1
        ops.extend(corrections)
        free.sort()
        batch += 1

    def parity(form):
        wires = list(bits(form.linear)) + [live[feature_index[pair]] for pair in form.products]
        if not wires:
            raise AssertionError('zero rank factor')
        pivot = wires[0]
        changes = [(q, pivot) for q in wires[1:]]
        ops.extend(Instruction('CX', pair, 'prepare') for pair in changes)
        return pivot, changes

    for mask in sorted(f.quadratic):
        qs = bits(mask)
        ops.append(Instruction('MINUS' if not qs else 'Z' if len(qs) == 1 else 'CZ',
                               qs if qs else (0,), 'consumer'))
    if storage == 'all':
        prepare(all_features)
    for position, j in enumerate(order):
        check_cancel(cancel)
        prepare(supports[j])
        left, undo_left = parity(forms[j][0])
        right, undo_right = parity(forms[j][1])
        ops.append(Instruction('AND', (0, left, helper), 'prepare'))
        ops.append(Instruction('CZ', (helper, right), 'consumer'))
        ops.append(Instruction('MX', (helper,), f'measure_factor_{position}', outcome=bit))
        ops.append(Instruction('CZ', (0, left), f'correct_factor_{position}', guard=bit))
        bit += 1
        # The correction must see the parity used to compute the measured helper.
        for pair in reversed(undo_left + undo_right):
            ops.append(Instruction('CX', pair, f'restore_factor_{position}'))
        if storage == 'stream':
            future = frozenset().union(*(supports[k] for k in order[position+1:]))
            erase(set(live) - future)
        elif storage == 'recompute':
            erase(tuple(live))
    erase(tuple(live))
    protocol = PhaseProtocol(p.digest, p.n + capacity, tuple(ops), f'exact_feature_factors_{storage}')
    receipt = {'schema': 'consumer-factor-construction-v1', 'factorization': f.payload(),
               'storage': storage, 'order': list(order), 'feature_computations': computed,
               'quadratic_compression': compress_quadratics,
               'forms': [[left.payload(), right.payload()] for left, right in forms],
               'feature_table': [list(pair) for pair in feature_table],
               'predicted_t': 4 * (f.rank + computed), 'predicted_peak': capacity,
               'schedule_status': schedule_status, 'schedule_certificate': schedule,
               'global_oracle_optimality': False}
    return protocol, receipt


def verify_factor_receipt(p: CleanupProblem, protocol: PhaseProtocol, receipt: dict) -> bool:
    """Bind arithmetic and schedule assertions to the actual emitted circuit.

    The complete symbolic checker proves physical correctness. This additional
    receipt checks the claimed fixed-factor peak and computation-count formulas.
    It is not an unrestricted synthesis-optimality certificate.
    """
    from .consumer import Factorization, verify_live_certificate
    try:
        if receipt.get('schema') != 'consumer-factor-construction-v1' or type(receipt.get('quadratic_compression')) is not bool:
            return False
        d = receipt['factorization']
        f = Factorization(frozenset(d['quadratic']), tuple(d['b_features']), tuple(d['x_features']),
                          tuple(d['rows']), tuple(tuple(pair) for pair in d['factors']))
        # Independent row-basis check, not a call to the factor-producing routine.
        pivots = {}
        for row in f.rows:
            if type(row) is not int or not 0 <= row < (1 << len(f.x_features)):
                return False
            while row:
                pivot = row.bit_length()-1
                if pivot in pivots:
                    row ^= pivots[pivot]
                else:
                    pivots[pivot] = row
                    break
        if d != f.payload() or f.polynomial() != p.target_polynomial or f.rank != len(pivots):
            return False
        # Reconstruct coefficient matrix from factors, independently of extraction.
        reconstructed = [0] * len(f.rows)
        for u, v in f.factors:
            if not 0 < u < (1 << len(f.rows)) or not 0 < v < (1 << len(f.x_features)):
                return False
            for i in bits(u):
                reconstructed[i] ^= v
        if tuple(reconstructed) != f.rows or receipt['global_oracle_optimality'] is not False:
            return False
        checked = verify_phase(p, protocol)
        if not checked['valid']:
            return False
        forms = tuple(tuple(QuadraticForm(form['linear'], tuple(tuple(pair) for pair in form['products']))
                            for form in pair_forms) for pair_forms in receipt['forms'])
        if len(forms) != f.rank or any(len(pair) != 2 for pair in forms):
            return False
        for (left, right), (u, v) in zip(forms, f.factors):
            if left.polynomial() != frozenset(f.b_features[i] for i in bits(u)):
                return False
            if right.polynomial() != frozenset(f.x_features[i] for i in bits(v)):
                return False
        features = tuple(sorted({pair for pair_forms in forms for form in pair_forms for pair in form.products}))
        if receipt['feature_table'] != [list(pair) for pair in features]:
            return False
        if any(not u & ~v or not v & ~u for u, v in features):
            return False
        index = {feature: j for j, feature in enumerate(features)}
        supports = tuple(frozenset(index[pair] for form in pair_forms for pair in form.products) for pair_forms in forms)
        order = tuple(receipt['order'])
        if sorted(order) != list(range(f.rank)):
            return False
        mode = receipt['storage']
        all_features = frozenset().union(*supports)
        computations = sum(map(len, supports)) if mode == 'recompute' else len(all_features)
        peak = (0 if not f.rank else 1 + len(all_features) if mode == 'all' else
                1 + max(map(len, supports), default=0) if mode == 'recompute' else order_peak(supports, order))
        if mode not in ('all', 'stream', 'recompute') or receipt['feature_computations'] != computations:
            return False
        if receipt['predicted_t'] != 4 * (f.rank + computations) or receipt['predicted_peak'] != peak:
            return False
        if checked['resources']['t_count'] != receipt['predicted_t'] or checked['resources']['peak_aux'] != peak:
            return False
        schedule = receipt['schedule_certificate']
        if schedule is not None:
            if mode != 'stream' or schedule['supports'] != [sorted(s) for s in supports] or schedule['order'] != list(order):
                return False
            if not verify_live_certificate(schedule) or schedule['peak'] != peak:
                return False
            if receipt['schedule_status'] != 'proved_fixed_factor_no_recomputation_peak':
                return False
        elif receipt['schedule_status'] not in ('fixed_order_upper_bound', 'exact_order_limit; canonical_order_upper_bound'):
            return False
        return True
    except (KeyError, ValueError, TypeError, IndexError):
        return False
