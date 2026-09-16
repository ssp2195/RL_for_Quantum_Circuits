"""Independent, exact branch/interface checks for the restricted protocol family.

No call to a policy, search engine, generator or dense global HybridState.
Coherent primitive identities are checked on all clean-input columns using the
existing independent cyclotomic backend. A square-free Boolean interpreter then
proves the complete corrected branch identity symbolically for ALL inputs and
ALL transcripts. The classical transcript is internal and cannot be read by a
subsequent external consumer under this contract.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
from fractions import Fraction
from typing import Callable

from ..native_exact import ExactMatrix, ONE, ZERO, add, sub, omega_times
from .contract import (ASSUMPTIONS, THEOREM, CleanupProblem, Layout, digest,
                       plus, times, evaluate)
from .ir import Protocol, Instruction, local_word, resources


@lru_cache(None)
def primitive_receipts(primitive: str) -> dict:
    """Check the actual native words, not an asserted cost or truth table."""
    targets = {
        'AND': tuple((x ^ (((x & 1) & ((x >> 1) & 1)) << 2)) for x in range(4)),
    }
    clean = ExactMatrix(tuple(tuple(ONE if i == j else ZERO for j in range(4)) for i in range(8)))
    expected = ExactMatrix(tuple(tuple(ONE if i == targets['AND'][j] else ZERO for j in range(4)) for i in range(8)))
    computed = clean
    for name, qs in local_word('AND', primitive): computed = computed.apply(name, qs)
    if computed != expected: raise AssertionError('AND primitive has an incorrect clean-input phase/action')
    undone = expected
    for name, qs in local_word('UNAND', primitive): undone = undone.apply(name, qs)
    if undone != clean: raise AssertionError('inverse AND promise is invalid')
    words = {}
    for kind in ('Z', 'CZ', 'X', 'H', 'CX', 'MINUS'):
        word = local_word(kind, primitive)
        n = 1+max(q for _, qs in word for q in qs)
        actual = ExactMatrix.identity(n)
        for name, qs in word: actual = actual.apply(name, qs)
        if kind != 'H':
            rows = []
            for i in range(1 << n):
                row = []
                for j in range(1 << n):
                    image = j ^ 1 if kind == 'X' else j ^ (((j & 1) << 1)) if kind == 'CX' else j
                    sign = (-1 if kind == 'MINUS' or (kind == 'Z' and j & 1) or (kind == 'CZ' and j == 3) else 1)
                    row.append(tuple(sign*x for x in ONE) if i == image else ZERO)
                rows.append(tuple(row))
            if actual != ExactMatrix(tuple(rows)): raise AssertionError(f'bad {kind} lowering')
        words[kind] = actual.digest
    # Verify the executable H, MZ, conditional X reset equals |0><s_X|.
    for s in (0, 1):
        h = ExactMatrix.identity(1).apply('H', (0,))
        projected = ExactMatrix(tuple(h.rows[i] if i == s else (ZERO, ZERO) for i in range(2)), h.denominator_power)
        if s:
            for name, qs in local_word('X'): projected = projected.apply(name, qs)
        expected_reset = ExactMatrix((h.rows[s], (ZERO, ZERO)), h.denominator_power)
        if projected != expected_reset: raise AssertionError('incorrect conditional reset')
    return {'primitive': primitive, 'clean_AND_isometry_digest': computed.digest,
            'native_word_digest': digest(local_word('AND', primitive)),
            'exact_Clifford_blocks': words, 'MX_reset_exact': True,
            'scope': 'exact integer-cyclotomic local identities; not a full-target Toffoli for and4'}


def _architecture(p: CleanupProblem, prot: Protocol) -> None:
    """Check the precise full-bank protocol, independently of its search state."""
    l = prot.layout
    if (l.r, l.m) != (p.r, p.m) or prot.problem_digest != p.digest:
        raise ValueError('target contract mismatch')
    expected_stages = (('prepare', 'consumer', 'measure_products', 'correct_products', 'measure_helpers', 'correct_helpers')
                       if prot.cleanup_mode == 'measured' else
                       ('prepare', 'consumer', 'uncompute_products', 'uncompute_helpers'))
    seen = []
    grouped = {}
    for op in prot.ops:
        if not seen or seen[-1] != op.stage:
            if op.stage in seen: raise ValueError('interleaved or repeated protocol stage')
            seen.append(op.stage)
        grouped.setdefault(op.stage, []).append(op)
    if any(s not in expected_stages for s in seen) or seen != [s for s in expected_stages if s in seen]:
        raise ValueError('unsupported live/garbage interaction or stage order')
    def signature(op): return (op.kind, op.wires, op.guard, op.outcome)
    def compare(stage, expected):
        if Counter(signature(op) for op in grouped.get(stage, [])) != Counter(expected):
            raise ValueError(f'{stage} is not the stated construction')
    compute = [('AND', l.helper_operands(j), None, None) for j in range(l.k)]
    compute += [('AND', l.product_operands(j), None, None) for j in range(p.r*p.m)]
    compare('prepare', compute)
    phase = []
    for term in p.consumer.terms:
        phase.append(('MINUS', (0,), None, None) if not term else
                     ('Z' if len(term) == 1 else 'CZ', tuple(l.wire(s) for s in term), None, None))
    compare('consumer', phase)
    if prot.cleanup_mode == 'measured':
        compare('measure_products', [('MX', (q,), None, k) for k, q in enumerate(l.products)])
        compare('correct_products', [('CZ', l.product_operands(k)[:2], k, None) for k in range(p.r*p.m)])
        compare('measure_helpers', [('MX', (q,), None, p.r*p.m+k) for k, q in enumerate(l.helpers)])
        compare('correct_helpers', [('CZ', l.helper_operands(k)[:2], p.r*p.m+k, None) for k in range(l.k)])
    else:
        compare('uncompute_products', [('UNAND', l.product_operands(k), None, None) for k in range(p.r*p.m)])
        compare('uncompute_helpers', [('UNAND', l.helper_operands(k), None, None) for k in range(l.k)])


def verify_protocol(problem: CleanupProblem, protocol: Protocol | dict, *, cancel: Callable[[], bool] | None = None) -> dict:
    """Fail closed. The proof is by exact polynomial coefficient identities.

    No input sampling or truth-table equivalence is used here. The polynomial
    semantics are LOCAL promises for a structured dynamic block, not the native
    phase-polynomial search engine and not an unrestricted synthesis solver.
    """
    try:
        prot = Protocol.from_payload(protocol) if isinstance(protocol, dict) else protocol
        if not isinstance(prot, Protocol): raise TypeError('protocol required')
        _architecture(problem, prot)
        receipt = primitive_receipts(prot.primitive)
        values = [frozenset({1 << i}) for i in range(problem.n)] + [frozenset()]*(prot.layout.width-problem.n)
        target_inputs = tuple(values[:problem.n])
        phase = frozenset(); coefficients: dict[int, frozenset[int]] = {}
        max_terms = 0
        for op in prot.ops:
            if cancel is not None and cancel(): raise InterruptedError('cancelled')
            if op.guard is not None and op.guard not in coefficients: raise ValueError('future/unknown measurement outcome')
            if op.kind in ('AND', 'UNAND'):
                c, d, q = op.wires
                product = times(values[c], values[d])
                if op.kind == 'AND':
                    if values[q]: raise ValueError('AND target is not clean')
                    values[q] = product
                else:
                    if values[q] != product: raise ValueError('inverse AND has lost its live relation')
                    values[q] = frozenset()
            elif op.kind == 'MX':
                if op.outcome in coefficients: raise ValueError('transcript overwritten')
                coefficients[op.outcome] = values[op.wires[0]]
                values[op.wires[0]] = frozenset()
            elif op.kind == 'CX':
                c, t = op.wires; values[t] = plus(values[t], values[c])
            else:
                term = (frozenset({0}) if op.kind == 'MINUS' else
                        values[op.wires[0]] if op.kind == 'Z' else times(values[op.wires[0]], values[op.wires[1]]))
                if op.guard is None: phase = plus(phase, term)
                else: coefficients[op.guard] = plus(coefficients[op.guard], term)
            max_terms = max(max_terms, len(phase)+sum(len(v) for v in values)+sum(len(c) for c in coefficients.values()))
        if tuple(values[:problem.n]) != target_inputs or any(values[problem.n:]):
            raise ValueError('logical inputs changed or workspace not coherently reset')
        if phase != problem.target_polynomial: raise ValueError('incorrect complete oracle phase')
        if any(coefficients.values()): raise ValueError('uncancelled measurement-dependent phase')
        r = resources(prot, problem).report()
        if not problem.limits.accepts(r): raise ValueError('resource contract exceeded')
        expected_m = problem.r*problem.m+prot.layout.k if prot.cleanup_mode == 'measured' else 0
        if r['measurements'] != expected_m: raise ValueError('incorrect transcript count')
        return {'valid': True, 'schema': 'cleanup-proof-v1', 'problem_digest': problem.digest,
                'protocol_digest': prot.digest, 'resources': r, 'primitive_receipt': receipt,
                'exact_branch_coefficient_identities': len(coefficients),
                'branch_count': 1 << len(coefficients),
                'branch_probability': str(Fraction(1, 1 << len(coefficients))),
                'branch_amplitude': f'2^(-{len(coefficients)}/2)',
                'target_anf': sorted(phase), 'max_live_polynomial_terms': max_terms,
                'all_outcomes_checked_symbolically': True, 'probability_sum': '1',
                'preserves_reference_entanglement': True, 'native_lowering_verified': True,
                'global_dense_arrays_allocated': False,
                'scope': 'complete exact phase oracle, internal classical transcript discarded after feedback'}
    except (ValueError, KeyError, TypeError, IndexError, AssertionError, InterruptedError) as e:
        return {'valid': False, 'reason': str(e)}


def workspace_certificate(problem: CleanupProblem, prot: Protocol) -> dict:
    checked = verify_protocol(problem, prot)
    if not checked['valid'] or prot.cleanup_mode != 'measured':
        raise ValueError('workspace theorem requires a valid measured lifecycle')
    return {'kind': THEOREM, 'problem_digest': problem.digest, 'protocol_digest': prot.digest,
            'assumptions': list(ASSUMPTIONS), 'lower_bound': problem.workspace_lower_bound,
            'upper_bound': checked['resources']['peak_aux'],
            'optimal_within_architecture': checked['resources']['peak_aux'] == problem.workspace_lower_bound,
            'global_oracle_optimality': False,
            'source': 'H - Prove Cleanup Theorem(1).pdf, Theorem 2, pages 59-60',
            'trusted_mathematical_converse': True}


def verify_workspace_certificate(problem: CleanupProblem, prot: Protocol | dict, certificate: dict) -> dict:
    """Re-derive all certificate fields. Never promotes a scope-limited bound."""
    try:
        p = Protocol.from_payload(prot) if isinstance(prot, dict) else prot
        expected = workspace_certificate(problem, p)
        if certificate != expected: raise ValueError('theorem fields, assumptions or scope were changed')
        return {'valid': True, 'lower_bound': expected['lower_bound'], 'upper_bound': expected['upper_bound'],
                'optimal_within_architecture': expected['optimal_within_architecture'],
                'global_oracle_optimality': False}
    except (ValueError, KeyError, TypeError, IndexError) as e:
        return {'valid': False, 'reason': str(e)}


def verify_small_matrix(problem: CleanupProblem, protocol: Protocol, *, max_width=8) -> dict:
    """Independent native/Kraus replay of every transcript on small registers.

    For large instances verify_protocol uses exact symbolic coefficient proofs.
    This test-only matrix path intentionally retains the small-width guard.
    """
    width, n = protocol.layout.width, problem.n
    if width > max_width: raise ValueError('dense branch check exceeds its explicit width guard')
    initial = ExactMatrix(tuple(tuple(ONE if i == j else ZERO for j in range(1 << n)) for i in range(1 << width)))
    expected = ExactMatrix(tuple(tuple(tuple(-x for x in ONE) if i == j and evaluate(problem.target_polynomial, j)
                                       else ONE if i == j else ZERO for j in range(1 << n)) for i in range(1 << width)))
    m = sum(op.kind == 'MX' for op in protocol.ops)
    def half_sqrt(a):
        return ExactMatrix(tuple(tuple(sub(omega_times(z, 1), omega_times(z, 3)) for z in r) for r in a.rows), a.denominator_power+1)
    for _ in range(m): expected = half_sqrt(expected)
    for transcript in range(1 << m):
        actual = initial
        for op in protocol.ops:
            if op.kind == 'MX':
                q = op.wires[0]; s = (transcript >> op.outcome) & 1
                # Explicit lowering: H, projector |s><s|, and conditional X.
                actual = actual.apply('H', (q,))
                actual = ExactMatrix(tuple(r if ((i >> q) & 1) == s else (ZERO,)*(1 << n) for i, r in enumerate(actual.rows)), actual.denominator_power)
                word = local_word('X') if s else ()
            else:
                word = local_word(op.kind, protocol.primitive) if op.guard is None or (transcript >> op.guard) & 1 else ()
            for name, qs in word:
                actual = actual.apply(name, tuple(op.wires[k] for k in qs))
        if actual != expected:
            return {'valid': False, 'failed_transcript': transcript}
    return {'valid': True, 'branches': 1 << m, 'input_columns': 1 << n, 'physical_qubits': width,
            'mode': 'all native gates and Kraus branches, exact cyclotomic arithmetic'}


def verify_truth_table_generators(problem: CleanupProblem, protocol: Protocol, *, max_inputs=16) -> dict:
    """Differential check using packed truth tables, not ANF multiplication.

    All Boolean inputs and every outcome generator are checked. Because emitted
    corrections depend linearly on transcript bits, generator equality implies
    equality for every transcript. This exponential-in-inputs diagnostic is not
    used by the scalable production verifier.
    """
    if problem.n>max_inputs:raise ValueError('truth-table diagnostic input guard')
    n=problem.n;size=1<<n;ones=(1<<size)-1
    values=[sum(((x>>i)&1)<<x for x in range(size)) for i in range(n)] + [0]*(protocol.layout.width-n)
    initial=values[:n];phase=0;coeff={}
    for op in protocol.ops:
        qs=op.wires
        if op.kind=='AND':
            if values[qs[2]]:return {'valid':False,'reason':'unclean target'}
            values[qs[2]]=values[qs[0]]&values[qs[1]]
        elif op.kind=='UNAND':values[qs[2]]^=values[qs[0]]&values[qs[1]]
        elif op.kind=='MX':coeff[op.outcome]=values[qs[0]];values[qs[0]]=0
        elif op.kind=='CX':values[qs[1]]^=values[qs[0]]
        else:
            term=ones if op.kind=='MINUS' else values[qs[0]] if op.kind=='Z' else values[qs[0]]&values[qs[1]]
            if op.guard is None:phase^=term
            else:coeff[op.guard]^=term
    # Direct evaluation of the supplied q on a,b,x,F, NOT target_polynomial.
    expected=0
    for x in range(size):
        z={'a':x&1};z.update({f'b{i}':(x>>(1+i))&1 for i in range(problem.r)})
        z.update({f'x{j}':(x>>(1+problem.r+j))&1 for j in range(problem.m)})
        z.update({f'f{i}_{j}':z['a']*z[f'b{i}']*z[f'x{j}'] for i in range(problem.r) for j in range(problem.m)})
        s=0
        for term in problem.consumer.terms:
            v=1
            for name in term:v&=z[name]
            s^=v
        expected|=s<<x
    return {'valid':phase==expected and not any(coeff.values()) and values[:n]==initial and not any(values[n:]),
            'raw_inputs':size,'outcome_generators':len(coeff),'input_generator_pairs':size*(len(coeff)+1),
            'method':'independent packed truth tables, all raw inputs and outcome generators'}
