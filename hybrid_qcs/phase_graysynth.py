# Adapted from Qiskit qiskit/synthesis/linear_phase/cnot_phase_synth.py,
# version 0.46.3, blob f1b49cc8d8df1ffba2d25830deb2875106f5c3f5.
# (C) Copyright IBM 2017, 2019.
# Licensed under the Apache License, Version 2.0. See publication/LICENSE-QISKIT.txt.
# Modified 2026: dependency-free token output, exact phase blocks, stable index
# order, identity-based stack alias handling, Gaussian/inverse restoration in
# place of Patel-Markov-Hayes restoration. NOT a full Qiskit performance claim.
"""GraySynth cofactor recursion with explicitly identified restoration variant.

Amy, Azimzadeh, Mosca, arXiv:1712.01859. All witnesses are separately certified.
This nonlearned control is never invoked by the learned discovery pipeline.
"""
from dataclasses import replace
from .phase_contract import root_state, successor, certify_phase


def graysynth_baseline(problem, *, dag=True):
    if problem.emission != 'deferred':
        raise ValueError('GraySynth control uses explicit phase emission tokens')
    p = problem
    n = p.n
    lookup = {pair: i for i, pair in enumerate(p.actions)}
    current, _ = root_state(p)
    tokens, cnots = [], []

    def emit(q):
        nonlocal current
        if current.remaining & p.phase_bits.get(current.rows[q], 0):
            token = lookup[q,q]
            current, _ = successor(p, current, token)
            tokens.append(token)

    def cx(c,t, collect=True):
        nonlocal current
        token = lookup[c,t]
        current, _ = successor(p, current, token)
        tokens.append(token)
        if collect:
            cnots.append((c,t))
            emit(t)

    for q in range(n):
        emit(q)
    matrix = [[(mask>>q)&1 for mask,_ in p.coefficients] for q in range(n)]
    stack = [(matrix,list(range(n)),None)] if p.coefficients else []
    while stack:
        subset,indices,target = stack.pop()
        if not subset or not subset[0]:
            continue
        if target is not None:
            while True:
                j = next((j for j in range(n) if j != target and all(subset[j])), None)
                if j is None:
                    break
                cx(j,target)
                seen = set()
                for other,_,_ in [*stack,(subset,indices,target)]:
                    if not other or id(other) in seen:
                        continue
                    seen.add(id(other))
                    other[j] = [x ^ y for x,y in zip(other[j],other[target])]
        if not indices:
            continue
        j = max(indices, key=lambda q: (max(subset[q].count(0),subset[q].count(1)),-q))
        zero = [k for k,v in enumerate(subset[j]) if not v]
        one = [k for k,v in enumerate(subset[j]) if v]
        remaining = [q for q in indices if q != j]
        if one:
            stack.append(([[row[k] for k in one] for row in subset],remaining,target if target is not None else j))
        if zero:
            stack.append(([[row[k] for k in zero] for row in subset],remaining,target))
    if current.remaining:
        raise AssertionError('GraySynth recursion omitted a phase obligation')
    # An explicitly documented restoration variant, not the PMH implementation.
    rows = list(current.rows[:n])
    cleanup = []
    for col in range(n):
        pivot = next(r for r in range(col,n) if rows[r]&(1<<col))
        if pivot != col:
            for c,t in ((pivot,col),(col,pivot),(pivot,col)):
                rows[t] ^= rows[c]
                cleanup.append((c,t))
        for r in range(n):
            if r != col and rows[r]&(1<<col):
                rows[r] ^= rows[col]
                cleanup.append((col,r))
    if len(cnots) < len(cleanup):
        cleanup = list(reversed(cnots))
    for c,t in cleanup:
        cx(c,t,False)
    relaxed = replace(p,max_cnot=max(p.max_cnot,len(tokens)),max_gates=max(p.max_gates,3*len(tokens)+16),
                      max_depth=max(p.max_depth,3*len(tokens)+16),max_t_depth=max(p.max_t_depth,p.t_count))
    cert = certify_phase(relaxed,tokens,dag=dag)
    resource = cert['resources']
    cert['within_original_contract'] = (resource['cnot'] <= p.max_cnot and resource['depth'] <= p.max_depth
        and resource['gates'] <= p.max_gates and resource['t_depth'] <= p.max_t_depth)
    cert['original_problem_digest'] = p.digest
    cert['source'] = 'GraySynth cofactor recursion + Gaussian/inverse restoration; Qiskit-0.46.3 adaptation'
    return cert
