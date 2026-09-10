"""Nonlearned rank/layer baseline and small independently checkable rank bounds.

The rank/ancilla connection and T-depth partitioning are established prior work
(Amy, Maslov, Mosca, arXiv:1303.2042; Selinger, arXiv:1210.0974). This module is a
small reference implementation, NOT a reproduction of the full T-par compiler.
Its witnesses are never passed to a learned discovery run.
"""
from __future__ import annotations
from functools import lru_cache
from dataclasses import replace
from .phase_contract import PhaseProblem, certify_phase, problem_from_manifest
from .resource_domain import canonical_digest


def rank(rows):
    pivots={}
    for value in rows:
        while value:
            bit=value.bit_length()-1
            if bit in pivots:value^=pivots[bit]
            else:
                pivots[bit]=value
                break
    return len(pivots)


def rank_tdepth_certificate(p):
    """A witness subset proves |S| <= d*(rank(S)+a) is necessary.

    The maximum of these elementary inequalities is exact for the associated
    matroid partition problem; our certificate only asserts the checked lower
    bound, and optimality is asserted only when an actual circuit meets it.
    """
    odd=tuple(m for m,c in p.coefficients if c%2)
    if len(odd)>16:raise ValueError('reference subset certificate is restricted to at most 16 odd terms')
    best,subset=0,()
    for bits in range(1,1<<len(odd)):
        selected=tuple(m for i,m in enumerate(odd) if bits&(1<<i))
        d=(len(selected)+rank(selected)+p.ancillas-1)//(rank(selected)+p.ancillas)
        if d>best:best,subset=d,selected
    payload={'schema':'phase-rank-tdepth-bound-v1','problem':p.manifest(),
             'subset':list(subset),'lower_t_depth':best,
             'scope':'fixed phase obligations; CNOT and diagonal phases; clean workspace'}
    return {**payload,'digest':canonical_digest(payload)}


def verify_rank_bound(certificate, expected=None):
    """Independent elimination, not a stored rank or a learned confidence score."""
    try:
        if certificate['schema']!='phase-rank-tdepth-bound-v1':return False
        if certificate['digest']!=canonical_digest({k:v for k,v in certificate.items() if k!='digest'}):return False
        p=problem_from_manifest(certificate['problem'])
        if expected is not None and expected.digest!=p.digest:return False
        subset=certificate['subset']
        if len(set(subset))!=len(subset) or any(type(x)is not int or p.phase_table.get(x,0)%2!=1 for x in subset):return False
        if type(certificate['lower_t_depth']) is not int or certificate['lower_t_depth']<0:return False
        if not subset:return certificate['lower_t_depth']==0
        # Reference bit-matrix elimination.
        matrix=[[(mask>>q)&1 for q in range(p.n)] for mask in subset]
        pivot=0
        for column in range(p.n):
            row=next((r for r in range(pivot,len(matrix)) if matrix[r][column]),None)
            if row is None:continue
            matrix[pivot],matrix[row]=matrix[row],matrix[pivot]
            for r in range(pivot+1,len(matrix)):
                if matrix[r][column]:matrix[r]=[a^b for a,b in zip(matrix[r],matrix[pivot])]
            pivot+=1
        lower=(len(subset)+pivot+p.ancillas-1)//(pivot+p.ancillas)
        return certificate['lower_t_depth']==lower
    except (ValueError,KeyError,TypeError,ZeroDivisionError):return False


def partition_terms(p):
    odd=tuple(m for m,c in p.coefficients if c%2)
    if len(odd)>12:raise ValueError('reference partition baseline supports at most 12 odd terms')
    independent=[]
    for bits in range(1,1<<len(odd)):
        group=tuple(m for i,m in enumerate(odd) if bits&(1<<i))
        if len(group)<=rank(group)+p.ancillas:independent.append(bits)
    by_first={i:sorted([b for b in independent if b&(1<<i)],key=lambda b:(-b.bit_count(),b)) for i in range(len(odd))}
    @lru_cache(None)
    def solve(bits):
        if not bits:return ()
        first=(bits&-bits).bit_length()-1
        best=None
        for group in by_first[first]:
            if group&bits!=group:continue
            tail=solve(bits^group)
            candidate=(group,)+tail
            if best is None or len(candidate)<len(best):best=candidate
        return best
    groups=[tuple(m for i,m in enumerate(odd) if bits&(1<<i)) for bits in solve((1<<len(odd))-1)]
    # Clifford phase blocks can be appended without increasing T-depth.
    groups.extend((m,) for m,c in p.coefficients if not c%2)
    return tuple(groups)


def frame_for_group(p,group):
    # Lift linearly dependent promised-input rows into independent physical rows
    # with distinct clean-ancilla coordinates, then complete an invertible basis.
    rows=[];extra=0
    for mask in group:
        vector=mask
        if rank((*rows,vector))==len(rows):
            if extra>=p.ancillas:raise ValueError('group exceeds clean-ancilla rank allowance')
            vector ^= 1<<(p.n+extra);extra+=1
        rows.append(vector)
    for q in range(p.width):
        if rank((*rows,1<<q))>len(rows):rows.append(1<<q)
        if len(rows)==p.width:break
    if len(rows)!=p.width:raise AssertionError('could not complete the frame')
    matrix=list(rows);reduction=[]
    for col in range(p.width):
        pivot=next(r for r in range(col,p.width) if matrix[r]&(1<<col))
        if pivot!=col:
            for c,t in ((pivot,col),(col,pivot),(pivot,col)):
                matrix[t]^=matrix[c];reduction.append((c,t))
        for r in range(p.width):
            if r!=col and matrix[r]&(1<<col):
                matrix[r]^=matrix[col];reduction.append((col,r))
    if matrix!=[1<<q for q in range(p.width)]:raise AssertionError('elimination failed')
    return tuple(reversed(reduction))


def rank_partition_baseline(p, *, dag=True):
    if p.emission!='deferred':raise ValueError('rank/layer baseline requires scheduled phase placement')
    groups=partition_terms(p)
    lookup={pair:i for i,pair in enumerate(p.actions)}
    tokens=[]
    for group in groups:
        frame=frame_for_group(p,group)
        tokens.extend(lookup[pair] for pair in frame)
        tokens.extend(lookup[q,q] for q in range(len(group)))
        tokens.extend(lookup[pair] for pair in reversed(frame))
    # A baseline may exceed the comparison cap; expose that, never silently
    # relax the target contract in a headline success rate.
    relaxed=replace(p,max_cnot=max(p.max_cnot,len(tokens)),max_depth=max(p.max_depth,4*len(tokens)+16),
                    max_gates=max(p.max_gates,4*len(tokens)+16),max_t_depth=max(p.max_t_depth,len(groups)))
    cert=certify_phase(relaxed,tokens,dag=dag)
    original=(cert['resources']['cnot']<=p.max_cnot and cert['resources']['depth']<=p.max_depth
              and cert['resources']['gates']<=p.max_gates and cert['resources']['t_depth']<=p.max_t_depth)
    cert['original_problem_digest']=p.digest
    cert['within_original_contract']=original
    cert['source']='deterministic_rank_partition_baseline'
    return cert


def uniform_cost_phase(p, *, limits=None, dag=True):
    """Dijkstra label-setting baseline; acceptance is at removal, not generation.

    CNOT count is the scalar queue key. Phase steps have zero CNOT cost. Every
    continuation-relevant resource coordinate participates in label dominance.
    A settled target is CNOT-optimal in this *bounded* fixed-word domain; a
    timeout is unknown. Its discoveries are labelled nonlearned, never RL.
    """
    import heapq
    from .phase_audit import reference_root, reference_step, _within, _covers
    from .resource_search import WorkLimits, WorkMeter
    limits = limits or WorkLimits(4096, 20000, 30., 30.)
    meter = WorkMeter(limits)
    root = reference_root(p)
    nodes = [(root, None, None)]
    groups = {root.key: [0]}
    queue = [(0, 0)] if _within(p, root) else []
    witness = None
    reason = 'exhausted_without_exported_proof'
    while queue:
        if meter.reason():
            reason = meter.reason()
            break
        _, rid = heapq.heappop(queue)
        state = nodes[rid][0]
        if rid not in groups[state.key]:
            continue
        if state.terminal(p):
            path = []
            while nodes[rid][1] is not None:
                path.append(nodes[rid][2])
                rid = nodes[rid][1]
            witness = certify_phase(p, reversed(path), dag=dag)
            if not witness['success']:
                raise AssertionError('uniform-cost witness failed native certification')
            reason = 'settled_goal'
            break
        for token in range(len(p.actions)):
            if meter.reason():
                break
            c,t = p.actions[token]
            if c == t and not state.remaining & p.phase_bits.get(state.rows[t], 0):
                continue
            child = reference_step(p, state, token)
            meter.edges += 1
            if not _within(p, child):
                continue
            group = groups.get(child.key, [])
            if any(_covers(nodes[i][0], child) for i in group):
                continue
            if len(nodes) >= limits.max_records:
                reason = 'record_limit'
                queue.clear()
                break
            new = len(nodes)
            groups[child.key] = [i for i in group if not _covers(child, nodes[i][0])] + [new]
            nodes.append((child, rid, token))
            heapq.heappush(queue, (child.cnot, new))
    return {'schema':'phase-discovery-v1', 'problem':p.manifest(), 'problem_digest':p.digest,
            'scheduler':'uniform_cost', 'status':'feasible' if witness else 'unknown',
            'reason':reason if witness or reason=='record_limit' else meter.reason() or reason,
            'witness':witness, 'witness_source':'uniform_cost' if witness else None,
            'policy_digest':None, 'edges':meter.edges, 'records':len(nodes),
            'wall_seconds':meter.wall, 'cpu_seconds':meter.cpu,
            'profile':{'lookahead_transitions':meter.edges, 'audit_calls':0},
            'audit_witness_used':False,
            'queue_optimality':'CNOT-optimal within contract' if witness else 'no claim'}
