"""Closed covers for the restored native grammar, never phase-domain proofs.

The verifier does not call a policy, search queue or discovery transition cache.
It rebuilds native witnesses and checks every legal continuation. It shares the
exact HybridState algebra (part of the trusted base) and uses independent dense
terminal comparisons. These are tolerance-domain numerical certificates, not
formal algebraic equality proofs for arbitrary floating-point input matrices.
"""
from __future__ import annotations
import numpy as np
from .model import Gate, HybridState
from .certify import unitary_from_gates
from .native_domain import (SCHEMA, digest, archive_key, legal, next_t_depths,
                            resources, matrix_error)
from .resource_search import WorkLimits, WorkMeter

COVER_SCHEMA='native-hybrid-closed-cover-v1'


def verify_native_cover(problem,certificate,*,limits=WorkLimits(1000000,100000,60.,60.)):
    meter=WorkMeter(limits)
    def failure(reason):return {'valid':False,'reason':reason,'checked_edges':meter.edges}
    try:
        if certificate.get('schema')!=COVER_SCHEMA or certificate.get('problem_digest')!=problem.digest:
            return failure('domain/schema mismatch: phase-polynomial proofs cannot certify native search')
        labels=certificate['labels']
        if len(labels)>limits.max_records:return failure('verification record limit')
        if certificate.get('digest')!=digest({'problem_digest':problem.digest,'labels':labels}):
            return failure('certificate digest mismatch')
        group={};states=[]
        for word in labels:
            if meter.reason():return failure(meter.reason())
            s=HybridState.identity(problem.width,problem.budget);td=(0,)*problem.width
            gates=[]
            for name,qs in word:
                if meter.reason():return failure(meter.reason())
                g=Gate(name,tuple(qs))
                if g not in problem.actions or not legal(problem,s,td,g):return failure('invalid witness')
                s=s.apply(g,partial_order_reduction=False);td=next_t_depths(td,g);gates.append(g)
                meter.edges+=1
            actual=unitary_from_gates(problem.width,gates)@problem.contract.input_embedding
            error=matrix_error(problem,actual)
            # Fail closed near the numerical decision boundary as well as at targets.
            if not np.isfinite(error) or error<=10*problem.tolerance:
                return failure('target or numerically unresolved label in exclusion cover')
            states.append((s,td));group.setdefault(archive_key(problem,s),[]).append(resources(s,td))
        def covered(s,td):
            r=resources(s,td)
            return any(all(a<=b for a,b in zip(old,r)) for old in group.get(archive_key(problem,s),[]))
        root=HybridState.identity(problem.width,problem.budget)
        if not covered(root,(0,)*problem.width):return failure('root not covered')
        for s,td in states:
            for g in problem.actions:
                if meter.reason():return failure(meter.reason())
                meter.edges+=1
                if not legal(problem,s,td,g):continue
                child=s.apply(g,partial_order_reduction=False)
                if not covered(child,next_t_depths(td,g)):return failure('feasible continuation not covered')
        return {'valid':True,'schema':COVER_SCHEMA,'checked_edges':meter.edges,'labels':len(states),
                'scope':'declared native grammar, resource caps, phase/ancilla contract and numerical terminal predicate',
                'formal_algebraic_proof':False}
    except (KeyError,TypeError,ValueError,IndexError,AssertionError):
        return failure('malformed certificate')


def audit_native(problem,*,limits=WorkLimits(100000,50000,30.,30.)):
    from .native_search import NativeSearch
    engine=NativeSearch(problem,limits)
    result=engine.run(scheduler='cost')
    if result['witness']:
        witness=dict(result['witness'],source='deterministic_native_audit')
        return {'status':'feasible','witness':witness,'certificate':None,'discovery':result}
    if result['reason']!='exhausted_without_independent_proof':
        return {'status':'unknown','reason':result['reason'],'certificate':None,'discovery':result}
    ids=sorted(rid for group in engine.archive.values() for rid in group)
    labels=[[[g.name,list(g.qubits)] for g in engine.records[rid].state.reconstruct_gates()] for rid in ids]
    cert={'schema':COVER_SCHEMA,'problem_digest':problem.digest,'labels':labels,
          'digest':digest({'problem_digest':problem.digest,'labels':labels})}
    verified=verify_native_cover(problem,cert)
    return {'status':'infeasible_under_numerical_contract' if verified['valid'] else 'unknown',
            'certificate':cert,'verification':verified,'discovery':result}
