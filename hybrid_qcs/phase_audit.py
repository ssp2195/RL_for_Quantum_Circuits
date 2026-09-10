"""Standalone closed-cover auditing for the finite scheduled phase domain.

The reference step does not call the discovery transition, feature extractor,
or any policy. A completed audit is a proof only after verify_phase_cover.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
from .phase_contract import PhaseProblem, PhaseState, problem_from_manifest, certify_phase
from .resource_domain import canonical_digest
from .resource_search import WorkLimits, WorkMeter

SCHEMA = 'phase-inductive-cover-v1'


def reference_root(p):
    rows = tuple([2**q for q in range(p.n)] + [0]*p.ancillas)
    missing = (1 << len(p.coefficients)) - 1
    depths, gates = [0]*p.width, 0
    t_depths = [0]*p.width
    if p.constant == 4:
        depths[0], gates = 12, 12
    lengths = (0, 1, 1, 2, 2, 2, 1, 1)
    for i, (mask, coefficient) in enumerate(p.coefficients):
        if p.emission == 'eager' and mask in rows:
            q = rows.index(mask)
            depths[q] += lengths[coefficient]
            t_depths[q] += coefficient % 2
            gates += lengths[coefficient]
            missing &= ~(1 << i)
    return PhaseState(rows, missing, tuple(depths), 0, gates, tuple(t_depths))


def reference_step(p, state, token):
    pairs = [(c,t) for c in range(p.width) for t in range(p.width) if c != t]
    if p.emission == 'deferred':
        pairs += [(q,q) for q in range(p.width)]
    c, t = pairs[token]
    rows = list(state.rows)
    if c != t:
        rows[t] = rows[t] ^ rows[c]
    depths = list(state.depths)
    t_depths = list(state.t_depths)
    if c != t:
        depths[c] = depths[t] = max(depths[c], depths[t]) + 1
        t_depths[c] = t_depths[t] = max(t_depths[c], t_depths[t])
    missing, gates = state.remaining, state.gates + int(c != t)
    lengths = (0, 1, 1, 2, 2, 2, 1, 1)
    for i, (mask, coefficient) in enumerate(p.coefficients):
        if (p.emission == 'eager' or c == t) and rows[t] == mask and missing & (1 << i):
            missing &= ~(1 << i)
            depths[t] += lengths[coefficient]
            t_depths[t] += coefficient % 2
            gates += lengths[coefficient]
    return PhaseState(tuple(rows), missing, tuple(depths), state.cnot + int(c != t), gates, tuple(t_depths))


def _covers(a,b):
    return a.rows == b.rows and a.remaining == b.remaining and a.cnot <= b.cnot and a.gates <= b.gates and all(x<=y for x,y in zip(a.depths,b.depths,strict=True)) and all(x<=y for x,y in zip(a.t_depths,b.t_depths,strict=True))


def _within(p,s):
    absent = sum(1 for i, (m, _) in enumerate(p.coefficients) if s.remaining & (1 << i) and m not in s.rows)
    repairs = sum(s.rows[q] != ((1 << q) if q < p.n else 0) for q in range(p.width))
    lower = max(absent, repairs)
    phase_cost = sum((0,1,1,2,2,2,1,1)[c] for i,(_,c) in enumerate(p.coefficients) if s.remaining & (1 << i))
    odd = sum(c % 2 for i,(_,c) in enumerate(p.coefficients) if s.remaining & (1 << i))
    capacity = sum(p.max_t_depth - d for d in s.t_depths)
    return odd <= capacity and all(d <= p.max_t_depth for d in s.t_depths) and s.cnot + lower <= p.max_cnot and s.gates + lower + phase_cost <= p.max_gates and all(d <= p.max_depth for d in s.depths)


def audit_phase(p, *, limits=WorkLimits(500000, 100000, 30., 30.), cancel=None):
    meter = WorkMeter(limits,cancel)
    root = reference_root(p)
    if not _within(p, root):
        proof = {'schema': SCHEMA, 'problem': p.manifest(), 'claim':'infeasible', 'root_exceeds':True, 'cover':[]}
        proof['digest'] = canonical_digest(proof)
        return {'status':'infeasible','proof':proof,'witness':None,'edges':0,'wall_seconds':meter.wall,'cpu_seconds':meter.cpu}
    nodes = [(root,None,None)]
    groups = {root.key:[0]}
    queue = deque([0])
    def finish(status,reason,witness=None,proof=None):
        return {'status':status,'reason':reason,'witness':witness,'proof':proof,'edges':meter.edges,
                'records':len(nodes),'wall_seconds':meter.wall,'cpu_seconds':meter.cpu,'witness_source':'auditor' if witness else None}
    def cert(rid):
        path=[]
        while nodes[rid][1] is not None:
            path.append(nodes[rid][2]);rid=nodes[rid][1]
        return certify_phase(p,reversed(path))
    while queue:
        if meter.reason():return finish('unknown',meter.reason())
        rid=queue.popleft()
        s=nodes[rid][0]
        if rid not in groups[s.key]:continue
        if s.terminal(p):return finish('feasible','auditor_discovery',cert(rid))
        # Phase continuations remain possible when the CNOT cap is reached.
        for token in range(len(p.actions)):
            if meter.reason():return finish('unknown',meter.reason())
            child=reference_step(p,s,token);meter.edges+=1
            if not _within(p,child):continue
            group=groups.get(child.key,[])
            if any(_covers(nodes[i][0],child) for i in group):continue
            if len(nodes)>=limits.max_records:return finish('unknown','record_limit')
            survivors=[i for i in group if not _covers(child,nodes[i][0])]
            new=len(nodes);nodes.append((child,rid,token));groups[child.key]=survivors+[new];queue.append(new)
            if child.terminal(p):return finish('feasible','auditor_discovery',cert(new))
    cover=[asdict(nodes[rid][0]) for key in sorted(groups) for rid in groups[key]]
    proof={'schema':SCHEMA,'problem':p.manifest(),'claim':'infeasible','root_exceeds':False,'cover':cover}
    proof['digest']=canonical_digest(proof)
    return finish('infeasible','closure_candidate',proof=proof)


def verify_phase_cover(proof, expected=None, *, limits=WorkLimits(2000000,200000,60.,60.),cancel=None):
    meter=WorkMeter(limits,cancel)
    def done(valid,reason):return {'valid':valid,'reason':reason,'edges':meter.edges,'wall_seconds':meter.wall,'cpu_seconds':meter.cpu}
    try:
        if proof.get('schema')!=SCHEMA or proof.get('claim')!='infeasible':return done(False,'schema_or_claim')
        if proof.get('digest')!=canonical_digest({k:v for k,v in proof.items() if k!='digest'}):return done(False,'digest')
        p=problem_from_manifest(proof['problem'])
        if expected is not None and p.digest!=expected.digest:return done(False,'wrong_contract')
        root=reference_root(p)
        if proof.get('root_exceeds'):
            return done(not _within(p,root) and proof['cover']==[], 'root_resource_check')
        if not isinstance(proof['cover'],list) or len(proof['cover'])>limits.max_records:return done(False,'cover_size')
        groups={}
        for label in proof['cover']:
            if meter.reason():return done(False,meter.reason())
            if set(label)!=set(('rows','remaining','depths','cnot','gates','t_depths')):return done(False,'label_fields')
            rows,depths=label['rows'],label['depths']
            tdepths=label['t_depths']
            if len(rows)!=p.width or len(depths)!=p.width or len(tdepths)!=p.width:return done(False,'label_width')
            if any(type(x)is not int or x<0 for x in [*rows,*depths,*tdepths,label['remaining'],label['cnot'],label['gates']]):return done(False,'invalid_integer')
            if any(x>=1<<p.n for x in rows) or label['remaining']>=1<<len(p.coefficients):return done(False,'label_range')
            s=PhaseState(tuple(rows),label['remaining'],tuple(depths),label['cnot'],label['gates'],tuple(tdepths))
            if not _within(p,s) or s.terminal(p):return done(False,'target_or_outside_bound')
            groups.setdefault(s.key,[]).append(s)
        if not any(_covers(s,root) for s in groups.get(root.key,())):return done(False,'missing_root')
        for states in groups.values():
            for s in states:
                # Do not omit phase-only continuations at the CNOT cap.
                for token in range(len(p.actions)):
                    if meter.reason():return done(False,meter.reason())
                    child=reference_step(p,s,token);meter.edges+=1
                    if _within(p,child) and not any(_covers(old,child) for old in groups.get(child.key,())):
                        return done(False,'uncovered_successor')
        return done(True,'closed_cover_verified')
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        return done(False,'invalid_certificate: '+str(exc))
