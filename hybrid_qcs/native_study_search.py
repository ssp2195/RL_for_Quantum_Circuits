"""Experimental controls around the unchanged authoritative hybrid transitions.

The raw native method is preserved. Work-axis context and three explicitly
labelled ablations are isolated here. Uniform-cost uses global goal-on-pop
ordering, not the legacy heuristic `cost` scheduler. Meet-in-the-middle is a
witness-finding baseline; its rounded candidate index never proves exclusion.
"""
from __future__ import annotations
from dataclasses import asdict
import heapq
import json
from pathlib import Path
import time
import numpy as np
from .model import HybridState, Gate, Budget, generate_gates
from .native_search import NativeSearch
from .native_domain import (matrix_error,certify_native,legal,next_t_depths,archive_key,resources,
                            apply_gate_to_isometry)
from .native_policy import NativeHierarchy,FAMILIES
from .native_exact import ExactMatrix,exact_word,verify_exact_word
from .resource_search import WorkLimits,WorkMeter


class StudyPolicy(NativeHierarchy):
    def __init__(self,seed=0,alpha=.01,ucb=.2,*,variant='full'):
        if variant not in ('full','no_inner_bootstrap','frontier_inner_bootstrap','no_structure'):
            raise ValueError('unknown native-study ablation')
        super().__init__(seed,alpha,ucb);self.variant=variant

    def _outer(self,x):
        x=np.array(x,copy=True)
        if self.variant=='no_structure':x[...,6:10]=0
        return x

    def _inner(self,x):
        x=np.array(x,copy=True)
        if self.variant=='no_structure':x[...,15:17]=0
        return x

    def score_outer(self,x):return super().score_outer(self._outer(x))
    def score_inner(self,x,family,explore=False):return super().score_inner(self._inner(x),family,explore)
    def update_outer(self,x,reward,next_x=None):
        return super().update_outer(self._outer(x),reward,None if next_x is None else self._outer(next_x))
    def update_inner(self,x,family,response):return super().update_inner(self._inner(x),family,response)

    @property
    def response_mode(self):
        return {'no_inner_bootstrap':'base','frontier_inner_bootstrap':'frontier'}.get(self.variant,'child')

    def payload(self):
        return {**super().payload(),'study_variant':self.variant}

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps({'schema':'native-study-policy-v1','model':self.payload()},separators=(',',':'))+'\n')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d.get('schema')!='native-study-policy-v1':raise ValueError('not a native-study checkpoint')
        d=d['model'];obj=cls(d['seed'],d['alpha'],d['ucb'],variant=d['study_variant'])
        for name in ('w','a','b','updates'):
            a=np.asarray(d[name],dtype=int if name=='updates' else float)
            if a.shape!=getattr(obj,name).shape or not np.isfinite(a).all():raise ValueError('malformed study checkpoint')
            setattr(obj,name,a)
        for i in range(len(FAMILIES)):
            if not np.allclose(obj.a[i],obj.a[i].T) or np.linalg.eigvalsh(obj.a[i]).min()<=0:raise ValueError('invalid covariance')
            obj.theta[i]=np.linalg.solve(obj.a[i],obj.b[i])
        if not d['frozen'] or type(d['episodes']) is not int or d['episodes']<1:raise ValueError('unfinished training')
        obj.episodes=d['episodes'];obj.outer_updates=d['outer_updates']
        obj.freeze()
        if json.dumps(obj.payload(),sort_keys=True)!=json.dumps(d,sort_keys=True):raise ValueError('checkpoint schema/context differs')
        return obj


class StudySearch(NativeSearch):
    def __init__(self,problem,limits,*,axis='edges',**kwargs):
        if axis not in ('edges','wall_seconds'):raise ValueError('unknown work axis')
        self.axis=axis
        super().__init__(problem,limits,**kwargs)

    def remaining_fraction(self):
        if self.axis=='edges':return super().remaining_fraction()
        return max(0.,1-self.meter.wall/max(1e-9,self.limits.wall_seconds))


def summarize_result(p,r,elapsed,axis,budget):
    """A terminal witness obtained after the wall deadline is NOT timely success."""
    timely = bool(r.get('witness')) and (axis!='wall_seconds' or elapsed<=budget)
    return {'status':r['status'],'reason':r['reason'],'witness':r.get('witness'),
            'timely_success':timely,'external_wall_seconds':elapsed,
            'wall_seconds':r['wall_seconds'],'cpu_seconds':r.get('cpu_seconds',0.),
            'edges':r['edges'],'records':r['records'],'profile':r.get('profile',{}),
            'late_witness':bool(r.get('witness')) and not timely,
            'audit_witness_used':False,'policy_digest':r.get('policy_digest'),
            'limits_are_cooperative':True}


def uniform_cost(p,limits):
    """Unit native-gate cost Dijkstra; exact symbolic/resource archive, goal-on-pop.

    No feature computation or learned priorities. A popped certified goal has
    minimum gate count under the fixed other bounds, but this routine exports
    only a witness; the independent cover checker is the proof path.
    """
    meter=WorkMeter(limits);heap=[];archive={};records=0;peak=0;iso_bytes=0
    root=HybridState.identity(p.width,p.budget)
    heapq.heappush(heap,(0,0,root,(0,)*p.width,np.array(p.contract.input_embedding)))
    archive[archive_key(p,root)]=[resources(root,(0,)*p.width)];records=1
    witness=None;reason='exhausted';seq=1
    while heap and not meter.reason():
        _,_,s,td,iso=heapq.heappop(heap)
        rs=resources(s,td)
        if rs not in archive.get(archive_key(p,s),[]):continue
        if matrix_error(p,iso)<=p.tolerance:
            witness=certify_native(p,s,provenance='uniform_cost_native');reason='certified';break
        for g in p.actions:
            if meter.reason():break
            if not legal(p,s,td,g):continue
            meter.edges+=1
            child=s.apply(g,partial_order_reduction=False);ctd=next_t_depths(td,g)
            key=archive_key(p,child);r=resources(child,ctd);group=archive.get(key,[])
            if any(all(a<=b for a,b in zip(old,r)) for old in group):continue
            if records>=limits.max_records:reason='record_limit';heap=[];break
            archive[key]=[old for old in group if not all(a<=b for a,b in zip(r,old))]+[r]
            ci=apply_gate_to_isometry(iso,g)
            heapq.heappush(heap,(child.gate_count,seq,child,ctd,ci));seq+=1;records+=1
            peak=max(peak,len(heap));iso_bytes=max(iso_bytes,len(heap)*ci.nbytes)
    return {'status':'feasible' if witness else 'unknown','reason':reason if witness else meter.reason() or reason,
            'witness':witness,'edges':meter.edges,'records':records,'wall_seconds':meter.wall,'cpu_seconds':meter.cpu,
            'profile':{'peak_frontier':peak,'isometry_bytes_lower_bound':iso_bytes,'learned_scores':0}}


def _fingerprint(u):
    # Retrieval, NOT a key for exact pruning or an infeasibility claim.
    a=np.round(u.real,10);b=np.round(u.imag,10);a[a==0]=0.;b[b==0]=0.
    return a.tobytes()+b.tobytes()


def meet_in_middle(p,limits):
    """Cold-table native MITM control using only the logical target matrix.

    It constructs ancilla-free words and embeds them into available workspace.
    Every generated half-edge and candidate replay is charged; table building
    is timed per call (no hidden pretraining). Rounded lookups only nominate
    candidates. No completeness or optimality is inferred from a miss.
    """
    from .certify import unitary_from_gates
    meter=WorkMeter(limits);n=len(p.contract.logical_qubits);gates=generate_gates(n)
    # All half-prefixes up to four gates, with inverse pairing for full circuits.
    half=min(4,(p.budget.max_gates+1)//2);entries=[((),np.eye(1<<n,dtype=complex),0,0)]
    table={_fingerprint(entries[0][1]):[0]};head=0;witness=None;reason='half_depth_limit'
    peak=1
    while head<len(entries) and not meter.reason():
        word,u,tc,cx=entries[head];head+=1
        # If Utarget = B A, then A = B^dagger Utarget.
        needed=u.conj().T@p.unitary
        joins=[(index,False) for index in table.get(_fingerprint(needed),[])]
        joins += [(index,True) for index in table.get(_fingerprint(p.unitary@u.conj().T),[])]
        for index,reverse in joins:
            aword=entries[index][0];full=word+aword if reverse else aword+word
            if len(full)>p.budget.max_gates:continue
            native=[Gate(g.name,tuple(p.contract.logical_qubits[q] for q in g.qubits)) for g in full]
            s=HybridState.identity(p.width,p.budget);td=(0,)*p.width;valid=True
            for g in native:
                if meter.reason() or not legal(p,s,td,g):valid=False;break
                meter.edges+=1;s=s.apply(g,partial_order_reduction=False);td=next_t_depths(td,g)
            if valid:
                cert=certify_native(p,s,provenance='cold_mitm_native')
                if cert['success']:witness=cert;reason='certified';break
        if witness:break
        if len(word)>=half:continue
        for g in gates:
            if meter.reason():break
            nt,nc=tc+int(g.is_non_clifford),cx+int(g.is_two_qubit)
            if nt>p.budget.max_t_count or nc>p.budget.max_cnot_count:continue
            if len(entries)>=limits.max_records:reason='record_limit';break
            meter.edges+=1
            cu=apply_gate_to_isometry(u,g);cw=word+(g,);key=_fingerprint(cu)
            # Keep different resource/depth witnesses; duplicate lookup entries
            # are not asserted to be continuation-equivalent.
            entries.append((cw,cu,nt,nc));table.setdefault(key,[]).append(len(entries)-1)
        peak=max(peak,len(entries))
        if reason=='record_limit':break
    return {'status':'feasible' if witness else 'unknown','reason':reason if witness else meter.reason() or reason,
            'witness':witness,'edges':meter.edges,'records':len(entries),'wall_seconds':meter.wall,'cpu_seconds':meter.cpu,
            'profile':{'peak_frontier':peak,'learned_scores':0,'cold_table':True,'half_depth':half,
                       'rounded_lookup_only':True,'excluded_domain':False}}


def run_discovery(p,model,method,*,axis,budget,record_limit=30000):
    if axis=='edges':limits=WorkLimits(int(budget),record_limit,30.,30.)
    else:limits=WorkLimits(10_000_000,record_limit,float(budget),30.)
    start=time.perf_counter()
    if method=='uniform_cost':raw=uniform_cost(p,limits)
    elif method=='mitm':raw=meet_in_middle(p,limits)
    else:
        engine=StudySearch(p,limits,axis=axis)
        raw=engine.run(model,scheduler=method)
        raw['profile']['isometry_bytes_lower_bound']=sum(r.isometry.nbytes for r in engine.records.values())
    elapsed=time.perf_counter()-start
    return summarize_result(p,raw,elapsed,axis,budget)


def run_anytime(p,model,method,*,axis,budget,record_limit=30000):
    """Discovery-only gate-cap tightening with one cumulative budget.

    No auditor or constructive reference can enter this function. Every method
    starts from identity under the same original contract, and cold MITM tables
    are regenerated and charged at each tighter trial.
    """
    start,cpu=time.perf_counter(),time.process_time();current=p;edges=records=0
    incumbent=None;trace=[];rounds=[];profile={};halt='budget';first=None
    while True:
        elapsed=time.perf_counter()-start
        remaining=budget-edges if axis=='edges' else budget-elapsed
        if remaining<=0 or records>=record_limit:break
        r=run_discovery(current,model,method,axis=axis,budget=remaining,record_limit=record_limit-records)
        edges+=r['edges'];records+=r['records']
        for k,v in r['profile'].items():
            if type(v) in (int,float):
                profile[k]=max(profile.get(k,0),v) if k.startswith(('peak','max','isometry_bytes')) else profile.get(k,0)+v
        rounds.append({'problem_digest':current.digest,'gate_cap':current.budget.max_gates,
                       'edges':r['edges'],'records':r['records'],'wall_seconds':r['external_wall_seconds'],
                       'reason':r['reason'],'late_witness':r['late_witness'],
                       'late_certificate':r['witness'] if r['late_witness'] else None})
        elapsed=time.perf_counter()-start
        timely=r['witness'] is not None and (axis!='wall_seconds' or elapsed<=budget)
        if r['witness'] is not None and not timely:
            rounds[-1]['late_witness']=True
            rounds[-1]['late_certificate']=r['witness']
        if not timely:
            halt=r['reason'] if r['witness'] is None else 'late_witness';break
        incumbent=r['witness'];cost=incumbent['resources']['gates']
        if first is None:first=elapsed
        trace.append({'wall_seconds':elapsed,'edges':edges,'resources':incumbent['resources'],
                      'source':incumbent['source'],'native':incumbent['native']})
        if cost==0:halt='zero_gate_bound';break
        current=current.cap('gates',cost-1)
    return {'witness':incumbent,'timely_success':incumbent is not None,'status':'upper_bound' if incumbent else 'unknown',
            'reason':halt,'external_wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'edges':edges,'records':records,'profile':profile,'rounds':rounds,'incumbent_trace':trace,
            'time_to_first':first,'strict_improvements':max(0,len(trace)-1),'audit_witness_used':False,
            'objective':'gates','late_witness':any(r['late_witness'] for r in rounds),
            'uses_references':False,'limits_are_cooperative':True}
