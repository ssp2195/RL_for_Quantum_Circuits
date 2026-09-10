"""Fair native hybrid frontier; SARSA selects records, LinUCB orders native gates.

The full frontier is retained. A panel bounds only scoring. Native gate
continuations are never filtered by target phase-polynomial obligations or a
hidden reference circuit. Numerical caches do not authorize semantic pruning.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
import time
import numpy as np
from .model import HybridState
from .native_domain import (SCHEMA, archive_key, resources, legal, next_t_depths,
                            matrix_error, certify_native, apply_gate_to_isometry)
from .native_policy import NativeHierarchy, outer_features, inner_features
from .phase_index import CandidateIndex  # generic indexed heap, no phase semantics
from .resource_search import WorkLimits, WorkMeter


@dataclass(slots=True)
class NativeRecord:
    record_id: int
    state: HybridState
    isometry: np.ndarray
    t_depths: tuple[int, ...]
    pending: int
    x: np.ndarray
    parent_id: int | None = None
    token: int | None = None
    contexts: dict = field(default_factory=dict)


class NativeSearch:
    def __init__(self, problem, limits=WorkLimits(), *, panel_size=32, fairness=32, cancel=None):
        if type(panel_size) is not int or panel_size < 1 or type(fairness) is not int or fairness < 1:
            raise ValueError('panel and fairness periods must be positive integers')
        self.p, self.limits = problem, limits
        self.panel_size, self.fairness = panel_size, fairness
        self.meter = WorkMeter(limits, cancel)
        self.records={};self.frontier=OrderedDict();self.archive={};self.index=CandidateIndex()
        self.halt=None;self.solution=None;self.allocations=[]
        self.profile={'hybrid_transitions':0,'lookahead_isometry_updates':0,'selection_seconds':0.,
                      'transition_seconds':0.,'certification_seconds':0.,'peak_frontier':0,
                      'max_scored_panel':0,'fairness_steps':0,'dominance_comparisons':0,
                      'phase_obligation_transitions':0,'full_dag_policy_reconstructions':0}
        root=HybridState.identity(problem.width,problem.budget)
        self.insert(root,np.array(problem.contract.input_embedding,copy=True),(0,)*problem.width)

    def insert(self,state,iso,td,parent_id=None,token=None):
        if type(state) is not HybridState:
            raise TypeError('every frontier record must contain the full HybridState')
        key=archive_key(self.p,state);rs=resources(state,td)
        group=self.archive.get(key,[])
        for rid in group:
            self.profile['dominance_comparisons']+=1
            if all(a<=b for a,b in zip(resources(self.records[rid].state,self.records[rid].t_depths),rs)):
                return None
        if len(self.records)>=self.limits.max_records:
            self.halt='record_limit';return None
        survivors=[]
        for rid in group:
            old=self.records[rid]
            if all(a<=b for a,b in zip(rs,resources(old.state,old.t_depths))):
                self.frontier.pop(rid,None);self.index.discard(rid)
            else:
                survivors.append(rid)
        pending=sum(1<<t for t,g in enumerate(self.p.actions) if legal(self.p,state,td,g))
        rid=len(self.records)
        x=outer_features(self.p,state,iso,pending.bit_count()/len(self.p.actions))
        record=NativeRecord(rid,state,iso,td,pending,x,parent_id,token)
        self.records[rid]=record;self.archive[key]=survivors+[rid]
        if pending:
            self.frontier[rid]=record;self.index.add(float(x[1]+.02*x[4]),rid)
        self.profile['peak_frontier']=max(self.profile['peak_frontier'],len(self.frontier))
        # Target equality need not share an incomplete symbolic normal form.
        if matrix_error(self.p,iso)<=self.p.tolerance:
            start=time.perf_counter()
            cert=certify_native(self.p,state)
            self.profile['certification_seconds']+=time.perf_counter()-start
            if not cert['success']:
                raise AssertionError('candidate cache disagrees with independent native certification')
            self.solution=cert
        return record

    def potential(self):
        return -self.index.minimum()[0] if self.index.minimum() is not None else 0.

    def tokens(self,record):
        return tuple(t for t in range(len(self.p.actions)) if record.pending & (1<<t))

    def step(self,rid,token):
        record=self.frontier[rid]
        if not record.pending & (1<<token):
            raise ValueError('continuation is not pending')
        gate=self.p.actions[token]
        record.pending &= ~(1<<token)
        record.contexts.pop(token,None)
        if not record.pending:
            self.frontier.pop(rid);self.index.discard(rid)
        start=time.perf_counter()
        child=record.state.apply(gate,partial_order_reduction=False)
        if child is None:
            raise AssertionError('native legal mask and deterministic transition disagree')
        td=next_t_depths(record.t_depths,gate)
        iso=apply_gate_to_isometry(record.isometry,gate)
        self.profile['hybrid_transitions']+=1;self.meter.edges+=1
        self.profile['transition_seconds']+=time.perf_counter()-start
        result=self.insert(child,iso,td,rid,token)
        self.allocations.append([rid,token])
        return result

    def context(self,record):
        """The same decision-time features for ranking and the SARSA update."""
        x=record.x.copy()
        x[-1]*=self.remaining_fraction()
        x[15]=record.pending.bit_count()/len(self.p.actions)
        return x

    def select(self,model,scheduler,train,epsilon):
        start=time.perf_counter()
        try:
            if self.meter.reason():
                return None
            forced=(self.meter.edges+1)%self.fairness==0
            if forced:
                record=next(iter(self.frontier.values()));token=self.tokens(record)[0]
                self.profile['fairness_steps']+=1
            else:
                panel=[self.records[rid] for rid in self.index.smallest(self.panel_size)]
                if scheduler=='cost':
                    record=min(panel,key=lambda r:(r.state.t_count,r.state.cnot_count,r.state.gate_count,r.record_id))
                elif scheduler in ('greedy','inner'):
                    record=panel[0]
                elif train=='outer' and model.rng.random()<epsilon:
                    record=panel[int(model.rng.integers(len(panel)))]
                else:
                    xs=np.array([self.context(r) for r in panel])
                    scores=model.score_outer(xs)
                    record=panel[int(np.argmax(scores))]
                    self.profile['max_scored_panel']=max(self.profile['max_scored_panel'],len(panel))
                candidates=self.tokens(record);scores=[]
                for token in candidates:
                    if self.meter.reason():
                        return None
                    if token not in record.contexts:
                        gate=self.p.actions[token]
                        projected=apply_gate_to_isometry(record.isometry,gate)
                        record.contexts[token]=inner_features(self.p,record,gate,projected)
                        self.profile['lookahead_isometry_updates']+=1
                    x=record.contexts[token]
                    score=float(x@model.prior) if scheduler in ('greedy','outer','cost') else model.score_inner(x,self.p.actions[token].name,train=='inner')
                    scores.append(score)
                token=candidates[int(np.argmax(scores))]
            if token not in record.contexts:
                gate=self.p.actions[token]
                record.contexts[token]=inner_features(self.p,record,gate,apply_gate_to_isometry(record.isometry,gate))
                self.profile['lookahead_isometry_updates']+=1
            return record,token,self.context(record),record.contexts[token]
        finally:
            self.profile['selection_seconds']+=time.perf_counter()-start

    def remaining_fraction(self):
        return max(0.,1-self.meter.edges/max(1,self.limits.max_edges))

    def run(self,model=None,*,scheduler='hierarchy',train=None,epsilon=.15):
        if scheduler not in ('hierarchy','untrained','greedy','outer','inner','cost') or train not in (None,'outer','inner') or not 0<=epsilon<=1:
            raise ValueError('invalid scheduler or training configuration')
        if model is None:
            model=NativeHierarchy().freeze()
            if scheduler=='hierarchy':
                raise ValueError('hierarchy requires an explicitly supplied trained frozen checkpoint')
        if scheduler in ('untrained','greedy','cost'):
            if train is not None:
                raise ValueError('baseline cannot train')
            model=NativeHierarchy(model.seed).freeze()
        if (train is None and not model.frozen) or (train is not None and model.frozen):
            raise ValueError('training/evaluation checkpoint state mismatch')
        if train is None and scheduler in ('hierarchy','outer','inner') and model.episodes<1:
            raise ValueError('trained-policy evaluation requires completed training')
        before_digest=model.digest;training=[]
        choice=None if self.solution or not self.frontier else self.select(model,scheduler,train,epsilon)
        while choice is not None and not self.solution and not self.halt:
            record,token,x,ix=choice
            before=self.potential()
            child=self.step(record.record_id,token)
            ended=bool(self.solution or self.halt or self.meter.reason() or not self.frontier)
            nxt=None if ended else self.select(model,scheduler,train,epsilon)
            ended=ended or nxt is None
            after=0. if ended else self.potential()
            base=float(self.solution is not None)-.002
            reward=base+after-before
            if train=='outer':
                model.update_outer(x,reward,None if ended else nxt[2])
            elif train=='inner':
                # Frozen outer baseline supplies a one-step TD-like response.
                # It never supplies pruning or certification decisions.
                child_value=0. if ended else float(model.score_outer(child.x if child else record.x))
                response=base+child_value-float(model.score_outer(x))
                model.update_inner(ix,self.p.actions[token].name,response)
            if train:
                training.append({'rid':record.record_id,'token':token,'family':self.p.actions[token].name,
                                 'reward':reward,'base_reward':base,'potential_before':before,
                                 'potential_after':after,'terminal':ended})
            choice=nxt
        reason='certified' if self.solution else self.halt or self.meter.reason() or 'exhausted_without_independent_proof'
        if train:
            model.episodes+=1
        elif model.digest!=before_digest:
            raise AssertionError('evaluation mutated the frozen model')
        return {'schema':SCHEMA,'problem':self.p.manifest(),'name':self.p.name,'family':self.p.family,
                'problem_digest':self.p.digest,'scheduler':scheduler,'status':'feasible' if self.solution else 'unknown',
                'reason':reason,'witness':self.solution,'witness_source':'native_frontier_discovery' if self.solution else None,
                'optimality':'not_established','policy_digest':before_digest,'policy_frozen':train is None,
                'edges':self.meter.edges,'records':len(self.records),'wall_seconds':self.meter.wall,
                'cpu_seconds':self.meter.cpu,'profile':self.profile,'allocations':self.allocations,
                'training_transitions':training,'audit_witness_used':False}


def optimize_native(problem,model,*,objective='cnot',limits=WorkLimits(),scheduler='hierarchy',cancel=None,**kwargs):
    """Compatibility entry point for proof-carrying native resource minimization.

    Pass objectives=(...) for lexicographic optimization. Proofs are native
    closed covers; historical phase-polynomial certificates are not accepted.
    """
    from .native_optimize import optimize_native_resources
    order=kwargs.pop('objectives',(objective,))
    return optimize_native_resources(problem,model,objectives=order,limits=limits,
                                     scheduler=scheduler,cancel=cancel,**kwargs)
