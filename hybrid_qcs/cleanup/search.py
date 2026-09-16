"""Fair, persistent frontier search over certified cleanup protocol continuations.

The native unitary search and its existing optimality certificates are untouched.
This planner has a DIFFERENT explicit grammar. Its constructive witnesses carry
local HybridStates; instrument correctness is established by cleanup.verify.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
import time
import numpy as np

from ..phase_index import CandidateIndex
from ..resource_search import WorkLimits, WorkMeter
from .contract import CleanupProblem, THEOREM, ASSUMPTIONS
from .ir import Protocol, ProtocolEvent, ResourceState, append_event, instructions
from .plan import Plan, Action, eligible, transition
from .policy import CleanupHierarchy, outer_context, inner_context
from .verify import verify_protocol, workspace_certificate, verify_workspace_certificate


@dataclass(slots=True)
class CleanupRecord:
    record_id: int
    plan: Plan
    resource: ResourceState
    tail: ProtocolEvent | None
    pending: tuple[Action, ...]
    parent_id: int | None = None
    action: Action | None = None
    contexts: dict = field(default_factory=dict)


def _resource_order(r: ResourceState) -> tuple:
    return (r.peak_aux,r.t_count,r.cnot,r.gates,r.expected_twice,*r.depths,*r.tdepths,*r.ticks,
            *[t for _,t in r.classical])


class CleanupSearch:
    def __init__(self, problem: CleanupProblem, limits=WorkLimits(4096,20000,10.,10.), *,
                 sides=('row','column'), primitive='and4', mode='measured', panel_size=32,
                 fairness=32, cancel=None):
        if not isinstance(problem,CleanupProblem): raise TypeError('cleanup problem required')
        if type(panel_size) is not int or panel_size<1 or type(fairness) is not int or fairness<1:
            raise ValueError('positive panel/fairness required')
        if primitive not in ('and4','ccx7') or mode not in ('measured','coherent'):
            raise ValueError('invalid protocol grammar')
        if not sides or len(set(sides))!=len(sides) or any(s not in ('row','column') for s in sides):
            raise ValueError('invalid helper-layout domain')
        self.p,self.limits,self.sides,self.primitive,self.mode=problem,limits,tuple(sides),primitive,mode
        self.panel,self.fairness=panel_size,fairness;self.meter=WorkMeter(limits,cancel)
        self.records={};self.frontier=OrderedDict();self.archive={};self.index=CandidateIndex()
        self.solution=None;self.verification=None;self.halt=None
        self.profile={'selection_seconds':0.,'transition_seconds':0.,'verification_seconds':0.,
                      'peak_frontier':0,'fairness_steps':0,'max_scored_panel':0,
                      'emitted_native_gate_work':0,'protocol_instructions':0,'dominated_records':0,
                      'global_dense_arrays_allocated':False}
        self.allocations=[]
        self.insert(Plan(),ResourceState.zero(problem.n),None)

    def insert(self,plan,resource,tail,parent_id=None,action=None):
        if not self.p.limits.accepts(resource.report()): return None
        # Plan equality fixes all live-value identities, internal correction
        # obligations and remaining consumers. It is NOT ideal-unitary equality
        # alone. Costs include incoming per-wire and classical readiness.
        key=(plan,resource.stage,tuple(b for b,_ in resource.classical),len(resource.depths))
        group=self.archive.get(key,[]); new=_resource_order(resource)
        for rid in group:
            old=_resource_order(self.records[rid].resource)
            if len(old)==len(new) and all(a<=b for a,b in zip(old,new)):
                self.profile['dominated_records']+=1;return None
        if len(self.records)>=self.limits.max_records:
            self.halt='record_limit';return None
        survivors=[]
        for rid in group:
            old=_resource_order(self.records[rid].resource)
            if len(old)==len(new) and all(a<=b for a,b in zip(new,old)):
                self.frontier.pop(rid,None);self.index.discard(rid)
            else:survivors.append(rid)
        rid=len(self.records);pending=eligible(self.p,plan,self.sides)
        record=CleanupRecord(rid,plan,resource,tail,pending,parent_id,action)
        self.records[rid]=record;self.archive[key]=survivors+[rid]
        if pending:
            self.frontier[rid]=record
            k=min(self.p.r,self.p.m) if plan.side is None else plan.layout(self.p).k
            self.index.add(-4*plan.progress(self.p)+.1*k/max(self.p.r,self.p.m),rid)
        self.profile['peak_frontier']=max(self.profile['peak_frontier'],len(self.frontier))
        if plan.terminal:
            protocol=Protocol(self.p.digest,plan.layout(self.p),self.primitive,self.mode,instructions(tail))
            start=time.perf_counter();checked=verify_protocol(self.p,protocol,cancel=self.meter.cancel)
            self.profile['verification_seconds']+=time.perf_counter()-start
            if not checked['valid']:
                if checked.get('reason')=='cancelled': self.halt='cancelled';return record
                raise AssertionError('generated protocol failed exact independent verification: '+checked['reason'])
            if checked['resources']!=resource.report():raise AssertionError('incremental and replay resource ledgers differ')
            self.solution=protocol;self.verification=checked
        return record

    def work_fraction(self):return max(0.,1-self.meter.edges/max(1,self.limits.max_edges))

    def potential(self):
        pair=self.index.minimum()
        return 0. if pair is None else self.records[pair[1]].plan.progress(self.p)-1.

    def select(self,model,scheduler,train,epsilon):
        if self.meter.reason() or not self.frontier:return None
        start=time.perf_counter()
        try:
            forced=(self.meter.edges+1)%self.fairness==0
            if forced:
                record=next(iter(self.frontier.values()));action=record.pending[0]
                self.profile['fairness_steps']+=1
            else:
                panel=[self.records[i] for i in self.index.smallest(self.panel)]
                self.profile['max_scored_panel']=max(self.profile['max_scored_panel'],len(panel))
                if scheduler=='cost':record=min(panel,key=lambda r:(r.resource.gates,r.record_id))
                elif train=='outer' and model.rng.random()<epsilon:
                    record=panel[int(model.rng.integers(len(panel)))]
                else:
                    score=np.array([outer_context(self.p,r,self.work_fraction()) for r in panel])@model.w
                    record=panel[int(np.argmax(score))]
                xs=[inner_context(self.p,record,a,self.primitive,self.mode,self.work_fraction()) for a in record.pending]
                scores=[model.score_inner(x,a.family,train=='inner') if scheduler not in ('outer','cost')
                        else float(x@model.prior) for a,x in zip(record.pending,xs)]
                if train=='inner' and model.rng.random()<epsilon:
                    pos=int(model.rng.integers(len(xs)))
                else:pos=int(np.argmax(scores))
                action=record.pending[pos]
            return (record,action,outer_context(self.p,record,self.work_fraction()),
                    inner_context(self.p,record,action,self.primitive,self.mode,self.work_fraction()))
        finally:self.profile['selection_seconds']+=time.perf_counter()-start

    def step(self,rid,action):
        record=self.frontier[rid]
        if action not in record.pending:raise ValueError('action is not pending')
        record.pending=tuple(a for a in record.pending if a!=action)
        if not record.pending:self.frontier.pop(rid);self.index.discard(rid)
        start=time.perf_counter()
        plan,ops=transition(self.p,record.plan,action,mode=self.mode,sides=self.sides)
        resource=ResourceState.zero(plan.layout(self.p).width) if action.family=='layout' else record.resource
        tail=record.tail
        for op in ops:
            if self.meter.cancel is not None and self.meter.cancel():
                self.halt='cancelled';return None
            resource=resource.append(op,self.primitive,self.p)
            tail=append_event(tail,op,self.primitive)
        self.meter.edges+=1
        self.profile['emitted_native_gate_work']+=resource.gates-record.resource.gates
        self.profile['protocol_instructions']+=len(ops)
        self.profile['transition_seconds']+=time.perf_counter()-start
        child=self.insert(plan,resource,tail,rid,action)
        self.allocations.append([rid,action.payload()])
        return child

    def run(self,model=None,*,scheduler='hierarchy',train=None,epsilon=.15):
        if scheduler not in ('hierarchy','untrained','outer','cost') or train not in (None,'outer','inner'):
            raise ValueError('invalid scheduler/training mode')
        if not 0<=epsilon<=1:raise ValueError('invalid exploration probability')
        if scheduler in ('untrained','cost'):
            if train is not None:raise ValueError('untrained control cannot train')
            model=CleanupHierarchy(0).freeze()
        if model is None:raise ValueError('explicit completed cleanup checkpoint required')
        if type(model) is not CleanupHierarchy:raise TypeError('native/phase checkpoints cannot be used in cleanup environment')
        if train is None and not model.frozen or train is not None and model.frozen:
            raise ValueError('training/evaluation checkpoint mismatch')
        if train is None and scheduler in ('hierarchy','outer') and (model.stage_episodes['inner']<1 or model.stage_episodes['outer']<1):
            raise ValueError('both training stages must complete before generation testing')
        before=model.digest;training=[]
        choice=self.select(model,scheduler,train,epsilon)
        while choice is not None and not self.solution and not self.halt:
            record,action,x,ix=choice
            phi0=self.potential(); child=self.step(record.record_id,action)
            ended=bool(self.solution or self.halt or self.meter.reason() or not self.frontier)
            nxt=None if ended else self.select(model,scheduler,train,epsilon)
            ended=ended or nxt is None
            phi1=0. if ended else self.potential()
            quality=0.
            if self.solution:
                r=self.verification['resources'];den=self.p.r*self.p.m+max(self.p.r,self.p.m)
                quality=max(.05,1-.5*r['peak_aux']/(den+1)-.5*r['t_count']/(8*den+1))
            reward=quality-.002+phi1-phi0
            response=None
            if train=='outer':model.update_outer(x,reward,None if ended else nxt[2])
            elif train=='inner':
                response=reward+(0. if ended else float(model.score_outer(nxt[2])))-float(model.score_outer(x))
                model.update_inner(ix,action.family,response)
            if train:
                training.append({'record':record.record_id,'action':action.payload(),'next_record':None if ended else nxt[0].record_id,
                                 'reward':reward,'quality':quality,'potential_before':phi0,'potential_after':phi1,
                                 'terminal':ended,'inner_response':response,'outer_context':x.tolist(),
                                 'next_outer_context':None if ended else nxt[2].tolist()})
            choice=nxt
        if train:
            model.episodes+=1;model.stage_episodes[train]+=1
            if self.p.digest not in model.training_contracts:model.training_contracts.append(self.p.digest)
        elif model.digest!=before:raise AssertionError('evaluation mutated frozen checkpoint')
        late=bool(self.solution and self.meter.reason())
        return {'schema':'cleanup-search-v1','problem':self.p.manifest(),'scheduler':scheduler,
                'status':'feasible' if self.solution else 'unknown','reason':'certified' if self.solution else self.halt or self.meter.reason() or 'exhausted_without_exclusion',
                'timely':bool(self.solution and not late),'policy_digest':before,'policy_frozen':train is None,
                'protocol':None if self.solution is None else self.solution.payload(),'verification':self.verification,
                'edges':self.meter.edges,'records':len(self.records),'wall_seconds':self.meter.wall,'cpu_seconds':self.meter.cpu,
                'profile':self.profile,'allocations':self.allocations,'training_transitions':training,
                'audit_witness_used':False,'grammar':'bounded certified isometry/consumer/cleanup macro continuations',
                'search_allocation_is_not_native_gate':True}


def optimize_cleanup(problem,model=None,*,scheduler='hierarchy',limits=WorkLimits(4096,20000,10.,10.),
                     sides=('row','column'),primitive='and4',mode='measured',cancel=None):
    """Minimize peak workspace with theorem-scoped lower bound and real witnesses.

    No claim is made about global T/CNOT/depth optima. A coherent inverse is a
    comparison construction, not a member of the theorem's measured architecture.
    """
    clock=WorkMeter(limits,cancel);rounds=[];incumbent=None;checked=None;cert=None;edges=records=0
    current=problem
    bound=problem.workspace_lower_bound
    if mode=='measured' and problem.limits.max_aux is not None and problem.limits.max_aux<bound:
        return {'status':'infeasible_within_theorem_architecture','protocol':None,'rounds':[],
                'lower_bound':bound,'assumptions':list(ASSUMPTIONS),'theorem':THEOREM,
                'global_oracle_optimality':False,'wall_seconds':clock.wall,'edges':0,'records':0}
    while not clock.reason():
        left=WorkLimits(max(0,limits.max_edges-edges),max(1,limits.max_records-records),
                        max(0.,limits.wall_seconds-clock.wall),max(0.,limits.cpu_seconds-clock.cpu))
        result=CleanupSearch(current,left,sides=sides,primitive=primitive,mode=mode,cancel=cancel).run(model,scheduler=scheduler)
        rounds.append(result);edges+=result['edges'];records+=result['records'];clock.edges=edges
        if not result['protocol']:break
        trial=Protocol.from_payload(result['protocol'])
        # Rebind a tighter-cap witness to the caller's original contract and
        # reverify; constraints were strengthened, not silently changed.
        trial=Protocol(problem.digest,trial.layout,trial.primitive,trial.cleanup_mode,trial.ops,trial.provenance)
        checked=verify_protocol(problem,trial,cancel=cancel)
        if not checked['valid']:raise AssertionError('optimized witness cannot be replayed under original contract')
        incumbent=trial
        if mode=='measured':
            cert=workspace_certificate(problem,trial)
            if not verify_workspace_certificate(problem,trial,cert)['valid']:raise AssertionError('invalid theorem receipt')
            if checked['resources']['peak_aux']==bound:break
            current=problem.with_aux_cap(checked['resources']['peak_aux']-1)
        else:break
        if records>=limits.max_records:break
    proved=bool(cert and cert['optimal_within_architecture'])
    return {'status':'optimal_workspace_within_architecture' if proved else 'upper_bound' if incumbent else 'unknown',
            'protocol':None if incumbent is None else incumbent.payload(),'verification':checked,
            'certificate':cert,'lower_bound':bound if mode=='measured' else None,'rounds':rounds,
            'global_oracle_optimality':False,'timely':bool(incumbent and not clock.reason()),
            'wall_seconds':clock.wall,'cpu_seconds':clock.cpu,'edges':edges,'records':records,
            'policy_digest':None if model is None else model.digest,
            'scope':'peak auxiliary minimum in the source theorem architecture; other resources are certified upper bounds'}
