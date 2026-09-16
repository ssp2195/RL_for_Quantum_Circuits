"""Versioned linear SARSA / disjoint LinUCB models for the protocol planner.

The original NativeHierarchy is unchanged. Here actions are certified protocol
macros, not primitive native gates or measurement outcomes. Updates use the
actual next selected frontier context. Gamma=1, and one allocation is one
classical decision; physical gates and measured outcomes have separate costs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .contract import digest
from .plan import FAMILIES
from .ir import local_word

SCHEMA = 'cleanup-sarsa-linucb-v1'
OUTER_NAMES = ('bias','progress','remaining','t_used','cnot_used','gates_used','depth_used',
               'helper_fraction','aux_fraction','ready_fraction','pending_fraction','r_fraction',
               'm_fraction','bank_fraction','helper_done','product_done','consumer_done',
               'cleanup_stage','work_remaining','progress_work')
INNER_NAMES = ('bias','parent_progress','child_progress','progress_gain','native_cost',
               't_cost','cnot_cost','operand_depth','helper_fraction','aux_fraction',
               'r_fraction','m_fraction','work_remaining','ready_fraction',
               'layout','helper','product','consumer','bank_cleanup','helper_cleanup',
               'remaining_products','remaining_consumers','cleanup_mode','depth_progress')


def outer_context(p, record, work_left=1.):
    plan, r = record.plan, record.resource
    k = min(p.r,p.m) if plan.side is None else plan.layout(p).k
    den = p.r*p.m+max(p.r,p.m)
    ready = len(record.pending); total = k+p.r*p.m+len(p.consumer.terms)+3
    prog = plan.progress(p)
    return np.array([1.,prog,1-prog,r.t_count/(8*den+1),r.cnot/(8*den+1),r.gates/(32*den+1),
                     max(r.depths,default=0)/(24*den+1),k/max(p.r,p.m),r.peak_aux/(den+1),
                     ready/total,ready/total,p.r/(p.r+p.m),p.m/(p.r+p.m),p.r*p.m/(den+1),
                     plan.helpers.bit_count()/max(1,k),plan.products.bit_count()/(p.r*p.m),
                     plan.consumers.bit_count()/max(1,len(p.consumer.terms)),plan.cleanup_stage/2,
                     work_left,prog*work_left],float)


def inner_context(p, record, action, primitive, mode, work_left=1.):
    plan, r = record.plan, record.resource
    k = (p.m if action.index else p.r) if action.family == 'layout' else plan.layout(p).k
    den = p.r*p.m+max(p.r,p.m)
    count = 0; operand_depth = 0
    if action.family in ('helper','product'):
        count = 1
        operands = plan.layout(p).helper_operands(action.index) if action.family == 'helper' else plan.layout(p).product_operands(action.index)
        operand_depth = max(r.depths[q] for q in operands)
    elif action.family in ('bank_cleanup','helper_cleanup'):
        count = p.r*p.m if action.family == 'bank_cleanup' else k
    word = local_word('AND',primitive)
    if action.family in ('helper','product'):
        native = len(word); t = sum(n in ('T','TDG') for n,_ in word); cnot = sum(n=='CNOT' for n,_ in word)
    elif action.family in ('bank_cleanup','helper_cleanup'):
        native = count*(8 if mode == 'measured' else len(word))
        t = 0 if mode == 'measured' else count*sum(n in ('T','TDG') for n,_ in word)
        cnot = count if mode == 'measured' else count*sum(n=='CNOT' for n,_ in word)
    elif action.family == 'consumer':
        native = 2 if len(p.consumer.terms[action.index]) == 1 else 3 if p.consumer.terms[action.index] else 12
        t = 0; cnot = int(len(p.consumer.terms[action.index]) == 2)
    else: native = t = cnot = 0
    prog = plan.progress(p)
    increment = 1/(k+p.r*p.m+len(p.consumer.terms)+3)
    ready = len(record.pending)/(k+p.r*p.m+len(p.consumer.terms)+3)
    return np.array([1,prog,prog+increment,increment,native/(32*den+1),t/(8*den+1),
                     cnot/(8*den+1),operand_depth/(24*den+1),k/max(p.r,p.m),(p.r*p.m+k)/(den+1),
                     p.r/(p.r+p.m),p.m/(p.r+p.m),work_left,ready,
                     *[float(action.family == f) for f in FAMILIES],
                     1-plan.products.bit_count()/(p.r*p.m),
                     1-plan.consumers.bit_count()/max(1,len(p.consumer.terms)),
                     float(mode == 'measured'),prog*operand_depth/(24*den+1)],float)


class CleanupHierarchy:
    def __init__(self, seed=0, alpha=.015, ucb=.15):
        if type(seed) is not int or not 0 < alpha <= 1 or not np.isfinite(ucb) or ucb < 0:
            raise ValueError('invalid linear policy parameters')
        self.seed, self.alpha, self.ucb = seed, float(alpha), float(ucb)
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(len(OUTER_NAMES)); self.w[1]=4.; self.w[2]=-1.; self.w[7]=-.4; self.w[6]=-.1
        self.prior = np.zeros(len(INNER_NAMES)); self.prior[3]=1.; self.prior[7]=-.2; self.prior[8]=-.4; self.prior[4]=-.02
        d=len(INNER_NAMES); self.a=np.array([np.eye(d) for _ in FAMILIES]); self.ainv=self.a.copy()
        self.b=np.zeros((len(FAMILIES),d));self.theta=self.b.copy()
        self.updates=np.zeros(len(FAMILIES),int);self.outer_updates=0;self.episodes=0;self.frozen=False
        self.training_contracts=[]; self.stage_episodes={'outer':0,'inner':0}

    def score_outer(self, x): return np.asarray(x)@self.w

    def score_inner(self, x, family, explore=False):
        i=FAMILIES.index(family); score=float(x@(self.prior+self.theta[i]))
        if explore: score+=self.ucb*np.sqrt(max(0.,float(x@self.ainv[i]@x)))
        return score

    def update_outer(self, x, reward, next_x=None):
        if self.frozen: raise ValueError('frozen policy cannot train')
        delta=reward+(0 if next_x is None else float(next_x@self.w))-float(x@self.w)
        self.w=np.clip(self.w+self.alpha*delta*x/max(1.,float(x@x)),-20.,20.)
        self.outer_updates+=1
        return float(delta)

    def update_inner(self, x, family, response):
        if self.frozen: raise ValueError('frozen policy cannot train')
        i=FAMILIES.index(family)
        self.a[i]+=np.outer(x,x); self.b[i]+=(response-float(x@self.prior))*x
        u=self.ainv[i]@x; self.ainv[i]-=np.outer(u,u)/(1+float(x@u))
        self.updates[i]+=1
        if self.updates[i]%64==0: self.ainv[i]=np.linalg.inv(self.a[i])
        self.theta[i]=self.ainv[i]@self.b[i]

    def payload(self):
        return {'schema':SCHEMA,'seed':self.seed,'alpha':self.alpha,'ucb':self.ucb,
                'outer_names':list(OUTER_NAMES),'inner_names':list(INNER_NAMES),'families':list(FAMILIES),
                'w':self.w.tolist(),'prior':self.prior.tolist(),'a':self.a.tolist(),'b':self.b.tolist(),
                'outer_updates':self.outer_updates,'updates':self.updates.tolist(),
                'episodes':self.episodes,'stage_episodes':self.stage_episodes,
                'training_contracts':self.training_contracts,'frozen':self.frozen}

    @property
    def digest(self): return digest(self.payload())

    def freeze(self):
        for i in range(len(FAMILIES)):
            self.ainv[i]=np.linalg.inv(self.a[i]);self.theta[i]=self.ainv[i]@self.b[i]
        self.frozen=True
        for value in (self.w,self.prior,self.a,self.b,self.ainv,self.theta,self.updates): value.setflags(write=False)
        return self

    def save(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(self.payload(),indent=2)+'\n')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if (d.get('schema')!=SCHEMA or d.get('outer_names')!=list(OUTER_NAMES) or
            d.get('inner_names')!=list(INNER_NAMES) or d.get('families')!=list(FAMILIES)):
            raise ValueError('checkpoint does not belong to the cleanup environment')
        obj=cls(d['seed'],d['alpha'],d['ucb'])
        for name in ('w','prior','a','b','updates'):
            arr=np.array(d[name],int if name=='updates' else float)
            if arr.shape!=getattr(obj,name).shape or not np.isfinite(arr).all(): raise ValueError('invalid coefficient shape/value')
            setattr(obj,name,arr)
        for a in obj.a:
            if not np.allclose(a,a.T) or np.linalg.eigvalsh(a).min()<=0: raise ValueError('invalid LinUCB covariance')
        if any(type(d[k]) is not int or d[k]<0 for k in ('episodes','outer_updates')): raise ValueError('invalid provenance')
        if any(type(d['stage_episodes'].get(k)) is not int or d['stage_episodes'][k]<0 for k in ('outer','inner')): raise ValueError('invalid staged training')
        obj.episodes=d['episodes'];obj.outer_updates=d['outer_updates'];obj.stage_episodes=d['stage_episodes']
        obj.training_contracts=d['training_contracts']
        if type(d['frozen']) is not bool or np.any(obj.updates<0): raise ValueError('invalid checkpoint state')
        for i in range(len(FAMILIES)):
            obj.ainv[i]=np.linalg.inv(obj.a[i]);obj.theta[i]=obj.ainv[i]@obj.b[i]
        return obj.freeze() if d['frozen'] else obj
