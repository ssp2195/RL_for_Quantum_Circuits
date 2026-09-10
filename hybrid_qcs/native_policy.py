"""Small native-hybrid SARSA / disjoint LinUCB policies. No phase-obligation state."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .native_domain import SCHEMA, distance, digest, next_t_depths

OUTER_NAMES = ('bias', 'distance', 't_used', 'cnot_used', 'gates_used', 'depth_used',
               'rotation_length', 'pauli_weight', 'anticommuting_fraction', 'clifford_mismatch',
               'phase_sin', 'phase_cos', 'workspace_fraction', 'leakage', 'workspace_depth',
               'pending_fraction', 'distance_t_slack', 'distance_cnot_slack', 'distance_depth_slack', 'work_remaining_distance')
INNER_NAMES = ('bias', 'parent_distance', 'child_distance', 'distance_gain', 't_cost', 'cnot_cost',
               'depth_increment', 'workspace_touch', 'workspace_control', 'workspace_target',
               't_slack', 'cnot_slack', 'gate_slack', 'depth_slack', 'operand_depth',
               'rotation_length', 'noncommuting_fraction', 'phase_sin', 'phase_cos',
               'progress_t_slack', 'progress_cnot_slack', 'progress_depth_slack', 't_depth_slack', 'cleanup')
FAMILIES = ('H', 'S', 'SDG', 'T', 'TDG', 'CNOT')


def outer_features(p, state, iso, pending_fraction=1.):
    b = p.budget
    d = distance(p, iso)
    t, c, g, depth = (state.t_count/(b.max_t_count+1), state.cnot_count/(b.max_cnot_count+1),
                      state.gate_count/(b.max_gates+1), state.depth/(b.max_depth+1))
    nrot = len(state.rotations)
    anti = state.anticommuting_pairs / max(1, nrot*(nrot-1)//2)
    ident = type(state.tableau).identity(p.width).canonical_payload()
    mismatch = sum(a != b for a, b in zip(ident, state.tableau.canonical_payload())) / (2*p.width)
    work = p.contract.clean_ancillas
    leakage = float(np.sum(np.abs(iso[p.contract.invalid_clean_output_rows]) ** 2) / p.contract.domain_dimension)
    return np.array([1., d, t, c, g, depth, nrot/(b.max_t_count+1),
                     state.mean_pauli_weight/p.width, anti, mismatch,
                     np.sin(np.pi*state.global_phase_eighths/8), np.cos(np.pi*state.global_phase_eighths/8),
                     len(work)/p.width, leakage,
                     sum(state.wire_depths[q] for q in work)/max(1,len(work)*(b.max_depth+1)),
                     pending_fraction, d*(1-t), d*(1-c), d*(1-depth), d], dtype=float)


def inner_features(p, record, gate, projected):
    s, b = record.state, p.budget
    before, after = record.x[1], distance(p, projected)
    gain = before-after
    level = 1+max(s.wire_depths[q] for q in gate.qubits)
    slack = [max(0., 1-v/(limit+1)) for v,limit in ((s.t_count+int(gate.is_non_clifford),b.max_t_count),
             (s.cnot_count+int(gate.is_two_qubit),b.max_cnot_count),(s.gate_count+1,b.max_gates),(max(s.depth,level),b.max_depth))]
    work = p.contract.clean_ancillas
    tds = next_t_depths(record.t_depths,gate)
    clean = float(np.sum(np.abs(projected[p.contract.invalid_clean_output_rows]) ** 2) / p.contract.domain_dimension)
    return np.array([1,before,after,gain,int(gate.is_non_clifford),int(gate.is_two_qubit),
                     (max(s.depth,level)-s.depth)/(b.max_depth+1),any(q in work for q in gate.qubits),
                     gate.is_two_qubit and gate.qubits[0] in work,gate.qubits[-1] in work,
                     *slack, level/(b.max_depth+1),record.x[6],record.x[8],record.x[10],record.x[11],
                     gain*slack[0],gain*slack[1],gain*slack[3],1-max(tds)/(p.max_t_depth+1),
                     record.x[13]-clean],dtype=float)


class NativeHierarchy:
    def __init__(self, seed=0, alpha=.01, ucb=.2):
        self.seed, self.alpha, self.ucb = int(seed), float(alpha), float(ucb)
        if not 0 < self.alpha <= 1 or not np.isfinite(self.ucb) or self.ucb < 0:
            raise ValueError('invalid learning parameters')
        self.rng=np.random.default_rng(seed)
        self.w=np.zeros(len(OUTER_NAMES)); self.w[1]=-4.; self.w[4]=-.1; self.w[13]=-.1
        self.prior=np.zeros(len(INNER_NAMES));self.prior[2]=-4.;self.prior[4:6]=-.02;self.prior[6]=-.01
        d=len(INNER_NAMES)
        self.a=np.array([np.eye(d) for _ in FAMILIES]); self.b=np.zeros((len(FAMILIES),d))
        self.theta=np.zeros_like(self.b);self.updates=np.zeros(len(FAMILIES),dtype=int)
        self.outer_updates=0;self.episodes=0;self.frozen=False

    @property
    def digest(self):
        return digest(self.payload())

    def payload(self):
        return {'schema':SCHEMA,'seed':self.seed,'alpha':self.alpha,'ucb':self.ucb,
                'outer_features':OUTER_NAMES,'inner_features':INNER_NAMES,'families':FAMILIES,
                'w':self.w.tolist(),'a':self.a.tolist(),'b':self.b.tolist(),
                'outer_updates':self.outer_updates,'updates':self.updates.tolist(),
                'episodes':self.episodes,'frozen':self.frozen}

    def freeze(self):
        self.frozen=True
        for array in (self.w,self.a,self.b,self.theta,self.updates):
            array.setflags(write=False)
        return self

    def score_outer(self,x):
        return np.asarray(x)@self.w

    def score_inner(self,x,family,explore=False):
        a=FAMILIES.index(family)
        score=float(x@(self.prior+self.theta[a]))
        if explore:
            score+=self.ucb*np.sqrt(max(0.,x@np.linalg.solve(self.a[a],x)))
        return score

    def update_outer(self,x,reward,next_x=None):
        if self.frozen:
            raise ValueError('cannot train a frozen checkpoint')
        delta=reward+(0. if next_x is None else float(next_x@self.w))-float(x@self.w)
        self.w+=self.alpha*delta*x/max(1.,float(x@x));self.w=np.clip(self.w,-20,20)
        self.outer_updates+=1

    def update_inner(self,x,family,response):
        if self.frozen:
            raise ValueError('cannot train a frozen checkpoint')
        a=FAMILIES.index(family)
        self.a[a]+=np.outer(x,x);self.b[a]+=(response-float(x@self.prior))*x
        self.theta[a]=np.linalg.solve(self.a[a],self.b[a]);self.updates[a]+=1

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(self.payload(),indent=2)+'\n')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d.get('schema')!=SCHEMA or tuple(d.get('outer_features',()))!=OUTER_NAMES or tuple(d.get('inner_features',()))!=INNER_NAMES or tuple(d.get('families',()))!=FAMILIES:
            raise ValueError('checkpoint is not compatible with the native hybrid environment')
        obj=cls(d['seed'],d['alpha'],d['ucb'])
        for name in ('w','a','b','updates'):
            value=np.asarray(d[name],dtype=int if name=='updates' else float)
            if value.shape!=getattr(obj,name).shape or not np.isfinite(value).all():
                raise ValueError('malformed checkpoint arrays')
            setattr(obj,name,value)
        for a in range(len(FAMILIES)):
            if not np.allclose(obj.a[a],obj.a[a].T) or np.linalg.eigvalsh(obj.a[a]).min()<=0:
                raise ValueError('invalid LinUCB covariance')
            obj.theta[a]=np.linalg.solve(obj.a[a],obj.b[a])
        if any(type(d[k]) is not int or d[k] < 0 for k in ('outer_updates','episodes')) or type(d['frozen']) is not bool or np.any(obj.updates<0):
            raise ValueError('invalid checkpoint training provenance')
        obj.outer_updates=d['outer_updates'];obj.episodes=d['episodes']
        return obj.freeze() if d['frozen'] else obj
