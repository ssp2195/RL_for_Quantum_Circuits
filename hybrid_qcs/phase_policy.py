"""Linear SARSA ranking and residual disjoint LinUCB for phase obligations.

The fixed prior is public and ablated against an untrained policy. All learned
parameters and covariance matrices are frozen before test generation. No circuit
witnesses, audit outputs or lower-bound labels are policy inputs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from .resource_domain import canonical_digest

OUTER_NAMES = ('bias', 'remaining', 'repair_wires', 'repair_bits', 'nearest_parity',
               'cnot_usage', 'depth_usage', 'gate_usage', 'clean_fraction',
               'remaining_cnot_slack', 'repair_depth_slack', 'work_depth_mean',
               'depth_spread', 'remaining_work', 'repair_work', 'last_progress', 't_depth_usage', 'remaining_t_depth_slack', 'available_phases', 't_capacity_margin')
INNER_NAMES = ('bias', 'phase_progress', 'repair_progress', 'bit_progress', 'nearest_progress',
               'child_remaining', 'child_repair', 'cnot_usage', 'depth_usage',
               'depth_increase', 'clean_change', 'workspace_target',
               'progress_cnot_slack', 'repair_depth_slack', 'progress_work', 'inverse', 't_depth_usage', 't_depth_increase', 'available_progress', 't_capacity_change')
ARMS = ('data', 'to_workspace', 'from_workspace', 'phase')
SCHEMA = 'linear-phase-hierarchy-v1'


def features(p, s, work=1., last_progress=0.):
    remaining = s.remaining.bit_count() / max(1, len(p.coefficients))
    wrong = sum(x != y for x, y in zip(s.rows, p.identity_rows)) / p.width
    bits = sum((x ^ y).bit_count() for x, y in zip(s.rows, p.identity_rows)) / (p.n * p.width)
    nearest = sum(min((m ^ r).bit_count() for r in s.rows) for i, (m, _) in enumerate(p.coefficients)
                  if s.remaining & (1 << i)) / (p.n * max(1, len(p.coefficients)))
    clean = sum(row == 0 for row in s.rows[p.n:]) / (p.ancillas + 1)
    usage = s.cnot / (p.max_cnot + 1)
    depth = s.depth / (p.max_depth + 1)
    return np.array((1., remaining, wrong, bits, nearest, usage, depth,
                     s.gates / (p.max_gates + 1), clean,
                     remaining * (1 - usage), wrong * (1 - depth),
                     sum(s.depths[p.n:]) / ((p.ancillas + 1) * (p.max_depth + 1)),
                     (max(s.depths) - min(s.depths)) / (p.max_depth + 1),
                     remaining * work, wrong * work, last_progress, s.t_depth / (p.max_t_depth+1), remaining*(1-s.t_depth/(p.max_t_depth+1)),
                     sum(bool(s.remaining & (1 << i)) and m in s.rows for i,(m,_) in enumerate(p.coefficients))/max(1,len(p.coefficients)),
                     (sum(p.max_t_depth-d for d in s.t_depths)-sum(c%2 for i,(_,c) in enumerate(p.coefficients) if s.remaining & (1<<i)))/(p.width*(p.max_t_depth+1))), dtype=float)


def inner_features(p, parent, child, token, last_token=None, work=1.):
    before, after = features(p, parent, work), features(p, child, work)
    c, t = p.actions[token]
    progress = before[1] - after[1]
    repair = before[2] - after[2]
    return np.array((1., progress, repair, before[3] - after[3], before[4] - after[4],
                     after[1], after[2], after[5], after[6],
                     (child.depth - parent.depth) / (p.max_depth + 1),
                     after[8] - before[8], float(t >= p.n),
                     progress * (1 - after[5]), repair * (1 - after[6]),
                     progress * work, float(c != t and token == last_token), child.t_depth/(p.max_t_depth+1), (child.t_depth-parent.t_depth)/(p.max_t_depth+1), after[18]-before[18], after[19]-before[19]), dtype=float)


def arm(p, token):
    c, t = p.actions[token]
    return 'phase' if c == t else 'to_workspace' if t >= p.n else 'from_workspace' if c >= p.n else 'data'


def outer_prior():
    w = np.zeros(len(OUTER_NAMES))
    w[1:5] = (-1., -.30, -.30, -.25)
    w[5:7] = (-.02, -.03)
    w[16] = -.03
    w[18] = .35
    w[19] = .05
    return w


def inner_prior():
    w = np.zeros(len(INNER_NAMES))
    w[1:5] = (1., .30, .30, .25)
    w[9] = -.03
    w[17] = -.03
    w[18] = .35
    w[19] = .05
    return w


@dataclass
class PhaseHierarchy:
    seed: int = 0
    rate: float = .01
    ucb: float = .15
    regularization: float = 5.
    outer: np.ndarray = field(default_factory=outer_prior)
    prior: np.ndarray = field(default_factory=inner_prior)
    outer_updates: int = 0
    inner_updates: int = 0
    stage: str = 'untrained'

    def __post_init__(self):
        if self.rate <= 0 or self.regularization <= 0 or self.ucb < 0:
            raise ValueError('invalid learning hyperparameters')
        self.rng = np.random.default_rng(self.seed)
        self.inverses = {a: np.eye(len(INNER_NAMES)) / self.regularization for a in ARMS}
        self.responses = {a: np.zeros(len(INNER_NAMES)) for a in ARMS}
        self.arm_updates = dict.fromkeys(ARMS, 0)
        self._theta = {}
        self.frozen = False
        self.ablation = None

    def masked_outer(self, matrix):
        x = np.asarray(matrix)
        if self.ablation in ('no_budget', 'no_workspace'):
            x = x.copy()
            indices = [5, 6, 7, 9, 10, 13, 14, 16, 17, 19] if self.ablation == 'no_budget' else [8, 11, 12, 19]
            x[..., indices] = 0
        return x

    def masked_inner(self, x):
        x = np.asarray(x)
        if self.ablation in ('no_budget', 'no_workspace'):
            x = x.copy()
            indices = [7, 8, 9, 12, 13, 14, 16, 17, 19] if self.ablation == 'no_budget' else [10, 11, 19]
            x[..., indices] = 0
        return x

    def outer_score(self, matrix):
        return self.masked_outer(matrix) @ self.outer

    def inner_score(self, x, a, explore=False):
        x = self.masked_inner(x)
        if a not in self._theta:
            self._theta[a] = self.inverses[a] @ self.responses[a]
        score = float(x @ (self.prior + self._theta[a]))
        if explore:
            score += self.ucb * np.sqrt(max(0., float(x @ self.inverses[a] @ x)))
        return score

    def update_outer(self, x, reward, next_x):
        if self.frozen:
            raise RuntimeError('cannot train a frozen hierarchy')
        x = self.masked_outer(x)
        value = float(x @ self.outer)
        target = reward + (0. if next_x is None else float(self.masked_outer(next_x) @ self.outer))
        # Normalized semi-gradient SARSA; the bootstrap target is not differentiated.
        self.outer += self.rate * (target - value) * x / max(1., float(x @ x))
        np.clip(self.outer, -20., 20., out=self.outer)
        self.outer_updates += 1

    def update_inner(self, x, a, response):
        if self.frozen:
            raise RuntimeError('cannot train a frozen hierarchy')
        x = self.masked_inner(x)
        v = self.inverses[a] @ x
        self.inverses[a] -= np.outer(v, v) / (1 + float(x @ v))
        self.inverses[a] = (self.inverses[a] + self.inverses[a].T) * .5
        self.responses[a] += (float(response) - float(x @ self.prior)) * x
        self._theta.pop(a, None)
        self.inner_updates += 1
        self.arm_updates[a] += 1

    def freeze(self):
        self.frozen = True
        self.stage = 'frozen'
        for a in ARMS:
            self._theta[a] = self.inverses[a] @ self.responses[a]
        return self

    def payload(self):
        return {'schema': SCHEMA, 'outer_names': list(OUTER_NAMES), 'inner_names': list(INNER_NAMES),
                'seed': self.seed, 'rate': self.rate, 'ucb': self.ucb, 'regularization': self.regularization,
                'outer': self.outer.tolist(), 'prior': self.prior.tolist(),
                'inverses': {k: v.tolist() for k, v in self.inverses.items()},
                'responses': {k: v.tolist() for k, v in self.responses.items()},
                'outer_updates': self.outer_updates, 'inner_updates': self.inner_updates,
                'arm_updates': self.arm_updates, 'stage': self.stage, 'frozen': self.frozen,
                'ablation': self.ablation}

    @property
    def digest(self):
        return canonical_digest(self.payload())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload(), sort_keys=True, indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if d.get('schema') != SCHEMA or d['outer_names'] != list(OUTER_NAMES) or d['inner_names'] != list(INNER_NAMES):
            raise ValueError('incompatible phase-policy schema')
        h = cls(d['seed'], d['rate'], d['ucb'], d['regularization'])
        h.outer, h.prior = np.array(d['outer']), np.array(d['prior'])
        if h.outer.shape != (len(OUTER_NAMES),) or h.prior.shape != (len(INNER_NAMES),):
            raise ValueError('bad linear parameter dimensions')
        for a in ARMS:
            inverse, response = np.array(d['inverses'][a]), np.array(d['responses'][a])
            if inverse.shape != (len(INNER_NAMES),) * 2 or response.shape != (len(INNER_NAMES),):
                raise ValueError('bad bandit dimensions')
            if not np.allclose(inverse, inverse.T) or np.min(np.linalg.eigvalsh(inverse)) <= 0:
                raise ValueError('invalid covariance inverse')
            h.inverses[a], h.responses[a] = inverse, response
        if not all(np.isfinite(v).all() for v in [h.outer, h.prior, *h.inverses.values(), *h.responses.values()]):
            raise ValueError('nonfinite model parameter')
        for k in ('outer_updates', 'inner_updates', 'arm_updates', 'stage', 'frozen', 'ablation'):
            setattr(h, k, d[k])
        return h
