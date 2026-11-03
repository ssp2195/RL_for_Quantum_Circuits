"""Persistent frontier search for exact general Boolean phase-oracle macros.

Outer semi-gradient SARSA ranks frontier records. Inner disjoint LinUCB ranks
the two exact construction choices for each nonlinear ANF monomial. Neither
learner determines legality, Boolean identities, resource limits or correctness.
All legal choices remain pending until explored (or a declared limit intervenes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .general_boolean import BooleanPhaseProblem, _compile_term, _monomials, compile_boolean_phase
from .ir import Instruction, ResourceState, ProtocolEvent, append_event
from .phase_protocol import verify_phase
from .causal_protocol import causalize_immediate_cleanup


OUTER_DIM = 6
INNER_DIM = 6
METHODS = ('measured', 'direct')


@dataclass
class GenericHierarchy:
    """Separate outer linear SARSA and inner LinUCB models; no fixed score priors."""
    seed: int = 0
    alpha: float = 0.02
    exploration: float = 0.15
    outer: np.ndarray = field(init=False, repr=False)
    covariances: np.ndarray = field(init=False, repr=False)
    targets: np.ndarray = field(init=False, repr=False)
    frozen: bool = False
    outer_updates: int = 0
    inner_updates: int = 0

    def __post_init__(self):
        self.outer = np.zeros(OUTER_DIM)
        self.covariances = np.array([np.eye(INNER_DIM) for _ in METHODS])
        self.targets = np.zeros((len(METHODS), INNER_DIM))
        self.rng = np.random.default_rng(self.seed)

    def score_outer(self, features: np.ndarray) -> float:
        return float(features @ self.outer)

    def score_inner(self, features: np.ndarray, mode: str, training: bool) -> float:
        index = METHODS.index(mode)
        inverse_times_feature = np.linalg.solve(self.covariances[index], features)
        mean = float(features @ np.linalg.solve(self.covariances[index], self.targets[index]))
        bonus = self.exploration * np.sqrt(max(0., float(features @ inverse_times_feature)))
        return mean + (bonus if training else 0.)

    def update_inner(self, features: np.ndarray, mode: str, reward: float):
        if self.frozen:
            raise ValueError('cannot train a frozen model')
        i = METHODS.index(mode)
        self.covariances[i] += np.outer(features, features)
        self.targets[i] += reward * features
        self.inner_updates += 1

    def update_outer(self, features: np.ndarray, reward: float, next_features: np.ndarray | None):
        if self.frozen:
            raise ValueError('cannot train a frozen model')
        next_value = 0. if next_features is None else self.score_outer(next_features)
        difference = reward + next_value - self.score_outer(features)
        self.outer += self.alpha * difference * features / max(1., float(features @ features))
        self.outer_updates += 1

    def freeze(self):
        self.frozen = True
        for array in (self.outer, self.covariances, self.targets):
            array.setflags(write=False)
        return self


@dataclass
class GenericRecord:
    record_id: int
    modes: tuple[str, ...]
    resource: ResourceState
    tail: ProtocolEvent | None
    outcomes: int
    pending: tuple[str, ...] = METHODS


class GenericFrontierSearch:
    """Explicit exact construction grammar; its optimum is library-relative."""

    def __init__(self, problem: BooleanPhaseProblem, *, max_nonlinear_terms: int = 12,
                 m1_guided: bool = True):
        self.problem = problem
        self.m1_guided = m1_guided
        self.nonlinear = tuple(m for m in _monomials(problem) if m.bit_count() >= 3)
        if len(self.nonlinear) > max_nonlinear_terms:
            raise ValueError('generic frontier exceeds declared exhaustive-term limit')
        self.t_upper = 1 + sum(max(4*(m.bit_count()-2), 4*(m.bit_count()-3)+7)
                               for m in self.nonlinear)
        width = problem.n + max((m.bit_count()-2 for m in self.nonlinear), default=0)
        starting = tuple(_compile_term(mask, '', problem.n, 0)
                         for mask in _monomials(problem) if mask.bit_count() < 3)
        ops = tuple(op for group in starting for op in group)
        resource = ResourceState.zero(width)
        tail = None
        for op in ops:
            resource = resource.append(op, 'and4', problem)
            tail = append_event(tail, op, 'and4')
        self.records = {0: GenericRecord(0, (), resource, tail, 0)}
        self.frontier = {0: self.records[0]}
        self.complete: list[dict] = []
        self.expansions = 0

    def outer_features(self, record: GenericRecord) -> np.ndarray:
        r = record.resource
        count = max(1, len(self.nonlinear))
        max_ancillas = max((m.bit_count()-2 for m in self.nonlinear), default=1)
        return np.array((1., len(record.modes)/count, r.peak_aux/max_ancillas,
                         r.t_count/self.t_upper, r.cnot/(1+12*self.t_upper),
                         (len(self.nonlinear)-len(record.modes))/count), dtype=float)

    def _extension(self, record: GenericRecord, mode: str):
        if mode not in record.pending or len(record.modes) >= len(self.nonlinear):
            raise ValueError('ineligible construction choice')
        emitted = _compile_term(self.nonlinear[len(record.modes)], mode,
                                self.problem.n, record.outcomes)
        if self.m1_guided:
            emitted = causalize_immediate_cleanup(emitted, self.problem.n)
        resource, tail = record.resource, record.tail
        for op in emitted:
            resource = resource.append(op, 'and4', self.problem)
            tail = append_event(tail, op, 'and4')
        return emitted, resource, tail

    def inner_features(self, record: GenericRecord, mode: str, new: ResourceState) -> np.ndarray:
        r = record.resource
        return np.array((1., float(mode == 'direct'),
                         self.nonlinear[len(record.modes)].bit_count()/self.problem.n,
                         (new.peak_aux-r.peak_aux)/max(1, self.problem.n),
                         (new.t_count-r.t_count)/self.t_upper,
                         (new.cnot-r.cnot)/(1+12*self.t_upper)), dtype=float)

    def _reward(self, old: ResourceState, new: ResourceState) -> float:
        # Exact lexicographic scalarization of peak ancillas then T-count,
        # because total T-count of every complete menu circuit < t_upper.
        return -(new.peak_aux-old.peak_aux) - (new.t_count-old.t_count)/self.t_upper

    def run(self, policy: GenericHierarchy | None = None, *,
            train: str | None = None, max_expansions: int = 8192) -> dict:
        if train not in (None, 'outer', 'inner') or max_expansions < 1:
            raise ValueError('invalid learning stage or expansion limit')
        if policy is None and train is not None:
            raise ValueError('training requires a model')
        if policy is not None and policy.frozen == (train is not None):
            raise ValueError('policy freeze/training mismatch')
        if not self.nonlinear:
            circuit = compile_boolean_phase(self.problem, m1_guided=self.m1_guided)
            verified = verify_phase(self.problem, circuit)
            if not verified['valid']:
                raise AssertionError('Clifford-only circuit failed verification')
            self.complete.append({'modes': (), 'resources': verified['resources'],
                                  'protocol': circuit})
            self.frontier.clear()
        while self.frontier and self.expansions < max_expansions:
            panel = list(self.frontier.values())
            record = min(panel, key=lambda r: r.record_id) if policy is None else max(
                panel, key=lambda r: (policy.score_outer(self.outer_features(r)), -r.record_id))
            actions = []
            for mode in record.pending:
                emitted, resource, tail = self._extension(record, mode)
                if self.problem.limits.accepts(resource.report()):
                    features = self.inner_features(record, mode, resource)
                    actions.append((mode, emitted, resource, tail, features))
            if not actions:
                self.frontier.pop(record.record_id)
                continue
            if policy is None:
                mode, emitted, resource, tail, features = actions[0]
            else:
                mode, emitted, resource, tail, features = max(
                    actions, key=lambda a: (policy.score_inner(a[4], a[0], train == 'inner'),
                                            -METHODS.index(a[0])))
            record.pending = tuple(m for m in record.pending if m != mode)
            if not record.pending:
                self.frontier.pop(record.record_id)
            self.expansions += 1
            modes = record.modes + (mode,)
            reward = self._reward(record.resource, resource)
            if len(modes) == len(self.nonlinear):
                completed = compile_boolean_phase(self.problem, modes,
                                                 m1_guided=self.m1_guided)
                verified = verify_phase(self.problem, completed)
                if not verified['valid']:
                    raise AssertionError('exact constructed candidate failed verification: ' + verified['reason'])
                completed.persistent_tail()
                self.complete.append({'modes': modes, 'resources': verified['resources'],
                                      'protocol': completed})
            else:
                index = len(self.records)
                child = GenericRecord(index, modes, resource, tail,
                                      record.outcomes+sum(op.kind=='MX' for op in emitted))
                self.records[index] = child
                self.frontier[index] = child
            if train == 'inner':
                policy.update_inner(features, mode, reward)
            elif train == 'outer':
                next_record = (max(self.frontier.values(),
                                   key=lambda r: (policy.score_outer(self.outer_features(r)), -r.record_id))
                               if self.frontier else None)
                policy.update_outer(self.outer_features(record), reward,
                                    self.outer_features(next_record) if next_record else None)
        selection = (min(self.complete, key=lambda r:
                         (r['resources']['peak_aux'], r['resources']['t_count'],
                          r['resources']['cnot'], r['modes']))
                     if self.complete else None)
        return {'status': 'complete' if not self.frontier else 'limit_reached',
                'expansions': self.expansions, 'records': len(self.records),
                'm1_guided': self.m1_guided,
                'candidate_count': len(self.complete), 'selected': selection,
                'global_oracle_optimality': False,
                'complete_library_search': not self.frontier}


def train_generic_hierarchy(problems: tuple[BooleanPhaseProblem, ...], *,
                            seed: int = 11, episodes: int = 16) -> GenericHierarchy:
    """Train both controllers on independent supplied logical targets."""
    if not problems or episodes < 2:
        raise ValueError('training requires targets and at least two episodes')
    model = GenericHierarchy(seed=seed)
    for episode in range(episodes):
        stage = 'inner' if episode % 2 else 'outer'
        problem = problems[episode % len(problems)]
        GenericFrontierSearch(problem).run(model, train=stage)
    if not model.outer_updates or not model.inner_updates:
        raise AssertionError('both controllers must receive updates')
    return model.freeze()
