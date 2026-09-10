"""Fair deferred exact search; discovery never invokes or imports the auditor."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import time
from collections import OrderedDict
from .phase_index import CandidateIndex
import numpy as np
from .phase_contract import PhaseProblem, PhaseState, root_state, successor, covers, certify_phase, completion_possible
from .phase_policy import PhaseHierarchy, features, inner_features, arm
from .resource_search import WorkLimits, WorkMeter

SCHEDULERS = ('hierarchy', 'outer', 'inner', 'untrained', 'greedy', 'ucs', 'random', 'no_budget', 'no_workspace')


@dataclass
class Node:
    rid: int
    state: PhaseState
    parent: int | None
    token: int | None
    pending: int
    x: np.ndarray
    projections: dict = field(default_factory=dict)


def search_phase(p: PhaseProblem, model: PhaseHierarchy | None = None, *,
                 scheduler='hierarchy', limits=WorkLimits(4096, 20000, 30., 30.),
                 train=None, epsilon=.15, fairness=32, dag=True, cancel=None, panel_size=32):
    if scheduler not in SCHEDULERS or train not in (None, 'outer', 'inner'):
        raise ValueError('unsupported scheduler or training stage')
    if fairness < 1 or not 0 <= epsilon <= 1 or panel_size < 0:
        raise ValueError('invalid fairness/epsilon')
    model = PhaseHierarchy() if model is None else model
    if train is not None and model.frozen:
        raise ValueError('cannot train frozen policy')
    if train is None and scheduler in ('hierarchy', 'outer', 'inner', 'no_budget', 'no_workspace') and not model.frozen:
        raise ValueError('evaluation requires a frozen checkpoint; use untrained for explicit prior baseline')
    if scheduler == 'untrained':
        model = PhaseHierarchy(seed=model.seed).freeze()
    snapshot = model.digest if train is None else None
    meter = WorkMeter(limits, cancel)
    nodes, frontier, archive = [], OrderedDict(), {}
    candidate_index = CandidateIndex()
    stats = {'lookahead_transitions': 0, 'cap_checks': 0, 'dominance_comparisons': 0,
             'feature_seconds': 0., 'transition_seconds': 0., 'certification_seconds': 0.,
             'peak_frontier': 0, 'fairness_steps': 0, 'policy_choices': 0,
             'audit_calls': 0, 'full_dag_policy_reconstructions': 0,
             'phase_emitting_edges': 0, 'workspace_edges': 0, 'cleanup_edges': 0, 'max_scored_panel': 0, 'outer_vectors_scored': 0, 'panel_size': panel_size}
    halt = None
    expanded_tokens, logs = [], []
    root, _ = root_state(p)

    def insert(s, parent, token):
        nonlocal halt
        group = archive.get(s.key, [])
        for old in group:
            stats['dominance_comparisons'] += 1
            if covers(nodes[old].state, s):
                return None
        if len(nodes) >= limits.max_records:
            halt = 'record_limit'
            return None
        survivors = []
        for old in group:
            stats['dominance_comparisons'] += 1
            if covers(s, nodes[old].state):
                frontier.pop(old, None)
                candidate_index.discard(old)
            else:
                survivors.append(old)
        rid = len(nodes)
        pending = 0
        for k, (c, t) in enumerate(p.actions):
            if (c != t and s.cnot < p.max_cnot) or (c == t and s.remaining & p.phase_bits.get(s.rows[t], 0)):
                pending |= 1 << k
        progress = 0. if parent is None else (nodes[parent].state.remaining.bit_count() - s.remaining.bit_count()) / max(1, len(p.coefficients))
        node = Node(rid, s, parent, token, pending, features(p, s, last_progress=progress))
        nodes.append(node)
        archive[s.key] = survivors + [rid]
        if pending:
            frontier[rid] = node
            candidate_index.add(float(node.x[1] + .3*node.x[2] + .3*node.x[3] + .25*node.x[4]), rid)
        stats['peak_frontier'] = max(stats['peak_frontier'], len(frontier))
        return node

    def tokens(node):
        mask = node.pending
        out = []
        while mask:
            low = mask & -mask
            out.append(low.bit_length() - 1)
            mask ^= low
        return out

    def projected(node, token):
        if token not in node.projections:
            s, _ = successor(p, node.state, token)
            x = inner_features(p, node.state, s, token, node.token)
            node.projections[token] = (s, x)
            stats['lookahead_transitions'] += 1
        return node.projections[token]

    def panel():
        if panel_size == 0:
            return tuple(frontier.values())
        return tuple(frontier[rid] for rid in candidate_index.smallest(panel_size))

    def potential():
        minimum = candidate_index.minimum()
        return -minimum[0] if minimum else 0.

    def selection():
        started = time.perf_counter()
        records = panel()
        forced = (meter.edges + 1) % fairness == 0
        if forced:
            node = next(iter(frontier.values()))
            token = tokens(node)[0]
            stats['fairness_steps'] += 1
        else:
            if scheduler == 'ucs':
                node = min(frontier.values(), key=lambda n: (n.state.cnot, n.rid))
            elif scheduler == 'random':
                node = records[int(model.rng.integers(len(records)))]
            elif scheduler in ('greedy', 'inner'):
                node = min(records, key=lambda n: (n.x[1] + .3 * n.x[2] + .3 * n.x[3] + .25 * n.x[4], n.rid))
            else:
                if train == 'outer' and model.rng.random() < epsilon:
                    node = records[int(model.rng.integers(len(records)))]
                else:
                    xs = np.array([n.x for n in records])
                    xs[:, 13] = xs[:, 1] * max(0., 1 - meter.edges / max(1, limits.max_edges))
                    xs[:, 14] = xs[:, 2] * max(0., 1 - meter.edges / max(1, limits.max_edges))
                    scores = model.outer_score(xs)
                    stats['outer_vectors_scored'] += len(records)
                    stats['max_scored_panel'] = max(stats['max_scored_panel'], len(records))
                    node = records[int(np.argmax(scores))]
            ts = tokens(node)
            if scheduler == 'ucs':
                token = ts[0]
            elif scheduler == 'random':
                token = ts[int(model.rng.integers(len(ts)))]
            else:
                candidates = []
                for t in ts:
                    s, x = projected(node, t)
                    if scheduler in ('greedy', 'outer'):
                        score = float(x @ model.prior)
                    else:
                        score = model.inner_score(x, arm(p, t), explore=train == 'inner')
                    candidates.append(score if completion_possible(p,s) else -np.inf)
                stats['cap_checks'] += len(ts)
                token = ts[int(np.argmax(candidates))]
            stats['policy_choices'] += 1
        s, ix = projected(node, token)
        x = node.x.copy()
        x[13:15] *= max(0., 1 - meter.edges / max(1, limits.max_edges))
        stats['feature_seconds'] += time.perf_counter() - started
        return node, token, s, x, ix

    def finish(witness=None, reason='unknown'):
        if train is None and model.digest != snapshot:
            raise AssertionError('evaluation modified frozen policy parameters')
        return {'schema': 'phase-discovery-v1', 'problem': p.manifest(), 'problem_digest': p.digest,
                'scheduler': scheduler, 'status': 'feasible' if witness else 'unknown', 'reason': reason,
                'witness': witness, 'witness_source': 'frozen_policy_discovery' if witness and scheduler in ('hierarchy', 'outer', 'inner', 'no_budget', 'no_workspace') else scheduler if witness else None,
                'policy_digest': snapshot, 'edges': meter.edges, 'records': len(nodes),
                'wall_seconds': meter.wall, 'cpu_seconds': meter.cpu, 'profile': stats,
                'training_transitions': logs if train else [], 'expanded_tokens': expanded_tokens,
                'audit_witness_used': False}

    def certify(node):
        path = []
        current = node
        while current.parent is not None:
            path.append(current.token)
            current = nodes[current.parent]
        started = time.perf_counter()
        cert = certify_phase(p, reversed(path), dag=dag)
        stats['certification_seconds'] += time.perf_counter() - started
        if not cert['success']:
            raise AssertionError('terminal generated circuit failed independent certification')
        return cert

    if not completion_possible(p,root):
        return finish(reason='root_completion_bound_excludes_cap')
    node = insert(root, None, None)
    if root.terminal(p):
        return finish(certify(node), 'certified_root')
    if not frontier or meter.reason():
        return finish(reason=meter.reason() or 'exhausted_without_proof')
    chosen = selection()
    while frontier:
        if meter.reason():
            return finish(reason=meter.reason())
        node, token, child_state, x, ix = chosen
        before = potential()
        started = time.perf_counter()
        node.pending &= ~(1 << token)
        if not node.pending:
            frontier.pop(node.rid, None)
            candidate_index.discard(node.rid)
        stats['cap_checks'] += 1
        child = insert(child_state, node.rid, token) if completion_possible(p,child_state) else None
        meter.edges += 1
        expanded_tokens.append(token)
        stats['transition_seconds'] += time.perf_counter() - started
        emitting = child_state.remaining != node.state.remaining
        stats['phase_emitting_edges'] += emitting
        stats['workspace_edges'] += any(q >= p.n for q in p.actions[token])
        stats['cleanup_edges'] += (child_state.rows[p.actions[token][1]] == p.identity_rows[p.actions[token][1]])
        hit = child is not None and child_state.terminal(p)
        certificate = certify(child) if hit else None
        reason = 'certified' if hit else halt or meter.reason() or ('exhausted_without_proof' if not frontier else None)
        next_selection = None if reason else selection()
        reason = reason or meter.reason()
        after = 0. if reason else potential()
        base = float(hit) - .002
        reward = base + after - before
        if train == 'outer':
            model.update_outer(x, reward, None if reason else next_selection[3])
        elif train == 'inner':
            # A contextual response informed by the frozen outer rank value.
            # It is not an admissible bound and never authorizes deletion.
            a = features(p, node.state)
            b = features(p, child_state)
            response = base + float(model.outer_score(b) - model.outer_score(a))
            if not completion_possible(p,child_state):
                response = -1.
            model.update_inner(ix, arm(p, token), response)
        if train:
            logs.append({'token': token, 'arm': arm(p, token), 'phase_emission': emitting,
                         'workspace': any(q >= p.n for q in p.actions[token]),
                         'reward': reward, 'terminal': bool(reason), 'potential_before': before, 'potential_after': after})
        if reason:
            return finish(certificate, reason)
        chosen = next_selection
    return finish(reason='exhausted_without_proof')


def constructive_phase(p, *, dag=True):
    """Public nonlearned parity compute/phase/uncompute baseline; no hidden witness."""
    state, _ = root_state(p)
    tokens = []
    lookup = {pair: i for i, pair in enumerate(p.actions)}
    while state.remaining:
        i = (state.remaining & -state.remaining).bit_length() - 1
        mask = p.coefficients[i][0]
        support = [q for q in range(p.n) if mask & (1 << q)]
        t = support[-1]
        path = [lookup[c, t] for c in support[:-1]]
        for token in path:
            state, _ = successor(p, state, token)
            tokens.append(token)
        if p.emission == 'deferred':
            token = lookup[t,t]
            state, _ = successor(p,state,token)
            tokens.append(token)
        for token in path[::-1]:
            state, _ = successor(p,state,token)
            tokens.append(token)
    return certify_phase(p, tokens, dag=dag)
