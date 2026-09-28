"""Proof-first synthesis, with learning optional for the unresolved schedule.

There are no manually weighted resource scores. The caller declares a
lexicographic objective; all feasible nondominated constructions are preserved.
Analytic construction and learned discovery have separate provenance.
Historical CleanupSearch/optimize_cleanup remain unchanged for reproduction.
"""
from __future__ import annotations
from dataclasses import replace
import time
import math
from typing import Callable

from ..resource_search import WorkLimits
from .contract import CleanupProblem, Limits
from .consumer import check_cancel, Interrupted
from .consumer_compile import compile_factors, verify_factor_receipt
from .phase_protocol import PhaseProtocol, verify_phase
from .plan import compile_deterministic
from .ir import Protocol
from .reference import compile_bank_free
from .search import CleanupSearch
from .policy import CleanupHierarchy
from .verify import verify_protocol, workspace_certificate, verify_workspace_certificate

RESOURCE_KEYS = ('t_count', 'peak_aux', 'cnot', 'native_gates', 'native_depth',
                 't_depth', 'worst_case_ticks', 'measurement_rounds')
DEFAULT_OBJECTIVE = ('t_count', 'peak_aux', 'worst_case_ticks', 'native_gates', 'cnot')


def _frozen(policy):
    if policy is not None and type(policy) is not CleanupHierarchy:
        raise TypeError('a cleanup SARSA/LinUCB checkpoint is required')
    if policy is not None and (not policy.frozen or min(policy.stage_episodes.values()) < 1):
        raise ValueError('both learning stages must finish and the policy must be frozen')


def _phase(p, circuit, provenance):
    return PhaseProtocol(p.digest, circuit.layout.width, circuit.ops, provenance, circuit.primitive)


def optimize_materialization(p: CleanupProblem, policy=None, *,
                             search_on_failure: bool = False,
                             limits: WorkLimits = WorkLimits(4096, 20000, 10., 10.),
                             cancel: Callable[[], bool] | None = None) -> dict:
    """Close workspace using its proof before asking a learner to find a schedule.

    A failed constructive schedule is not an infeasibility theorem. Optional
    search first investigates minimum-width layouts, then alternatives. No
    depth/latency/global oracle optimum is asserted.
    """
    _frozen(policy)
    before = None if policy is None else policy.digest
    started = time.perf_counter()
    cpu_started = time.process_time()
    lower = p.workspace_lower_bound
    try:
        check_cancel(cancel)
        if p.limits.max_aux is not None and p.limits.max_aux < lower:
            return {'status': 'infeasible_within_theorem_architecture', 'lower_bound': lower,
                    'global_oracle_optimality': False, 'learned_discovery': False,
                    'scope': 'full-bank architecture only', 'wall_seconds': time.perf_counter()-started}
        circuit = compile_deterministic(p, side='smaller')
        checked = verify_protocol(p, circuit, cancel=cancel)
        trace = []
        learned = False
        used_edges = used_records = 0
        if not checked['valid'] and search_on_failure:
            # Preserve larger-helper layouts: they can have better depth.
            small = 'column' if p.m <= p.r else 'row'
            for side in (small, 'row' if small == 'column' else 'column'):
                check_cancel(cancel)
                left = limits.wall_seconds - (time.perf_counter()-started)
                cpu_left = limits.cpu_seconds - (time.process_time()-cpu_started)
                if min(left, cpu_left) <= 0 or used_edges >= limits.max_edges or used_records >= limits.max_records:
                    break
                if p.limits.max_aux is not None and p.r*p.m+(p.m if side=='column' else p.r)>p.limits.max_aux:
                    continue
                result = CleanupSearch(p, replace(limits, wall_seconds=left, cpu_seconds=cpu_left,
                                                      max_edges=limits.max_edges-used_edges,
                                                      max_records=limits.max_records-used_records),
                                       sides=(side,), cancel=cancel).run(policy, scheduler='hierarchy' if policy else 'untrained')
                trace.append(result)
                used_edges += result['edges']
                used_records += result['records']
                if result['protocol']:
                    circuit = Protocol.from_payload(result['protocol'])
                    checked = result['verification']
                    learned = policy is not None
                    break
        if not checked['valid']:
            return {'status': 'unknown', 'reason': 'analytic schedule violates a cap; no exclusion proved',
                    'construction_failure': checked, 'search': trace, 'lower_bound': lower,
                    'global_oracle_optimality': False, 'wall_seconds': time.perf_counter()-started}
        certificate = workspace_certificate(p, circuit)
        if not verify_workspace_certificate(p, circuit, certificate)['valid']:
            raise AssertionError('invalid materialization receipt')
        return {'status': 'optimal_workspace_within_architecture' if certificate['optimal_within_architecture'] else 'upper_bound',
                'protocol': circuit.payload(), 'verification': checked, 'certificate': certificate,
                'lower_bound': lower, 'search': trace, 'learned_discovery': learned,
                'policy_digest': before, 'global_oracle_optimality': False,
                'wall_seconds': time.perf_counter()-started}
    except Interrupted:
        return {'status': 'unknown', 'reason': 'cancelled', 'global_oracle_optimality': False,
                'wall_seconds': time.perf_counter()-started}
    finally:
        if policy is not None and policy.digest != before:
            raise AssertionError('evaluation changed a frozen policy')


def nondominated(entries: list[dict]) -> list[dict]:
    def dominates(a, b):
        x, y = a['verification']['resources'], b['verification']['resources']
        return all(x[k] <= y[k] for k in RESOURCE_KEYS) and any(x[k] < y[k] for k in RESOURCE_KEYS)
    return [a for a in entries if not any(dominates(b, a) for b in entries)]


def synthesize_oracle(p: CleanupProblem, policy=None, *, objective=DEFAULT_OBJECTIVE,
                      include_learned: bool = False, max_factors: int = 16,
                      seconds: float = 10., cancel: Callable[[], bool] | None = None) -> dict:
    """Choose complete oracles, not an obligatorily materialized intermediate bank.

    Always compare full-bank, bank-free and exact consumer-factor candidates.
    Optional SARSA/LinUCB search uses the same minimum-width full-bank interface,
    and is labelled separately. The selected circuit may still be analytic.
    All preprocessing and candidate verification is charged to wall time.
    """
    _frozen(policy)
    if (not objective or len(set(objective)) != len(objective) or
            any(k not in RESOURCE_KEYS for k in objective)):
        raise ValueError('declare distinct supported lexicographic resource coordinates')
    if not math.isfinite(seconds):
        raise ValueError('time budget must be finite')
    if seconds <= 0:
        return {'status': 'unknown', 'reason': 'zero time budget', 'candidates': [], 'global_oracle_optimality': False}
    if include_learned and policy is None:
        raise ValueError('learned candidate requires a completed frozen policy')
    started = time.perf_counter()
    candidates: list[dict] = []
    failures: list[dict] = []
    before = None if policy is None else policy.digest
    partial = False

    def stopped():
        return (cancel is not None and cancel()) or time.perf_counter()-started >= seconds

    def add(name, circuit, receipt=None, learned=False):
        check_cancel(stopped)
        checked = verify_phase(p, circuit, cancel=stopped)
        if not checked['valid']:
            failures.append({'method': name, 'reason': checked['reason']})
            return
        if receipt is not None and not verify_factor_receipt(p, circuit, receipt):
            raise AssertionError('invalid rank/streaming certificate')
        # Retain actual hybrid-block DAG witness, not only a formula for costs.
        circuit.persistent_tail()
        candidates.append({'method': name, 'protocol': circuit.payload(), 'verification': checked,
                           'receipt': receipt, 'learned_discovery': learned})

    try:
        # Existing complete constructions are retained as explicit controls.
        unconstrained = replace(p, limits=Limits())
        for side in ('smaller', 'row' if p.m <= p.r else 'column'):
            check_cancel(stopped)
            c = compile_deterministic(unconstrained, side=side)
            add('full_bank_' + side, _phase(p, c, 'analytic_full_bank_' + side))
        check_cancel(stopped)
        c = compile_bank_free(unconstrained)
        add('bank_free_monomials', _phase(p, c, 'deterministic_bank_free_monomials'))
        for mode in ('all', 'stream', 'recompute'):
            check_cancel(stopped)
            c, receipt = compile_factors(p, storage=mode, max_factors=max_factors, cancel=stopped)
            add('rank_' + mode, c, receipt)
        if include_learned:
            check_cancel(stopped)
            remaining = seconds - (time.perf_counter()-started)
            side = 'column' if p.m <= p.r else 'row'
            # Do not rediscover the analytically inferior helper count. This
            # candidate explores schedules, not the selection of bank-free form.
            tight = p.with_aux_cap(p.workspace_lower_bound)
            if p.limits.max_aux is None or p.limits.max_aux >= p.workspace_lower_bound:
                result = CleanupSearch(tight, WorkLimits(4096, 20000, remaining, remaining),
                                       sides=(side,), cancel=stopped).run(policy)
                if result['protocol']:
                    c = Protocol.from_payload(result['protocol'])
                    add('learned_full_bank_schedule', _phase(p, c, 'frozen_SARSA_LinUCB_schedule'), learned=True)
                    if candidates and candidates[-1]['method'] == 'learned_full_bank_schedule':
                        candidates[-1]['search'] = {k: result[k] for k in ('edges', 'records', 'profile', 'allocations', 'policy_digest')}
                else:
                    failures.append({'method': 'learned_full_bank_schedule', 'reason': result['reason']})
            else:
                failures.append({'method': 'learned_full_bank_schedule', 'reason': 'full bank exceeds workspace cap'})
    except Interrupted:
        partial = True
    finally:
        if policy is not None and policy.digest != before:
            raise AssertionError('evaluation changed the frozen policy')
    if not candidates:
        return {'status': 'unknown', 'reason': 'no verified feasible candidate', 'candidates': [],
                'failures': failures, 'menu_complete': not partial, 'global_oracle_optimality': False,
                'wall_seconds': time.perf_counter()-started}
    frontier = nondominated(candidates)
    selected = min(frontier, key=lambda c: (tuple(c['verification']['resources'][k] for k in objective), c['method']))
    return {'schema': 'consumer-aware-synthesis-v1', 'status': 'certified_upper_bound',
            'problem': p.manifest(), 'logical_contract': 'exact complete phase oracle; full bank not required',
            'objective': list(objective), 'selected': selected, 'candidates': candidates,
            'pareto_methods': [c['method'] for c in frontier], 'failures': failures,
            'menu_complete': not partial, 'policy_digest': before,
            'selected_is_learned': selected['learned_discovery'],
            'global_oracle_optimality': False, 'wall_seconds': time.perf_counter()-started,
            'timely': time.perf_counter()-started <= seconds,
            'scope': 'best verified completed candidate in declared finite menu; not unrestricted optimality'}
