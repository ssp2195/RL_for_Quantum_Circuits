"""Proof-carrying resource optimization over authoritative HybridState records.

Discovery, deterministic audit, and certificate checking have separate provenance
and share one cumulative work ledger. No phase-obligation solver is called.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import time
from typing import Callable

from .model import Gate, HybridState
from .native_domain import SCHEMA, NativeProblem, certify_native, legal, next_t_depths, digest
from .native_audit import audit_native, verify_native_cover
from .resource_search import WorkLimits, WorkMeter

OPT_SCHEMA = 'native-hybrid-optimization-v1'
OBJECTIVES = ('t_count', 'cnot', 'depth', 't_depth', 'gates')
DEFAULT_ORDER = ('t_count', 'cnot', 'depth', 'gates')
OPTIMAL = ('optimal_under_numerical_contract', 'optimal_nonnegative_resource_bound')


def _order(objectives):
    order = tuple(objectives)
    if not order or len(set(order)) != len(order) or any(o not in OBJECTIVES for o in order):
        raise ValueError('objectives must be distinct native resource names')
    return order


class _Ledger:
    def __init__(self, limits, cancel=None):
        self.clock = WorkMeter(limits, cancel)
        self.limits, self.records = limits, 0
        self.phase_edges = {'discovery': 0, 'audit': 0, 'verification': 0}
        self.phase_seconds = {k: 0. for k in self.phase_edges}

    def reason(self):
        return self.clock.reason() or ('record_limit' if self.records >= self.limits.max_records else None)

    def remaining(self, quota=None):
        edges = max(0, self.limits.max_edges - self.clock.edges)
        return WorkLimits(min(edges, quota) if quota is not None else edges,
                          max(1, self.limits.max_records - self.records),
                          max(0., self.limits.wall_seconds - self.clock.wall),
                          max(0., self.limits.cpu_seconds - self.clock.cpu))

    def charge(self, phase, edges, records, seconds):
        self.clock.edges += edges
        self.records += records
        self.phase_edges[phase] += edges
        self.phase_seconds[phase] += seconds

    def report(self):
        return {'edges': self.clock.edges, 'records': self.records,
                'phase_edges': dict(self.phase_edges), 'phase_seconds': dict(self.phase_seconds),
                'wall_seconds': self.clock.wall, 'cpu_seconds': self.clock.cpu,
                'limits_are_cooperative': True,
                'record_accounting': 'cumulative admitted search records and checked cover labels'}


def _replay(problem, witness, *, provenance=None, cancel=None):
    """Never trust stored resource counts or a witness's success flag."""
    state = HybridState.identity(problem.width, problem.budget)
    td = (0,) * problem.width
    for name, qubits in witness['native']:
        if cancel is not None and cancel():
            raise InterruptedError('cancelled')
        gate = Gate(name, tuple(qubits))
        if gate not in problem.actions or not legal(problem, state, td, gate):
            raise ValueError('witness violates the native synthesis domain')
        state = state.apply(gate, partial_order_reduction=False)
        td = next_t_depths(td, gate)
    certificate = certify_native(problem, state, provenance=provenance or witness.get('source', 'independent_replay'))
    if not certificate['success']:
        raise ValueError('native witness does not implement the target contract')
    return certificate


def optimize_native_resources(
    problem: NativeProblem, model=None, *, objectives=DEFAULT_ORDER,
    limits=WorkLimits(50000, 100000, 60., 60.), scheduler='hierarchy',
    discovery_edges=512, audit_edges=10000, verification_edges=20000,
    audit=True, allow_audit_witness=True, max_rounds=128,
    cancel: Callable[[], bool] | None = None,
):
    """Minimize resources lexicographically within one fixed ancilla contract.

    Later objectives are entered only after the current minimum is proved. A
    numerical-domain optimum requires a fresh native witness and a checked
    exclusion at k-1 (or the universal nonnegative-resource bound when k=0).
    Audit-discovered circuits are explicitly attributed, never RL successes.
    """
    from .native_search import NativeSearch

    order = _order(objectives)
    for name, value in (('discovery_edges', discovery_edges), ('audit_edges', audit_edges),
                        ('verification_edges', verification_edges), ('max_rounds', max_rounds)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if scheduler not in ('hierarchy', 'untrained', 'greedy', 'outer', 'inner', 'cost'):
        raise ValueError('unsupported native scheduler')
    if scheduler in ('hierarchy', 'outer', 'inner') and (
            model is None or not model.frozen or model.episodes < 1):
        raise ValueError('optimization requires completed, frozen native training')
    if model is not None and not model.frozen:
        raise ValueError('optimization never trains or mutates a policy')
    policy_digest = model.digest if model is not None else None
    ledger = _Ledger(limits, cancel)
    current = problem
    incumbent = discovery_incumbent = None
    stages, journal, trace = [], [], []
    rounds = 0
    first_correct = best_at = None

    def finish(status, reason):
        if model is not None and model.digest != policy_digest:
            raise AssertionError('optimization changed a frozen checkpoint')
        return {'schema': OPT_SCHEMA, 'frontier_schema': SCHEMA, 'status': status, 'reason': reason,
                'problem': problem.manifest(), 'problem_digest': problem.digest,
                'objectives': list(order), 'objective': order[0] if len(order) == 1 else None,
                'witness': incumbent, 'discovery_incumbent': discovery_incumbent,
                'stages': stages, 'journal': journal, 'incumbent_trace': trace,
                'attempts': [j['discovery'] for j in journal if 'discovery' in j],
                'policy_digest': policy_digest, 'scheduler': scheduler,
                'audit_witness_used': any(t['source'] == 'deterministic_native_audit' for t in trace),
                'work': ledger.report(), 'limits': asdict(limits),
                'phase_quotas': {'discovery': discovery_edges, 'audit': audit_edges,
                                'verification': verification_edges},
                'time_to_first_correct': first_correct, 'time_to_best': best_at,
                'proof': stages if status in OPTIMAL else None,
                'infeasibility_certificate': journal[-1].get('certificate') if status == 'infeasible_under_numerical_contract' else None,
                'scope': 'native H/S/SDG/T/TDG/CNOT grammar; declared resource, ancilla, phase and tolerance contract',
                'formal_algebraic_proof': False}

    def unknown(reason):
        return finish('upper_bound' if incumbent is not None else 'unknown', reason)

    def remember(witness, source, trial):
        nonlocal incumbent, discovery_incumbent, first_correct, best_at
        # NativeSearch and audit_native have independently replayed this witness
        # under trial. Reject metadata inconsistent with that exact trial.
        if (not witness.get('success') or witness.get('problem_digest') != trial.digest
                or witness.get('schema') != SCHEMA):
            raise AssertionError('unbound or uncertified incumbent')
        if incumbent is not None and tuple(witness['resources'][o] for o in order) > tuple(incumbent['resources'][o] for o in order):
            raise AssertionError('incumbent worsened under lexicographic tightening')
        incumbent = dict(witness, source=source)
        if source == 'native_frontier_discovery':
            if discovery_incumbent is None or tuple(witness['resources'][o] for o in order) < tuple(discovery_incumbent['resources'][o] for o in order):
                discovery_incumbent = incumbent
        best_at = ledger.clock.wall
        if first_correct is None:
            first_correct = best_at
        trace.append({'source': source, 'trial_digest': trial.digest,
                      'wall_seconds': best_at, 'edges': ledger.clock.edges,
                      'resources': witness['resources']})

    for objective in order:
        stage = {'objective': objective, 'lower_bound': 0, 'upper_bound': None, 'proved': False}
        stages.append(stage)
        while True:
            cost = incumbent['resources'][objective] if incumbent is not None else None
            stage['upper_bound'] = cost
            if cost == 0:
                stage.update(proved=True, lower_bound=0, proof={'kind': 'nonnegative_integer_resource'})
                current = current.cap(objective, 0)
                break
            if ledger.reason():
                return unknown(ledger.reason())
            if rounds >= max_rounds:
                return unknown('round_limit')
            trial = current if cost is None else current.cap(objective, cost - 1)
            rounds += 1
            row = {'round': rounds, 'objective': objective, 'trial_digest': trial.digest,
                   'trial_problem': trial.manifest()}
            journal.append(row)
            start = time.perf_counter()
            discovered = NativeSearch(trial, ledger.remaining(discovery_edges), cancel=cancel).run(model, scheduler=scheduler)
            ledger.charge('discovery', discovered['edges'], discovered['records'], time.perf_counter() - start)
            row['discovery'] = discovered
            if discovered['witness'] is not None:
                remember(discovered['witness'], 'native_frontier_discovery', trial)
                continue
            if not audit or ledger.reason():
                return unknown(ledger.reason() or 'audit_disabled')
            start = time.perf_counter()
            audited = audit_native(trial, limits=ledger.remaining(audit_edges), cancel=cancel, verify=False)
            row['audit'] = audited
            work = audited['discovery']
            ledger.charge('audit', work['edges'], work['records'], time.perf_counter() - start)
            if audited.get('witness') is not None:
                if not allow_audit_witness:
                    return unknown('audit_found_witness_not_adopted')
                remember(audited['witness'], 'deterministic_native_audit', trial)
                continue
            if not audited.get('certificate') or ledger.reason():
                return unknown(ledger.reason() or audited.get('reason', 'audit_incomplete'))
            start = time.perf_counter()
            checked = verify_native_cover(trial, audited['certificate'], limits=ledger.remaining(verification_edges), cancel=cancel)
            ledger.charge('verification', checked['checked_edges'], checked.get('checked_records', 0), time.perf_counter() - start)
            row['verification'] = checked
            if not checked['valid']:
                return unknown('exclusion_not_verified: ' + checked['reason'])
            row['certificate'] = audited['certificate']
            if incumbent is None:
                return finish('infeasible_under_numerical_contract', 'checked native cover excludes the original bounded domain')
            stage.update(proved=True, lower_bound=cost, upper_bound=cost,
                         proof={'kind': 'checked_native_exclusion', 'trial_digest': trial.digest,
                                'certificate': audited['certificate'], 'verification': checked})
            current = current.cap(objective, cost)
            break
    zero_only = all(s['proof']['kind'] == 'nonnegative_integer_resource' for s in stages)
    return finish('optimal_nonnegative_resource_bound' if zero_only else 'optimal_under_numerical_contract',
                  'certified witness and independently checked lower bounds for every ordered objective')


def verify_native_optimization(problem, result, *, limits=WorkLimits(1000000, 200000, 60., 60.), cancel=None):
    """Replay the final witness and recheck each bound without trusting the optimizer.

    The caller supplies the original problem. Embedded replacement targets or
    resource caps cannot redefine it. This verifies the numerical domain, not
    exact algebraic equality to an arbitrary floating-point target matrix.
    """
    ledger = _Ledger(limits, cancel)
    def fail(reason):
        return {'valid': False, 'optimality_verified': False, 'reason': reason, 'work': ledger.report()}
    try:
        if result['schema'] != OPT_SCHEMA or result['problem_digest'] != problem.digest or digest(result['problem']) != problem.digest:
            return fail('optimization domain mismatch')
        order = _order(result['objectives'])
        if result['status'] == 'infeasible_under_numerical_contract':
            if result.get('witness') is not None:
                return fail('infeasibility result also contains a witness')
            check = verify_native_cover(problem, result['infeasibility_certificate'], limits=ledger.remaining(), cancel=cancel)
            return {'valid': check['valid'], 'optimality_verified': False, 'infeasibility_verified': check['valid'], 'verification': check}
        witness = result.get('witness')
        if witness is None:
            return fail('no witness or checked infeasibility certificate')
        if ledger.reason() or len(witness['native']) > ledger.remaining().max_edges:
            return fail(ledger.reason() or 'witness verification edge limit')
        start = time.perf_counter()
        actual = _replay(problem, witness, cancel=lambda: bool(ledger.reason()))
        ledger.charge('verification', len(witness['native']), 1, time.perf_counter() - start)
        if ledger.reason():
            return fail(ledger.reason())
        if actual['resources'] != witness['resources']:
            return fail('stored witness resource counts differ from replay')
        current, proved = problem, 0
        for objective, stage in zip(order, result['stages']):
            if stage['objective'] != objective:
                return fail('objective order mismatch')
            if not stage['proved']:
                break
            cost = actual['resources'][objective]
            if stage['lower_bound'] != cost or stage['upper_bound'] != cost:
                return fail('resource lower/upper bound mismatch')
            proof = stage['proof']
            if proof['kind'] == 'nonnegative_integer_resource':
                if cost != 0:
                    return fail('nonnegative bound cannot prove a positive optimum')
            elif proof['kind'] == 'checked_native_exclusion':
                if cost <= 0 or ledger.reason():
                    return fail(ledger.reason() or 'invalid exclusion cap')
                trial = current.cap(objective, cost - 1)
                if proof['trial_digest'] != trial.digest:
                    return fail('exclusion not bound to the k-1 subproblem')
                start = time.perf_counter()
                checked = verify_native_cover(trial, proof['certificate'], limits=ledger.remaining(), cancel=cancel)
                ledger.charge('verification', checked['checked_edges'], checked.get('checked_records', 0), time.perf_counter() - start)
                if not checked['valid']:
                    return fail(checked['reason'])
            else:
                return fail('unrecognized proof type')
            current = current.cap(objective, cost)
            proved += 1
        if len(result['stages']) > len(order):
            return fail('unexpected extra objective stages')
        optimal = proved == len(order)
        if result['status'] in OPTIMAL and not optimal:
            return fail('claimed optimum has an unproved objective')
        if result['status'] == 'optimal_nonnegative_resource_bound' and any(actual['resources'][o] != 0 for o in order):
            return fail('false zero-cost optimum status')
        if result['status'] not in (*OPTIMAL, 'upper_bound'):
            return fail('unsupported feasible-result status')
        return {'valid': True, 'witness_verified': True, 'optimality_verified': optimal,
                'proved_objectives': list(order[:proved]), 'work': ledger.report(),
                'formal_algebraic_proof': False}
    except (KeyError, ValueError, TypeError, IndexError, AssertionError, InterruptedError):
        return fail('malformed, invalid or interrupted optimization certificate')


def optimize_native_ancillas(problem, model=None, *, ancilla_budgets=(0, 1, 2), **kwargs):
    """Separate clean-width archives; prove a minimum only by excluding all smaller widths.

    ``limits`` is PER WIDTH and is reported as such. Each width retains the same
    logical unitary and non-width resource caps. No cross-width state is merged.
    """
    from .ancilla_contract import AncillaContract
    widths = tuple(sorted(ancilla_budgets))
    if not widths or any(type(a) is not int or a < 0 for a in widths) or len(set(widths)) != len(widths):
        raise ValueError('clean ancilla budgets must be distinct nonnegative integers')
    if problem.contract.borrowed_ancillas:
        raise ValueError('the width sweep supports clean ancillas only')
    n = len(problem.contract.logical_qubits)
    if n + max(widths) > 8:
        raise ValueError('width sweep exceeds the native small-register limit')
    rows = []
    for a in widths:
        contract = AncillaContract(n+a, tuple(range(n)), tuple(range(n, n+a)), phase_mode=problem.contract.phase_mode)
        p = replace(problem, contract=contract, name=f'{problem.name}-clean-{a}')
        rows.append({'available_ancillas': a, 'result': optimize_native_resources(p, model, **kwargs)})
    feasible = [r['available_ancillas'] for r in rows if r['result']['witness'] is not None]
    excluded = {r['available_ancillas'] for r in rows if r['result']['status'] == 'infeasible_under_numerical_contract'}
    smallest = min(feasible) if feasible else None
    proved = smallest is not None and all(a in excluded for a in range(smallest))
    return {'schema': 'native-hybrid-width-optimization-v1', 'results': rows,
            'smallest_feasible_tested_budget': smallest,
            'minimum_clean_ancillas': smallest if proved else None,
            'minimum_proved': proved, 'limits_scope': 'per width; total is the sum of width ledgers',
            'scope': 'same logical unitary, resource caps, phase and numerical acceptance contract'}
