"""Exact staged frontier for M1/M2-guided measurement cleanup.

Domain: a declared bank of measured cubic byproducts
  g[i,j] = x[i] * y[2*j] * y[2*j+1].
One correction stage may prepare and measure each chosen quadratic helper ONCE.
A helper cannot be measured, reset and reused in the same stage. This makes
the number of stages a meaningful resource; stage order is explicit.

M2 rejects partial records whose remaining byproduct rank exceeds the
necessary future Clifford correction capacity. M1 synthesizes the actual
conditional Z/CZ corrections for every completed stage schedule. The resulting
complete instrument is checked independently by verify_phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from functools import lru_cache
from itertools import combinations

from .contract import times
from .contextual import LiveStage, MeasuredValue, causal_corrections, necessary_peak_ancillas
from .general_boolean import BooleanPhaseProblem
from .ir import Instruction
from .phase_protocol import PhaseProtocol, verify_phase


@dataclass(frozen=True)
class StageRecord:
    """Persistent decision path: one immutable helper mask per correction stage."""
    selected: tuple[int, ...]
    remaining_mask: int


def _popcount(mask: int) -> int:
    return mask.bit_count()


def _subsets(remaining: int, maximum: int) -> tuple[int, ...]:
    """Enumerate the entire legal helper-subset grammar, without a heuristic."""
    indices = tuple(i for i in range(remaining.bit_length()) if remaining & (1 << i))
    return tuple(sum(1 << i for i in group)
                 for size in range(1, min(maximum, len(indices)) + 1)
                 for group in combinations(indices, size))


def _m3_values(q: int):
    x = tuple(frozenset({1 << i}) for i in range(q))
    y = tuple(frozenset({1 << (q+i)}) for i in range(2*q))
    quadratic = tuple(times(y[2*j], y[2*j+1]) for j in range(q))
    return x, y, quadratic


def _pending(q: int, remaining: int) -> tuple[MeasuredValue, ...]:
    x, _, quadratic = _m3_values(q)
    return tuple(MeasuredValue(times(x[i], quadratic[j]), 0)
                 for j in range(q) if remaining & (1 << j)
                 for i in range(q))


def compile_staged_corrections(q: int, stages: tuple[int, ...]) -> tuple[BooleanPhaseProblem, PhaseProtocol]:
    """Synthesize a complete identity-phase instrument and its M1 correction.

    The measured garbage is PHYSICALLY created first (not an assumed resource
    free operation). Its preparation cost is included in resource accounting.
    Correction-stage ancilla occupancy is reported separately from total peak.
    """
    if type(q) is not int or not 1 <= q <= 6 or not stages or any(
        type(mask) is not int or mask <= 0 or mask >= 1 << q for mask in stages
    ):
        raise ValueError('invalid q or correction-stage helper mask')
    seen = 0
    for mask in stages:
        if mask & seen:
            raise ValueError('a helper feature cannot be prepared in two stages')
        seen |= mask
    if seen != (1 << q) - 1:
        raise ValueError('every measured garbage feature must be corrected')

    n = 3*q
    x, y, quadratic = _m3_values(q)
    initial = LiveStage(x+y)
    outcomes = _pending(q, (1 << q)-1)
    contexts = tuple(LiveStage(initial.values+tuple(
        quadratic[j] for j in range(q) if mask & (1 << j))) for mask in stages)
    correction = causal_corrections(contexts, outcomes)
    if correction is None:
        raise ValueError('M1: no causal correction in the selected live contexts')

    peak_correction = max(_popcount(mask) for mask in stages)
    width = n + max(2, peak_correction)
    ops: list[Instruction] = []
    # Generate all q^2 cubic garbage measurements. Every temporary is
    # prepared by AND, then X-measured; no garbage cost is suppressed.
    for j in range(q):
        left, right = q+2*j, q+2*j+1
        ops.append(Instruction('AND', (left, right, n), 'garbage_prepare'))
        for i in range(q):
            ops.append(Instruction('AND', (i, n, n+1), 'garbage_prepare'))
            ops.append(Instruction('MX', (n+1,), 'garbage_measure', outcome=j*q+i))
        ops.append(Instruction('MX', (n,), 'source_helper_measure', outcome=q*q+j))
        ops.append(Instruction('CZ', (left, right), 'source_helper_correct', guard=q*q+j))

    by_stage = [[] for _ in stages]
    for gate in correction:
        by_stage[gate.stage].append(gate)
    extra_outcome = q*q+q
    for stage_index, mask in enumerate(stages):
        features = tuple(j for j in range(q) if mask & (1 << j))
        for slot, j in enumerate(features):
            ops.append(Instruction('AND', (q+2*j,q+2*j+1,n+slot),
                                   f'correction_prepare_{stage_index}'))
        physical = tuple(range(n)) + tuple(n+k for k in range(len(features)))
        for gate in by_stage[stage_index]:
            ops.append(Instruction(gate.kind, tuple(physical[i] for i in gate.wires),
                                   f'causal_m1_stage_{stage_index}',
                                   guard=gate.measurement))
        for slot, j in enumerate(features):
            ops.append(Instruction('MX', (n+slot,),f'correction_measure_{stage_index}',
                                   outcome=extra_outcome))
            ops.append(Instruction('CZ',(q+2*j,q+2*j+1),
                                   f'correction_reset_{stage_index}',
                                   guard=extra_outcome))
            extra_outcome += 1
    problem = BooleanPhaseProblem(
        f'm3-garbage-q{q}-stages{len(stages)}', n, frozenset())
    circuit = PhaseProtocol(problem.digest, width, tuple(ops),
                            'M1_generated_M2_pruned_staged_cleanup')
    return problem, circuit


class CausalCorrectionFrontier:
    """Exhaust all legal stage partitions; M2 prunes only proven impossibility."""

    def __init__(self, q: int, max_stages: int, peak_correction_ancillas: int,
                 *, m2_enabled: bool = True, max_records: int = 50000):
        if any(type(v) is not int for v in (q, max_stages, peak_correction_ancillas, max_records)):
            raise ValueError('integer search bounds required')
        if not (1 <= q <= 6 and 1 <= max_stages <= q and
                0 <= peak_correction_ancillas <= q and max_records >= 1):
            raise ValueError('invalid declared correction frontier bounds')
        self.q = q
        self.max_stages = max_stages
        self.cap = peak_correction_ancillas
        self.m2_enabled = m2_enabled
        self.max_records = max_records
        self.baseline = LiveStage(_m3_values(q)[0]+_m3_values(q)[1])

    def run(self) -> dict:
        frontier = deque([StageRecord((), (1 << self.q)-1)])
        @lru_cache(maxsize=None)
        def suffix_m2(remaining: int, stages_left: int) -> int:
            return necessary_peak_ancillas(
                self.baseline, _pending(self.q,remaining),stages_left)
        generated = 1
        expanded = edges = m2_pruned = 0
        solutions: list[dict] = []
        while frontier:
            record = frontier.popleft()
            remaining_stages = self.max_stages - len(record.selected)
            if record.remaining_mask == 0:
                problem, circuit = compile_staged_corrections(self.q,record.selected)
                checked = verify_phase(problem,circuit)
                if not checked['valid']:
                    raise AssertionError('independent verification rejected M1 circuit: '+checked['reason'])
                circuit.persistent_tail()
                solutions.append({
                    'schedule': record.selected,
                    'stages_used':len(record.selected),
                    'peak_correction_ancillas':max(mask.bit_count() for mask in record.selected),
                    'resources':checked['resources'],
                    'protocol_digest':circuit.digest,
                    'exact_all_branches':checked['all_inputs_and_outcomes'],
                })
                continue
            if remaining_stages <= 0:
                continue
            if self.m2_enabled and (
                suffix_m2(record.remaining_mask,remaining_stages) > self.cap
            ):
                m2_pruned += 1
                continue
            expanded += 1
            for subset in _subsets(record.remaining_mask,self.cap):
                next_record = StageRecord(record.selected+(subset,),
                                          record.remaining_mask ^ subset)
                frontier.append(next_record)
                generated += 1
                edges += 1
                if generated > self.max_records:
                    raise ValueError('explicit finite search record budget exceeded')
        selected = min(solutions,key=lambda s: (
            s['resources']['peak_aux'],
            s['resources']['t_count'],s['resources']['cnot'],
            s['resources']['native_depth'],s['schedule'])) if solutions else None
        return {
            'status':'certified_finite_stage_optimum' if selected else 'infeasible_in_stage_grammar',
            'q':self.q,'max_stages':self.max_stages,
            'max_correction_ancillas':self.cap,'m2_enabled':self.m2_enabled,
            'generated_records':generated,'expanded_records':expanded,
            'expanded_edges':edges,'m2_pruned_records':m2_pruned,
            'verified_schedules':len(solutions),'selected':selected,
            'm2_rank_checks':suffix_m2.cache_info().misses,
            'global_oracle_optimality':False,
            'stage_model':'one simultaneous helper preparation/measurement cycle per stage',
        }
