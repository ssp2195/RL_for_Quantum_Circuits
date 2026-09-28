"""Complete phase-oracle protocols, outside the full-bank lower-bound domain.

Independent symbolic branch verification uses exact local cyclotomic identities.
The checker does not call factorization, scheduling, search, or a trained policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from .contract import CleanupProblem, Polynomial, plus, times, digest
from .consumer import check_cancel, Interrupted
from .ir import Instruction, Protocol, ResourceState, append_event, instructions
from .reference import ReferenceLayout
from .verify import primitive_receipts


@dataclass(frozen=True)
class PhaseProtocol:
    problem_digest: str
    width: int
    ops: tuple[Instruction, ...]
    provenance: str
    primitive: str = 'and4'

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1 or self.primitive not in ('and4', 'ccx7'):
            raise ValueError('invalid phase protocol')
        if not self.provenance or any(max(op.wires) >= self.width for op in self.ops):
            raise ValueError('invalid phase instruction or provenance')

    @property
    def layout(self):
        return ReferenceLayout(self.width)

    @property
    def digest(self):
        return digest(self.payload())

    def payload(self):
        return {'schema': 'consumer-phase-protocol-v1', 'problem_digest': self.problem_digest,
                'width': self.width, 'primitive': self.primitive, 'provenance': self.provenance,
                'ops': [op.payload() for op in self.ops],
                'scope': 'complete phase oracle; no full-bank materialization requirement'}

    @classmethod
    def from_payload(cls, value):
        if value.get('schema') != 'consumer-phase-protocol-v1':
            raise ValueError('not a complete phase protocol')
        obj = cls(value['problem_digest'], value['width'],
                  tuple(Instruction.from_payload(o) for o in value['ops']),
                  value['provenance'], value['primitive'])
        if obj.payload() != value:
            raise ValueError('noncanonical protocol or false architecture scope')
        return obj

    def qasm3(self):
        return Protocol.qasm3(self)

    def persistent_tail(self):
        tail = None
        for op in self.ops:
            tail = append_event(tail, op, self.primitive)
        if instructions(tail) != self.ops:
            raise AssertionError('persistent circuit witness changed')
        return tail


def verify_phase(p: CleanupProblem, protocol: PhaseProtocol | dict, *,
                 cancel: Callable[[], bool] | None = None) -> dict:
    """Prove K_s J = 2^(-m/2) J O_p for every input and every transcript.

    Reversible CNOT changes of coordinates are permitted. Every AND target must
    be clean. Input wires cannot be measured. Conditional corrections may only
    use available classical bits. All bits remain internal after return.
    """
    try:
        pr = PhaseProtocol.from_payload(protocol) if isinstance(protocol, dict) else protocol
        if type(pr) is not PhaseProtocol or pr.problem_digest != p.digest or not p.n <= pr.width <= p.n + 8192:
            raise ValueError('phase contract or width mismatch')
        primitive_receipts(pr.primitive)
        values: list[Polynomial] = [frozenset({1 << i}) for i in range(p.n)]
        values += [frozenset()] * (pr.width - p.n)
        original = values[:p.n]
        phase: Polynomial = frozenset()
        byproduct: dict[int, Polynomial] = {}
        reserved: set[int] = set()
        peak = 0
        ledger = ResourceState.zero(pr.width)
        for op in pr.ops:
            check_cancel(cancel)
            if op.kind in ('AND', 'UNAND'):
                c, d, t = op.wires
                if t < p.n:
                    raise ValueError('structured AND must target an auxiliary wire')
                value = times(values[c], values[d])
                if op.kind == 'AND':
                    if values[t] or t in reserved:
                        raise ValueError('AND target is not available and clean')
                    values[t] = value
                    reserved.add(t)
                else:
                    if values[t] != value or t not in reserved:
                        raise ValueError('inverse AND promise fails')
                    values[t] = frozenset()
                    reserved.remove(t)
            elif op.kind == 'CX':
                c, t = op.wires
                if t >= p.n and t not in reserved:
                    raise ValueError('CNOT cannot allocate unaccounted auxiliary storage')
                values[t] = plus(values[t], values[c])
            elif op.kind == 'MX':
                t = op.wires[0]
                if t < p.n or t not in reserved or op.outcome in byproduct:
                    raise ValueError('invalid measurement, reset, or duplicate transcript')
                byproduct[op.outcome] = values[t]
                values[t] = frozenset()
                reserved.remove(t)
            elif op.kind in ('Z', 'CZ', 'MINUS'):
                term = (frozenset({0}) if op.kind == 'MINUS' else values[op.wires[0]]
                        if op.kind == 'Z' else times(values[op.wires[0]], values[op.wires[1]]))
                if op.guard is None:
                    phase = plus(phase, term)
                else:
                    if op.guard not in byproduct:
                        raise ValueError('correction reads an unavailable outcome')
                    byproduct[op.guard] = plus(byproduct[op.guard], term)
            else:
                raise ValueError('unsupported structured operation')
            peak = max(peak, len(reserved))
            ledger = ledger.append(op, pr.primitive, p)
        if values[:p.n] != original or any(values[p.n:]) or reserved:
            raise ValueError('logical input or auxiliary return is incorrect')
        if phase != p.target_polynomial or any(byproduct.values()):
            raise ValueError('wrong logical phase or uncancelled measurement phase')
        report = ledger.report()
        if report['peak_aux'] != peak:
            raise ValueError('independent occupancy accounting disagrees')
        if not p.limits.accepts(report):
            raise ValueError('resource contract exceeded')
        return {'valid': True, 'protocol_digest': pr.digest, 'problem_digest': p.digest,
                'resources': report, 'measurements': len(byproduct), 'branch_count': 1 << len(byproduct),
                'all_inputs_and_outcomes': True, 'global_oracle_optimality': False,
                'branch_identity': 'K_s J = 2^(-measurements/2) J O_p',
                'scope': 'exact complete phase oracle; no materialization lower bound used'}
    except (TypeError, ValueError, KeyError, IndexError, Interrupted) as exc:
        return {'valid': False, 'reason': str(exc)}


def compose_blocks(target: CleanupProblem,
                   blocks: tuple[tuple[CleanupProblem, PhaseProtocol, tuple[int, ...]], ...]) -> PhaseProtocol:
    """Reuse clean workspace between blocks; input embeddings must be injective.

    Full verification at each boundary and at the composed boundary is mandatory.
    The target phase must be the XOR of the embedded block phases. Original
    logical wires may overlap between blocks, but auxiliary wires may not survive.
    """
    ops = []
    bit_offset = 0
    auxiliary = 0
    for number, (problem, block, mapping) in enumerate(blocks):
        checked = verify_phase(problem, block)
        if not checked['valid']:
            raise ValueError('invalid component: ' + checked['reason'])
        if len(mapping) != problem.n or len(set(mapping)) != len(mapping) or any(
                type(q) is not int or not 0 <= q < target.n for q in mapping):
            raise ValueError('invalid logical block embedding')
        auxiliary = max(auxiliary, block.width - problem.n)
        wires = mapping + tuple(range(target.n, target.n + block.width - problem.n))
        for op in block.ops:
            # Non-Clifford computation stays in the named prepare/consumer
            # stages. Corrections/batches are uniquely labelled across blocks.
            stage = op.stage if op.stage in ('prepare', 'consumer') else f'block{number}_{op.stage}'
            ops.append(Instruction(op.kind, tuple(wires[q] for q in op.wires), stage,
                                   None if op.guard is None else bit_offset + op.guard,
                                   None if op.outcome is None else bit_offset + op.outcome))
        bit_offset += 1 + max((o.outcome for o in block.ops if o.outcome is not None), default=-1)
    result = PhaseProtocol(target.digest, target.n + auxiliary, tuple(ops), 'verified_clean_block_composition')
    checked = verify_phase(target, result)
    if not checked['valid']:
        raise ValueError('invalid composed oracle: ' + checked['reason'])
    return result
