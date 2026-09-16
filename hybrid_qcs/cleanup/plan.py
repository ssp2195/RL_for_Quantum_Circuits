"""Deterministic, guarded protocol transitions and compiler controls.

The finite action grammar includes certified AND isometries, individual consumer
phases, and all-outcome cleanup batches. It is an explicit structured synthesis
domain; completeness is NOT asserted for arbitrary quantum instruments.
"""
from __future__ import annotations
from dataclasses import dataclass, replace

from .contract import CleanupProblem, Layout
from .ir import Instruction, Protocol

FAMILIES = ('layout', 'helper', 'product', 'consumer', 'bank_cleanup', 'helper_cleanup')


@dataclass(frozen=True, order=True)
class Action:
    family: str
    index: int = 0

    def __post_init__(self):
        if self.family not in FAMILIES or type(self.index) is not int or self.index < 0:
            raise ValueError('invalid protocol action')

    def payload(self): return [self.family, self.index]


@dataclass(frozen=True)
class Plan:
    side: str | None = None
    helpers: int = 0
    products: int = 0
    consumers: int = 0
    cleanup_stage: int = 0

    @property
    def terminal(self): return self.cleanup_stage == 2

    def layout(self, problem):
        if self.side is None: raise ValueError('layout not selected')
        return Layout(problem.r, problem.m, self.side)

    def progress(self, p):
        k = min(p.r, p.m) if self.side is None else self.layout(p).k
        denominator = k + p.r*p.m + len(p.consumer.terms) + 3
        return (int(self.side is not None)+self.helpers.bit_count()+self.products.bit_count()+
                self.consumers.bit_count()+self.cleanup_stage)/denominator


def eligible(p: CleanupProblem, plan: Plan, sides=('row', 'column')) -> tuple[Action, ...]:
    if any(s not in ('row','column') for s in sides) or not sides: raise ValueError('invalid layout domain')
    if plan.terminal: return ()
    if plan.side is None:
        return tuple(Action('layout', int(s == 'column')) for s in sides
                     if p.limits.max_aux is None or p.r*p.m+Layout(p.r,p.m,s).k <= p.limits.max_aux)
    l = plan.layout(p)
    if plan.products != (1 << (p.r*p.m))-1:
        hs = [Action('helper', j) for j in range(l.k) if not plan.helpers >> j & 1]
        fs = [Action('product', j) for j in range(p.r*p.m)
              if not plan.products >> j & 1 and plan.helpers >> l.helper_for(j) & 1]
        return tuple(hs+fs)
    if plan.consumers != (1 << len(p.consumer.terms))-1:
        return tuple(Action('consumer', j) for j in range(len(p.consumer.terms)) if not plan.consumers >> j & 1)
    return (Action('bank_cleanup' if plan.cleanup_stage == 0 else 'helper_cleanup'),)


def transition(p: CleanupProblem, plan: Plan, action: Action, *, mode='measured', sides=('row','column')) -> tuple[Plan, tuple[Instruction, ...]]:
    if mode not in ('measured','coherent') or action not in eligible(p, plan, sides):
        raise ValueError('ineligible protocol continuation')
    family, k = action.family, action.index
    if family == 'layout': return replace(plan, side='column' if k else 'row'), ()
    l = plan.layout(p)
    if family == 'helper':
        return replace(plan, helpers=plan.helpers | (1 << k)), (Instruction('AND', l.helper_operands(k), 'prepare'),)
    if family == 'product':
        return replace(plan, products=plan.products | (1 << k)), (Instruction('AND', l.product_operands(k), 'prepare'),)
    if family == 'consumer':
        term = p.consumer.terms[k]
        op = Instruction('MINUS', (0,), 'consumer') if not term else Instruction('Z' if len(term) == 1 else 'CZ', tuple(l.wire(s) for s in term), 'consumer')
        return replace(plan, consumers=plan.consumers | (1 << k)), (op,)
    bank = family == 'bank_cleanup'
    wires = l.products if bank else l.helpers
    operands = l.product_operands if bank else l.helper_operands
    if mode == 'coherent':
        stage = 'uncompute_products' if bank else 'uncompute_helpers'
        ops = tuple(Instruction('UNAND', operands(j), stage) for j in reversed(range(len(wires))))
    else:
        offset = 0 if bank else p.r*p.m
        stage = 'products' if bank else 'helpers'
        ops = tuple(Instruction('MX', (q,), 'measure_'+stage, outcome=offset+j) for j, q in enumerate(wires))
        ops += tuple(Instruction('CZ', operands(j)[:2], 'correct_'+stage, guard=offset+j) for j in range(len(wires)))
    return replace(plan, cleanup_stage=plan.cleanup_stage+1), ops


def compile_deterministic(p: CleanupProblem, *, side='smaller', primitive='and4', mode='measured') -> Protocol:
    """Strong, explicit known-construction control; no learning or hidden target word.

    This uses the same action grammar as search. Its cheaper helper choice is
    deliberately NOT withheld from the untrained baseline.
    """
    if side == 'smaller': side = 'column' if p.m <= p.r else 'row'
    plan = Plan(); ops = []
    while not plan.terminal:
        actions = eligible(p, plan, (side,))
        if not actions: raise ValueError('deterministic construction exceeds available workspace')
        action = actions[0]
        plan, emitted = transition(p, plan, action, mode=mode, sides=(side,))
        ops.extend(emitted)
    return Protocol(p.digest, plan.layout(p), primitive, mode, tuple(ops), 'deterministic_constructive_control')
