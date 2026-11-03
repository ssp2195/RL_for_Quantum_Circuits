"""M1-guided conditional phase synthesis for immediately measured AND temporaries.

The independent phase verifier remains unchanged. This transformation consumes
the deterministic AND/MX/CZ grammar and derives each feedback gate from live
Boolean functions, rather than assuming the AND operands are still available.
It is exact GF(2) algebra, never a learned correctness decision.
"""
from __future__ import annotations

from .contextual import LiveStage, MeasuredValue, causal_corrections
from .contract import Polynomial, times, plus
from .ir import Instruction


def causalize_immediate_cleanup(
    ops: tuple[Instruction, ...], logical_qubits: int
) -> tuple[Instruction, ...]:
    """Replace each immediate measured-AND correction with its M1 witness.

    Input circuits must have AND preparations, MX followed by its original CZ
    feedback, and optional unguarded diagonal gates. Temporary values are ANFs.
    A deferred correction is not silently promoted to an immediate correction.
    """
    if type(logical_qubits) is not int or logical_qubits < 1:
        raise ValueError('positive logical width required')
    width = max((q for op in ops for q in op.wires), default=logical_qubits-1) + 1
    values: list[Polynomial] = [frozenset({1 << i}) for i in range(logical_qubits)]
    values += [frozenset()] * (width-logical_qubits)
    occupied: set[int] = set()
    result: list[Instruction] = []
    expected_guard: int | None = None
    for op in ops:
        if expected_guard is not None:
            if op.kind != 'CZ' or op.guard != expected_guard:
                raise ValueError('expected the original immediate CZ correction')
            expected_guard = None
            continue
        if op.kind == 'AND':
            a, b, target = op.wires
            if target < logical_qubits or target in occupied or values[target]:
                raise ValueError('AND requires a clean auxiliary wire')
            values[target] = times(values[a], values[b])
            occupied.add(target)
            result.append(op)
        elif op.kind == 'MX':
            target = op.wires[0]
            if target not in occupied or op.outcome is None:
                raise ValueError('measurement must target a live AND temporary')
            byproduct = values[target]
            occupied.remove(target)
            values[target] = frozenset()
            live = tuple(range(logical_qubits)) + tuple(sorted(occupied))
            witness = causal_corrections(
                (LiveStage(tuple(values[q] for q in live)),),
                (MeasuredValue(byproduct, 0),),
            )
            if witness is None:
                raise ValueError('M1: unavailable causal Clifford correction')
            result.append(op)
            for gate in witness:
                if gate.kind not in ('Z','CZ'):
                    raise ValueError('unsupported M1 generator')
                result.append(Instruction(gate.kind, tuple(live[w] for w in gate.wires),
                                          'correct_m1', guard=op.outcome))
            expected_guard = op.outcome
        elif op.guard is None and op.kind in ('Z','CZ','CCZ','MINUS'):
            result.append(op)
        else:
            raise ValueError('M1 cleanup expects AND, MX, and diagonal phase operations')
    if expected_guard is not None:
        raise ValueError('missing immediate correction after final measurement')
    if occupied:
        raise ValueError('unclean temporary remains after M1 rewrite')
    return tuple(result)
