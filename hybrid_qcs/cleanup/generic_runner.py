"""Reproducible comparison on existing phase-oracle problems and unitary controls.

The generic phase compiler targets diagonal sign oracles. Toffoli, QFT-3 and
4-wire SWAP are deliberately evaluated as separate non-diagonal native controls;
the new phase-oracle theorems do not claim to synthesize arbitrary unitaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..ancilla_benchmarks import (
    qft3_clean_ancilla_witness, swap_decomposition, toffoli_decomposition,
)
from ..native_exact import ExactMatrix, ONE, ZERO
from .benchmarks import candidate_problems
from .consumer_search import synthesize_oracle
from .general_boolean import BooleanPhaseProblem, compile_boolean_phase
from .phase_protocol import verify_phase


def _result(p: BooleanPhaseProblem, mode: str) -> dict:
    choices = (mode,) * sum(m.bit_count() >= 3 for m in p.target_polynomial)
    circuit = compile_boolean_phase(p, choices)
    checked = verify_phase(p, circuit)
    if not checked['valid']:
        raise AssertionError((p.name, mode, checked))
    return checked['resources']


def _unitary_controls() -> dict:
    # Exact native witnesses already present in the project; no synthesis
    # success or global optimality is attributed to the generic phase compiler.
    items = {
        '3-Toffoli': (3, toffoli_decomposition(0, 1, 2)),
        '3-QFT': (4, qft3_clean_ancilla_witness()),
        '4-SWAP': (4, swap_decomposition(0, 3)),
    }
    result = {}
    for name, (width, gates) in items.items():
        matrix = ExactMatrix.identity(width)
        for gate in gates:
            matrix = matrix.apply(gate.name, gate.qubits)
        if name != '3-QFT':
            expected = []
            for i in range(1 << width):
                row = []
                for j in range(1 << width):
                    mapped = (j ^ (4 if j & 3 == 3 else 0)) if name == '3-Toffoli' else (
                        j ^ ((1 << 0) | (1 << 3)) if ((j >> 0) & 1) != ((j >> 3) & 1) else j)
                    row.append(ONE if i == mapped else ZERO)
                expected.append(tuple(row))
            if matrix != ExactMatrix(tuple(expected)):
                raise AssertionError(f'{name} native reference is not exact')
        result[name] = {
            'width': width,
            'native_gates': len(gates),
            't_count': sum(g.name in ('T', 'TDG') for g in gates),
            'cnot': sum(g.name == 'CNOT' for g in gates),
            'exact_matrix_built': True,
            'native_reference_only': True,
            'qft_exact_target_validated_by_existing_qft_tests': name == '3-QFT',
        }
    return result


def run() -> dict:
    examples = {
        'CCZ-3': BooleanPhaseProblem('CCZ-3', 3, frozenset({7})),
        'C4Z': BooleanPhaseProblem('C4Z', 4, frozenset({15})),
        'majority-3': BooleanPhaseProblem.from_truth_table('majority-3',
                       tuple(int(x.bit_count() >= 2) for x in range(8))),
    }
    individual = {name: {'measured': _result(p, 'measured'), 'direct': _result(p, 'direct')}
                  for name, p in examples.items()}
    earlier = []
    for p in candidate_problems():
        old = synthesize_oracle(p, seconds=10.)
        if not old.get('selected'):
            raise AssertionError(f'existing benchmark unexpectedly failed: {p.name}')
        generic = BooleanPhaseProblem(p.name, p.n, p.target_polynomial)
        earlier.append({
            'problem': p.name,
            'previous_selected_method': old['selected']['method'],
            'previous': old['selected']['verification']['resources'],
            'generic_measured': _result(generic, 'measured'),
            'generic_direct': _result(generic, 'direct'),
            'previous_optimality_claim': old['global_oracle_optimality'],
        })
    return {
        'schema': 'generic-boolean-phase-comparison-v1',
        'scope': 'finite candidate constructions; no global ancilla or T optimality',
        'objective': ['peak_aux', 't_count', 'cnot', 'native_depth', 'measurement_rounds'],
        'isolated_phase_examples': individual,
        'existing_five_phase_benchmarks': earlier,
        'non_diagonal_unitary_controls': _unitary_controls(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    summary = {
        'phase_examples': list(result['isolated_phase_examples']),
        'existing_benchmarks': len(result['existing_five_phase_benchmarks']),
        'unitary_controls': list(result['non_diagonal_unitary_controls']),
        'output': str(args.output),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
