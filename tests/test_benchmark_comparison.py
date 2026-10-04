"""Circuit-generating comparison tests, not post-training RL claims."""
import json
from pathlib import Path
import pytest
from hybrid_qcs.cleanup.benchmark_comparison import (Network, input_network, lower_network,
    parity_pair, extension_problems, candidate_problems)
from hybrid_qcs.cleanup.contract import CleanupProblem, Consumer
from hybrid_qcs.cleanup.phase_protocol import verify_phase
from hybrid_qcs.cleanup.verify import verify_small_matrix, verify_truth_table_generators
from hybrid_qcs.cleanup.exact_t_comparison import certify_evaluator, parse_qasm, wrapper_qasm, counts


@pytest.mark.parametrize('problem',candidate_problems(),ids=lambda p:p.name)
@pytest.mark.parametrize('form',('source','anf','davio'))
def test_source_forms_have_identical_boolean_phase(problem,form):
    n=input_network(problem,form)
    assert Network.parse(n.text())==n
    assert n.polynomial()==problem.target_polynomial
    assert 'module oracle(' in n.verilog()


@pytest.mark.parametrize('problem',extension_problems(),ids=lambda p:p.name)
def test_new_comparison_oracles_from_boolean_network(problem):
    n=input_network(problem,'davio')
    for parallel in (True,False):
        circuit=lower_network(problem,n,parallel=parallel)
        checked=verify_phase(problem,circuit)
        assert checked['valid']
        assert checked['resources']['cleanup_t']==0
        assert verify_truth_table_generators(problem,circuit)['valid']
        assert circuit.qasm3().startswith('OPENQASM 3.0;')


@pytest.mark.parametrize('ca',(0,1))
@pytest.mark.parametrize('cb',(0,1))
def test_complemented_fanins_and_entire_native_branches(ca,cb):
    # (a+ca)(b+cb)x = abx + ca*bx + cb*ax + ca*cb*x.
    terms=[('f0_0',)]
    if ca:terms.append(('b0','x0'))
    if cb:terms.append(('a','x0'))
    if ca and cb:terms.append(('x0',))
    p=CleanupProblem('complemented',1,1,Consumer(tuple(terms)))
    net=Network(3,(('A',2+ca,4+cb),('A',8,6)),10)
    circuit=lower_network(p,net)
    assert verify_phase(p,circuit)['valid']
    assert verify_small_matrix(p,circuit)['valid']


def test_no_unnecessary_predicate_output_qubit():
    p=CleanupProblem('quadratic',1,1,Consumer((('a','b0'),)))
    net=Network(3,(('A',2,4),),8)
    c=lower_network(p,net)
    r=verify_phase(p,c)
    assert r['valid'] and r['resources']['t_count']==r['resources']['peak_aux']==0


def test_exhaustive_two_parity_coordinates():
    for a in range(1,8):
        for b in range(1,8):
            if a==b:continue
            u=frozenset(i for i in range(3) if a>>i&1)
            v=frozenset(i for i in range(3) if b>>i&1)
            i,j,cxs=parity_pair(u,v)
            for x in range(8):
                bits=[x>>q&1 for q in range(3)]
                for c,t in cxs:bits[t]^=bits[c]
                assert bits[i]==(a&x).bit_count()%2
                assert bits[j]==(b&x).bit_count()%2
                for c,t in reversed(cxs):bits[t]^=bits[c]
                assert bits==[x>>q&1 for q in range(3)]


@pytest.mark.parametrize('text',('3 1 8\nA 2 8\n','3 1 8\nY 2 4\n','3 2 8\nA 2 4\n','3 0 18\n'))
def test_reject_invalid_external_network(text):
    with pytest.raises(ValueError):Network.parse(text)


def test_wrong_external_function_is_not_accepted():
    p=candidate_problems()[0]
    with pytest.raises(ValueError):lower_network(p,Network(p.n,(),0))


def test_sample_manifest_is_reproducible_and_unique():
    ps=extension_problems()
    assert len(ps)==36 and len({p.oracle_digest for p in ps})==36
    assert not {p.oracle_digest for p in ps}&{p.oracle_digest for p in candidate_problems()}
    assert [p.digest for p in ps]==[p.digest for p in extension_problems()]


def test_exact_external_wrapper_checks_every_input_not_one_branch():
    p=CleanupProblem('zero',1,1,Consumer(()))
    q='OPENQASM 2.0;\ninclude "qelib1.inc";\n// input_qubits: 0,1,2\n// output_qubits: 3\nqreg q[4];\nh q[0];\nh q[0];\n'
    assert certify_evaluator(p,q)['valid']
    wrong=q+'cx q[0],q[3];\n'
    assert not certify_evaluator(p,wrong)['valid']
    wrapper,word=wrapper_qasm(q)
    assert 'z q[3];' in wrapper and counts(4,3,word)['t_count']==0


@pytest.mark.parametrize('line',('measure q[0];','rz(0.3) q[0];','cx q[0],q[0];'))
def test_unsupported_external_instructions_fail_closed(line):
    q='// input_qubits: 0,1,2\n// output_qubits: 3\nqreg q[4];\n'+line+'\n'
    with pytest.raises(ValueError):parse_qasm(q)


@pytest.mark.parametrize('text',('qreg q[4];','// input_qubits: 0,1,2\n// output_qubits: 3\n'))
def test_missing_external_mapping_is_explicit_error(text):
    with pytest.raises(ValueError):parse_qasm(text)
