"""Render claims from saved comparison records; never invoke a synthesizer."""
from pathlib import Path
import argparse
import json
import statistics


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def build(root: Path, destination: Path):
    main=json.loads((root/'benchmark_comparison_v1/summary.json').read_text())
    verified=json.loads((root/'benchmark_comparison_v1/verification.json').read_text())
    exact=json.loads((root/'exact_t_comparison_v1/results.json').read_text())
    order=['enabled-three-feature-coupling','two-channel-feature-ring','mixed-five-row-predicate','three-channel-cross-feature-bank','wide-shared-mask-oracle']
    names=['3 x 1','4 x 2','5 x 2','6 x 3','8 x 3']
    original={p['name']:p for p in main['problems'] if p['cohort']=='original'}
    original_table=table(['Product bank','T: current / XAG','Aux: current / XAG','CNOT: current / XAG','Ticks: current / XAG'],[
        [shape]+[f"{original[name]['current'][key]} / {original[name]['baseline'][key]}" for key in ('t_count','peak_aux','cnot','worst_case_ticks')]
        for shape,name in zip(names,order)])
    extension=main['comparisons']['extension']
    extension_table=table(['Resource (lower is better)','Current better','Tie','XAG better'],[
        [title]+[extension[key][v] for v in ('current_better','equal','baseline_better')]
        for key,title in [('t_count','T-count'),('peak_aux','Peak auxiliary qubits'),('cnot','CNOT count'),('native_gates','Native gate count'),('worst_case_ticks','Measurement-inclusive latency'),('measurement_rounds','Measurement rounds')]])
    exact_table=table(['Product bank','Current T','Best executed Exact-T coherent wrapper T','Wrapper auxiliaries'],[
        [shape,original[name]['current']['t_count'],min(x['wrapper_resources']['t_count'] for x in exact if x['name']==name and x['valid']),
         min([x for x in exact if x['name']==name and x['valid']],key=lambda x:x['wrapper_resources']['t_count'])['wrapper_resources']['peak_aux']]
        for shape,name in zip(names,order)])
    a=main['by_method']['current_menu']['median_ms'];b=main['by_method']['mockturtle_measured']['median_ms']
    sums={k:tuple(sum(p[v][k] for p in main['problems'] if p['cohort']=='extension') for v in ('current','baseline')) for k in ('t_count','peak_aux','worst_case_ticks')}
    text=f'''# External benchmark comparison and publication assessment

## 1. Executive conclusion

The consumer-aware construction has a real workspace advantage against the executed measured Boolean-network control, but it does not have uniform T-count or latency superiority. The five original examples give a substantially more favorable impression than the additional 36 structured examples. The experiment supports a space-versus-non-Clifford-cost-versus-latency trade-off, not a superior general-purpose quantum synthesizer and not SARSA-LinUCB superiority.

The existing synthesizer, learning rules, trained checkpoints, native hybrid representation, and historical experiments are unchanged. New code only constructs baseline inputs, invokes pinned authors' tools, translates their results under explicit contracts, checks complete circuits, and reports the comparisons. No new learning campaign or theorem has been introduced in this assessment.

Source snapshot before this assessment: consumer-aware-cleanup-20260928 at 2ca775e70ec3b6da94a7246a57874f5d6cee6d3f. Its production synthesis code is unchanged from 8509fc8dd1c095560af20bf7373af829a90cdb29; the intervening two commits prepare external tools.

## 2. What was actually compared

**Primary current method.** The existing deterministic complete-oracle menu computes the required Boolean sign phase, considers full-bank and bank-free constructions, factors separated operand features, and schedules their first and last uses. It chooses lexicographically by T-count, peak auxiliary workspace, modelled latency, native gates, and CNOT count. This is the existing default; its choices are not labelled as learned discoveries.

**Primary external control.** Unmodified mockturtle algorithms at commit 47d1e70fdf775e1a295016c3c17a1ad206db24c0 optimize XOR/AND networks. The fixed driver uses five-input cut rewriting with the minimum-multiplicative-complexity database, cut limit 16, and two rewriting/resubstitution rounds. Resubstitution uses at most eight window inputs and three inserted gates. Three independently correct source forms are supplied: the original consumer computation, expanded algebraic normal form, and a fixed positive-Davio factorization. No source form contains the current method's discovered circuit.

The optimized network is converted to a complete measured phase oracle by our explicit adapter, using the SAME verified four-T clean-target AND, parity CNOTs, X-basis measurement, reset, and conditional Clifford correction as the proposed method. Affine fanins do not require separate parity ancillas. Terminal ANDs used only in the final phase are applied directly as CZ phases on their parity operands, rather than unnecessarily computing an output flag. Both serial and parallel reverse-dependency cleanup are evaluated. The best admissible result uses the same lexicographic objective as the current method.

This is a concrete execution of authors' Boolean-optimization algorithms with a transparent, matched quantum lowering. It is NOT a reproduction of all configurations of caterpillar, an optimal XAG synthesizer, or the strongest possible XAG workspace scheduler. Its retained AND intermediates favor parallel cleanup; another schedule or recomputation can change its workspace frontier. We do not infer a universal workspace lower bound for XAG methods from this comparison.

**Supplementary external control.** The unmodified Exact-T Library mapper at commit bffe54c38b6bfd689a04e0d7d5afdbd949bdac3a is executed with `-n 6 --cut-size 5 --clean-ancilla` on all three source forms of the five original examples. Its emitted result is a function evaluator, not automatically our requested clean phase oracle. We explicitly produce the complete coherent circuit U-dagger Z-output U. Its gate counts include the entire inverse. This is a contract-conversion diagnostic, NOT a measurement-matched reproduction of the paper's complete optimized flow. It must not be used as a headline claim of beating the Exact-T Library method.

AlphaTensor-Quantum, Polytof, Reqomp, full caterpillar, and relational optimization were not executed in this assessment. Prepared PyZX wheels are not an executed PyZX comparison. No performance claim against those tools follows from this report.

## 3. Problems and experimental units

The primary experiment contains 41 labelled Boolean phase functions: the same five development examples and 36 additional synthetic functions. The latter have 12 bilinear, 12 quadratic-feature, and 12 mixed consumers, with operand shapes cycling through (3,2), (4,2), (4,3), and (5,3). They use six to nine logical inputs; the full collection uses five to twelve. The fixed sample seed is 20261004. The only exclusions are duplicate labelled Boolean functions and the absence of a non-Clifford high-degree term.

The sample is fixed independently of final solver outcomes. It is not separated by all input-permutation or affine-equivalence classes, and it is not a held-out reinforcement-learning generalization study. There is no new training. Both methods receive exactly the same final phase function and clean-output requirement. Connectivity is all-to-all. The inherited timing model assigns one tick per native gate, five per Z measurement, and two until an outcome is available; conditional reset and correction gates are counted. These are schedule estimates, not physical hardware observations.

There are three timing repetitions for each method on each function: {main['records']} records, of which {main['valid']} returned verified circuits. The statistical unit is the logical target, not a repetition. No population-level significance or state-of-the-art superiority is claimed from these descriptive comparisons.

## 4. The five original examples

{original_table}
The current construction improves T-count on three examples and ties on two. It uses fewer auxiliaries on four and ties on one. The parallel measured XAG control has lower latency on all five. Even within this favorable development corpus, the selected points are not componentwise superior.

These numbers should replace a comparison based only on the old full-bank construction when discussing external competitiveness. The earlier 48.15--80% T reductions were against that internal construction, not external state-of-the-art tools.

## 5. Additional structured examples

{extension_table}
On these 36 functions the current method uses fewer T gates in five, the same in thirteen, and more in eighteen. Its clearer strength is reduced workspace: thirty improvements and six ties. Its default T-first selections have worse latency on thirty-two examples.

For an additional transparent aggregate, sums over the 36 distinct targets are: T-count {sums['t_count'][0]} versus {sums['t_count'][1]}, auxiliary count {sums['peak_aux'][0]} versus {sums['peak_aux'][1]}, and ticks {sums['worst_case_ticks'][0]} versus {sums['worst_case_ticks'][1]} (current versus XAG). These sums are NOT the resources of one combined circuit, and do not weight targets by practical frequency.

The source of the trade-off is understandable. The current method reuses a small collection of features and cleans them after their last use. This can serialize corrections. The XAG optimizer can find nonlinear identities beyond the current fixed separated feature basis, and retains more intermediates to permit parallel measured cleanup. A minimum rank in one feature basis is not a minimum T-count over all Boolean or quantum constructions.

A latency-first selection in the current menu can choose another existing implementation, but it must be compared with a latency-first baseline as a separate experiment. We have not silently switched objectives to conceal the losses in this table.

## 6. Exact-T Library: a deliberately qualified comparison

{exact_table}
All 15 evaluator outputs were checked on all logical basis inputs using exact cyclotomic arithmetic. The checker established that the evaluator output qubit has the required Boolean value on the entire promised input subspace. Therefore Z on that qubit contributes exactly the desired sign, and the literal inverse restores the complete input and all initial zero qubits. This establishes coherent correctness by linearity, including arbitrary reference entanglement; it does not assume that the evaluator has no intermediate garbage.

The checked inputs total {sum(r.get('verification',{}).get('basis_inputs',0) for r in exact)} across the 15 jobs. The emitted full wrapper QASM is saved. Auxiliary counts are allocated non-input qubits of that wrapper, not a separately optimized live-storage schedule. All best T-count results in this table came from the Davio input form.

The difference from the current adaptive implementation includes the cost of an explicitly coherent inverse. It would be misleading to attribute the entire gap to a new theorem or to call it a fair comparison with the best measurement-enabled Exact-T compilation. The primary measured XAG comparison is the more informative control for that purpose.

## 7. Classical runtime

The median recorded current-menu time is {a:.3f} ms; the external measured-XAG workflow takes {b:.3f} ms. Their ratio is {b/a:.2f}, using pooled medians. Each time includes all candidates considered, full oracle construction, verification, and persistent coherent-block witnesses. External subprocess startup, three input constructions and optimization calls, and both cleanup schedules are included. Compiler builds, target-independent primitive preparation, training, and final output serialization are outside this timed interval. There was no training in this comparison.

This is a workflow-specific descriptive result. The methods do different amounts of classical optimization, and the external method often produces lower T-count or latency. It is not a general algorithmic speedup theorem or an equal-quality time-to-solution comparison. Future experiments should compare resource-quality-versus-time curves and a shared hard workspace/latency contract.

## 8. Correctness, testing, and provenance

Independent primary replay checked {verified['records']} recorded outcomes, {verified['distinct_protocols']} distinct returned protocols, and {verified['input_outcome_generator_pairs']:,} input/outcome-generator pairs. The last number is verification work, not that many independent experiments. The all-outcome phase coefficients are checked symbolically; independent truth tables check the selected instances. Local coherent primitives use the existing exact arithmetic, and small tests additionally evaluate every native Kraus branch. Exact-T evidence has its own sparse-column and inverse-wrapper check.

The new regression module contains 69 tests, including actual complete-oracle generation and exact measurement-branch tests. They are NOT labelled as post-training RL tests. All original 1,272 tests are retained, giving 1,341 collected tests; actual execution receipts are stored separately and must be checked before stating that the suite passed. No new policy coefficients are fitted. Historical experiments and source-theorem claims are not overwritten.

Two developmental control versions were completed and retained in the delivery: a four-cut source/ANF baseline and a five-cut/Davio baseline that still materialized the terminal Boolean output. The final control removes that unnecessary flag and supports parallel cleanup. Strengthening the control substantially weakens the apparent T-count advantage. All target functions and the proposed method remained fixed. Because the comparator was improved during inspection, this is a developmental audit, not a preregistered confirmation study. Pilot numbers are not pooled into the primary comparison.

## 9. Publication judgment

**Not supported:** a generally superior SARSA-LinUCB quantum synthesizer; globally optimal complete-oracle T-count or workspace; uniformly lower physical latency; arbitrary large-register native synthesis; or superiority over AlphaTensor-Quantum and every applicable published compiler.

The new experiment does not train or evaluate new SARSA/LinUCB coefficients. The historical same-menu-plus-learning experiment still reports algebraically constructed winners and additional policy cost. Neither frontier ordering nor a small action-value model causes the consumer factorization. A positive construction result must not be relabelled as a reinforcement-learning result.

**Supported strengths:** exact full-branch coherent correctness; explicit clean-workspace and measurement accounting; a resource-scoped converse for full-bank materialization; constructive consumer-aware alternatives; exact no-recomputation ordering for fixed factors; and a reproducible measured space/T/latency trade-off against an executed external Boolean optimizer. The production method is small and mathematically inspectable.

**Scientific limitations:** the 36 new functions are synthetic and remain inside one structural class; the feature basis and factorization are restricted; the exact scheduling recurrence is exponential in the number of factors and has a declared limit; the complete-method comparison does not establish any unrestricted optimum; the XAG control does not include all possible space-constrained schedules; the Exact-T comparison has a clearly different coherent-wrapper contract; and no hardware routing, magic-state factory, or logical-error model is tested.

**Novelty limitation:** temporary-AND termination, Boolean factoring, quadratic-form elimination, and workspace/recomputation trade-offs have substantial precedents. A finite menu's minimum is no worse than its included candidates by definition. These observations are valuable implementation and proof lemmas, not sufficient principal novelty claims. The strongest mathematical candidate remains the supplied converse over the specified helper encodings and garbage instruments, not choosing the smaller operand side. Independent proof scrutiny and a final nearest-literature comparison are still required; this benchmark does not establish priority.

## 10. Defensible claim and next research threshold

A suitable current description is:

> We implement certified consumer-aware synthesis for a specified family of Boolean phase oracles. Exact algebra avoids unnecessary temporary values, and a restricted exact lifetime schedule reduces peak clean workspace. Comparison with measured compilation of externally optimized XOR/AND networks exposes a resource trade-off: the proposed construction typically uses less workspace, but does not uniformly minimize T-count or measurement latency. A separate materialization theorem certifies workspace optimality only when the complete temporary bank is required.

A theory-led paper could center on the source theorem's matching lower bound and construction, with the compiler and counterexamples illustrating when its architecture is useful or avoidable. A compiler paper needs a stronger natural application family and a matched workspace-constrained external baseline. An RL paper needs a new, independently confirmed benefit from trained decisions on an identical deterministic engine. These are separate publication routes, not interchangeable claims.

The current artifact is a substantially verified research prototype and a useful comparative study. I would not submit it unchanged as a broadly competitive QCS or RL paper. The next decisive experiment should impose the same scarce-workspace and latency contract on both methods, allow both to schedule or recompute temporary values, and use complete application-derived oracles. More passing regression tests alone will not close that scientific gap.

## 11. Primary sources and exact scope

1. Meuli et al., *The Role of Multiplicative Complexity in Compiling Low T-count Oracle Circuits*, arXiv:1908.01609. Existing four-T-per-AND oracle construction and qubit/T trade-offs. https://arxiv.org/abs/1908.01609
2. Wang, Yu, Wu, and Cong, *Quantum Circuit Synthesis Using an Exact T Library*, arXiv:2605.15476 (2026). Direct T-aware library mapping; its published benchmark claims are not reproduced by our coherent wrapper. https://arxiv.org/abs/2605.15476
3. mockturtle documentation and pinned authors' source. Cut rewriting permits an AND-count cost, with resynthesis and resubstitution algorithms. https://mockturtle.readthedocs.io/en/latest/algorithms/cut_rewriting.html
4. Amy and Lunderville, *Linear and non-linear relational analyses for Quantum Program Optimization*, arXiv:2410.23493. Relevant prior nonlinear relation-aware compilation. https://arxiv.org/abs/2410.23493
5. Paradis et al., *Reqomp: Space-Constrained Uncomputation*, Quantum 8,1258 (2024). Space and recomputation are existing research mechanisms. https://doi.org/10.22331/q-2024-02-19-1258
6. *Quantum circuit optimization with AlphaTensor*, Nature Machine Intelligence (2025), DOI 10.1038/s42256-025-01001-1. Not an executed comparator here. https://www.nature.com/articles/s42256-025-01001-1
7. Supplied *H - Prove Cleanup Theorem* PDF, pp.59--62: full-bank bound and explicit bank-free counterexample. Newer constructive proof note: docs/consumer_cleanup/PROOFS.md. Neither file is replaced by this assessment.
'''
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'REPORT.md').write_text(text)
    return text

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiments',type=Path,default=Path('experiments'))
    p.add_argument('--output',type=Path,default=Path('docs/benchmark_comparison'))
    a=p.parse_args();build(a.experiments,a.output)
