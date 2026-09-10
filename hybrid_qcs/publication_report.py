"""Generate the manuscript's result tables from saved, verified measurements."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def generate(out: Path, publication: Path):
    out, publication = Path(out), Path(publication)
    a = json.loads((out/'analysis.json').read_text())
    g = json.loads((out/'guard_amendment/analysis.json').read_text())
    v = json.loads((out/'complete_verification.json').read_text())
    post = json.loads((out/'post_audit.json').read_text())
    app = json.loads((out/'bnn_application.json').read_text())
    labels = {'hierarchy':'Trained hierarchy', 'untrained':'Identical untrained prior',
              'greedy':'Greedy', 'outer':'Outer only', 'inner':'Inner only',
              'uniform_cost':'Uniform cost', 'audit_only':'Audit only'}
    lines = [r'\subsection{Locked discovery experiment}',
             r'Every failure remains in the endpoint. Table~\ref{tab:primary} reports the complete locked campaign, not only successful trajectories. The main comparison is against the identical untrained prior; that control shares the representation, candidate panel, fairness, native certifier, and feature computation.']
    lines += [r'\begin{table}[htbp]\centering\small',
              r'\begin{tabular}{@{}lrrrr@{}}\toprule',
              r'Method & Correct/375 & Savings (\%) & Time (s) & Improving runs \\\midrule']
    for m in labels:
        s = a['methods'][m]
        lines.append(f"{labels[m]} & {s['successes']} & {100*s['mean_normalized_savings_all_runs']:.2f} & {s['mean_wall_seconds']:.4f} & {s['strict_improvement_runs']} \\")
        lines[-1] += '\\'
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{Primary discovery outcomes. Savings are normalized CNOT reductions relative to the specification-derived parity-star count, with zero credit for failures. Time includes the whole anytime job, not merely first-solution latency. Improving runs contain at least two successively better, self-discovered certified incumbents. Each method has 25 targets, five seeds, and three timings.}\label{tab:primary}\end{table}']
    contrast = a['paired_hierarchy_minus_control']['untrained']['savings']
    lines.append(f"The trained hierarchy has {100*contrast['mean']:.2f} percentage points less mean normalized savings than the untrained prior; the paired target-cluster 95\\% interval is [{100*contrast['lower_95']:.2f}, {100*contrast['upper_95']:.2f}]. This is negative transfer, not evidence of a learned speed or resource advantage. The 121 hierarchy runs with strict incumbent improvement demonstrate working anytime synthesis, but the untrained control also improves incumbents. Correct output circuits and improvement during search therefore do not establish a benefit from training.")
    lines.append(f"The mean final-model training cost is {a['mean_training_cpu_seconds']:.3f} CPU seconds ({a['mean_training_wall_seconds']:.3f} wall seconds), across five models of 184 episodes each. At 1,000 uses this contributes {a['training_amortization_cpu_seconds_per_target']['1000']:.4f} CPU seconds per target. These figures exclude hyperparameter selection, ablation retraining, and the later shrinkage-selection experiment; all are separate artifact records. A small training bill does not compensate for worse discovery performance.")
    lines += [r'\subsection{Constructive and feature controls}',
              r'\begin{table}[htbp]\centering\small',
              r'\begin{tabular}{@{}lrrr@{}}\toprule',
              r'Constructive control & In-contract/75 & Savings (\%) & Time (s) \\\midrule']
    for m, label in [('graysynth','Adapted GraySynth'),('parity_star','Parity star'),('rank_partition','Rank-partition reference')]:
        s = a['constructive'][m]
        lines.append(f"{label} & {s['successes']} & {100*s['mean_savings']:.2f} & {s['mean_wall_seconds']:.4f} \\\\")
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{Constructive controls on the same 25 contracts, with three timings. These are one-shot algorithms, not algorithms padded to the search timeout. All emitted circuits are independently replayed, including those outside the original caps.}\label{tab:constructive}\end{table}']
    lines.append(r'The adapted GraySynth control is stronger than the trained hierarchy in this domain. Its three failed in-contract repetitions refer to one target with a correctly synthesized circuit outside the CNOT cap. The rank-partition reference prioritizes T-depth rather than CNOT count; its circuits are correct but exceed these tight CNOT comparison caps. It is not accurate to call those circuits semantically incorrect or to hide the relaxed resource checks.')
    lines += [r'\begin{table}[htbp]\centering\small',
              r'\begin{tabular}{@{}lrrr@{}}\toprule',
              r'Secondary control & Correct/125 & Savings (\%) & Maximum panel \\\midrule']
    for m, label in [('no_budget','Retrained without budget features'),('no_workspace','Retrained without workspace features'),('full_panel','Full-frontier scoring')]:
        s = a['secondary'][m]
        lines.append(f"{label} & {s['successes']} & {100*s['mean_savings']:.2f} & {s['max_panel']} \\\\")
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{Separately trained feature ablations and identical-weight full-frontier scoring, one timing per seed/target. These are descriptive secondary comparisons; the primary untrained comparison remains unfavorable.}\label{tab:ablation}\end{table}']
    panel = a['secondary']['full_panel']['hierarchy_minus_ablation']
    lines.append(f"The bounded panel scores at most 32 candidates, compared with an observed maximum of {a['secondary']['full_panel']['max_panel']} under full-frontier scoring. With identical trained weights, the bounded version improves normalized savings by {100*panel['mean']:.2f} percentage points in the matched secondary contrast (descriptive 95\\% interval [{100*panel['lower_95']:.2f}, {100*panel['upper_95']:.2f}]). This supports limiting ranking overhead within this implementation; it does not establish that learning itself is beneficial. The secondary timings are not additional independent replications of the primary contrast.")
    boundary = json.loads((out/'boundary_evaluation.json').read_text())
    ns = {m: sum(r['result']['witness'] is not None for r in boundary if r['method']==m) for m in ('hierarchy','untrained')}
    lines.append(f"On 40 rank-tight T-depth jobs per scheduler, the hierarchy finds {ns['hierarchy']} circuits and the untrained prior {ns['untrained']}; this exploratory subset uses only eight original targets. The nonlearned rank-partition reference succeeds on all of these widened-CNOT contracts. This local contrast cannot overturn the unfavorable prespecified primary result.")
    lines += [r'\subsection{Validation safeguard and fresh confirmation}',
              r'Validation selects $\beta=0$ from the declared shrinkage grid. Hence the guarded deployment uses \emph{no learned correction}. The first five coefficients in the grid tie on validation savings and success; the declared smaller-coefficient tie rule selects zero. No new confirmation outcome is used to reselect the coefficient.',
              r'\begin{table}[htbp]\centering\small',
              r'\begin{tabular}{@{}lrr@{}}\toprule',
              r'Fresh-confirmation method & Correct/540 & Savings (\%) \\\midrule']
    for m,label in [('hierarchy','Raw trained hierarchy'),('untrained','Untrained prior'),('guarded',r'Validation-selected guard ($\beta=0$)')]:
        s=g['methods'][m]
        lines.append(f"{label} & {s['successes']} & {100*s['mean_savings']:.2f} \\\\")
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{New logical target orbits after the separately declared amendment. Twelve targets are crossed with three ancilla widths, five seeds, and three timings. Intervals cluster by twelve logical targets.}\label{tab:confirmation}\end{table}']
    h = g['hierarchy_minus_untrained']; q=g['guarded_minus_untrained']
    lines.append(f"The raw hierarchy remains worse than the prior: {100*h['mean']:.2f} percentage points (95\\% interval [{100*h['lower_95']:.2f}, {100*h['upper_95']:.2f}]). The guarded-minus-prior effect is {100*q['mean']:.3f} percentage points, interval [{100*q['lower_95']:.3f}, {100*q['upper_95']:.3f}]. Because their scores are identical at $\\beta=0$, small wall-budget differences reflect timing and termination variability, not learned generalization. This amendment prevents unsupported deployment of a degraded learned controller; it does not rescue the positive learning hypothesis.")
    lines += [r'\subsection{Certified resource trade-off and independent audits}',
              r'For controlled-S, $2x_0x_1=x_0+x_1-(x_0\oplus x_1)$ modulo eight. Frozen trained models generate three-T circuits with the following verified costs:',
              r'\begin{center}\small\begin{tabular}{@{}rrrrrr@{}}\toprule',
              r'Available clean & Used & T-depth & CNOT & Native depth & Gates \\\midrule',
              r'0 & 0 & 2 & 2 & 4 & 5 \\',
              r'1 & 1 & 1 & 4 & 5 & 7 \\',
              r'2 & 1 & 1 & 4 & 5 & 7 \\\bottomrule\end{tabular}\end{center}',
              r'All five seeds reproduce all three calibration settings. Independent rank bounds certify T-depth two without workspace and one with at least one clean ancilla; independently verified tighter-CNOT covers certify the stated CNOT minima under the corresponding depth contracts. Thus one auxiliary qubit removes a T-layer but costs two CNOTs and two native gates. The construction and rank principle are known; this is a calibration of the learned-generation and proof pipeline, not a newly discovered quantum identity.']
    opt=sum(r['status']=='bounded_optimal' for r in post)
    lines.append(f"Post-campaign auditing closes {opt} of the 25 primary CNOT gaps; the other {25-opt} remain certified upper bounds. No audit result repairs a failed discovery record. The complete evidence verifier replays {v['unique_native_witness_contract_pairs']} unique native witness/contract pairs across {v['outcome_records']:,} recorded outcomes, checks {v['closed_covers_verified']} closed covers and {v['rank_bounds_verified']} rank-bound records, and verifies the coherent evaluator reference. The 5,501 outcomes include calibration, validation, constructive controls and the amendment; they are not 5,501 independent target problems.")
    lines.append(r'The learned seed-zero BNN phase oracle uses no auxiliary qubits, seven T gates, six CNOTs, eleven native depth levels, sixteen native gates, and five T-layers. The separately constructed compute--phase--uncompute reference uses two auxiliaries, 42 T gates, 42 CNOTs, native depth 66, and 98 gates. This is a representation comparison, not a learning ablation. Running the actual synthesized oracle and diffusion gives marked probability $0.8437499999999996$, agreeing with $27/32$ to $4.45\times10^{-16}$; workspace leakage is zero in the simulation.')
    publication.mkdir(parents=True,exist_ok=True)
    (publication/'results.tex').write_text('\n\n'.join(lines)+'\n')
    return {'primary_rows': a['primary_rows'], 'verified_outcomes': v['outcome_records'],
            'raw_hierarchy_primary_effect': contrast['mean'], 'selected_beta': g['selected_beta']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('experiments/publication_v1'))
    p.add_argument('--publication-dir',type=Path,default=Path('publication'))
    a=p.parse_args();print(json.dumps(generate(a.output_dir,a.publication_dir),indent=2))


if __name__=='__main__':main()
