"""Build every numerical manuscript statement from one locked native campaign."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import numpy as np
from .native_study_analysis import read_rows,analyze,paired_interval
from .native_study_corpus import dump

LABELS={'hierarchy':'Trained hierarchy','untrained':'Untrained prior','outer':'Outer only','inner':'Inner only',
        'greedy':'Greedy','uniform_cost':'Uniform cost','mitm':'Cold MITM'}


def escaped(s):
    for a,b in (('\\',r'\textbackslash{}'),('_',r'\_'),('%',r'\%'),('&',r'\&'),('#',r'\#')):s=s.replace(a,b)
    return s


def build(output,publication):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output);pub=Path(publication);pub.mkdir(parents=True,exist_ok=True);(pub/'figures').mkdir(exist_ok=True)
    if not (out/'ALL_CAMPAIGNS_COMPLETE').exists():raise ValueError('cannot report an incomplete campaign')
    analysis=analyze(out);protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text())
    verify=json.loads((out/'verification.json').read_text());diag=json.loads((out/'diagnostics.json').read_text())['summary']
    endpoint=next(r for r in analysis['budget_results'] if r['axis']=='wall_seconds' and r['budget']==1.)
    effect=endpoint['paired']['untrained']['resource_score']
    lines=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{lrrrrr}\toprule',
           r'Method & Successes/runs & $Q$ & Gates$^\dagger$ & Seconds & Edges \\ \midrule']
    for key,label in LABELS.items():
        m=endpoint['methods'][key];g='--' if m['mean_gates_successful'] is None else f"{m['mean_gates_successful']:.2f}"
        lines.append(f"{label} & {m['successes']}/{m['runs']} & {m['resource_score']:.3f} & {g} & {m['wall']:.3f} & {m['edges']:.0f}"+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{One-second primary endpoint. $Q$ is the failure-aware native-gate resource score; $\dagger$ denotes successful runs only, which must not be interpreted without success rate. Learned variants use five training seeds and two timing repetitions; deterministic controls use two repetitions. Table entries include all executed jobs, not only successes.}\label{tab:primary}',r'\end{table}']
    lines += [f"The locked campaign contains {analysis['runs']:,} primary jobs on {analysis['logical_test_targets']} distinct logical targets. "
              f"At the primary one-second endpoint the hierarchy--prior difference in $Q$ is {effect['mean']:.4f}, "
              f"with a paired target-cluster 95\\% interval [{effect['lower_95']:.4f},{effect['upper_95']:.4f}]. "
              "The target, not a timing repetition or training seed, is the resampling unit."]
    if effect['lower_95']>0:
        verdict='The primary endpoint supports a positive target-average effect in this sampled domain; it does not establish superiority on other resource contracts or larger registers.'
    elif effect['upper_95']<0:
        verdict='The primary endpoint favors the untrained prior. Correct output circuits and exact resource proofs do not reverse this controlled negative learning result.'
    else:
        verdict='The primary interval includes zero. This experiment does not resolve a positive advantage of the trained hierarchy over its untrained prior.'
    lines.append(verdict)
    lines += [r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{llrrr}\toprule',
              r'Budget axis & Budget & Hierarchy $Q$ & Prior $Q$ & Paired difference [95\%] \\ \midrule']
    for b in analysis['budget_results']:
        e=b['paired']['untrained']['resource_score'];h=b['methods']['hierarchy'];u=b['methods']['untrained']
        axis='Native edges' if b['axis']=='edges' else 'Wall seconds'
        lines.append(f"{axis} & {b['budget']} & {h['resource_score']:.3f} & {u['resource_score']:.3f} & {e['mean']:.3f} [{e['lower_95']:.3f},{e['upper_95']:.3f}]"+r' \\')
    lines += [r'\bottomrule\end{tabular}\caption{Budget-dependent comparisons. Only the one-second $Q$ contrast is primary; the other intervals are descriptive and unadjusted for multiple comparisons.}\label{tab:budgets}\end{table}']
    # Separate figures, default matplotlib colors. Tables are data-derived too.
    for axis in protocol['primary_budget_axes']:
        fig,ax=plt.subplots(figsize=(6.3,3.7))
        bs=[r for r in analysis['budget_results'] if r['axis']==axis]
        for method in LABELS:
            ax.plot([r['budget'] for r in bs],[r['methods'][method]['resource_score'] for r in bs],marker='o',label=LABELS[method])
        ax.set_xlabel('Attempted native-edge budget' if axis=='edges' else 'Wall-time budget (s)')
        ax.set_ylabel('Failure-aware native-gate score Q');ax.set_ylim(bottom=0)
        ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(pub/'figures'/f'{axis}.pdf');fig.savefig(pub/'figures'/f'{axis}.png',dpi=150);plt.close(fig)
    # Independently retrained ablations, compared only at equal 1024 edges.
    specs={r['name']:r for r in protocol['splits']['test']};arows=[]
    for r in read_rows(out/'ablations.jsonl'):
        result=r['result'];gcap=specs[r['name']]['problem']['budget']['max_gates']
        q=(gcap+1-result['witness']['resources']['gates'])/(gcap+1) if result['timely_success'] else 0.
        arows.append({'name':r['name'],'seed':r['seed'],'method':r['variant'],'resource_score':q,'success':int(result['timely_success'])})
    for r in read_rows(out/'primary.jsonl'):
        if r['method']=='hierarchy' and r['axis']=='edges' and r['budget']==1024 and r['seed'] in protocol['ablation_seeds']:
            v=r['result'];gcap=specs[r['name']]['problem']['budget']['max_gates']
            arows.append({'name':r['name'],'seed':r['seed'],'method':'full',
                          'resource_score':(gcap+1-v['witness']['resources']['gates'])/(gcap+1) if v['timely_success'] else 0.,
                          'success':int(v['timely_success'])})
    abl={}
    alines=[r'\begin{table}[htbp]\centering\small\begin{tabular}{lrrr}\toprule',r'Variant & Successes/runs & Mean $Q$ & Difference from full [95\%] \\ \midrule']
    for variant in ('full',*protocol['exploratory_ablations']):
        rows=[r for r in arows if r['method']==variant];e=paired_interval(arows,variant,'full','resource_score')
        abl[variant]={'runs':len(rows),'successes':sum(r['success'] for r in rows),'Q':float(np.mean([r['resource_score'] for r in rows])), 'effect':e}
        detail='--' if variant=='full' else f"{e['mean']:.3f} [{e['lower_95']:.3f},{e['upper_95']:.3f}]"
        alines.append(f"{escaped(variant)} & {abl[variant]['successes']}/{len(rows)} & {abl[variant]['Q']:.3f} & {detail}"+r' \\')
    alines += [r'\bottomrule\end{tabular}\caption{Exploratory, independently retrained ablations at 1024 edges, with three matched training seeds. No variant is selected using these test results.}\label{tab:ablations}\end{table}']
    lines += alines
    lines.append(f"Off-policy withheld-prefix probes contain {diag['probes']} states. The trained inner policies select an action with a verified two-gate completion in {diag['trained_short_completion_hits']}/{diag['trained_selections']} selections; "
                 f"the prior does so in {diag['prior_short_completion_hits']}/{diag['probes']} probes. "
                 f"There are {diag['near_alias_probes']} probes with same-family feature vectors within $10^{{-12}}$ but different bounded completion outcomes, and {diag['nonmonotone_completion_probes']} with a verified completion whose first step increases target discrepancy. "
                 "These are diagnostic states from withheld construction prefixes after testing, not a claim about the policies' on-policy state distribution. Unresolved continuations are not labelled globally impossible.")
    ranking=json.loads((out/'ranking_profile.json').read_text())['snapshots']
    lines.append(f"On {len(ranking)} fixed validation frontiers, full-frontier scoring selects a different record from panel scoring in {sum(r['different_selected_record'] for r in ranking)} snapshots. "
                 "Consequently, panel/full-frontier performance cannot be attributed solely to arithmetic overhead; the allowed policy selection set changes too. The microbenchmark records raw scoring with precomputed vectors separately.")
    proofrows=read_rows(out/'proofs.jsonl');exactcount=sum(r['exact_check'] is not None and r['exact_check']['valid'] for r in proofrows)
    lines.append(f"Of the {len(proofrows)} predetermined post-campaign optimization problems, {exactcount} close all requested native resource gaps and pass the separate exact cyclotomic verifier. "
                 f"Independent replay verifies {verify['unique_exact_discovery_contracts']} distinct saved discovery-circuit/contract pairs and {verify['construction_specifications_checked']} constructor specifications. "
                 "Certificates and unchanged upper-bound/unknown outcomes are archived individually; these counts are not independent targets.")
    # Native hard cases, not conflated with the historical phase campaign.
    challenge=read_rows(out/'challenges.jsonl');ch={}
    lines += [r'\begin{table}[htbp]\centering\small\begin{tabular}{lrrr}\toprule',r'Challenge family & Hierarchy & Prior & Cold MITM \\ \midrule']
    for family in sorted({r['family'] for r in challenge}):
        cells=[]
        for method in ('hierarchy','untrained','mitm'):
            rows=[r for r in challenge if r['family']==family and r['method']==method]
            ch[f'{family}:{method}']={'successes':sum(r['result']['timely_success'] for r in rows),'runs':len(rows)}
            cells.append(f"{ch[f'{family}:{method}']['successes']}/{len(rows)}")
        lines.append(f"{escaped(family)} & "+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule\end{tabular}\caption{Unchanged 43-contract challenge registry, three-second discovery-only limits and one frozen seed. This is not a broad generalization estimate. Known exact determinant obstructions are reported separately, not inferred from these time limits.}\label{tab:challenges}\end{table}']
    reach=json.loads((out/'reachability.json').read_text())
    lines += [r'\begin{table}[htbp]\centering\small\begin{tabular}{lrrrrr}\toprule',r'Known one-clean-wire reference & $N_T$ & $N_{CX}$ & $D$ & $N_G$ & Min. clean width \\ \midrule']
    for r in reach:
        c=r['one_ancilla_reference']['resources']
        lines.append(f"{escaped(r['name'])} & {c['t_count']} & {c['cnot']} & {c['depth']} & {c['gates']} & 1"+r' \\')
    lines += [r'\bottomrule\end{tabular}\caption{Exact reference replay plus determinant exclusions establishes minimum clean width for these ideal exact-phase targets. These are existing constructions and are not learned discoveries or claims of gate optimality.}\label{tab:reach}\end{table}']
    workspace=json.loads((out/'workspace_calibration.json').read_text())
    lines += [r'\begin{table}[htbp]\centering\small\begin{tabular}{lrrrrrr}\toprule',r'Controlled-$S$ reference & Anc. & $N_T$ & $D_T$ & $N_{CX}$ & $D$ & $N_G$ \\ \midrule']
    for r in workspace['constructive_references']:
        c=r['certificate']['resources']
        lines.append(f"Exact native word & {r['ancillas']} & {c['t_count']} & {c['t_depth']} & {c['cnot']} & {c['depth']} & {c['gates']}"+r' \\')
    lines += [r'\bottomrule\end{tabular}\caption{Known ancilla--$T$-depth trade-off, independently replayed. These witnesses are not used as native search inputs. No fixed-polynomial rank certificate is imported into the general native search domain.}\label{tab:cs}\end{table}']
    csdisc=sum(r['result']['timely_success'] for r in workspace['discovery'])
    lines.append(f"Native discovery finds a timely controlled-$S$ witness in {csdisc}/{len(workspace['discovery'])} separately budgeted calibration attempts. The complete six-target, three-width crossed workspace study is archived; availability of an ancilla is not itself evidence that a policy uses it effectively.")
    application=json.loads((out/'application.json').read_text());ap={}
    for r in application['outcomes']:
        w=r['result']['witness'];aa=r['result'].get('amplitude_amplification')
        ap[r['method']]={'success':bool(w),'resources':w['resources'] if w else None,
                         'marked_probability':aa['marked_probability'] if aa else None}
    lines.append("For the new two-input threshold-network conjunction predicate, " + '; '.join(
        f"{LABELS.get(m,m)} {'finds a certified oracle' if r['success'] else 'does not find an oracle under the budget'}" for m,r in ap.items()) + ". "
        "All returned oracles are checked by exact promised-input replay and executed coherently with a diffusion operator. The ideal marked-state probability is one after one iteration. Diffusion cost is not included in the reported oracle resources; no large-BNN quantum advantage is claimed.")
    # Plot and draw one actual learned test circuit, never a construction reference.
    candidates=[r for r in read_rows(out/'primary.jsonl') if r['method']=='hierarchy' and r['result']['witness']]
    if candidates:
        candidates.sort(key=lambda r:(specs[r['name']]['n']>1,r['result']['witness']['resources']['gates']),reverse=True)
        chosen=candidates[0];w=chosen['result']['witness'];n=specs[chosen['name']]['n'];draw_circuit(w['native'],n,pub/'figures'/'generated_circuit.tex')
        dump(pub/'figures'/'generated_circuit.json',{'name':chosen['name'],'seed':chosen['seed'],'native':w['native'],
              'scope':'actual frozen-hierarchy discovery, not a reference','source_run_key':chosen['key']})
        lines.append(r'\begin{figure}[htbp]\centering\resizebox{\linewidth}{!}{\input{figures/generated_circuit.tex}}'+
              r'\caption{A circuit actually generated by the frozen native hierarchy for '+escaped(chosen['name'])+
              f" (seed {chosen['seed']}). Gate order comes directly from the saved native witness."+r'}\label{fig:circuit}\end{figure}')
    pub.joinpath('results.tex').write_text('\n\n'.join(lines)+'\n')
    fitting=selection['all_fitting_cpu_seconds']
    macro={'TrainingCases':len(protocol['splits']['training']),'ValidationCases':len(protocol['splits']['validation']),
           'TestCases':len(protocol['splits']['test']),'PrimaryRuns':analysis['runs'],
           'ExactCircuitPairs':verify['unique_exact_discovery_contracts'],'ExactOptima':exactcount,
           'SelectedAlpha':selection['selected_alpha'],'FittingCPU':f'{fitting:.2f}'}
    pub.joinpath('numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k,v in macro.items())+'\n')
    claim={'primary_resource_effect':effect,'verdict':verdict,'exact_optima':exactcount,'proof_sample':len(proofrows),
           'unique_exact_circuits':verify['unique_exact_discovery_contracts'],'primary_runs':analysis['runs'],
           'selected_alpha':selection['selected_alpha'],'aggregate_fitting_cpu_seconds':fitting,
           'ablations':abl,'challenge_results':ch,'diagnostics':diag,'application':ap}
    dump(out/'claims.json',claim)
    (out/'RESULTS.md').write_text('# Locked native publication study\n\n'+verdict+'\n\n'+
        f"Primary jobs: {analysis['runs']}; logical test targets: {analysis['logical_test_targets']}.\n"+
        f"Hierarchy minus untrained primary Q: {effect['mean']:.6f}, target-cluster 95% interval [{effect['lower_95']:.6f}, {effect['upper_95']:.6f}].\n"+
        f"Exact circuit/contract replays: {verify['unique_exact_discovery_contracts']}; exact optimality receipts: {exactcount}/{len(proofrows)}.\n"+
        f"Aggregate fitting CPU: {fitting:.2f} seconds; selected alpha: {selection['selected_alpha']}.\n\n"+
        'Full results, failures, limits and provenance are in analysis.json, claims.json and the raw JSONL records. The current manuscript is publication_native/main.tex. Historical phase-polynomial findings are not transferred.\n')
    # Every number/paragraph can be regenerated from this exact source list.
    inputs=['protocol.json','analysis.json','selection.json','verification.json','diagnostics.json','proofs.jsonl',
            'reachability.json','workspace_calibration.json','application.json','challenges.jsonl','ablations.jsonl','ranking_profile.json','primary.jsonl']
    dump(pub/'RESULT_PROVENANCE.json',{'inputs':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in inputs},
                                      'code':'hybrid_qcs/native_study_report.py','scope':'one native campaign only; no historical phase data copied'})
    pub.joinpath('abstract_results.tex').write_text(
        f"The locked study contains {analysis['runs']:,} primary jobs on {analysis['logical_test_targets']} held-out logical targets. "
        f"The primary hierarchy--prior resource-score difference is {effect['mean']:.4f} "
        f"(target-cluster 95\\% interval [{effect['lower_95']:.4f},{effect['upper_95']:.4f}]). "
        f"Independent exact replay verifies {verify['unique_exact_discovery_contracts']} distinct discovery-circuit contracts and closes {exactcount} predetermined bounded optimization problems. "
        +verdict+'\n')
    return claim


def draw_circuit(native,n,path):
    """TikZ source from a saved native gate word; no illustrative invented gates."""
    # Gate labels are generated by trusted enum names, not arbitrary TeX input.
    out=[r'\begin{tikzpicture}[x=0.62cm,y=0.65cm, every node/.style={font=\small}]']
    length=len(native)
    for q in range(n):out.append(fr'\draw (0,{-q}) node[left]{{$q_{q}$}} -- ({length+1},{-q});')
    for i,(name,qs) in enumerate(native,1):
        if name=='CNOT':
            c,t=qs;out.append(fr'\draw ({i},{-c}) -- ({i},{-t});\fill ({i},{-c}) circle (2pt);\draw ({i},{-t}) circle (4pt);\draw ({i},{-t}-.15)--({i},{-t}+.15);')
        else:
            q=qs[0];label={'SDG':r'S^\dagger','TDG':r'T^\dagger'}.get(name,name)
            out.append(fr'\node[draw,fill=white,minimum size=0.43cm] at ({i},{-q}) {{$ {label} $}};')
    out.append(r'\end{tikzpicture}');Path(path).write_text('\n'.join(out)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',default='experiments/native_publication_v1')
    p.add_argument('--publication-dir',default='publication_native');a=p.parse_args();print(json.dumps(build(a.output_dir,a.publication_dir),indent=2))


if __name__=='__main__':main()
