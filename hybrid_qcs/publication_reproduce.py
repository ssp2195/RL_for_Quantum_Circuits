"""Sequential reproduction of the locked study, with all failures retained.

This reuses the prespecified hyperparameters. It does not reselect them using
held-out outcomes. Use a NEW output directory to produce independent timings.
The later shrinkage amendment remains separate from the original primary study.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from .publication_runner import (lock_protocol, read_lock, train_models, benchmark_batch,
    constructive_controls, calibration, boundary_evaluation, application, post_audit, write_json, environment)
from .publication_analysis import analyze
from .publication_guard import validation_select, confirm, analyze_guard
from .publication_evidence import export_and_verify
from .publication_report import generate


def reproduce(out: Path, *, resume=False, publication_dir=None):
    out=Path(out)
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError('Use a new output directory, or explicitly pass --resume; published evidence is never overwritten by default.')
    lock_protocol(out)
    plan=read_lock(out)
    write_json(out/'campaign_environment.json',environment())
    train_models(out)
    for repeat in range(plan['timing_repeats']):
        for seed in plan['seeds']:
            benchmark_batch(out,seed,repeat)
    (out/'PRIMARY_COMPLETE').write_text('all locked primary seed/repetition batches completed\n')
    # Preserve ordering: amendment selection uses validation only, after the
    # primary results have been recorded. It cannot mutate the primary models.
    validation_select(out)
    for ablation in ('no_budget','no_workspace'):
        train_models(out,ablation=ablation)
    for seed in plan['seeds']:
        benchmark_batch(out,seed,0,secondary=True)
    constructive_controls(out)
    calibration(out)
    boundary_evaluation(out)
    application(out)
    post_audit(out)
    analyze(out)
    confirm(out)
    analyze_guard(out)
    export_and_verify(out)
    if publication_dir is not None:
        generate(out,Path(publication_dir))
    (out/'ALL_CAMPAIGNS_COMPLETE').write_text('all primary, secondary, constructive, application, proof, confirmation and verification stages completed\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/publication-reproduction'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--publication-dir',type=Path)
    args=p.parse_args();reproduce(args.output_dir,resume=args.resume,publication_dir=args.publication_dir)


if __name__=='__main__':main()
