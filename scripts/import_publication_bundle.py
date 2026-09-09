#!/usr/bin/env python3
"""Import the approved publication snapshot without rewriting any existing branch.

The archive is checked before any contained source is executed. Run from a Git
checkout. This helper belongs only to the transfer branch, not the publication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BRANCH = 'publication-linear-certified-qcs-v1'
COMMIT = '12f6cdfcfd2835bd36f3870faaf0db9abbd0c7cf'
TREE = '2172b41561d4583539e6721c711153a5194cfc1c'
BASE = 'dfa7f9081a289a0b645eea70d8192c89cc17c1eb'
ARCHIVE_SHA256 = '8bbef25aa9f6f7f371ec53ac94c2aac09e6dfc75458d3b4f41fe25c28d704df3'


def git(*args: str) -> str:
    result = subprocess.run(['git', *args], text=True, capture_output=True,
                            timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(f'git {args[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def heads(remote: str) -> dict[str, str]:
    rows = git('ls-remote', '--heads', remote)
    return {ref: sha for sha, ref in (r.split() for r in rows.splitlines())}


def verify_archive(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f'Upload {BRANCH}.bundle to the transfer branch root.')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError(f'Archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}')
    git('bundle', 'verify', str(path.resolve()))
    refs = {ref: sha for sha, ref in
            (r.split() for r in git('bundle', 'list-heads', str(path.resolve())).splitlines())}
    if refs.get(f'refs/heads/{BRANCH}') != COMMIT:
        raise ValueError('Archive does not contain the approved publication branch.')
    git('fetch', '--no-tags', str(path.resolve()), f'refs/heads/{BRANCH}')
    if git('rev-parse', 'FETCH_HEAD') != COMMIT:
        raise ValueError('Fetched commit differs from approved snapshot.')
    if git('rev-parse', f'{COMMIT}^{{tree}}') != TREE:
        raise ValueError('Source-tree identity check failed.')
    if git('rev-parse', f'{COMMIT}^') != BASE:
        raise ValueError('Publication parent differs from the approved base.')
    git('fsck', '--full', '--strict', '--no-dangling')
    # Workflow updates are pre-staged via the authorized GitHub connection.
    # The Actions token must never introduce unreviewed workflow content.
    changed = git('diff', '--name-only', BASE, COMMIT, '--', '.github/workflows')
    for path_name in changed.splitlines():
        if git('rev-parse', f'HEAD:{path_name}') != git('rev-parse', f'{COMMIT}:{path_name}'):
            raise ValueError(f'Workflow not pre-staged with identical contents: {path_name}')
    return {'branch': BRANCH, 'commit': COMMIT, 'tree': TREE,
            'base': BASE, 'archive_sha256': actual}


def publish(remote: str, receipt: dict[str, str]) -> dict:
    before = heads(remote)
    target = f'refs/heads/{BRANCH}'
    existing = before.get(target)
    if existing not in (None, COMMIT):
        raise ValueError('Target branch already has a different commit; refusing to overwrite it.')
    if existing is None:
        # No force flag, no wildcard refspec, no branch deletion.
        git('push', '--porcelain', remote, f'{COMMIT}:{target}')
    after = heads(remote)
    if after.get(target) != COMMIT:
        raise RuntimeError('Remote readback did not confirm the publication commit.')
    changes = {ref: {'before': sha, 'after': after.get(ref)}
               for ref, sha in before.items() if ref != target and after.get(ref) != sha}
    if changes:
        raise RuntimeError(f'Unrelated refs changed concurrently; inspect before proceeding: {changes}')
    return {**receipt, 'published': True, 'idempotent': existing == COMMIT,
            'before_refs': before, 'after_refs': after,
            'publication_url': f'https://github.com/ssp2195/RL_for_Quantum_Circuits/tree/{BRANCH}'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(f'{BRANCH}.bundle'))
    parser.add_argument('--remote', default='origin')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--receipt', type=Path, default=Path('/tmp/qcs-import/receipt.json'))
    args = parser.parse_args()
    receipt = verify_archive(args.archive)
    result = {**receipt, 'published': False} if args.validate_only else publish(args.remote, receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
