# Article V1 ten-minute protocol preflight

Recorded 2026-08-21 before campaign integration.

## Source boundary

- Branch: `ten-minute-protocol`
- HEAD: `2ab3361a8e7f5629055696f17432b9a744fe4cfd`
- Plan reference differed: `frontier-rl-exper@51667b704d3605ce8cdcb9441733b3c4f121663d`
- Resolution: the user-directed local branch and its existing partial V3 work are
  treated as authoritative; no reset, artifact deletion, or legacy-config edit
  was performed.
- Worktree at preflight: dirty with the local ten-minute protocol changes.

## Preserved legacy configurations

| Profile | File SHA-256 | Canonical config digest | Easy/medium/hard horizons | Episodes/target |
|---|---|---|---|---:|
| pilot | `f10a74e4d6de1799b5f7e87e4dc451a2c196748f87785f7ab458145c10583f62` | `sha256:fbaaead71126203068c22b9d3fb384c593f5d8cd066effc1b6f585f79de762d9` | 256 / 2,048 / 8,192 | 2 |
| publication | `202b4067beadaac7cd60cea9735b7fe0f09754974ece2e6db5a0c84bdcda5516` | `sha256:fdf44d9d6f89c063f3c5d2037712d64efad634cc326c607f1caa51711a5a78d3` | 1,000 / 10,000 / 100,000 | 10 |

The legacy files were not modified. Existing outputs and recovery artifacts
remain in place and are not eligible for V3 resume.

## Verification

- `python -m compileall -q .`: source compilation completed; traversal reported
  access-denied messages only for pre-existing `.pytest-*` cache directories.
- `pytest -q tests/article_v1`: 371 passed, one compatibility failure initially
  exposed a V3 CPU field leaking into the exact legacy V4 raw schema.
- The compatibility boundary was corrected and the failed audit regression now
  passes. A clean full-suite rerun remains required after implementation.
- `git diff --check`: passed at the preflight boundary (line-ending notices only).

No scientific pilot or held-out evaluation has been launched from this branch.
