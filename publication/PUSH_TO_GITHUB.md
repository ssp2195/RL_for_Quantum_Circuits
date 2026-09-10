# Publish the prepared branch

Remote writes were not available in the execution session. These commands publish the supplied, committed Git branch without changing the original branch and without force-pushing.

## Fresh clone from the delivered bundle

Run in a directory containing `publication-linear-certified-qcs-v1.bundle`:

```bash
git clone --branch publication-linear-certified-qcs-v1 publication-linear-certified-qcs-v1.bundle RL_for_Quantum_Circuits
cd RL_for_Quantum_Circuits
git remote set-url origin https://github.com/ssp2195/RL_for_Quantum_Circuits.git
git status --short
git push --set-upstream origin publication-linear-certified-qcs-v1
```

The push uses your existing GitHub authentication. No token is embedded in the bundle or scripts. The branch contains a new `.github/workflows/publication-ci.yml`; the account/token performing a push must permit the intended repository and workflow changes. Do not send credentials in chat.

## Existing local checkout

From the existing repository, with an unused branch name:

```bash
git fetch /absolute/path/publication-linear-certified-qcs-v1.bundle refs/heads/publication-linear-certified-qcs-v1:refs/heads/publication-linear-certified-qcs-v1
git switch publication-linear-certified-qcs-v1
git status --short
git push --set-upstream origin publication-linear-certified-qcs-v1
```

If the branch already exists, inspect it before switching or replacing anything. The prepared commands do not force a ref update. The expected parent is `dfa7f9081a289a0b645eea70d8192c89cc17c1eb`. The delivery's `DELIVERY_STATUS.json` records the exact new commit and bundle checksum.

The workflow retrains five models, then runs the full 481-test suite, independently replays archived circuits/proofs, builds the manuscript, and exports its exact checkout. It is prepared but was not remotely executed during this session. A local passing suite is not labelled remote CI success.
