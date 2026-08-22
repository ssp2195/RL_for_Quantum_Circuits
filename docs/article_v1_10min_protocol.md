# Article V1 ten-minute protocol

This protocol is a versioned V3 envelope around the existing Article V1 native
corpus. Legacy `article-v1-corpus-config-v2` profiles remain valid and are not
silently reinterpreted.

The primary learner uses `fixed-total-expansions-curriculum-v1` over train
easy/medium targets only. Evaluation uses one
`fixed-max-horizon-anytime-v1` execution per target and derives threshold rows
from the first certified hit. The primary comparison remains fixed expansion
budget; equal process-CPU time is supplementary and reported separately.

Process CPU is measured with `time.process_time_ns()`. The watchdog is checked
only after a complete exhaustive expansion and before beginning the next one.
`OPERABILITY_TIMEOUT` is incomplete evidence and cannot be aggregated as an
ordinary unsolved run.

The pilot envelope is in `configs/article_v1_10min_pilot.json`; the publication
template is unresolved by design. A publication config may be produced only by
`freeze-10min-protocol` from calibration and validation evidence that explicitly
asserts no held-out target access, with a clean worktree and a recorded source
commit.

Validation:

```powershell
python article_benchmark.py validate-10min --config configs/article_v1_10min_pilot.json
```

No pilot or publication campaign is authorized by the presence of a V3 config
alone; calibration, validation, freeze, and the existing campaign audit gates
remain required.
