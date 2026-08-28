# 2026-08-20 pose_solver mesh-split memory exhaustion incident

Daily operating rules derived from this incident are maintained in
[Execution and resource safety](../execution_safety.md).

## Impact

Two pose_solver mesh diagnostics triggered global OOM kills at 10:55 and
12:04 UTC.  The shared server was severely degraded and was eventually
rebooted at 2026-08-21 02:50 UTC.

## Direct evidence

The kernel log identifies two Python victims in the same user session and
account (`uid=1029`, `ziang`):

| Kernel time | PID | Anonymous RSS | Approx. GiB | Result |
| --- | ---: | ---: | ---: | --- |
| 10:55:12 | 1413745 | 298,301,888 KiB | 284.5 GiB | global OOM kill |
| 12:04:26 | 1415943 | 564,885,068 KiB | 538.7 GiB | global OOM kill |

The first process used approximately 571 MiB of page tables and the second
approximately 1.06 GiB, in addition to their anonymous RSS.  Both command
records ended with exit code 137.

The local Codex execution record makes the command-to-kernel correlation
unambiguous:

1. At 10:37:55 a `.venv/bin/python -` diagnostic loaded the Object-3 `lid`
   and `motor` GLBs and executed
   `m.split(only_watertight=False)`.  It ran for 1,041.9 seconds and was
   recorded as killed at 10:55:17, five seconds after PID 1413745 was killed
   by the kernel.
2. At 10:38:31 a concurrent `.venv/bin/python -` diagnostic from another
   active Codex session loaded five Object-4/Object-9 meshes and also executed
   `m.split(only_watertight=False)`.  It ran for 5,186.2 seconds and was
   recorded as killed at 12:04:57, 31 seconds after PID 1415943 was killed by
   the kernel.

Thus the two OOM events were not one unexplained leak: two unsafe diagnostics
started 36 seconds apart and independently expanded until the kernel killed
them.

The Object-3 inputs to the first diagnostic were unusually dense
reconstruction meshes:

| Part | Vertices | Faces |
| --- | ---: | ---: |
| motor | 337,581 | 494,001 |
| lid | 466,539 | 471,116 |

## Root cause

The diagnostics called `trimesh.Trimesh.split(only_watertight=False)` to print
mesh component statistics. `split()` materializes a new mesh for every
connected component. On dense, fragmented/non-manifold reconstruction meshes,
the intermediate graph and copied component geometry expanded to hundreds of
GiB.

Both commands were ad-hoc `python -` invocations running concurrently in
separate agent sessions, so neither had a process-group memory limit.

At 14:37 the same Object-3 split diagnostic was mistakenly attempted twice
again.  A later process sample found those two workers at 247.7 GiB and
237.9 GiB RSS (approximately 485.5 GiB combined).  This was a separate
post-incident recurrence, not the source of the 10:55/12:04 kernel entries.
It confirms that retrying an apparently silent diagnostic without inspecting
its child processes was a second operational failure.

## Ruled out as the direct cause

- The single-frame pose and scale optimizations completed independently in
  seconds and did not own the two dominant RSS allocations.
- The review-sheet renderer ran sequentially and `SceneRenderer` clears each
  temporary scene in a `finally` block.
- No full trajectory render, video encode, or Isaac run was active when the
  two high-RSS workers were identified.

This does not prove every renderer path is leak-free; it identifies the direct
cause of these two OOM events from matching kernel and command records.

## Corrective actions

1. The unified `python -m pose_solver run` entry point now automatically runs
   inside a process-group memory guard.
2. The guard rejects startup below the host-memory safety threshold, samples
   once per second, and stops a task above its RSS limit.
3. Guard interruption and unexpected monitor exceptions now clean up the
   child process group in `finally`, preventing orphan workers.
4. Core pose/render command-line tools reject unguarded direct execution.
5. Dense collision proxies fail closed before component materialization;
   physics export requires a low-poly proxy.
6. Repository execution policy forbids ad-hoc mesh diagnostics and forbids
   `trimesh.split()` on reconstruction meshes.

Object-3 is configured for GPUs 6 and 7, a 32 GiB maximum task RSS, and a
128 GiB minimum host-available-memory reserve.

## Validation

- 27 resource/config/simulation unit tests pass.
- Start-time rejection was exercised with an intentionally impossible memory
  reserve and returned guard code 125 without starting a child.
- Runtime enforcement was exercised with a safe 20 MiB allocation and a
  10 MiB test limit; the full process group was stopped within one sample.
- Object-3 unified preflight passes with exactly GPUs 6 and 7.
