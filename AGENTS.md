# Repository execution safety

The full runbook and the 2026-08-20 OOM evidence are documented in
`docs/execution_safety.md` and
`docs/incidents/2026-08-20-mesh-split-memory-exhaustion.md`.

- Run production mask/depth/pose work through `python -m pose_solver run`; its
  mandatory memory guard must remain enabled for heavy work.
- Do not launch mesh, render, pose, or video diagnostics with ad-hoc
  `python -` snippets. Add a bounded diagnostic CLI and run it through
  `tools/diagnostics/run_with_memory_guard.py`.
- Never call `trimesh.Trimesh.split()` on reconstruction meshes. For component
  counts use `body_count` or sparse component labels. Materialize components
  only from a deliberately low-poly collision proxy.
- Before any GPU task, honor `runtime.devices`; this repository's checked-in
  Object-3 configuration permits only GPUs 6 and 7 and at most two devices.
- A command that exceeds 32 GiB process-group RSS or leaves less than 128 GiB
  host memory available must be stopped, investigated, and not retried without
  changing the algorithm.
- Never start a second heavy command because the first command is silent or a
  tool call returned early. Inspect exact project PIDs, process groups, guard
  logs, and host memory first. Do not run concurrent heavy jobs for the same
  data/output from separate terminals or agent sessions.
- Treat guard exit 125 as a resource refusal/limit event and exit 137 as a
  possible OOM incident. Preserve evidence, verify that the whole process group
  is gone, and do not retry the unchanged command.
- Never clear swap, reboot services, or terminate another user's processes on
  the shared server.
