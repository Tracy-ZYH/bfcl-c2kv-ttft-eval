# Frozen BCRO-120 Artifacts

These files freeze the routing inputs and runtime configuration used by the
BCRO-120 experiment. The exact integrated runtime snapshot is Git commit
`a271a2d59b69835aca5301c31280098efa4460a0`.

Use `c2kv_eval/portable/ops/run_bfcl_long_bcro120.sh` for the frozen
120-task configuration. Set `BCRO120_REQUIRE_EXACT_RUNTIME=1` to reject later
runtime performance changes; later commits are not silently claimed as the
frozen timing implementation. They are byte-for-byte copies of the original files under
`results/` or `runtime/configs/`; no model, routing, detector, recovery, or
generation setting was changed during migration.

The source paths and SHA256 digests are recorded in `manifest.sha256` and
`provenance.json`. The original result directories remain valid for audit and
backward compatibility.

Runtime model weights and the embedding checkpoint remain external assets and
are intentionally not stored in Git.
