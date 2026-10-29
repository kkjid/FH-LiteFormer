# Historical computational snapshots

These modules document the verified research stages described in
[MODEL_EVOLUTION.md](../MODEL_EVOLUTION.md). They are outside the installed
`fh_liteformer` package and never replace the final C-r8 entry points.

`manifest.json` identifies the original commits and exact extracted definitions.
Only module docstrings and imports were adapted. Shared type encoders, relation
parameters, FiLM and topology are imported from their already verified package
definitions. Unrelated comparison models, third-party implementations, research
drivers and internal reports are excluded.

The diffs compare actual selected computational definitions. They are reading
aids, not patches for migrating checkpoints or commands for historical
end-to-end reproduction. Historical checkpoint results are not newly validated
by importing these source snapshots.
