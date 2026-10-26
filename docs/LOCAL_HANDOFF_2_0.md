# MorphTree2.0.0 — local owner handoff

The prepared source targets Windows AMD64/CPython3.12–3.14 and Ubuntu24.04 WSL2
x86_64/CPython3.12. Python3.15 is experimental and retained on the roadmap.
The owner approved this definitive release candidate for manual transfer on
26October2026. Local technical preparation is complete and accepted.
Consult QUALIFICATION_2_0.json for exact evidence and retained limitations.

The owner copies only `Repository GitHub/morph-tree/` into the official checkout,
including hidden `.github` files, source, tests, examples and documentation.
Use docs/PUBLIC_FILES_2_0.json and LOCAL_SOURCE_MANIFEST.json to review exact bytes.
Never copy PRIVATE_CODEX, datasets, qualification records, environments, grants
or private staging files. Preserve the destination .git and history.

The local `.git` directory, existing history and preexisting edits were preserved.
If updating an existing clone of the official repository, copy the public files
into its working tree while preserving that clone's `.git`. Review `git diff`
and `git status`, stage the reviewed changed-file list and commit it yourself.
Do not initialize a replacement history, force-push or alter historical v1.0.0.
The agent has not staged, committed, tagged, pushed, dispatched hosted CI or
published anything. No Git operation is required to use the source locally.

After the owner transfers the reviewed files to the official repository, run
CI on the definitive commit. Then verify TestPyPI before seeking separate
PyPI approval. The existing workflow verifies source identity, builds wheel/sdist
and checks SHA-256, installed origins and software contracts outside the checkout. Every changed artifact has its own identity; do not
substitute a previous 1.0.0 file. Declared Python metadata is limited to the explicitly approved and verified
first-release scope; future expansion needs its own qualification.

Remote CI, a future v2.0.0 tag, Trusted Publishing OIDC, TestPyPI verification,
identical-artifact promotion to PyPI and GitHub Release are future manual owner
steps after complete qualification and approval. Existing manual workflows and
provenance checks are preserved for that purpose. They do not run on push and
their release preflight refuses an incomplete 2.0.0 decision. See
`docs/CI_PREPARATION.md` and `docs/TESTPYPI_PUBLISHING.md`.

