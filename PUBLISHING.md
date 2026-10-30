# Publishing procedure — MorphTree 2.1.0

The staged repository is prepared for a manual update of the **existing**
[GitHub project](https://github.com/lucalullo/morph-tree). It is not a new
repository. The local source is qualified within
[the stated scope](docs/QUALIFICATION.md); it is **not yet approved for
publication**. The owner performs all Git and package-index actions.

## Before GitHub transfer

Read [LOCAL_HANDOFF.md](docs/LOCAL_HANDOFF.md). Preserve the destination `.git`,
back up the existing checkout, remove stale files only after reviewing the
file-change report, and run:

```bash
python -I -B tools/check_repository.py
```

No private laboratory results, account credentials or local development paths
belong in the repository.

## After the owner transfers the sources

1. Verify the exact final commit, version `2.1.0`, file inventory and correct
   native source inclusion. Ensure any old `morphtree/` root does not shadow
   `python/morphtree/`.
2. Trigger the **manual** `.github/workflows/test.yml` workflow with its
   explicit zero-fit confirmation. Verify every required Windows/Linux
   Python 3.12–3.14 job and review the uploaded artifacts/hashes.
3. Check the built distributions and verify clean installed behavior.
   These zero-fit checks do not replace past LIVE evidence or authorize new
   learning experiments.
4. Before TestPyPI, review the manual
   `.github/workflows/publish.yml` configuration, its environment protections,
   required secret and the exact selected build run. The guard in
   `tools/publish_reviewed.py` deliberately blocks publication while
   `docs/RELEASE_STATUS.json` is `NO_GO_PUBLICATION`.
5. Only after owner review may those publication fields and the corresponding
   manifest be updated, followed by new source/build checks and a **separate
   TestPyPI authorization**. Download and check actual TestPyPI files, hashes
   and clean installations before any subsequent PyPI decision.
6. Authorize PyPI **separately** and verify that the reviewed artifacts are
   identical to those approved in the staged TestPyPI step. Never overwrite
   an existing PyPI/TestPyPI release using the same version.

MorphTree currently has a review-controlled Twine-based publishing gate.
Switching to trusted publishing is a separate security-sensitive change
requiring maintainer configuration and verification.

For exact conditions see [the release checklist](RELEASE_CHECKLIST.md),
[the version policy](VERSIONING.md), and [handoff](docs/LOCAL_HANDOFF.md).
