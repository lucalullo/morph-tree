# MorphTree 2.1.0 — release readiness checklist

The repository may be complete for **manual local handoff** while publication
remains blocked. This checklist must be reviewed against the **actual final
commit**, not the private working-tree qualification.

- [ ] Review the destination checkout and preserve its `.git`, existing tags and a backup.
- [ ] Transfer the staged files and review deletions using [the handoff guide](docs/LOCAL_HANDOFF.md).
- [ ] Confirm Python package version, public metadata, license, native ABI and source manifest match the approved release.
- [ ] Run `python -I -B tools/check_repository.py`, `git diff --check` and review `git status`.
- [ ] Verify each built wheel and sdist is derived from the *final* source, not a stale previous candidate.
- [ ] Install wheel and sdist into clean environments and run zero-fit API, native, persistence and source-integrity checks.
- [ ] Obtain the required approved finite LIVE training evidence for any behavior-affecting changes. Prior evidence is not automatically transferable.
- [ ] Review support scope: Windows and Ubuntu WSL2 CPython 3.12–3.14 are qualified as described; macOS, universal Linux/manylinux, 3.15 and million-row training are **not** thereby qualified.
- [ ] Require exact-commit GitHub CI for all declared supported build profiles; investigate all failures rather than waiving them.
- [ ] Independently check platform wheels, package metadata, archive inclusion and artifact hashes; do not promise universal Linux portability.
- [ ] Verify TestPyPI under a separate explicit owner decision, download those artifacts and test their actual installation.
- [ ] Approve any PyPI publication separately, with reviewed artifact identity and no silent rebuild or replacement.

The repository's publication guard is intentionally **NO_GO** until the owner
explicitly reviews and authorizes the remaining gates. No automatic or agent
commit, tag, push, account operation or publishing is permitted.
