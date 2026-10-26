# MorphTree 2.0.0 — current local delivery boundary

The owner explicitly requires the existing local public checkout and its Git
history to be preserved. Transfer reviewed public files manually into the
existing official repository only after qualification and approval. No new
repository, reset history, tag, push, workflow dispatch or publication is part
of this agent's task. See `LOCAL_HANDOFF_2_0.md` and `RELEASE_QUALIFICATION.md`.

## Historical 1.0.0 preparation instructions, not the current handoff procedure

This repository contains a **manual-only, zero-fit** GitHub Actions workflow. It does not create releases, publish packages, run scientific experiments or upload artifacts. A push does not trigger it automatically.

## First upload

1. Create a new **empty private** repository on GitHub (no auto-generated README or license). Do not copy the history of a previous MorphTree repository.
2. Upload **the content of this `MorphTree/` folder**, not a nested folder, private research workspace or previous release-candidate tree.
3. Prefer Git CLI over the web file uploader because this repository contains thousands of research records. Before the first commit, run `python -I -B tools/check_repository.py` from the root and confirm `PASS_LOCAL_SOURCE_INTEGRITY`.
4. Initialize Git only locally, add all source files, commit, and run `python -I -B tools/check_repository.py --git-head` to verify the committed bytes. You may then add the owner-chosen remote and push personally. **No remote push is authorized by this package.**
5. In repository settings review Actions permissions, billing (including safeguards against unexpected charges) and log retention. The workflow requests only read-only permissions, has one Windows Python 3.12 job, and a 20-minute timeout. Account billing and log retention are not enforced by these YAML settings.

## Manual qualification

After checking budget and permissions, open GitHub **Actions → MorphTree 1.0.0 - zero-fit qualification → Run workflow**, choose the intended `main` commit, and explicitly enable `confirm_zero_fit_scope`. The workflow:

- verifies the exact committed source against the included SHA-256 manifest;
- builds 1.0.0 wheel and sdist from the commit;
- installs the two builds into separate fresh environments;
- checks installed metadata, wheel RECORD, Optuna version and package module identity;
- runs zero-fit product tests that must not train trees or create Optuna studies.

A skipped job, a cancelled run, a metadata-only check or a locally parsed workflow is **not** a hosted CI PASS. Save the logs in your private workspace before the selected retention window ends. Any failure requires diagnosis and a consciously authorized new run. A passing zero-fit job supplements, but does not replace, earlier LIVE training evidence and final maintainer release approval.

## Owner note

If you delete an older MorphTree repository, create the new one from scratch and upload only this clean folder. Do not use force-push to disguise old commits; old public forks/clones can persist. Do not include private datasets, access tokens or owner-restricted scientific dossiers.
