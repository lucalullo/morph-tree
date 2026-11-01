# Official PyPI — owner-controlled MorphTree 2.1.0 publication

Preparation is not permission to upload. The owner reports successful TestPyPI
publication and an existing protected `pypi` environment and Pending Trusted
Publisher. The preparation agent has not checked these services online.
`publish.yml` remains the separate, unchanged TestPyPI workflow. Official PyPI
uses **only `publish-pypi.yml`**, without a permanent token or password.

## Transfer and the new commit

Copy the reviewed local repository into your own checkout, including `.github`,
new tools, documentation and `SOURCE_MANIFEST.json`. Preserve the destination
`.git`. Review changes and perform commit/push personally. The agent does neither.
From the repository root run:

```powershell
python -I -B tools/check_repository.py
```

Preserve file bytes; `.gitattributes` disables automatic text conversion. Do
not regenerate manifests merely to conceal unexplained differences.

Adding the workflow changes the SHA of `main`. **Old CI run IDs, attempts and
manifest hashes cannot be reused.** Record the new full 40-character commit SHA
and run fresh CI after the last source edit. Local simulation hashes are not
publication approval.

## Owner-side settings

The official PyPI Pending Trusted Publisher must use:

| Field | Exact value |
|---|---|
| Project | `morphtree` |
| Repository owner | `lucalullo` |
| Repository | `morph-tree` |
| Workflow filename | `publish-pypi.yml` |
| Environment | `pypi` |

In GitHub, `pypi` must have required **user** reviewer `lucalullo`. Deployment
branches/tags must be **Selected branches and tags**, with exactly one **branch**
policy `main`, no tags or wildcards. Preflight checks the reviewer and selected
branch-protection mode using the workflow's read-only token; the CI/context
guards themselves allow only the exact `main` branch, never a tag or other branch.
The owner confirms the environment's actual branch-policy list in Settings:
this procedure does not request administrative credentials to enumerate it.
If the environment API cannot verify the reviewer/protection mode, it stops;
do not bypass that check. If self-review prevention is enabled, another authorized
person must start the run so the required reviewer can approve it. Review
administrator bypass settings. No API-token secret needs to be added.

## Fresh six-profile CI and reviewed manifests

In Actions run **MorphTree native candidate - manual zero-fit build** (`test.yml`)
on the new `main` commit with `confirm_zero_fit_scope=true`. All six jobs must
succeed: Windows/Linux × Python 3.12/3.13/3.14. Skipped jobs, fork builds,
different commits or mixed rerun artifacts do not qualify. Prefer a fresh
complete run. Record its numeric run ID and exact attempt.

Download and extract these artifact sets separately into `windows` and `linux`:

- `morphtree-windows-latest-3.12`
- `morphtree-ubuntu-24.04-3.12`

Each contains a wheel, sdist and `BUILD_PROVENANCE.json`. Inspect the manifest's
commit/run/attempt, source and README hashes, version and distribution checksums.
It is generated only after installed-package, metadata, strict Twine and
publication-protocol tests pass. Compute the manifest hashes:

```powershell
(Get-FileHash -LiteralPath '.\windows\BUILD_PROVENANCE.json' -Algorithm SHA256).Hash.ToLower()
(Get-FileHash -LiteralPath '.\linux\BUILD_PROVENANCE.json' -Algorithm SHA256).Hash.ToLower()
```

Review these hashes before using them as dispatch inputs. Committing final
artifact hashes back into that same build's source would be circular.

The already qualified local packages are preserved, not rebuilt or replaced by
this preparation. Fresh CI builds from the new commit. The new source archives
include the workflow/tools/docs and necessarily differ from the old TestPyPI
sdist. Do not describe those new archives as already installed from TestPyPI.
`docs/PYPI_RUNTIME_REFERENCE.json` pins the qualified Python/C source, README
and build configuration; preflight rejects drift. Never update that reference
silently. The TestPyPI metadata check proves release existence, not binary
equivalence or installation of the newly built CI artifacts. Review them.

## Preflight without upload

Run **MorphTree official PyPI - reviewed OIDC publication** (`publish-pypi.yml`)
manually on the same unchanged `main` commit, with:

| Input | Value |
|---|---|
| `expected_version` | `2.1.0` |
| `reviewed_commit` | Full new final commit SHA |
| `reviewed_build_run` | New successful `test.yml` run ID |
| `reviewed_build_attempt` | Its exact attempt, normally `1` |
| `windows_manifest_sha256` | Reviewed Windows 3.12 manifest SHA-256 |
| `linux_manifest_sha256` | Reviewed Linux 3.12 manifest SHA-256 |
| `publish_pypi` | **`false`** |
| `approval` | Empty |

Only preflight runs and it cannot request OIDC. It checks repository, branch,
workflow/commit/run/attempt, six successful jobs, artifacts, reviewer and branch
policy, version, runtime reference, source inventory, metadata and SHA-256.
It requires published non-yanked TestPyPI 2.1.0 evidence and HTTP404 from official
PyPI's 2.1.0 metadata endpoint. Existing version, redirects, timeout or
inconclusive responses stop the run.

Inspect logs and `pypi-approved-<run>-<attempt>`. The sealed bundle contains two
wheels and one canonical Windows sdist. The independently built Linux sdist must
match all source member names and bytes; TAR permissions may differ. No silent
deduplication, overwrite, source substitution or build occurs during upload.

## Explicit authorization and protected upload

After reviewing the successful preflight, start a new manual run on the **same
unchanged commit**, using the same CI run/attempt and manifest hashes. Set:

```text
publish_pypi: true
approval: PUBLISH morphtree 2.1.0 TO PYPI
```

Preflight runs again. Then explicitly approve deployment to the `pypi`
environment. Only `publish-pypi` has `id-token: write`. It downloads its own
run's sealed bundle, rechecks receipt/file hashes and commit/run/attempt, and
checks official-index absence immediately before the PyPA action. There is no
checkout, build, package installation or repository-script execution in this job.
The fixed upload URL is `https://upload.pypi.org/legacy/`, using
`pypa/gh-action-pypi-publish@release/v1` with OIDC, attestations enabled and
`skip-existing=false`. No permanent token is used.

Global `publication_approved` and `pypi_upload_enabled` preparation flags remain
false. They grant no upload approval. The dedicated workflow is the only
production route, with explicit dispatch and protected-environment approval.
The legacy TestPyPI tool still rejects `--target pypi`.

If `main` changes, repeat CI and review its new manifests. Do not automatically
retry a partially uploaded release: inspect the actual files first. Do not
overwrite files, change the version, reuse old provenance or weaken checks.
TestPyPI 2.1.0 also must not be uploaded again or replaced.

## Owner verification after publication

Review actual PyPI files and attestations against the approved SHA-256 bundle.
Install a compatible wheel in a fresh environment and check import/native
loading, prediction, persistence and historical models. Any physical training
needs its own finite approved scope. Preserve CI/publish run evidence.
Local tests do not verify hosted CI, actual environment protection, OIDC exchange
or an upload. This procedure is owner-executed; preparation performs none of it.
