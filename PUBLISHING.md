# TestPyPI publication — MorphTree 2.1.0

This guide covers **TestPyPI only**, using GitHub Actions and Trusted
Publishing (OIDC). Nothing is uploaded by local checks. Official PyPI is
hard-blocked: no selectable production target, token secret, Twine upload or
production publisher exists in this TestPyPI procedure. The separately reviewed
official PyPI workflow and its own approval gates are documented in
[Official PyPI publication](docs/PYPI_PUBLISHING.md). `publish.yml` is unchanged.

The owner performs the account setup, file transfer, Git operations and remote
workflow runs. The preparation agent does not access GitHub or personal accounts.
README145, the Python+C learning engine, defaults and version remain unchanged.

## 1. Configure the protected environment and Trusted Publisher

In the owner's GitHub repository `lucalullo/morph-tree`, create the environment
named **`testpypi`** in Settings → Environments. Add at least one trusted required
reviewer and restrict deployment branches to `main`. Review bypass settings.
The workflow refuses to proceed without the required-reviewer protection; if
your repository plan cannot provide it, keep publishing blocked and review an
alternative approval design separately.

On **TestPyPI**, manage the `morphtree` project and add a GitHub Trusted Publisher:

| Field | Exact value |
|---|---|
| Repository owner | `lucalullo` |
| Repository name | `morph-tree` |
| Workflow filename | `publish.yml` |
| Environment name | `testpypi` |

If no TestPyPI project exists, use TestPyPI's pending-publisher registration
with the same values and project name `morphtree`. TestPyPI and PyPI have
separate account/publisher configurations. Do not configure this publisher on
official PyPI. No `MORPHTREE_PUBLICATION_TOKEN`, API token, username or password
secret is needed. Do not add one to the workflow.

Official setup references: [configure a publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/),
[publish with OIDC and TestPyPI](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
The action follows the documented `pypa/gh-action-pypi-publish@release/v1`
interface. Maintainers should review third-party action revisions as part of
their normal release review.

## 2. Transfer and verify one final commit

Transfer the locally reviewed files manually, preserving `.git`. The latest
workflow changes must be included; an earlier README145 build alone is not
sufficient provenance for the new publication workflow.

Run `python -I -B tools/check_repository.py`. Commit and push through the owner's
normal process, then record the complete 40-character SHA on `main`. Do not
change any source after this point without creating and testing a new commit.

## 3. Build and review all six CI profiles

Run `.github/workflows/test.yml` manually on `main`, enabling
`confirm_zero_fit_scope`. Wait for all six `zero-fit-<OS>-<Python>` jobs to pass:
Windows/Linux × Python 3.12, 3.13 and 3.14. Skipped jobs do not qualify.

Each artifact set contains its wheel, sdist and **`BUILD_PROVENANCE.json`**.
That manifest records the commit, repository, workflow/run/attempt, source
inventory, README and distribution SHA-256 values. It is written only after
installed-package, metadata, strict Twine and publication-protocol tests pass.

Download and inspect the artifact sets named:

- `morphtree-windows-latest-3.12`
- `morphtree-ubuntu-24.04-3.12`

Review their files, metadata and checksums. Record the SHA-256 of each manifest
itself, for example in PowerShell:

```powershell
(Get-FileHash -LiteralPath 'windows/BUILD_PROVENANCE.json' -Algorithm SHA256).Hash.ToLower()
(Get-FileHash -LiteralPath 'linux/BUILD_PROVENANCE.json' -Algorithm SHA256).Hash.ToLower()
```

Record the successful build run ID and attempt. Use a fresh complete build run
rather than mixing artifacts from different attempts. Old local artifact hashes
are historical evidence, not substitutes for hashes of these final CI bytes.

## 4. Run publication preflight without uploading

Manually run `publish.yml` on that same commit on `main` with these inputs:

| Input | Value |
|---|---|
| `expected_version` | `2.1.0` |
| `reviewed_commit` | Full reviewed commit SHA |
| `reviewed_build_run` | Successful `test.yml` run ID |
| `reviewed_build_attempt` | Its exact attempt, normally `1` |
| `windows_manifest_sha256` | Reviewed Windows manifest SHA-256 |
| `linux_manifest_sha256` | Reviewed Linux manifest SHA-256 |
| `publish_testpypi` | **`false`** |
| `approval` | Empty |

Only `preflight` runs. It has no `id-token: write`. It verifies the run's
repository, branch, workflow, commit, attempt, six successful jobs and immutable
artifact sets, then verifies the owner-approved manifest hashes and every
distribution's bytes, source inventory, version and README. It runs strict
Twine checks and requires HTTP404 for the TestPyPI version endpoint. Network
failure, unexpected redirects or existing version stop the run without retries.

The approved bundle contains exactly two native wheels and **one canonical
Windows-built sdist**. The Linux-built sdist is also checked; it must have exactly
the same member names and bytes. Different TAR permissions may yield different
archive hashes, so the workflow never silently overwrites or arbitrarily picks
a duplicate filename.

Inspect the `testpypi-approved-<run>-<attempt>` artifact and preflight logs.
Local preparation cannot verify live GitHub environment settings, OIDC exchange,
hosted CI or an actual index upload; these steps belong to the owner.

## 5. Explicitly authorize TestPyPI upload

After reviewing the successful preflight and exact artifact hashes, start a new
manual `publish.yml` run on the **same unchanged commit**, using the same build
run/attempt and manifest hashes. Set:

```text
publish_testpypi: true
approval: PUBLISH morphtree 2.1.0 TO TESTPYPI
```

Preflight runs again. Then review and explicitly approve deployment to the
protected **`testpypi`** environment. Only this job has `id-token: write`.
It checks the exact preflight bundle hash, file hashes and commit again, and
rechecks TestPyPI immediately before calling the PyPA publish action with the
fixed URL `https://test.pypi.org/legacy/`. No checkout, repository-code execution,
build or package installation runs in the OIDC job. Attestations are enabled;
`skip-existing` is false. Never rerun a partially uploaded attempt automatically.

The former global `NO_GO_PUBLICATION`/`publication_approved: false` is preserved
as the production block. The new target-specific preparation policy permits
verification and requires runtime manual consent plus protected environment
approval for TestPyPI; it does not approve a publication by itself. Artifact
hashes are approved as dispatch inputs because committing the final build
hashes into that same build's source would create a circular dependency.

## 6. Verify TestPyPI before considering official PyPI

After an owner-performed upload, inspect the actual TestPyPI files and compare
their SHA-256 with the approved bundle. Install the downloaded compatible wheel
in a fresh environment and run the supplied package checks. Dependencies may
need the owner's normal PyPI dependency workflow; avoid ambiguous mixed-index
installation commands. The source build requires the documented compiler.

If `2.1.0` is already present, stop. Do not overwrite, skip existing files,
change artifact names or change the version automatically. Obtain an explicit
owner decision. A404 observed during local preparation is time-specific, so
both workflow jobs repeat the check. Official PyPI uses only the separately
reviewed `publish-pypi.yml`, TestPyPI evidence and its own explicit approval.
