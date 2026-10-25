# MorphTree 1.0.0 - TestPyPI first-public-release checklist

This repository prepares a **manual** GitHub -> TestPyPI -> PyPI flow for version `1.0.0`. No publication happens by committing these files. TestPyPI is a public, independent package index. Uploading there permanently consumes the uploaded filename/version on that index and **cannot be overwritten**; do not trigger the publish workflow just to discover a packaging bug.

## 1. Qualify the source on GitHub

1. Upload the reviewed repository files to **`lucalullo/morph-tree`** on GitHub (private until you decide otherwise). Do not upload raw private experiment archives, tokens, Python environments or `dist/`.
2. Commit and push the exact source snapshot. Check `python -I -B tools/check_repository.py --git-head` in that checkout.
3. GitHub **Actions -> MorphTree 1.0.0 - zero-fit qualification -> Run workflow** on `main`; tick `confirm_zero_fit_scope`. Require full **PASS** on the exact source commit. Save the job logs. This is not a new LIVE training run.
4. Once that commit is approved and unchanged, tag it **`v1.0.0`** and push the tag. From a local clone at that commit:

   ```bash
   git tag -a v1.0.0 -m "MorphTree 1.0.0"
   git push origin v1.0.0
   ```

   Check the tag resolves to the exact passing commit. Do not retag/rewrite it after publication.

## 2. Configure TestPyPI Trusted Publishing (once)

Use your separate [TestPyPI account](https://test.pypi.org/account/register/). On [TestPyPI](https://test.pypi.org/), create a **pending trusted publisher** for the new project if it does not exist yet, or add a publisher to the existing project if you own it. Register these fields using the **exact owner/repository names actually shown on GitHub**:

| Field | Value |
| --- | --- |
| Project name | `morphtree` |
| Owner | `lucalullo` |
| Repository | `morph-tree` |
| Workflow file | `publish.yml` |
| Environment | `testpypi` |

In GitHub -> repository Settings -> Environments, create `testpypi`. Review Actions permissions and billing/usage controls personally. No tokens/secrets are required: this workflow uses OIDC with `id-token: write` only on its publishing job. Do not configure the `pypi` environment or official PyPI publisher unless you separately decide to publish the official package.

**Before uploading:** check that the name and version are available to your account on TestPyPI. A package listing on PyPI does not reserve a TestPyPI name, or vice versa.

## 3. Publish to TestPyPI only

In GitHub -> Actions -> **MorphTree 1.0.0 - manually publish validated artifacts** -> Run workflow:

- Choose the approved `main` commit (or its `v1.0.0` tag) as workflow ref.
- `destination`: **`testpypi`**.
- `confirm_release`: **true**, only when you intentionally authorize the public TestPyPI upload.
- `testpypi_run_id`: leave blank.

The workflow refuses mismatched commits, tags or missing successful hosted CI. It builds **one** wheel and **one** sdist on Windows/Python 3.12, writes SHA-256 provenance, keeps the exact files as a short-lived Actions artifact, and uploads only to TestPyPI. It will fail without the correct Trusted Publisher identity and GitHub `testpypi` environment. A failed upload should **not** be retried without checking whether either distribution was already accepted by the index.

Record the successful run ID, commit SHA, tag, published artifact names and SHA-256. The artifact retention is **1 day**, so future promotion to official PyPI must happen while those exact files remain available or fail closed.

## 4. Verify what users will download

Inspect `https://test.pypi.org/project/morphtree/1.0.0/` and confirm both `.whl` and `.tar.gz` are listed. Download the file **without its dependencies** from TestPyPI, then install the downloaded wheel while satisfying dependencies from regular PyPI, in a clean **Windows x64 / CPython 3.12** environment:

```powershell
py -3.12 -m venv .venv-testpypi
.venv-testpypi\Scripts\python -m pip download --no-deps --index-url https://test.pypi.org/simple/ --dest test-download morphtree==1.0.0
# Replace <downloaded-wheel-file> with the exact .whl filename in test-download\.
.venv-testpypi\Scripts\python -m pip install .\test-download\<downloaded-wheel-file>
.venv-testpypi\Scripts\python -m pip check
.venv-testpypi\Scripts\python -c "import morphtree; print(morphtree.__version__)"
```

Compare the downloaded wheel SHA-256 against the workflow's proven SHA-256. To verify the source distribution as well, download it explicitly from the file listing and compare its SHA-256. Run a **separately authorized** small LIVE `fit()/predict()` smoke test for each task after the install; the publishing workflow intentionally performs no native training. TestPyPI's dependency index is incomplete, so do not use its index for dependencies.

## 5. Future official PyPI: NOT authorized by TestPyPI approval

The **same** `publish.yml` also defines a separately gated `pypi` option. It requires a successful TestPyPI origin run ID, the `pypi` GitHub environment with human approval, and a Trusted Publisher independently configured at pypi.org. It downloads the exact TestPyPI run's artifacts (no rebuild), verifies SHA-256 and provenance, then can publish to official PyPI **only when separately authorized**. Do not select `pypi` during the current TestPyPI experiment. If artifacts expire or anything differs, stop; do not silently rebuild under the original `1.0.0` version.

## Official references

- [TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/)
- [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Pending Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
