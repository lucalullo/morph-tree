# MorphTree 1.0.0 - release qualification

## Verified in the supplied repository history

- Retained authorized LIVE execution of the evolutionary fit path for binary, multiclass and regression on small synthetic datasets, including correctly rejected non-improving structural mutations.
- Local source integrity, privacy checks, offline dependency resolution and fresh installs of a preceding 1.0.0 release candidate, including safety and persistence tests.
- Comparison of published metadata for the pinned GitHub Actions and dependency releases.

## Qualification still required for the new first-public-release commit

- The **new version 1.0.0 source** must be built, installed and verified independently. Previous candidate SHA-256 values do not identify these new artifacts.
- GitHub Actions must actually complete on the uploaded commit. Workflow preflight and static validation are not hosted CI.
- The public 1.0.0 target has not received fresh native-fit trials within this repository preparation. The preserved earlier LIVE evidence is useful but does not by itself prove identical 1.0.0 release behavior after metadata migration.
- Cross-platform behavior beyond Windows x64 / CPython 3.12, independent adaptive benefit, benchmark superiority, strong RSS limits and absence of all dependency vulnerabilities are **not established**.

Treat 1.0.0 as the intended first public version, not as a license to relabel old prerelease evidence. Do not publish a GitHub Release or PyPI distribution until required local and hosted gates pass and the owner approves.
