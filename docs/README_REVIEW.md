# Public description verification

MorphTree 2.1.0 uses `README.md` as its sole Markdown long description. The owner
explicitly authorized this editorial revision on 31 October 2026, superseding
the earlier immutable-README instruction. The revised text is ready for owner
review; authorization to edit is not a claim that the new wording was already
approved. Earlier README bytes and qualified deliveries remain preserved.

Current source SHA-256: `7d9ffb26fb493feb1ee3c25573ec6e2480e663349b7e7550de290a243552517a`.

`pyproject.toml` selects `README.md` with `text/markdown`. The internal description
contract records the source hash, normalized metadata hash and explicit revision
authority. Installed tests require that authority and exact METADATA text.
The distribution checker requires the complete wheel METADATA and sdist
PKG-INFO descriptions to equal the current README and rejects obsolete duplicate
descriptions. No integrity or publication guard has been weakened.

Examples cover the three tasks, NumPy output shapes and class-column ordering,
inspection, persistence and explicit NumPy standalone inference. Simulated example
checks verify orchestration without training; installed retained-model checks and
the completed CAPACITY132 physical fits are reported separately. The initial
two-round capacity is distinguished from automatic final numerical capacity;
fixed 4/8-round results do not establish a universal default. See [capacity](CAPACITY.md).

## Scope of release checks

- `pip install morphtree` selects the latest compatible release available on the
  index. Local checks do not assert index availability or execute an index install.
- README links use absolute public URLs; local targets and example syntax are
  checked. Remote reachability is not verified because network access is prohibited.
- Windows/Linux local builds and tests do not replace hosted CI, maintainer
  TestPyPI validation or final publication approval.
- Linux ABI auditing does not qualify every glibc distribution. Python 3.15,
  macOS and other architectures remain outside runtime qualification.
- No new default, numerical planner or diagnostic rule was adopted in this revision.

Current README makes no universal predictive or evolutionary superiority claim.
Historical compatibility tests, changelog entries and scientific evidence remain
preserved. No new training is implied by metadata-only artifact rebuilding.
