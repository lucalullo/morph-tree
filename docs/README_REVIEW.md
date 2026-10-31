# Public description verification

MorphTree 2.1.0 uses `README.md` as its sole Markdown long description. The owner
explicitly authorized this editorial revision on 31 October 2026, superseding
the earlier immutable-README instruction. The revised text is ready for owner
review; authorization to edit is not a claim that the new wording was already
approved. This revision explains evolutionary discovery cost and the separate
contracts for saved-model inference and frozen-program training. Earlier README
bytes and qualified deliveries remain preserved.

Current source SHA-256: `0c567465e0db8cf91112fef749f83f31d78e5aa48f7088817a3617c91154d9f4`.

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

`get_algorithm_program()` describes the executable family and rules;
`get_recipe()` adds fitted preprocessing, classes and numerical parameters.
Neither inspection method trains a model. Saved weights and standalone inference
exports predict without repeating search. `export_algorithm()` instead creates
an exact-runtime-bound `train_encoded()` fragment: new weights can be trained
with the frozen family, program, parameters and encoder, using compatible encoded
TRAIN data and explicit provenance. That path does not repeat screening, OOF,
mutations or Optuna and is not a raw-data estimator replacement. The README links
to the advanced encoded-data contract rather than implying an unsupported simple
trainer API. Fresh validation is required for reuse on a new problem.

Export inspection and mocked trainer delegation are zero-fit checks, distinct
from physical frozen-program training. Retained-model prediction and save/load
checks provide inference evidence only. The text promises neither a universal
speedup nor an independent predictive benefit from mutations or program reuse.

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
