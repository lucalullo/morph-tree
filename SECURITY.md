# Security

Models are declarative typed JSON. Loaders validate duplicate keys, non-finite
values, bounded memory use, classes, encoders, model graphs, provenance and
recorded integrity. Loading does not execute pickle or arbitrary source code.
An expected SHA-256 establishes byte custody, not an unknown author's identity.

Keep original inputs and model checksums when reproducibility matters. Loading
does not restore artifact-selected output destinations. Existing working
directories reject link/reparse ancestors. Writes use atomic publication where
supported; partial campaign evidence is kept separately from usable models.

Time and memory estimates and callback stops are cooperative. They are not an
absolute OS guarantee for arbitrary workloads. Returned prediction vectors and
model serialization still need memory. Use process isolation and external
resource policies when handling untrusted workloads.

Exported Python is source code: inspect its provenance before executing it.
Training fragments and inference exports have different documented contracts.
Neither model files nor exports grant access to sealed scientific data or owner
execution budgets. Private datasets, permits, credentials and raw research logs
are not distributed. No blanket absence-of-vulnerabilities claim is made.

Native kernels validate owned buffer lengths, indices and graphs. A hash-bound package manifest restricts library loading. Signed-zero keys and RLock protect the cache; PyDLL retains the GIL. Scoped sanitizer and malformed-buffer tests do not prove absence of all memory/concurrency defects.
