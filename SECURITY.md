# Security policy - MorphTree 1.0.0

MorphTree uses bounded declarative JSON for model loading. Input and saved model structures are validated, including duplicate keys, non-finite numbers and model consistency. Do not open untrusted artifacts without checking provenance and, where applicable, the expected SHA-256. Model loading does not use pickle or unrestricted code evaluation.

The public API uses only user-provided datasets and does not inherit permissions over research archives or sealed evaluation sets. Full private scientific receipts, datasets and owner authorizations are **not** part of this repository.

Time and memory controls are partly **cooperative**. They do not provide an operating-system-enforced hard cap on peak RSS, CPU time or the safety of arbitrary input workloads. Isolate untrusted work and apply suitable external resource limits.

Qualified retained evidence covers Windows x64 / CPython 3.12 on previously built development candidates. The first public 1.0.0 build and hosted CI require their own gate. No general absence-of-vulnerabilities claim is made.

Report suspected security issues to the maintainer privately. Do not place exploit credentials, personal data or confidential research documents in public issues. The curated [research archive](research/README.md) explains what has been withheld.
