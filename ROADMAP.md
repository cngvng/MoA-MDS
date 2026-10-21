# Maintenance roadmap

This is an early-stage research implementation. The priorities below describe
planned maintenance, with no guaranteed delivery dates. The maintainer is
[Tuan-Cuong Vuong](https://github.com/cngvng).

## Maintenance foundation

- [x] Correct publication metadata and provide a machine-readable citation.
- [x] Add a code license and contributor documentation.
- [x] Add offline regression tests and a GitHub Actions workflow.
- [x] Provide issue and pull-request templates.

These items describe repository infrastructure. CI run results are visible in
[Actions](https://github.com/cngvng/MoA-MDS/actions/workflows/ci.yml).

## Next priorities

- [ ] Expand malformed-input and dataset-integrity checks, including mismatched
  source/reference counts and empty documents after citation cleanup.
- [ ] Expand orchestration tests for failed samples, interrupted runs, and resume
  behavior; verify all required artifacts before skipping completed cases.
- [ ] Extend mocked backend tests for provider errors, retries, structured content,
  and missing token usage. Validate real hosted/local backends separately when
  credentials and hardware are available.
- [ ] Document a complete reproducibility manifest: dataset revisions/checksums,
  exact sample IDs, prompts, model identifiers, generation settings, tokenization,
  multi-reference aggregation, and metric scales.
- [ ] Re-run documented experiments and publish provenance with results before
  claiming that this release reproduces published scores.
- [ ] Prepare a tagged release with installation checks and reviewed release notes.
- [ ] Evaluate opt-in Codex-assisted issue triage, pull-request review, regression
  test drafting, and release-note preparation, with maintainer review before
  changes are merged or experimental claims are published.

Discuss priorities in [Issues](https://github.com/cngvng/MoA-MDS/issues). Backend
benchmarks, full scientific reproduction, and Codex automation remain planned;
offline CI does not establish those results.
