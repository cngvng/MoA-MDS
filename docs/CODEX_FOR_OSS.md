# Codex for Open Source application draft

Prepared on 21 October 2026. Review the live form before submitting. This file
contains proposed application text, not an application submission or an award.

## Maintainer and evidence

- GitHub: [cngvng](https://github.com/cngvng)
- Repository: [cngvng/MoA-MDS](https://github.com/cngvng/MoA-MDS)
- Role: primary/release maintainer, as identified in the [README](../README.md).
- Publication: [Neural Computing and Applications, 2026](https://doi.org/10.1007/s00521-026-12249-y).
- Code license: [MIT](../LICENSE).
- Software checks: [offline tests](../tests) and [CI run history](https://github.com/cngvng/MoA-MDS/actions/workflows/ci.yml).
- Contributor support: [CONTRIBUTING](../CONTRIBUTING.md), issue/PR templates,
  [reproducibility notes](REPRODUCIBILITY.md), and [roadmap](../ROADMAP.md).

No adoption, usage, citation counts, or maintainer-workload hours are asserted
without supporting evidence. Synthetic tests do not verify published scores.

## Why does this repository qualify? (455/500 characters)

I am the first author and release maintainer of MoA-MDS, the MIT-licensed implementation accompanying our 2026 Neural Computing and Applications paper (DOI: 10.1007/s00521-026-12249-y). It supports training-free multi-document summarization across four English/Vietnamese datasets with modular LLM and knowledge-graph agents. Offline regression tests, CI, citation metadata, and contributor guidance support researchers reproducing and extending the work.

## How will you use API credits for your project? (458/500 characters)

I would use API credits for maintainer workflows: assisting pull-request review, triaging reproducible bug reports, drafting regression tests for dataset adapters and model backends, and preparing release notes. I plan to expand offline CI coverage and add opt-in Codex-assisted workflows with maintainer review. Credits would support OSS maintenance; full scientific experiments would be documented separately with model, dataset, and evaluation provenance.

## Anything else we should know? (422/500 characters)

MoA-MDS is an early-stage research OSS project maintained by the paper's first author. I do not claim broad adoption or independent reproduction of published scores. The repository now includes an MIT license, offline tests, CI, contribution guidance, and a public roadmap. My next priorities are data-integrity checks, backend compatibility, failure/resume tests, reproducibility manifests, and a reviewed tagged release.

## Before submitting

Use your own ChatGPT-linked email and account details in the live form; keep
organization IDs and other account details out of this public repository. Check
that the latest CI run passed and that the repository links above remain valid.
Only claim release, backend, or experimental verification when evidence exists.
Codex-assisted automation and a tagged release are still roadmap items.

The [official program page](https://developers.openai.com/community/codex-for-oss)
and [program terms](https://learn.chatgpt.com/docs/codex-for-oss-terms) describe
eligibility and benefits. Selection is discretionary; API credits and Codex
Security require separate consideration. This repository does not claim program
acceptance or OpenAI endorsement.
