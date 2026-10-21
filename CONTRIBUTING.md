# Contributing to MoA-MDS

Contributions to dataset adapters, model compatibility, tests, reproducibility,
and documentation are welcome. Tuan-Cuong Vuong ([@cngvng](https://github.com/cngvng))
maintains this research implementation and reviews changes.

## Development setup

Fork and clone the repository, then install [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --group dev
uv run --locked python -m pytest
uv run --locked python -m compileall -q src scripts tests
```

Python 3.10 and 3.12 are the CI targets. The tests use temporary synthetic data,
an injected LLM for pipeline tests, and a mocked SDK transport for backend tests.
`pytest-socket` blocks network calls during tests. No dataset downloads, API keys,
or model weights are required. Installing dependencies still requires network
access on a fresh machine.

For a real run, follow the README, configure local storage paths, and create a
local `.env` from `.env.example`. Hosted inference can incur provider charges.

## Reporting a bug or proposing a change

Use [GitHub Issues](https://github.com/cngvng/MoA-MDS/issues) with the relevant
template. Include your commit, Python version, dataset, model/backend, exact
command, expected behavior, and a sanitized traceback. Prefer a small synthetic
example; omit API keys, private documents, and confidential model responses.

For a larger change, describe the proposed behavior in an issue before doing
substantial implementation work. Documentation corrections and focused fixes can
be submitted directly as pull requests.

## Pull requests

1. Create a branch for one focused change.
2. Add regression coverage for behavior changes; keep tests offline.
3. Update documentation if configuration, artifacts, or usage changes.
4. Run the checks above and describe their results in the pull request.

Keep the unified sample schema and existing case-study artifacts compatible, or
document a migration. Changes to prompts, selection policies, tokenization,
reference aggregation, or metric scales affect scientific comparisons: state
them explicitly. A synthetic test or alternate-model run does not reproduce the
paper's results. Report the dataset revision, sample selection, model, config,
and evaluation settings with any experimental claim.

AI-assisted contributions follow the same review and evidence requirements.
Review generated code and disclose material limitations in the pull request.
Contributions are made under the repository's [MIT license](LICENSE); third-party
datasets, model weights, dependencies, and the published article retain their
own terms.
