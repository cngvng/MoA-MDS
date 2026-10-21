# Reproducibility notes

## What the automated checks establish

The [CI workflow](../.github/workflows/ci.yml) installs the locked dependencies,
checks Python syntax and CLI imports, and runs offline tests on Python 3.10 and
3.12. Tests generate temporary, synthetic raw files for each dataset format,
check normalization and sample selection, load release config overlays, exercise
the LLM wrapper with a mocked SDK call, and run the actual MoA agents with an
injected deterministic LLM to verify artifact creation.

These are software checks. Synthetic summaries and test ROUGE values are not
experimental results. CI does not download full datasets or weights, invoke a
hosted model, validate a live inference server, or reproduce the paper's tables.

## Before running experiments

1. Install dependencies with `uv sync --locked` and record the Git commit and
   Python version. Keep `uv.lock` with your experiment records.
2. Download the raw datasets using the README instructions and respect the
   upstream terms. Set all storage paths in `configs/storage.yaml` to writable
   locations matching the downloaded data.
3. Prepare samples and inspect `metadata.json` alongside each processed JSONL.
   Check `count_matches_expected`, sample IDs, reference availability, and the
   raw source/reference counts. A count mismatch is not a successful full run.
4. Review the agent and dataset overlays. `src.run` loads the base/storage
   configuration, then `configs/agents/default.yaml`, then the dataset overlay,
   then any `--overlay-config` paths in command order.
5. Record the exact model, endpoint/server version, prompts, generation settings,
   and dataset selection. Changing the default model or provider can change the
   outputs. Start with one sample before scheduling a full run.

For an alternate backend, use an overlay without editing tracked defaults:

```yaml
# local-backend.yaml
llm:
  model_id: "your-model-name"
  base_url: "http://localhost:8000/v1"
  generation:
    reasoning_effort: null
    max_tokens_param: "max_tokens"
```

```bash
uv run --locked python -m src.run --dataset multi_news --num-samples 1 \
  --overlay-config local-backend.yaml --output-root outputs/local-check
```

Choose parameters supported by your server. This example is not a verified
compatibility claim for every OpenAI-compatible implementation.

## Evaluation details that affect comparisons

- The Vietnamese adapters use the first configured number of naturally sorted
  clusters for the test selection. Record the selected IDs rather than assuming
  the entire raw collection is evaluated.
- ViMs retains multiple gold summaries. The dataset overlays configure the
  multi-reference aggregation policy; record it with the results.
- `RougeMetric` returns fractional scores (0–1). For Vietnamese inputs, the
  current code uses a whitespace-stripped character tokenizer. Report that
  policy explicitly; those scores are not interchangeable with word-tokenized
  ROUGE. A percentage in a paper or table requires a documented conversion.
- The default lexical extractor and `gpt-5-nano` settings are release defaults.
  Their presence does not show that the published experimental setup or scores
  have been reproduced.

## Preserve evidence

Keep dataset-level `metrics.json` and `metadata.json`, case-study artifacts,
configuration snapshots, raw-data revisions/checksums, and the exact sample IDs.
Check attempted/completed/failed counts and failures before reporting success.
Resume currently recognizes a case by completed metadata, an agent log, and a
final-summary file; inspect the reference and knowledge-graph artifacts too.
Use a fresh output root after changing a model, prompt, config, or dataset to
avoid mixing resumed results from different setups.

Agent logs and config snapshots can contain source text, summaries, and endpoint
URLs. Review them before sharing; never publish credentials or private inputs.
The full provenance manifest and independent paper-score verification are
[planned maintenance work](../ROADMAP.md).
