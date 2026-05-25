# A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs

This repository contains the release implementation for the paper:

**A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs**  
Accepted/published in **Neural Computing and Applications**, 2025.

**Code release maintained by: Tuan-Cuong Vuong**  
First author of the paper and maintainer of this implementation.

![MoA overview](assets/overview-architecture.png)


## Installation

This project uses `uv` for Python environment and dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Create your environment file:

```bash
cp .env.example .env
```

For OpenAI-compatible hosted models, set:

```bash
OPENAI_API_KEY=...
```

The default release config uses `gpt-5-nano` for low-cost reproduction smoke tests. Edit [configs/agents/default.yaml](configs/agents/default.yaml) or [configs/base.yaml](configs/base.yaml) to change:

```yaml
llm:
  model_id: "gpt-5-nano"
  base_url:
  api_key_env: "OPENAI_API_KEY"
```

## Self-Hosted Models

The LLM wrapper is OpenAI-compatible. To use an open-source model, start an OpenAI-compatible endpoint and update `llm.model_id` and `llm.base_url`.

Ollama example:

```bash
ollama serve
ollama pull llama3.1
```

```yaml
llm:
  model_id: "llama3.1"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
```

vLLM example:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000
```

```yaml
llm:
  model_id: "meta-llama/Llama-3.1-8B-Instruct"
  base_url: "http://localhost:8000/v1"
  api_key_env: "OPENAI_API_KEY"
```

For llama.cpp, run its OpenAI-compatible server and set `base_url` to that server's `/v1` endpoint.

## Data

The four release datasets are:

- `multi_news`
- `multi_xscience`
- `vn_mds`
- `vims`

Download raw datasets:

```bash
ROOT_DIR=/data/mds HF_CACHE_DIR=/data/hf_cache DATASETS=all \
  bash scripts/download_raw_datasets.sh
```

Processed data and output paths are configured in [configs/storage.yaml](configs/storage.yaml). The current default processed layout is:

```text
/data/moa/processed/
  multinews/test.jsonl
  multixscience/test.jsonl
  vnmds/test.jsonl
  vims/test.jsonl
```

Prepare/check one normalized sample:

```bash
uv run python -m src.data.main --dataset multi_news --split test --num-samples 1
uv run python -m src.data.main --dataset multi_xscience --split test --num-samples 1
uv run python -m src.data.main --dataset vn_mds --split test --num-samples 1
uv run python -m src.data.main --dataset vims --split test --num-samples 1
```

Each sample follows:

```python
{
    "sample_id": str,
    "dataset_name": str,
    "split": str,
    "language": "en" | "vi",
    "documents": list[str],
    "reference_summary": str,
    "metadata": dict,
}
```

## One-Sample Reproduction

Run one sample through the full MoA release path:

```bash
uv run python -m src.run --dataset multi_news --split test --num-samples 1
```

Or use the dataset scripts:

```bash
scripts/run_one_multi_news.sh
scripts/run_one_multi_xscience.sh
scripts/run_one_vn_mds.sh
scripts/run_one_vims.sh
```

Case-study artifacts are written under:

```text
/data/moa/outputs/{dataset_name}/case_studies/{sample_id}_moa/
  agent_log.json
  metadata.json
  final_summary.txt
  reference_summary.txt
  knowledge_graph.json
```

`knowledge_graph.json` stores the KGSum case graph for later KG metrics:

```text
nodes, edges, communities, community_summaries, graph_metrics, centrality, key_entities
```

## Full Main-Results Runs

The full scripts are nohup-capable and support resume/skip-existing:

```bash
nohup scripts/run_multi_news.sh > logs/main_results/multi_news_full.nohup.log 2>&1 &
nohup scripts/run_multi_xscience.sh > logs/main_results/multi_xscience_full.nohup.log 2>&1 &
nohup scripts/run_vn_mds.sh > logs/main_results/vn_mds_full.nohup.log 2>&1 &
nohup scripts/run_vims.sh > logs/main_results/vims_full.nohup.log 2>&1 &
```

## Citation for this paper can be cited as:

Authors: **Tuan-Cuong Vuong**, Mai Xuan Trang, Nguyen Tien Cuong, Vu-Duc Ngo, Thien Van Luong.

```bibtex
@article{Vuong2025MixtureAgents,
  title = {A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs},
  author = {T. -C. Vuong and M. X. Trang and N. T. Cuong and V. -D. Ngo and T. V. Luong},
  year = {2025},
  date = {2025-01-01},
  journal = {Neural Computing and Applications},
  pubstate = {published},
  tppubtype = {article}
}
```