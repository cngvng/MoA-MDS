import json
from pathlib import Path

import pytest

from src.agents import MixtureOfAgents
from src.agents.prompts import PI_E_SYSTEM_PROMPT, PI_REL_SYSTEM_PROMPT
from src.evaluation.rouge_score import RougeMetric
from src.llms.llm import LLMResult, LLMUsage
from src.utils.config import load_config


class SyntheticLLM:
    """Deterministic plumbing fixture; its output is not a research result."""

    def __init__(self):
        self.calls = []

    def generate_answer(self, system_prompt, user_prompt, **kwargs):
        self.calls.append(system_prompt)
        if system_prompt == PI_E_SYSTEM_PROMPT:
            text = '["Atlas|Harbor"]'
        elif system_prompt == PI_REL_SYSTEM_PROMPT:
            text = '[{"head":"Atlas","tail":"Harbor","relation":"supports","confidence":0.9}]'
        else:
            text = "Atlas supports Harbor research. Harbor shares research findings with Atlas."
        return LLMResult(text, "synthetic", LLMUsage(10, 5, 15))


def test_full_pipeline_writes_case_artifacts(tmp_path):
    config = load_config(overlay_configs=["configs/agents/default.yaml"])
    llm = SyntheticLLM()
    pipeline = MixtureOfAgents(config, llm=llm, output_root=tmp_path)
    sample = dict(sample_id="synthetic-1", dataset_name="multi_news", split="test", language="en",
                  documents=["Atlas supports Harbor research. Harbor shares research findings with Atlas.",
                             "Atlas and Harbor publish research together."],
                  reference_summary="Atlas supports Harbor research.", metadata={})
    result = pipeline.process_sample(sample)
    assert list(result["agents"]) == ["extractor", "kgsum", "abstractor", "amf"]
    assert result["final_summary"].strip()
    assert PI_E_SYSTEM_PROMPT in llm.calls and PI_REL_SYSTEM_PROMPT in llm.calls
    paths = {name: Path(value) for name, value in result["artifact_paths"].items()}
    assert set(paths) == {"agent_log", "metadata", "final_summary", "reference_summary", "knowledge_graph"}
    assert all(path.is_file() and tmp_path in path.parents for path in paths.values())
    assert paths["final_summary"].read_text() == result["final_summary"]
    assert paths["reference_summary"].read_text() == sample["reference_summary"]
    assert json.loads(paths["metadata"].read_text())["status"] == "completed"
    log = json.loads(paths["agent_log"].read_text())
    assert set(log["agents"]) == {"extractor", "kgsum", "abstractor", "amf"}
    graph = json.loads(paths["knowledge_graph"].read_text())
    assert graph["nodes"] and graph["edges"]
    assert graph["schema_version"] == "kg_case_study_v1"


@pytest.mark.parametrize("language,text", [("en", "Research findings are shared."), ("vi", "Nghiên cứu tại Hà Nội.")])
def test_rouge_identity_and_multi_reference(language, text):
    metric = RougeMetric(language=language)
    result = metric.calculate_multi_reference(text, ["Unrelated content", text])
    assert result["rouge1"]["f1"] == pytest.approx(1.0)
    assert result["rougeL"]["f1"] == pytest.approx(1.0)
    assert result["selected_reference_index"] == 1
    assert result["reference_count"] == 2
