from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.agents.abstractor import AbstractorAgent
from src.agents.amf import AMFFusion
from src.agents.extractor import ExtractorAgent
from src.agents.kgsum_agent import KGSumAgent
from src.evaluation.rouge_score import RougeMetric
from src.llms.factory import build_llm, load_llm_factory_config
from src.utils.agent_logger import AgentLogger, atomic_write_json
from src.utils.config import Config
from src.utils.cost import calculate_token_cost


class MixtureOfAgents:
    """MoA pipeline orchestrator: Extractor Agent, KGSum Agent, Abstractor Agent, AMF Fusion."""

    def __init__(self, config: Config, llm: Optional[Any] = None, output_root: Optional[str | Path] = None) -> None:
        self.config = config
        self.llm = llm or build_llm(config)
        self.llm_config = load_llm_factory_config(config)
        self.output_root = Path(output_root or config.get("outputs.root_dir", "outputs"))
        self.extractor = ExtractorAgent(config)
        self.kgsum = KGSumAgent(self.llm, config)
        self.abstractor = AbstractorAgent(self.llm, config)
        self.amf = AMFFusion(self.llm, config)
        self.rouge = RougeMetric(use_stemmer=config.get("evaluation.rouge.use_stemmer", False))

    def process_sample(
        self,
        sample: Mapping[str, Any],
        *,
        write_case_study: bool = True,
        sample_id_suffix: str = "",
    ) -> Dict[str, Any]:
        results: Dict[str, Dict[str, Any]] = {}
        results["extractor"] = self.extractor.process(sample)
        results["kgsum"] = self.kgsum.process(sample)
        results["abstractor"] = self.abstractor.process(sample)
        results["amf"] = self.amf.process(results)

        final_summary = results["amf"]["output_text"]
        reference_summary = str(sample.get("reference_summary", ""))
        metrics = self.rouge.calculate(final_summary, reference_summary) if reference_summary else {}
        artifact_paths: Dict[str, str] = {}

        if write_case_study:
            artifact_paths = self._write_case_study(
                sample=sample,
                results=results,
                final_summary=final_summary,
                reference_summary=reference_summary,
                metrics=metrics,
                sample_id_suffix=sample_id_suffix,
            )

        return {
            "sample_id": sample.get("sample_id"),
            "dataset_name": sample.get("dataset_name"),
            "final_summary": final_summary,
            "reference_summary": reference_summary,
            "agents": results,
            "metrics": metrics,
            "artifact_paths": artifact_paths,
        }

    def _write_case_study(
        self,
        *,
        sample: Mapping[str, Any],
        results: Mapping[str, Dict[str, Any]],
        final_summary: str,
        reference_summary: str,
        metrics: Dict[str, Any],
        sample_id_suffix: str,
    ) -> Dict[str, str]:
        def cost_calculator(model_id: str, usage: Dict[str, Any]) -> Dict[str, Any]:
            return calculate_token_cost(model_id, usage).to_dict()

        sample_id = str(sample["sample_id"]) + sample_id_suffix
        logger = AgentLogger(
            dataset_name=str(sample["dataset_name"]),
            sample_id=sample_id,
            output_root=self.output_root,
            model_id=self.llm_config.model_id,
            base_url=self.llm_config.base_url,
            config_snapshot=self.config.to_dict(),
            cost_calculator=cost_calculator,
        )
        for agent_name in ["extractor", "kgsum", "abstractor", "amf"]:
            logger.log_agent_start(agent_name)
            logger.log_agent_result(agent_name, results[agent_name])
        logger.finalize_sample(
            final_summary=final_summary,
            reference_summary=reference_summary,
            sample_metrics=metrics,
            status="completed",
        )
        knowledge_graph_path = logger.case_dir / "knowledge_graph.json"
        atomic_write_json(knowledge_graph_path, self._knowledge_graph_payload(results["kgsum"]))
        return {
            "agent_log": str(logger.agent_log_path.resolve()),
            "metadata": str(logger.metadata_path.resolve()),
            "final_summary": str(logger.final_summary_path.resolve()),
            "reference_summary": str(logger.reference_summary_path.resolve()),
            "knowledge_graph": str(knowledge_graph_path.resolve()),
        }

    @staticmethod
    def _knowledge_graph_payload(kgsum_result: Mapping[str, Any]) -> Dict[str, Any]:
        metadata = kgsum_result.get("metadata", {})
        kg_metadata = metadata.get("M_kg", {}) if isinstance(metadata, Mapping) else {}
        return {
            "schema_version": "kg_case_study_v1",
            "nodes": kg_metadata.get("entities", []),
            "edges": kg_metadata.get("relations", []),
            "communities": kg_metadata.get("communities", []),
            "community_summaries": kg_metadata.get("community_summaries", []),
            "graph_metrics": kg_metadata.get("graph_metrics", {}),
            "centrality": kg_metadata.get("centrality", {}),
            "key_entities": kg_metadata.get("key_entities", []),
            "fact_density": kg_metadata.get("fact_density"),
            "params": kg_metadata.get("params", {}),
        }
