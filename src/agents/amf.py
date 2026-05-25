from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, List, Mapping, Tuple

from src.agents.common import (
    call_llm,
    elapsed,
    get_method_config,
    make_agent_output,
    normalize_whitespace,
    sum_token_usage,
    timed,
    tokenize,
)
from src.agents.prompts import PI_AMF_SYSTEM_PROMPT, PI_AMF_USER_PROMPT


class AMFFusion:
    """Adaptive Multi-Perspective Fusion from the MoA paper."""

    agent_name = "amf"

    def __init__(self, llm: Any, config: Any = None) -> None:
        method = get_method_config(config, "amf")
        self.llm = llm
        self.output_key = str(method.get("output_key", "S_MoA"))
        self.metadata_key = str(method.get("metadata_key", "M_AMF"))
        self.dominant_weight_threshold = float(method.get("dominant_weight_threshold", 0.5))
        self.dominant_quality_threshold = float(method.get("dominant_quality_threshold", 0.7))
        self.max_summary_tokens = int(method.get("max_summary_tokens", 512))

    def process(self, agent_results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        start = timed()
        normalized = self._normalize_agent_results(agent_results)
        quality_signals = self._extract_quality_signals(normalized)
        consistency = self._validate_cross_perspective_consistency(normalized)
        weights_payload = self._calculate_priority_weights(quality_signals, consistency)
        output_text, llm_call = self._synthesize(normalized, quality_signals, consistency, weights_payload)

        metadata = {
            self.metadata_key: {
                "quality_signals": quality_signals,
                "discrepancies": consistency["discrepancies"],
                "contradictions": consistency["contradictions"],
                "unique_contributions": consistency["unique_contributions"],
                "consistency_score": consistency["consistency_score"],
                "priority_weights": weights_payload["weights"],
                "bonuses": weights_payload["bonuses"],
                "penalties": weights_payload["penalties"],
                "strategy": weights_payload["strategy"],
                "prompt_metadata": {
                    "prompt_id": "Pi_AMF",
                    "inputs": ["r_ext", "r_kg", "r_abs"],
                    "analysis_fields": ["quality_signals", "priority_weights", "discrepancies"],
                },
                "final_llm_call": llm_call,
            },
            "output_key": self.output_key,
        }

        return make_agent_output(
            agent_name=self.agent_name,
            output_text=output_text,
            metadata=metadata,
            llm_calls=[llm_call],
            runtime_seconds=elapsed(start),
            token_usage=sum_token_usage([llm_call["usage"]]),
        )

    def _normalize_agent_results(
        self,
        agent_results: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        required = {"extractor", "kgsum", "abstractor"}
        missing = sorted(required - set(agent_results))
        if missing:
            raise ValueError(f"AMF requires extractor, kgsum, abstractor results. Missing: {missing}")
        return {
            name: {
                "output_text": normalize_whitespace(str(result.get("output_text", ""))),
                "metadata": result.get("metadata", {}),
            }
            for name, result in agent_results.items()
            if name in required
        }

    def _extract_quality_signals(self, results: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        ext_meta = _nested_metadata(results["extractor"]["metadata"], "M_ext")
        kg_meta = _nested_metadata(results["kgsum"]["metadata"], "M_kg")
        abs_meta = _nested_metadata(results["abstractor"]["metadata"], "M_abs")

        return {
            "extractor": {
                "salience": _float(ext_meta.get("salience"), 0.0),
                "density": _float(ext_meta.get("density"), 0.0),
                "coverage": _float(ext_meta.get("coverage"), 0.0),
            },
            "kgsum": {
                "relation_confidence": _float(
                    (kg_meta.get("relation_confidence") or {}).get("average"),
                    0.0,
                ),
                "fact_density": _float(kg_meta.get("fact_density"), 0.0),
                "community_coverage": min(1.0, _float((kg_meta.get("graph_metrics") or {}).get("num_communities"), 0.0) / 5.0),
                "contrastive_signal": min(1.0, _float(kg_meta.get("contrastive_relation_count"), 0.0) / 3.0),
            },
            "abstractor": {
                "quality_score": _float((abs_meta.get("quality_scores") or {}).get("final_quality_score"), 0.0),
                "factual_accuracy": _float((abs_meta.get("quality_scores") or {}).get("factual_accuracy"), 0.0),
                "coverage": _float((abs_meta.get("quality_scores") or {}).get("coverage"), 0.0),
                "fluency": _float((abs_meta.get("quality_scores") or {}).get("coherence"), 0.0),
            },
        }

    def _validate_cross_perspective_consistency(
        self,
        results: Mapping[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        outputs = {name: result["output_text"] for name, result in results.items()}
        overlaps: Dict[str, float] = {}
        discrepancies: List[Dict[str, Any]] = []
        contradictions: List[Dict[str, str]] = []

        for left, right in combinations(outputs, 2):
            left_tokens = set(tokenize(outputs[left]))
            right_tokens = set(tokenize(outputs[right]))
            overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
            overlaps[f"{left}_vs_{right}"] = round(overlap, 6)
            if overlap < 0.08:
                discrepancies.append({"pair": [left, right], "type": "low_overlap", "score": round(overlap, 6)})
            contradictions.extend(_numeric_contradictions(left, outputs[left], right, outputs[right]))

        unique_contributions = {
            name: _unique_terms(name, outputs, limit=12)
            for name in outputs
        }
        consistency_score = max(0.0, 1.0 - (len(discrepancies) * 0.12 + len(contradictions) * 0.18))
        return {
            "overlaps": overlaps,
            "discrepancies": discrepancies,
            "contradictions": contradictions,
            "unique_contributions": unique_contributions,
            "consistency_score": round(consistency_score, 6),
        }

    def _calculate_priority_weights(
        self,
        quality_signals: Mapping[str, Mapping[str, float]],
        consistency: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base_scores = {
            "extractor": _mean(quality_signals["extractor"].values()),
            "kgsum": _mean(quality_signals["kgsum"].values()),
            "abstractor": _mean(quality_signals["abstractor"].values()),
        }
        bonuses = {
            "extractor": 0.05 if quality_signals["extractor"].get("salience", 0.0) > 0.7 else 0.0,
            "kgsum": 0.08 if quality_signals["kgsum"].get("relation_confidence", 0.0) >= 0.6 else 0.0,
            "abstractor": 0.06 if quality_signals["abstractor"].get("fluency", 0.0) >= 0.7 else 0.0,
        }
        consistency_score = _float(consistency.get("consistency_score"), 1.0)
        penalties = {
            agent: 0.10 if consistency_score < 0.7 else 0.0
            for agent in base_scores
        }
        adjusted = {
            agent: max(0.01, base_scores[agent] + bonuses[agent] - penalties[agent])
            for agent in base_scores
        }
        adjusted_sum = sum(adjusted.values()) or 1.0
        weights = {
            agent: round(score / adjusted_sum, 6)
            for agent, score in adjusted.items()
        }
        strategy = self._select_strategy(weights, base_scores)
        return {
            "weights": weights,
            "bonuses": bonuses,
            "penalties": penalties,
            "base_scores": base_scores,
            "strategy": strategy,
        }

    def _select_strategy(self, weights: Mapping[str, float], base_scores: Mapping[str, float]) -> str:
        dominant_agent, dominant_weight = max(weights.items(), key=lambda item: item[1])
        if (
            dominant_weight > self.dominant_weight_threshold
            and base_scores.get(dominant_agent, 0.0) > self.dominant_quality_threshold
        ):
            return f"dominant_{dominant_agent}"
        return "balanced_multi_perspective"

    def _synthesize(
        self,
        results: Mapping[str, Dict[str, Any]],
        quality_signals: Mapping[str, Mapping[str, float]],
        consistency: Mapping[str, Any],
        weights_payload: Mapping[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        text, call = call_llm(
            self.llm,
            prompt_id="Pi_AMF",
            system_prompt=PI_AMF_SYSTEM_PROMPT,
            user_prompt=PI_AMF_USER_PROMPT.format(
                rext=results["extractor"]["output_text"],
                rkg=results["kgsum"]["output_text"],
                rabs=results["abstractor"]["output_text"],
                analysis_metadata=json.dumps(
                    {
                        "q": quality_signals,
                        "Delta_disc": consistency,
                        "w": weights_payload.get("weights", {}),
                        "fusion_strategy": weights_payload.get("strategy"),
                    },
                    ensure_ascii=False,
                ),
            ),
            max_tokens=self.max_summary_tokens,
        )
        if not normalize_whitespace(text):
            ordered = sorted(
                results,
                key=lambda name: weights_payload.get("weights", {}).get(name, 0.0),
                reverse=True,
            )
            text = " ".join(results[name]["output_text"] for name in ordered if results[name]["output_text"])
        return text, call


def _nested_metadata(metadata: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = metadata.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Any) -> float:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 6)


def _numeric_contradictions(left_name: str, left_text: str, right_name: str, right_text: str) -> List[Dict[str, str]]:
    left_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", left_text))
    right_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", right_text))
    if not left_numbers or not right_numbers or left_numbers & right_numbers:
        return []
    return [
        {
            "type": "numeric_mismatch",
            "left_agent": left_name,
            "right_agent": right_name,
            "left_numbers": ", ".join(sorted(left_numbers)[:5]),
            "right_numbers": ", ".join(sorted(right_numbers)[:5]),
        }
    ]


def _unique_terms(name: str, outputs: Mapping[str, str], limit: int) -> List[str]:
    own_tokens = set(tokenize(outputs[name]))
    other_tokens = set()
    for other_name, text in outputs.items():
        if other_name != name:
            other_tokens.update(tokenize(text))
    candidates = [token for token in own_tokens - other_tokens if len(token) > 4]
    return sorted(candidates)[:limit]
