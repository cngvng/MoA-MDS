from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, List, Mapping, Tuple

from src.agents.common import (
    call_llm,
    clean_summary_surface,
    elapsed,
    get_method_config,
    is_language_mismatch,
    language_instruction,
    language_retry_instruction,
    make_agent_output,
    normalize_whitespace,
    repetition_ratio,
    sum_token_usage,
    timed,
    tokenize,
    trim_to_word_budget,
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
        self.max_summary_words = method.get("max_summary_words")
        self.max_summary_words = int(self.max_summary_words) if self.max_summary_words else None
        self.extractor_anchor_threshold = float(method.get("extractor_anchor_threshold", 0.55))
        self.kg_surface_noise_penalty = float(method.get("kg_surface_noise_penalty", 0.30))
        self.abstractor_repetition_penalty = float(method.get("abstractor_repetition_penalty", 0.18))
        self.force_extractor_anchor = bool(method.get("force_extractor_anchor", False))
        self.max_kgsum_weight = method.get("max_kgsum_weight")
        self.max_kgsum_weight = float(self.max_kgsum_weight) if self.max_kgsum_weight is not None else None
        self.direct_anchor_output = bool(method.get("direct_anchor_output", False))
        self.direct_anchor_policy = str(method.get("direct_anchor_policy", "drift_guard"))
        self.direct_anchor_min_overlap = float(method.get("direct_anchor_min_overlap", 0.55))
        self.dataset_guidance = str(method.get("dataset_guidance", "") or "").strip()

    def process(self, agent_results: Mapping[str, Mapping[str, Any]], target_language: str | None = None) -> Dict[str, Any]:
        start = timed()
        normalized = self._normalize_agent_results(agent_results)
        target_language = target_language or _target_language_from_results(normalized)
        quality_signals = self._extract_quality_signals(normalized)
        consistency = self._validate_cross_perspective_consistency(normalized)
        language_mismatch = {
            name: is_language_mismatch(result["output_text"], target_language)
            for name, result in normalized.items()
        }
        surface_signals = self._surface_signals(normalized)
        weights_payload = self._calculate_priority_weights(
            quality_signals,
            consistency,
            language_mismatch,
            surface_signals,
        )
        output_text, llm_calls = self._synthesize(
            normalized,
            quality_signals,
            consistency,
            weights_payload,
            target_language,
        )
        output_text = trim_to_word_budget(output_text, self.max_summary_words)

        metadata = {
            self.metadata_key: {
                "quality_signals": quality_signals,
                "surface_signals": surface_signals,
                "discrepancies": consistency["discrepancies"],
                "contradictions": consistency["contradictions"],
                "unique_contributions": consistency["unique_contributions"],
                "consistency_score": consistency["consistency_score"],
                "priority_weights": weights_payload["weights"],
                "bonuses": weights_payload["bonuses"],
                "penalties": weights_payload["penalties"],
                "language_mismatch": language_mismatch,
                "strategy": weights_payload["strategy"],
                "prompt_metadata": {
                    "prompt_id": "Pi_AMF",
                    "inputs": ["r_ext", "r_kg", "r_abs"],
                    "analysis_fields": ["quality_signals", "priority_weights", "discrepancies"],
                    "max_summary_words": self.max_summary_words,
                    "target_language": target_language,
                },
                "final_llm_call": llm_calls[-1],
                "final_llm_calls": llm_calls,
            },
            "output_key": self.output_key,
        }

        return make_agent_output(
            agent_name=self.agent_name,
            output_text=output_text,
            metadata=metadata,
            llm_calls=llm_calls,
            runtime_seconds=elapsed(start),
            token_usage=sum_token_usage([call["usage"] for call in llm_calls]),
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
                "evidence_anchor": min(
                    1.0,
                    _float(ext_meta.get("coverage"), 0.0) * 2.5
                    + _float(ext_meta.get("density"), 0.0),
                ),
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

    def _surface_signals(self, results: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "kg_surface_noise": _kg_surface_noise(result["output_text"]),
                "repetition": repetition_ratio(result["output_text"]),
                "word_count": float(len(result["output_text"].split())),
            }
            for name, result in results.items()
        }

    def _calculate_priority_weights(
        self,
        quality_signals: Mapping[str, Mapping[str, float]],
        consistency: Mapping[str, Any],
        language_mismatch: Mapping[str, bool] | None = None,
        surface_signals: Mapping[str, Mapping[str, float]] | None = None,
    ) -> Dict[str, Any]:
        surface_signals = surface_signals or {}
        base_scores = {
            "extractor": _mean(quality_signals["extractor"].values()),
            "kgsum": _mean(quality_signals["kgsum"].values()),
            "abstractor": _mean(quality_signals["abstractor"].values()),
        }
        extractor_anchor = (
            self.force_extractor_anchor
            or quality_signals["extractor"].get("evidence_anchor", 0.0) >= self.extractor_anchor_threshold
        )
        kg_noise = surface_signals.get("kgsum", {}).get("kg_surface_noise", 0.0)
        abstractor_repetition = surface_signals.get("abstractor", {}).get("repetition", 0.0)
        bonuses = {
            "extractor": 0.18 if extractor_anchor else (0.05 if quality_signals["extractor"].get("salience", 0.0) > 0.7 else 0.0),
            "kgsum": 0.04 if quality_signals["kgsum"].get("relation_confidence", 0.0) >= 0.6 and kg_noise < 0.15 else 0.0,
            "abstractor": 0.04 if quality_signals["abstractor"].get("fluency", 0.0) >= 0.7 and abstractor_repetition < 0.45 else 0.0,
        }
        consistency_score = _float(consistency.get("consistency_score"), 1.0)
        language_mismatch = language_mismatch or {}
        penalties = {}
        for agent in base_scores:
            penalty = 0.10 if consistency_score < 0.7 else 0.0
            if language_mismatch.get(agent):
                penalty += 0.35
            if agent == "kgsum":
                penalty += min(self.kg_surface_noise_penalty, kg_noise)
            if agent == "abstractor" and abstractor_repetition > 0.45:
                penalty += min(self.abstractor_repetition_penalty, abstractor_repetition - 0.35)
            penalties[agent] = penalty
        adjusted = {
            agent: max(0.01, base_scores[agent] + bonuses[agent] - penalties[agent])
            for agent in base_scores
        }
        if extractor_anchor:
            adjusted["extractor"] += 0.25
            adjusted["kgsum"] *= max(0.30, 1.0 - kg_noise)
        if self.max_kgsum_weight is not None and 0.0 <= self.max_kgsum_weight < 1.0:
            non_kg_sum = adjusted["extractor"] + adjusted["abstractor"]
            adjusted["kgsum"] = min(
                adjusted["kgsum"],
                (self.max_kgsum_weight * non_kg_sum) / max(1.0 - self.max_kgsum_weight, 1e-9),
            )
        adjusted_sum = sum(adjusted.values()) or 1.0
        weights = {
            agent: round(score / adjusted_sum, 6)
            for agent, score in adjusted.items()
        }
        strategy = self._select_strategy(weights, base_scores, extractor_anchor)
        return {
            "weights": weights,
            "bonuses": bonuses,
            "penalties": penalties,
            "base_scores": base_scores,
            "strategy": strategy,
        }

    def _select_strategy(
        self,
        weights: Mapping[str, float],
        base_scores: Mapping[str, float],
        extractor_anchor: bool = False,
    ) -> str:
        if extractor_anchor and weights.get("extractor", 0.0) >= 0.30:
            return "extractor_anchored"
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
        target_language: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        user_prompt = PI_AMF_USER_PROMPT.format(
            rext=results["extractor"]["output_text"],
            rkg=results["kgsum"]["output_text"],
            rabs=results["abstractor"]["output_text"],
            analysis_metadata=json.dumps(
                {
                    "q": quality_signals,
                    "Delta_disc": consistency,
                    "w": weights_payload.get("weights", {}),
                    "fusion_strategy": weights_payload.get("strategy"),
                    "length_constraint": {
                        "max_summary_words": self.max_summary_words,
                    },
                    "dataset_guidance": self.dataset_guidance,
                    "target_language": target_language,
                },
                ensure_ascii=False,
            ),
            target_language_instruction=language_instruction(target_language),
            dataset_guidance=self.dataset_guidance or "No additional dataset-level guidance.",
        )
        text, call = call_llm(
            self.llm,
            prompt_id="Pi_AMF",
            system_prompt=PI_AMF_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.max_summary_tokens,
        )
        calls = [call]
        text = clean_summary_surface(text)
        if is_language_mismatch(text, target_language):
            text, retry_call = call_llm(
                self.llm,
                prompt_id="Pi_AMF_language_retry",
                system_prompt=PI_AMF_SYSTEM_PROMPT,
                user_prompt=user_prompt + language_retry_instruction(target_language),
                max_tokens=self.max_summary_tokens,
            )
            calls.append(retry_call)
            text = clean_summary_surface(text)
        if is_language_mismatch(text, target_language):
            ordered = sorted(
                results,
                key=lambda name: weights_payload.get("weights", {}).get(name, 0.0),
                reverse=True,
            )
            language_aligned = [
                results[name]["output_text"]
                for name in ordered
                if results[name]["output_text"] and not is_language_mismatch(results[name]["output_text"], target_language)
            ]
            if language_aligned:
                text = " ".join(language_aligned)
        if not normalize_whitespace(text):
            ordered = sorted(
                results,
                key=lambda name: weights_payload.get("weights", {}).get(name, 0.0),
                reverse=True,
            )
            text = " ".join(results[name]["output_text"] for name in ordered if results[name]["output_text"])
        if _kg_surface_noise(text) > 0.15 or repetition_ratio(text) > 0.55:
            text = _fallback_extractor_anchored_summary(results, weights_payload)
        if self._is_multixscience_mode() and weights_payload.get("strategy") == "extractor_anchored":
            text = self._multi_xscience_anchor_guard(text, results)
        if self.direct_anchor_output:
            text = self._direct_anchor_if_fusion_drifted(text, results, weights_payload)
        return clean_summary_surface(text), calls

    def _is_multixscience_mode(self) -> bool:
        return "multi-xscience" in self.dataset_guidance.lower() or "multi_xscience" in self.dataset_guidance.lower()

    def _multi_xscience_anchor_guard(
        self,
        fused_text: str,
        results: Mapping[str, Dict[str, Any]],
    ) -> str:
        extractor_text = clean_summary_surface(results["extractor"]["output_text"], remove_kg_markers=False)
        abstractor_text = clean_summary_surface(results["abstractor"]["output_text"], remove_kg_markers=False)
        if not extractor_text:
            return fused_text
        overlap = _token_overlap(fused_text, extractor_text)
        fused_relation = _relation_signal_score(fused_text)
        extractor_relation = _relation_signal_score(extractor_text)
        abstractor_relation = _relation_signal_score(abstractor_text)
        if overlap < 0.55 and fused_relation < extractor_relation * 0.85:
            return extractor_text
        if fused_relation < 0.85 and extractor_relation >= max(1.0, abstractor_relation):
            return extractor_text
        return fused_text

    def _direct_anchor_if_fusion_drifted(
        self,
        fused_text: str,
        results: Mapping[str, Dict[str, Any]],
        weights_payload: Mapping[str, Any],
    ) -> str:
        anchor_name, anchor_diagnostics = self._select_direct_anchor(results, weights_payload)
        if isinstance(weights_payload, dict):
            weights_payload["direct_anchor"] = anchor_diagnostics
        anchor_text = clean_summary_surface(results[anchor_name]["output_text"], remove_kg_markers=False)
        if not anchor_text:
            return fused_text
        if self.direct_anchor_policy == "always":
            return anchor_text
        if not fused_text:
            return anchor_text

        overlap = _token_overlap(fused_text, anchor_text)
        fused_words = len(fused_text.split())
        anchor_words = len(anchor_text.split())
        source_texts = [
            clean_summary_surface(result["output_text"], remove_kg_markers=False)
            for result in results.values()
        ]
        fused_score = _anchor_candidate_score(
            fused_text,
            source_texts,
            weight=max(weights_payload.get("weights", {}).values(), default=0.0),
            target_words=self.max_summary_words,
            is_kgsum=False,
        )["score"]
        anchor_score = anchor_diagnostics["candidate_scores"][anchor_name]["score"]
        fused_problem = (
            _kg_surface_noise(fused_text) > 0.15
            or repetition_ratio(fused_text) > 0.55
            or _list_like_noise(fused_text) > 0.35
            or fused_words < max(40, int(anchor_words * 0.45))
        )
        if (
            repetition_ratio(fused_text) > 0.45
            and overlap < self.direct_anchor_min_overlap * 0.40
        ):
            return anchor_text
        if (
            fused_problem
            and overlap < self.direct_anchor_min_overlap * 0.75
            and anchor_score >= fused_score - 0.03
        ):
            return anchor_text
        if (
            anchor_score > fused_score + 0.08
            and (fused_problem or overlap < self.direct_anchor_min_overlap * 0.75)
        ):
            return anchor_text
        return fused_text

    def _select_direct_anchor(
        self,
        results: Mapping[str, Dict[str, Any]],
        weights_payload: Mapping[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        weights = weights_payload.get("weights", {})
        candidate_scores: Dict[str, Dict[str, float]] = {}
        for name, result in results.items():
            text = clean_summary_surface(result["output_text"], remove_kg_markers=False)
            other_texts = [
                clean_summary_surface(other["output_text"], remove_kg_markers=False)
                for other_name, other in results.items()
                if other_name != name
            ]
            candidate_scores[name] = _anchor_candidate_score(
                text,
                other_texts,
                weight=_float(weights.get(name), 0.0),
                target_words=self.max_summary_words,
                is_kgsum=name == "kgsum",
            )

        selected = max(candidate_scores, key=lambda name: candidate_scores[name]["score"])
        if selected == "kgsum":
            best_non_kg = max(
                ("extractor", "abstractor"),
                key=lambda name: candidate_scores.get(name, {}).get("score", -1.0),
            )
            kg_margin = candidate_scores["kgsum"]["score"] - candidate_scores[best_non_kg]["score"]
            if kg_margin < 0.08 or candidate_scores["kgsum"]["kg_surface_noise"] > 0.05:
                selected = best_non_kg
        return selected, {"selected": selected, "candidate_scores": candidate_scores}


def _nested_metadata(metadata: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = metadata.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _target_language_from_results(results: Mapping[str, Dict[str, Any]]) -> str:
    for result in results.values():
        metadata = result.get("metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("target_language"):
            return str(metadata["target_language"]).lower()
        if isinstance(metadata, Mapping) and metadata.get("language"):
            return str(metadata["language"]).lower()
    return "en"


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


def _kg_surface_noise(text: str) -> float:
    lowered = normalize_whitespace(text).lower()
    if not lowered:
        return 0.0
    markers = [
        "cộng đồng",
        "knowledge graph",
        "đồ thị tri thức",
        "centrality",
        "metadata",
        "cấu trúc trung tâm",
        "mức độ liên kết",
        "community",
    ]
    marker_hits = sum(1 for marker in markers if marker in lowered)
    id_hits = len(re.findall(r"\bC\d+\b", text))
    return round(min(1.0, marker_hits * 0.12 + id_hits * 0.04), 6)


def _fallback_extractor_anchored_summary(
    results: Mapping[str, Dict[str, Any]],
    weights_payload: Mapping[str, Any],
) -> str:
    extractor = clean_summary_surface(results["extractor"]["output_text"], remove_kg_markers=False)
    abstractor = clean_summary_surface(results["abstractor"]["output_text"], remove_kg_markers=False)
    if weights_payload.get("weights", {}).get("abstractor", 0.0) > weights_payload.get("weights", {}).get("extractor", 0.0):
        return abstractor or extractor
    return extractor or abstractor


def _dominant_non_kg_agent(weights_payload: Mapping[str, Any]) -> str:
    weights = weights_payload.get("weights", {})
    extractor_weight = _float(weights.get("extractor"), 0.0)
    abstractor_weight = _float(weights.get("abstractor"), 0.0)
    return "abstractor" if abstractor_weight > extractor_weight else "extractor"


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / max(len(right_tokens), 1), 6)


def _jaccard_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _anchor_candidate_score(
    text: str,
    other_texts: List[str],
    *,
    weight: float,
    target_words: int | None,
    is_kgsum: bool,
) -> Dict[str, float]:
    consensus = _mean(_jaccard_overlap(text, other_text) for other_text in other_texts)
    length_fit = _length_fit(len(text.split()), target_words)
    kg_noise = _kg_surface_noise(text)
    repetition = repetition_ratio(text)
    list_noise = _list_like_noise(text)
    surface_quality = max(0.0, 1.0 - kg_noise - repetition - list_noise)
    score = (
        weight * 0.25
        + consensus * 0.35
        + length_fit * 0.25
        + surface_quality * 0.15
    )
    if is_kgsum:
        score -= min(0.12, kg_noise + 0.03)
    return {
        "weight": round(weight, 6),
        "consensus": round(consensus, 6),
        "length_fit": round(length_fit, 6),
        "surface_quality": round(surface_quality, 6),
        "kg_surface_noise": round(kg_noise, 6),
        "repetition": round(repetition, 6),
        "list_like_noise": round(list_noise, 6),
        "score": round(score, 6),
    }


def _length_fit(word_count: int, target_words: int | None) -> float:
    if word_count <= 0:
        return 0.0
    if target_words:
        return max(0.0, 1.0 - abs(word_count - target_words) / max(target_words, 1))
    if 80 <= word_count <= 260:
        return 1.0
    if word_count < 80:
        return max(0.0, word_count / 80)
    return max(0.0, 1.0 - (word_count - 260) / 260)


def _list_like_noise(text: str) -> float:
    words = text.split()
    if len(words) < 80:
        return 0.0
    sentence_count = len(re.findall(r"[.!?]", text))
    if sentence_count >= 3:
        return 0.0
    separator_count = text.count(",") + text.count(";") + text.count("\n")
    capitalized = sum(1 for word in words if word[:1].isupper())
    capitalized_ratio = capitalized / max(len(words), 1)
    separator_ratio = separator_count / max(len(words), 1)
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "by", "from", "that", "which", "as", "is", "are", "was", "were",
        "has", "have", "had", "will", "would", "this", "these", "those",
    }
    stopword_ratio = sum(1 for word in words if word.lower().strip(".,;:!?()[]{}\"'") in stopwords) / max(len(words), 1)
    noise = 0.0
    if sentence_count <= 1:
        noise += 0.45
    if stopword_ratio < 0.18:
        noise += min(0.45, (0.18 - stopword_ratio) * 2.5)
    if capitalized_ratio > 0.45:
        noise += min(0.35, (capitalized_ratio - 0.45) * 1.2)
    if separator_ratio > 0.08:
        noise += min(0.20, (separator_ratio - 0.08) * 1.5)
    return round(min(1.0, noise), 6)


def _relation_signal_score(text: str) -> float:
    lowered = normalize_whitespace(text).lower()
    if not lowered:
        return 0.0
    markers = (
        "however",
        "while",
        "whereas",
        "compared",
        "in contrast",
        "prior work",
        "previous work",
        "related work",
        "our work",
        "unlike",
        "similar",
    )
    hits = sum(1 for marker in markers if marker in lowered)
    return float(hits)


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
