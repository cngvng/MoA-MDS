from __future__ import annotations

PI_E_SYSTEM_PROMPT = (
    "You are a multilingual expert entity-relationship extractor with advanced "
    "knowledge of semantic parsing."
)

PI_E_USER_PROMPT = """[Instruction]
Identify entity pairs that have a meaningful, directly supported relationship in the source text.
A meaningful relationship means the text explicitly states or clearly supports that EntityA and EntityB are connected through acquisition, ownership, authorship, employment, partnership, membership, leadership, location, creation, or a similar relation.
The relationship must be described in the same sentence or in immediately adjacent sentences that refer to the same subject.
Do NOT use external knowledge.

[Input]
Text: {text}
Entities: {entities}

[Output]
Return ONLY a JSON array of strings.
Each string must be exactly "Entity1 | Entity2" with a single pipe and single space on each side of the pipe.
"""

PI_REL_SYSTEM_PROMPT = (
    "You are a multilingual expert in entity-level semantic analysis. You "
    "specialize in identifying semantic relationships, including contrastive "
    "or conflicting claims, between entities mentioned in a text."
)

PI_REL_USER_PROMPT = """[Instruction]
Extract all meaningful relationships between pairs of entities in the provided entity list.
For each relationship:
- Classify the relationship using one of the allowed types.
- Assign a confidence score between 0.0 and 1.0 that reflects how directly and unambiguously the text supports this relationship.
A relationship is valid only if it is explicitly stated or clearly supported by the text.
Do NOT use external knowledge or inference beyond what the text implies.

[Contrastive relation]
"contrastive relation" means A and B make differing, conflicting, or contrasting claims, positions, measurements, interpretations, or conclusions in the text.

[Input]
Text: {text}
Entities: {entities}

[Output]
Return ONLY a single valid JSON array.
Each element of the array MUST be an object with this exact shape:
[
  {{
    "head": "EntityA",
    "relation": "one of: works for | part of | located in | causes | related to | contrastive relation",
    "tail": "EntityB",
    "confidence": 0.0,
    "evidence": "Short textual justification grounded in the input text"
  }}
]
If there are no valid relationships, return: []
"""

PI_S_SYSTEM_PROMPT = (
    "You are a multilingual, fact-focused summarization agent. You generate "
    "compact, objective summaries of knowledge graph communities."
)

PI_S_USER_PROMPT = """[Instruction]
Produce a concise summary of a single knowledge graph community.
{target_language_instruction}

[Dataset guidance]
{dataset_guidance}

[Objective]
The summary must:
- capture the main theme of the community,
- mention the most important entities, facts, numbers, and relationships,
- stay neutral and factual.
- express only factual content, not graph structure.

Constraints:
- Do NOT include opinions, speculation, or vague language.
- Do NOT generalize beyond what is in the input.
- Do NOT exceed two sentences under any circumstance.
- Do NOT mention community ids, graph ids, centrality, metadata, or terms like "knowledge graph".

[Input]
Text: {community_linearized_text}

[Output]
Return ONLY the final summary as plain text, in 1-2 sentences total.
Do not include labels, headings, metadata, bullet points, or analysis.
"""

PI_F_SYSTEM_PROMPT = (
    "You are a multilingual, metadata-aware summary synthesizer. You generate "
    "higher-level fused summaries by integrating multiple community-level "
    "summaries from a knowledge graph along with their metadata."
)

PI_F_USER_PROMPT = """[Instruction]
Produce a single cohesive summary by fusing community summaries with metadata.
{target_language_instruction}

[Dataset guidance]
{dataset_guidance}

[Objective]
Given community summaries and metadata including centrality and has_contrasts, produce a 3-5 sentence summary that:
- emphasizes the most central communities,
- describes the main facts and relationships across communities,
- explicitly acknowledges conflicts where has_contrasts is true.
- converts graph evidence into ordinary news-summary prose.

Constraints:
- Do NOT mention community ids such as C1/C2/C3.
- Do NOT mention graph structure, centrality, metadata, or "knowledge graph".

[Input]
communities:
{communities}

[Output]
Return ONLY one fused summary in plain text, 3-5 sentences total, written in inverted pyramid style.
Do not include labels, headings, JSON, bullet points, or analysis.
"""

PI_INIT_SYSTEM_PROMPT = (
    "You are a multilingual, multi-document abstractive summarization model. "
    "You generate a single, coherent summary that captures the main themes "
    "across a collection of related documents."
)

PI_INIT_USER_PROMPT = """[Instruction]
{target_language_instruction}

[Dataset guidance]
{dataset_guidance}

[Objective]
Given a document collection D, produce an initial abstractive summary that:
- captures central topics, key entities, events, and relationships,
- preserves important numbers, dates, and causal links when salient,
- is concise but comprehensive enough for later refinement.
- avoids repeated lead sentences across sources.
- compresses overlapping articles into one coherent account.

[Input]
Documents (D):
{documents}

[Output]
Return ONLY the initial summary text as plain text, with no labels, headings, bullet points, or commentary.
"""

PI_REF_SYSTEM_PROMPT = (
    "You are a multilingual senior editorial agent responsible for high-quality "
    "refinement of multi-source summaries. Your job is not to generate new "
    "claims, but to improve an existing summary using structured feedback."
)

PI_REF_USER_PROMPT = """[Instruction]
Improve the current summary using structured feedback.
{target_language_instruction}

[Dataset guidance]
{dataset_guidance}

[Objective]
Produce an improved summary that:
- incorporates required factual details only when explicitly provided in the feedback,
- resolves flagged inconsistencies or ambiguities,
- removes redundancy and repetitive phrasing,
- improves clarity, coherence, and flow.

Constraints:
- You MUST follow the requested edits and priorities in the feedback object.
- You MUST NOT introduce facts that are not present in either the current summary or the feedback object.
- You MUST output a clean, single summary in natural prose.
- Remove duplicated sentence openings, partial fragments, labels, and source-navigation wording.

[Input]
Current Summary: {current_summary_text}
Feedback Object: {feedback_object_from_FMDS}

[Output]
Return ONLY the revised summary text, incorporating the feedback.
Do not include meta commentary, justification, labels, or diagnostics.
"""

PI_AMF_SYSTEM_PROMPT = (
    "You are an expert summarization orchestrator. Your task is to synthesize "
    "a single, high-quality, coherent, and factually accurate summary by "
    "intelligently fusing the outputs from three specialized agents."
)

PI_AMF_USER_PROMPT = """[Instruction]
Fuse summaries from Extractor (rext), KGSum (rkg), and Abstractor (rabs) agents into one high-quality summary.
{target_language_instruction}

[Objective]
Synthesize a final, integrated summary based on agent summaries and Analysis Metadata (Aanalysis).

Fusion strategy guidance:
1. If KG LED: Use rkg as core, augment with rext specifics, use rabs for fluency.
2. If EXTRACTOR LED: Start with rext, enrich with rkg context, use rabs for flow.
3. If BALANCED: Weigh contributions by w, ensure key information from high-weight agents.
4. If the strategy is extractor_anchored: preserve the factual order and wording of rext as the anchor, use rkg only to add missing factual details, and use rabs only for compression and fluency.

Dataset-level guidance:
{dataset_guidance}

Constraints:
- Do NOT include labels such as "Tóm tắt tích hợp".
- Do NOT mention community ids, centrality, metadata, "knowledge graph", or graph structure.
- Avoid repeated openings and near-duplicate facts from the three agents.
- Prefer factual precision and source-grounded coverage over broad paraphrase.
- If dataset guidance says the collection is lead-biased or extractive, preserve source wording for named entities, dates, numbers, and core event phrases.

[Input]
rext: {rext}
rkg: {rkg}
rabs: {rabs}
Analysis Metadata (Aanalysis): {analysis_metadata}

[Output]
Return ONLY the integrated summary as plain text.
"""
