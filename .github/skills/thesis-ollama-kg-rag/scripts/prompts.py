"""Fixed prompts and JSON schemas. Never edit after the test split is frozen; PROMPT_VERSION is logged."""
from __future__ import annotations

import hashlib
import json

from common import options_block

# ---------------------------------------------------------------- answer generation (all configs)
GEN_SYSTEM_RAG = (
    "You are a 3GPP telecommunications standards expert. Answer the multiple-choice question using ONLY "
    "the numbered evidence items. Passage items are labelled P1, P2, ...; knowledge-graph statements are "
    "labelled G1, G2, ... and are derived from the cited specification clause. Use standards terminology. "
    "Do not add technical facts that the evidence does not support; if the evidence is insufficient or "
    "conflicting, still choose the most likely option and say so in the justification. "
    "Return JSON with: option (integer option number), justification (at most 120 words), "
    "citations (list of evidence labels that support the answer)."
)
GEN_SYSTEM_CLOSED = (
    "You are a 3GPP telecommunications standards expert. Answer the multiple-choice question from your own "
    "knowledge. Return JSON with: option (integer option number), justification (at most 120 words), "
    "citations (empty list)."
)
GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "option": {"type": "integer"},
        "justification": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["option", "justification", "citations"],
}


def gen_messages(q: dict, evidence_block: str | None) -> list[dict]:
    body = f"Question: {q['question']}\nOptions:\n{options_block(q)}"
    if evidence_block is None:
        return [{"role": "system", "content": GEN_SYSTEM_CLOSED}, {"role": "user", "content": body}]
    return [{"role": "system", "content": GEN_SYSTEM_RAG},
            {"role": "user", "content": f"Evidence:\n{evidence_block}\n\n{body}"}]


# ---------------------------------------------------------------- answerability verifier
VERIFY_SYSTEM = (
    "You verify whether 3GPP Release 17 specification passages support a given answer to a question. "
    "Be strict: a passage supports the answer only if it explicitly states the information needed to "
    "derive the keyed option. For every supporting passage copy a VERBATIM quote (a contiguous span of at "
    "least one full sentence, copied character for character) and give its passage_id and clause. "
    "verdict: 'supports' (keyed option fully derivable), 'partial' (only part derivable), 'no_support'. "
    "release_conflict: true if the passages contradict the keyed option or the question targets another release."
)
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supports", "partial", "no_support"]},
        "release_conflict": {"type": "boolean"},
        "evidence": {"type": "array", "items": {
            "type": "object",
            "properties": {"passage_id": {"type": "string"}, "clause": {"type": "string"},
                           "quote": {"type": "string"}},
            "required": ["passage_id", "quote"]}},
    },
    "required": ["verdict", "release_conflict", "evidence"],
}


def verify_messages(q: dict, passages: list[dict]) -> list[dict]:
    ev = "\n\n".join(f"passage_id: {p['id']}\n{p['header']}\n{p['text']}" for p in passages)
    keyed = q["options"][q["answer"] - 1]
    user = (f"Question: {q['question']}\nOptions:\n{options_block(q)}\n"
            f"Keyed answer: option {q['answer']}: {keyed}\n\nPassages:\n{ev}")
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


ADJUDICATE_SYSTEM = (
    "You audit a multiple-choice question about 3GPP standards against supporting evidence. "
    "other_option_defensible: true if, given the evidence, a different option is also defensible. "
    "answerable_without_standards_text: true ONLY if the question is basic elementary common sense or generic trivia "
    "that requires no telecommunications or engineering knowledge. Set to false for 3GPP architectural, procedural, "
    "protocol, timer, or technical standards questions."
)
ADJUDICATE_SCHEMA = {
    "type": "object",
    "properties": {"other_option_defensible": {"type": "boolean"},
                   "answerable_without_standards_text": {"type": "boolean"}},
    "required": ["other_option_defensible", "answerable_without_standards_text"],
}


def adjudicate_messages(q: dict, quotes: list[str]) -> list[dict]:
    keyed = q["options"][q["answer"] - 1]
    user = (f"Question: {q['question']}\nOptions:\n{options_block(q)}\n"
            f"Keyed answer: option {q['answer']}: {keyed}\n\nEvidence quotes:\n" + "\n".join(f"- {x}" for x in quotes))
    return [{"role": "system", "content": ADJUDICATE_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- KG extraction
def kg_extract_messages(passage: dict, entity_types: list[str], relations: list[str], max_triples: int) -> list[dict]:
    system = (
        "You extract a knowledge graph from 3GPP specification text. Output only facts EXPLICITLY stated in "
        "the passage; never use outside knowledge. Use entity names exactly as written (keep abbreviations "
        f"such as AMF, SMF, N2, RRCSetup). Allowed entity types: {', '.join(entity_types)}. "
        f"Allowed relations: {', '.join(relations)}. Return at most {max_triples} triples."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{passage['header']}\n{passage['text']}"}]


def kg_extract_schema(entity_types: list[str], relations: list[str], max_triples: int) -> dict:
    return {
        "type": "object",
        "properties": {"triples": {"type": "array", "maxItems": max_triples, "items": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"}, "subject_type": {"type": "string", "enum": entity_types},
                "relation": {"type": "string", "enum": relations},
                "object": {"type": "string"}, "object_type": {"type": "string", "enum": entity_types},
            },
            "required": ["subject", "subject_type", "relation", "object", "object_type"]}}},
        "required": ["triples"],
    }


TRIPLE_VERIFY_SYSTEM = (
    "You check knowledge-graph triples against a source passage from a 3GPP specification. "
    "For each numbered triple answer entailed=true only if the passage explicitly states that relation "
    "between those two entities."
)
TRIPLE_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object", "properties": {"index": {"type": "integer"}, "entailed": {"type": "boolean"}},
        "required": ["index", "entailed"]}}},
    "required": ["results"],
}


def triple_verify_messages(passage: dict, triples: list[dict]) -> list[dict]:
    lines = "\n".join(f"{i}. ({t['subject']}) -[{t['relation']}]-> ({t['object']})" for i, t in enumerate(triples))
    return [{"role": "system", "content": TRIPLE_VERIFY_SYSTEM},
            {"role": "user", "content": f"Passage:\n{passage['header']}\n{passage['text']}\n\nTriples:\n{lines}"}]


TRIPLE_RELEVANCE_SYSTEM = (
    "Given a question, its correct answer and a knowledge-graph statement with its source passage, decide "
    "whether the statement helps establish that the answer is correct. helpful=true or false."
)
TRIPLE_RELEVANCE_SCHEMA = {"type": "object", "properties": {"helpful": {"type": "boolean"}}, "required": ["helpful"]}


def triple_relevance_messages(q: dict, statement: str, source_text: str) -> list[dict]:
    keyed = q["options"][q["answer"] - 1]
    user = (f"Question: {q['question']}\nCorrect answer: {keyed}\n\nStatement: {statement}\n\n"
            f"Source passage:\n{source_text}")
    return [{"role": "system", "content": TRIPLE_RELEVANCE_SYSTEM}, {"role": "user", "content": user}]


PROMPT_VERSION = hashlib.sha256(json.dumps([
    GEN_SYSTEM_RAG, GEN_SYSTEM_CLOSED, GEN_SCHEMA, VERIFY_SYSTEM, VERIFY_SCHEMA, ADJUDICATE_SYSTEM,
    TRIPLE_VERIFY_SYSTEM, TRIPLE_RELEVANCE_SYSTEM]).encode()).hexdigest()[:12]
