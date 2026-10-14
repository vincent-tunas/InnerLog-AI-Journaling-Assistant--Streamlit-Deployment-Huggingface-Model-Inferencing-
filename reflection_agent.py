"""InnerLog reflection agent: sentiment + optional history + local RAG -> reflection.

Each call retrieves from an existing local Chroma DB; no document indexing occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os

from dotenv import load_dotenv
from llm_model_inference import generate_response, inference_provider

load_dotenv()
from rag import retrieve_sources, KnowledgeBaseError


class ReflectionAgentError(RuntimeError):
    pass



SYSTEM_PROMPT = """
You are InnerLog, an empathetic AI journaling and personal growth assistant.

Your goal is to help users understand their emotions, reflect on their experiences, and take meaningful steps forward.

You receive:
- The user's current journal entry.
- Sentiment analysis results.
- Relevant previous journal entries, if available.
- Supporting knowledge retrieved through RAG.

Your responsibilities:

1. UNDERSTAND: Identify the user's main concern, emotions, and possible underlying needs. Consider their personal context and previous experiences.
2. REFLECT: Provide meaningful insights into their thoughts, feelings, and behavioral patterns. Help them see their situation from a constructive perspective.
3. ADVISE: Offer practical, personalized, and realistic guidance based on their situation and relevant RAG knowledge.
4. ACTION: Suggest 1–2 specific, achievable steps that help the user move forward.
5. ENCOURAGE: End with one thoughtful reflection question that promotes self-awareness and growth.

Guidelines:
- Be warm, empathetic, natural, and concise.
- Prioritize understanding over generic reassurance.
- Give actionable advice, not just emotional validation.
- Adapt recommendations to the user's emotional state.
- Use journal history and RAG evidence when relevant.
- Do not invent personal history, facts, or research.
- Avoid assumptions, judgment, diagnoses, and repetitive advice.
- Do not treat sentiment predictions as certain facts.
- Respect the user's autonomy; avoid forcing solutions.
- Prioritize immediate safety and human support when the entry suggests imminent danger or self-harm.

Respond using these sections:
Understanding, Reflection, Advice, Next Steps, Reflection Question.

Ensure every response is personalized, insightful, and useful.
Return ONLY a JSON object: {"reflection": "...", "question": "..."}.
Do not add any additional JSON keys. Put the open-ended question ONLY in question.
"""



@dataclass(frozen=True)
class ReflectionResult:
    reflection: str
    question: str
    sources: list[dict]
    model: str
    prompt_version: str
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string"},
        "question": {"type": "string"},
    },
    "required": ["reflection", "question"],
    "additionalProperties": False,
}


class ReflectionAgent:
    def __init__(self, model: str | None = None, timeout: float = 180):
        self.model = model or (os.getenv("HF_REFLECTION_MODEL", "google/gemma-3-12b-it"))
                               
        self.timeout = timeout

    def reflect(self, entry: dict, history: list[dict] | None = None) -> ReflectionResult:
        content = entry.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("A saved journal entry is required.")
        analysis = entry.get("analysis")
        if not analysis:
            raise ReflectionAgentError("Analyze sentiment for this entry before generating a reflection.")
        if len(content.encode("utf-8")) > 6000:
            raise ReflectionAgentError("This MVP accepts journal entries up to 6000 UTF-8 bytes.")

        # Never include arbitrary full historical entries in the LLM's context.
        history_summary = []
        for old in (history or []):
            if old.get("id") == entry.get("id") or not old.get("analysis"):
                continue
            previous = old["analysis"]
            history_summary.append({
                "date": str(old.get("created_at", ""))[:10],
                "title": str(old.get("title", ""))[:120],
                "sentiment": previous.get("sentiment", "unclear"),
                "emotions": previous.get("emotions", [])[:3],
            })
            if len(history_summary) >= 5:
                break

        # Query embedding only, NOT document reindexing.
        retrieval_query = (
            f"Emotional support and coping strategies for: {content[:850]} "
            f"Emotions: {', '.join(analysis.get('emotions', []))}"
        )
        try:
            sources = retrieve_sources(retrieval_query, k=3)
        except KnowledgeBaseError as exc:
            raise ReflectionAgentError(str(exc)) from exc

        input_data = {
            "current_entry": {"title": entry.get("title", ""), "content": content},
            "sentiment_analysis": {
                "sentiment": analysis.get("sentiment"),
                "emotions": analysis.get("emotions"),
                "explanation": analysis.get("explanation"),
            },
            "previous_entries_summary": history_summary,
            "knowledge_base_excerpts": [
                {"reference": i + 1, "text": src["text"]} for i, src in enumerate(sources)
            ],
        }
        try:
            raw_content = generate_response(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=json.dumps(input_data, ensure_ascii=False),
                model=self.model,
                temperature=0.3,
                max_tokens=1000,
                timeout=self.timeout,
                json_schema=REFLECTION_SCHEMA,
            )
            try:
                payload = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                raise ValueError("Inference provider returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object, got {type(payload).__name__}")
            for key in ("reflection", "question"):
                value = payload.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Missing or empty '{key}' in inference response; returned keys: {list(payload)}"
                    )
                if len(value) > 5000:
                    raise ValueError(f"Unexpectedly long '{key}' in inference response")
        except Exception as exc:
            raise ReflectionAgentError(f"Reflection generation failed: {exc}") from exc

        return ReflectionResult(
            reflection=payload["reflection"].strip(),
            question=payload["question"].strip(),
            sources=[{"source": src["source"], "page": src["page"]} for src in sources],
            model=self.model,
            prompt_version="innerlog-reflection-v1",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
