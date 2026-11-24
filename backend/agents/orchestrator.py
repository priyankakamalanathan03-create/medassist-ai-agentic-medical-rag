"""
agents/orchestrator.py — Master Orchestrator Agent

The Orchestrator is the entry point for ALL user queries. It:
  1. Classifies the intent of the query
  2. Routes to the appropriate specialist agent (RAG, Symptom)
  3. Scores prompt clarity and answer evidence
  4. Returns the final validated answer + full agent trace

This is what makes the system truly agentic — a coordinating meta-agent.
"""
from __future__ import annotations
import re
from typing import Generator

from agents.rag_agent import rag_agent
from agents.symptom_agent import symptom_agent
from agents.critique_agent import critique_agent
from memory.conversation_memory import ConversationMemory


class OrchestratorAgent:
    """
    Meta-agent that classifies, routes, critiques, and returns results.
    """

    AGENT_NAME = "Medical Orchestrator"

    # Intent categories → agent mapping
    INTENT_MAP = {
        "symptom_analysis": symptom_agent,
        "document_qa":      rag_agent,
        "summarization":    rag_agent,
    }

    def __init__(self):
        self.memory = ConversationMemory()
        print(f"[Orchestrator] {self.AGENT_NAME} initialized")

    # ── Public API ────────────────────────────────────────────────────
    def handle(self, query: str, session_id: str = "default") -> dict:
        """
        Full pipeline: classify → route → critique → return.

        Returns:
            {
                "answer":       str,
                "sources":      [str],
                "agent_trace":  [{type, content}, ...],
                "intent":       str,
                "quality":      str,
                "session_id":   str,
            }
        """
        agent_trace: list[dict] = []

        # ── Step 1: Classify intent ───────────────────────────────────
        intent = self._classify_intent(query)
        agent_trace.append({
            "type":    "orchestrator",
            "content": f"Intent classified as: **{intent}** — routing to {self.INTENT_MAP[intent].agent_name}"
        })

        # ── Step 2: Get conversation history ─────────────────────────
        history = self.memory.get_history(session_id)

        # ── Step 3: Route to specialist agent ─────────────────────────
        specialist = self.INTENT_MAP[intent]
        agent_trace.append({
            "type":    "orchestrator",
            "content": f"Delegating to: **{specialist.agent_name}**"
        })

        primary_result = specialist.run(query, conversation_history=history)

        # Add agent steps to trace
        agent_trace.extend(primary_result["steps"])

        # ── Step 4: Critique & validate ───────────────────────────────
        agent_trace.append({
            "type":    "orchestrator",
            "content": "Scoring prompt clarity and answer evidence..."
        })

        critique_result = critique_agent.critique(
            query=query,
            draft_answer=primary_result["answer"],
            sources=primary_result["sources"],
        )

        agent_trace.append({
            "type":    "critique",
            "content": (
                f"Quality Score: **{critique_result['quality_score']}** | "
                f"Issues: {critique_result['issues_found'] or 'None detected'}"
            )
        })

        # ── Step 5: Save to memory ────────────────────────────────────
        self.memory.add_exchange(
            session_id=session_id,
            user_message=query,
            assistant_message=critique_result["validated_answer"],
        )

        return {
            "answer":      critique_result["validated_answer"],
            "sources":     primary_result["sources"],
            "agent_trace": agent_trace,
            "intent":      intent,
            "quality":     critique_result["quality_score"],
            "quality_metrics": critique_result["quality_metrics"],
            "session_id":  session_id,
        }

    def handle_stream(self, query: str, session_id: str = "default") -> Generator[dict, None, None]:
        """
        Streaming version. Yields SSE-compatible dicts:

          {"type": "trace",  "data": {type, content}}   — agent reasoning steps
          {"type": "token",  "data": str}                — answer character tokens
          {"type": "done",   "data": {sources, intent, quality, session_id}}
        """
        intent = self._classify_intent(query)
        specialist = self.INTENT_MAP[intent]
        trace_step = {
            "type": "orchestrator",
            "content": f"Query routed to **{specialist.agent_name}** ({intent})",
        }
        yield {"type": "trace", "data": trace_step}
        history = self.memory.get_history(session_id)
        primary_result = None
        for event in specialist.run_stream(query, conversation_history=history):
            if event["type"] == "step":
                yield {"type": "trace", "data": event["data"]}
            elif event["type"] == "token":
                yield event
            elif event["type"] == "done":
                primary_result = event["data"]

        if primary_result is None:
            raise RuntimeError("The medical agent did not return a completed answer.")

        critique_step = {
            "type": "orchestrator",
            "content": "Scoring prompt clarity, answer relevance, and citation evidence...",
        }
        yield {"type": "trace", "data": critique_step}
        critique_result = critique_agent.critique(
            query=query,
            draft_answer=primary_result["answer"],
            sources=primary_result["sources"],
        )
        final_answer = critique_result["validated_answer"]
        if final_answer.startswith(primary_result["answer"]):
            additional_text = final_answer[len(primary_result["answer"]):]
            if additional_text:
                yield {"type": "token", "data": additional_text}

        yield {
            "type": "trace",
            "data": {
                "type": "critique",
                "content": f"Source citation check: **{critique_result['quality_score']}**",
            },
        }
        self.memory.add_exchange(
            session_id=session_id,
            user_message=query,
            assistant_message=final_answer,
        )

        yield {
            "type": "done",
            "data": {
                "sources": primary_result["sources"],
                "intent": intent,
                "quality": critique_result["quality_score"],
                "quality_metrics": critique_result["quality_metrics"],
                "session_id": session_id,
            },
        }

    # ── Intent classification ─────────────────────────────────────────
    def _classify_intent(self, query: str) -> str:
        """
        Routes directly from the query wording so no model request is needed.
        Returns one of: symptom_analysis | document_qa | summarization
        """
        normalized = query.lower()
        if re.search(r"\b(summarize|summarise|summary|overview|key findings)\b", normalized):
            return "summarization"
        personal_symptoms = re.search(
            r"\b(i have|i'm having|i am having|i've had|i feel|i'm experiencing|"
            r"i am experiencing|my symptoms|patient has|patient reports|"
            r"experiencing symptoms)\b",
            normalized,
        )
        symptom_question = re.search(
            r"\b(chest pain|shortness of breath|difficulty breathing|wheezing|"
            r"fainting|severe headache|high fever|sudden weakness|nighttime cough|"
            r"dizziness|nausea|rash|fever)\b.*"
            r"\b(what could|what might|should i|what should|possible cause|"
            r"possible condition|causes should be considered)\b",
            normalized,
        )
        if personal_symptoms or symptom_question:
            return "symptom_analysis"
        return "document_qa"


# ── Singleton ─────────────────────────────────────────────────────────
orchestrator = OrchestratorAgent()
