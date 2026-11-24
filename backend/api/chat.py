"""
api/chat.py — Agentic Chat Endpoint

Accepts user questions, runs them through the Orchestrator Agent,
and streams the response as Server-Sent Events (SSE).

SSE event format:
  data: {"type": "trace",  "data": {type, content}}   — agent reasoning steps
  data: {"type": "token",  "data": "char"}             — answer tokens
  data: {"type": "done",   "data": {sources, intent, quality, session_id}}
"""
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.orchestrator import orchestrator
from analytics import log_query
from llm.groq_client import llm_client


router = APIRouter()


class ChatRequest(BaseModel):
    question:   str
    session_id: str = "default"


@router.post("/chat")
def chat(request: ChatRequest):
    """
    Main agentic chat endpoint. Streams the full agent pipeline.
    """
    def event_stream():
        answer_length = 0
        completion = None
        try:
            for event in orchestrator.handle_stream(
                query=request.question,
                session_id=request.session_id,
            ):
                if event["type"] == "token":
                    answer_length += len(event["data"])
                elif event["type"] == "done":
                    completion = event["data"]
                yield f"data: {json.dumps(event)}\n\n"
            quality_score = {"High": 3, "Moderate": 2, "Low": 1}
            log_query(
                session_id=request.session_id,
                query=request.question,
                model=llm_client.get_model_name(),
                token_count=(answer_length + 3) // 4,
                intent=completion.get("intent") if completion else None,
                quality=quality_score.get(completion.get("quality")) if completion else None,
                answer_score=(
                    completion.get("quality_metrics", {})
                    .get("answer_quality", {})
                    .get("score")
                ) if completion else None,
                prompt_score=(
                    completion.get("quality_metrics", {})
                    .get("prompt_quality", {})
                    .get("score")
                ) if completion else None,
            )
        except Exception as e:
            error_msg = str(e)
            # Detect FortiGuard / firewall block
            if "FortiGuard" in error_msg or "Web Filter" in error_msg or "Access Blocked" in error_msg:
                error_msg = (
                    "The Groq API (api.groq.com) is blocked by your network firewall (FortiGuard). "
                    "Please connect to a different network (e.g., mobile hotspot) to use the AI features."
                )
            elif "PermissionDenied" in error_msg or "permission" in error_msg.lower():
                error_msg = (
                    "API access denied. Your network may be blocking AI services. "
                    "Try switching to mobile hotspot or a home network."
                )
            log_query(
                session_id=request.session_id,
                query=request.question,
                model=llm_client.get_model_name(),
                token_count=(answer_length + 3) // 4,
            )
            yield f"data: {json.dumps({'type': 'error', 'data': error_msg})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@router.delete("/chat/history/{session_id}")
def clear_history(session_id: str):
    """Clear conversation memory for a session."""
    orchestrator.memory.clear_session(session_id)
    return {"message": f"History cleared for session: {session_id}"}


@router.get("/chat/sessions")
def list_sessions():
    """List all active sessions."""
    sessions = orchestrator.memory.list_sessions()
    return {
        "sessions": [
            {
                "id":     sid,
                "turns":  orchestrator.memory.get_session_count(sid),
            }
            for sid in sessions
        ]
    }
