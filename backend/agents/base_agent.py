"""
agents/base_agent.py — ReAct Agent Base Class

Implements the ReAct (Reasoning + Acting) loop:
  Thought → Action → Observation → Thought → ... → Final Answer

Every specialist agent inherits from BaseAgent and gets the loop for free.
"""
from __future__ import annotations
import json
import re
from typing import Generator

from llm.groq_client import llm_client
from tools.tool_registry import tool_registry


class AgentStep:
    """A single step in the agent's reasoning trace."""

    def __init__(self, step_type: str, content: str):
        self.step_type = step_type   # "thought" | "action" | "observation" | "answer"
        self.content   = content

    def to_dict(self) -> dict:
        return {"type": self.step_type, "content": self.content}


class BaseAgent:
    """
    ReAct agent base. Subclasses override:
        - system_prompt  (str property)
        - agent_name     (str property)
    """

    MAX_ITERATIONS = 2

    # ── Override in subclass ──────────────────────────────────────────
    @property
    def agent_name(self) -> str:
        return "BaseAgent"

    @property
    def system_prompt(self) -> str:
        return self._build_system_prompt()

    # ── Core ReAct loop ───────────────────────────────────────────────
    def run(
        self,
        user_query: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Execute the ReAct loop for a given query.

        Returns:
            {
                "answer":  str,
                "sources": [str],
                "steps":   [{type, content}, ...],   # agent reasoning trace
                "agent":   str,
            }
        """
        steps:   list[AgentStep] = []
        sources: list[str]       = []
        history  = conversation_history or []

        # Build initial messages
        messages = [
            {"role": "system",    "content": self.system_prompt},
            *history,
            {"role": "user",      "content": user_query},
        ]

        for iteration in range(self.MAX_ITERATIONS):
            # ── Ask LLM for next thought/action ──────────────────────
            response = llm_client.generate(messages, temperature=0.1)
            messages.append({"role": "assistant", "content": response})

            # ── Parse response ────────────────────────────────────────
            thought, action, action_input, final_answer = self._parse_response(response)

            if thought:
                steps.append(AgentStep("thought", thought))

            # ── Final answer? ─────────────────────────────────────────
            if final_answer:
                steps.append(AgentStep("answer", final_answer))
                return {
                    "answer":  final_answer,
                    "sources": sources,
                    "steps":   [s.to_dict() for s in steps],
                    "agent":   self.agent_name,
                }

            # ── Call a tool ───────────────────────────────────────────
            if action:
                steps.append(AgentStep("action", f"{action}({json.dumps(action_input)})"))
                observation = tool_registry.call(action, **action_input)

                # Collect source citations from tool results
                self._collect_sources(observation, sources)

                obs_text = json.dumps(observation, indent=2)
                steps.append(AgentStep("observation", obs_text[:1500]))   # truncate for context

                # Feed observation back to LLM
                messages.append({
                    "role": "user",
                    "content": self._observation_message(observation, obs_text),
                })
            else:
                # LLM produced neither action nor final_answer → treat full response as answer
                steps.append(AgentStep("answer", response))
                return {
                    "answer":  response,
                    "sources": sources,
                    "steps":   [s.to_dict() for s in steps],
                    "agent":   self.agent_name,
                }

        # ── Max iterations reached — ask for final answer ──────────────
        messages.append({
            "role": "user",
            "content": "You have reached the maximum reasoning steps. Provide your Final Answer now based on what you have gathered."
        })
        final_response = llm_client.generate(messages, temperature=0.1)
        _, _, _, final_answer = self._parse_response(final_response)
        answer = final_answer or final_response

        steps.append(AgentStep("answer", answer))
        return {
            "answer":  answer,
            "sources": sources,
            "steps":   [s.to_dict() for s in steps],
            "agent":   self.agent_name,
        }

    def run_stream(
        self,
        user_query: str,
        conversation_history: list[dict] | None = None,
    ) -> Generator[dict, None, None]:
        """
        Run the ReAct loop while forwarding final-answer tokens as they arrive.
        """
        steps: list[AgentStep] = []
        sources: list[str] = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            *(conversation_history or []),
            {"role": "user", "content": user_query},
        ]

        for _ in range(self.MAX_ITERATIONS):
            response_parts: list[str] = []
            answer_offset = 0
            answer_started = False

            for chunk in llm_client.generate_stream(messages, temperature=0.1, max_tokens=768):
                response_parts.append(chunk)
                response = "".join(response_parts)
                answer_match = re.search(r"Final Answer:\s*", response, re.IGNORECASE)
                if answer_match:
                    answer_started = True
                    answer_text = response[answer_match.end():]
                    if len(answer_text) > answer_offset:
                        yield {"type": "token", "data": answer_text[answer_offset:]}
                        answer_offset = len(answer_text)

            response = "".join(response_parts)
            messages.append({"role": "assistant", "content": response})
            thought, action, action_input, final_answer = self._parse_response(response)

            if thought:
                thought_step = AgentStep("thought", thought).to_dict()
                steps.append(AgentStep("thought", thought))
                yield {"type": "step", "data": thought_step}

            if final_answer:
                if not sources:
                    yield from self._retrieve_sources_for_answer(user_query, sources, steps)
                if not answer_started:
                    for offset in range(0, len(final_answer), 80):
                        yield {"type": "token", "data": final_answer[offset:offset + 80]}
                steps.append(AgentStep("answer", final_answer))
                yield {
                    "type": "done",
                    "data": {
                        "answer": final_answer,
                        "sources": sources,
                        "steps": [step.to_dict() for step in steps],
                        "agent": self.agent_name,
                    },
                }
                return

            if not action:
                if not sources:
                    yield from self._retrieve_sources_for_answer(user_query, sources, steps)
                if not answer_started:
                    for offset in range(0, len(response), 80):
                        yield {"type": "token", "data": response[offset:offset + 80]}
                steps.append(AgentStep("answer", response))
                yield {
                    "type": "done",
                    "data": {
                        "answer": response,
                        "sources": sources,
                        "steps": [step.to_dict() for step in steps],
                        "agent": self.agent_name,
                    },
                }
                return

            action_step = AgentStep("action", f"{action}({json.dumps(action_input)})")
            steps.append(action_step)
            yield {"type": "step", "data": action_step.to_dict()}
            observation = tool_registry.call(action, **action_input)

            self._collect_sources(observation, sources)

            observation_text = json.dumps(observation, indent=2)
            observation_step = AgentStep("observation", observation_text[:1500])
            steps.append(observation_step)
            yield {"type": "step", "data": observation_step.to_dict()}
            messages.append({
                "role": "user",
                "content": self._observation_message(observation, observation_text),
            })

        final_prompt = (
            "Provide your final answer now using only the evidence gathered. "
            "Cite the source filename and page for each factual claim. Do not request another tool."
        )
        messages.append({"role": "user", "content": final_prompt})
        final_parts: list[str] = []
        final_offset = 0
        final_started = False
        for chunk in llm_client.generate_stream(messages, temperature=0.1, max_tokens=768):
            final_parts.append(chunk)
            response = "".join(final_parts)
            answer_match = re.search(r"Final Answer:\s*", response, re.IGNORECASE)
            if answer_match:
                final_started = True
                answer = response[answer_match.end():]
                if len(answer) > final_offset:
                    yield {"type": "token", "data": answer[final_offset:]}
                    final_offset = len(answer)

        final_response = "".join(final_parts)
        _, _, _, answer = self._parse_response(final_response)
        answer = answer or final_response
        if not final_started:
            for offset in range(0, len(answer), 80):
                yield {"type": "token", "data": answer[offset:offset + 80]}
        steps.append(AgentStep("answer", answer))
        yield {
            "type": "done",
            "data": {
                "answer": answer,
                "sources": sources,
                "steps": [step.to_dict() for step in steps],
                "agent": self.agent_name,
            },
        }

    @staticmethod
    def _collect_sources(observation: dict, sources: list[str]) -> None:
        for result in observation.get("results", []):
            if "source" in result and "page" in result:
                citation = f"{result['source']} (p. {result['page']})"
                if citation not in sources:
                    sources.append(citation)

        evidence = [
            item
            for condition in observation.get("possible_conditions", [])
            for item in condition.get("evidence", [])
        ]
        evidence.extend(observation.get("document_evidence", []))
        evidence.extend(observation.get("summary_chunks", []))
        for item in evidence:
            match = re.search(r"\[([^,\]]+),\s*p\.?\s*(\d+)\]", item, re.I)
            if match:
                citation = f"{match.group(1).strip()} (p. {match.group(2)})"
                if citation not in sources:
                    sources.append(citation)

    def _retrieve_sources_for_answer(
        self,
        query: str,
        sources: list[str],
        steps: list[AgentStep],
    ) -> Generator[dict, None, None]:
        action_input = {"query": query, "k": 5}
        action = AgentStep("action", f"search_documents({json.dumps(action_input)})")
        steps.append(action)
        yield {"type": "step", "data": action.to_dict()}
        observation = tool_registry.call("search_documents", **action_input)
        self._collect_sources(observation, sources)
        observation_text = json.dumps(observation, indent=2)
        observation_step = AgentStep("observation", observation_text[:1500])
        steps.append(observation_step)
        yield {"type": "step", "data": observation_step.to_dict()}

    @staticmethod
    def _observation_message(observation: dict, observation_text: str) -> str:
        message = f"Observation:\n{observation_text[:2000]}"
        if observation.get("error"):
            message += (
                "\nThe tool call failed and returned no evidence. Retry with valid arguments "
                "if possible; never treat this error as retrieved information."
            )
        return message

    # ── Parsing helpers ───────────────────────────────────────────────
    def _parse_response(self, text: str) -> tuple[str, str, dict, str]:
        """
        Parse LLM response for ReAct patterns.

        Expected format from LLM:
            Thought: <reasoning>
            Action: <tool_name>
            Action Input: {"param": "value"}

        OR final answer:
            Thought: <reasoning>
            Final Answer: <answer text>

        Returns: (thought, action_name, action_input_dict, final_answer)
        """
        thought      = ""
        action       = ""
        action_input = {}
        final_answer = ""

        # Extract Thought
        t_match = re.search(r'Thought:\s*(.*?)(?=Action:|Final Answer:|$)', text, re.DOTALL | re.IGNORECASE)
        if t_match:
            thought = t_match.group(1).strip()

        # Extract Final Answer
        fa_match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if fa_match:
            final_answer = fa_match.group(1).strip()
            return thought, "", {}, final_answer

        # Extract Action
        a_match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
        if a_match:
            action = a_match.group(1).strip()

        # Extract Action Input (JSON block)
        ai_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
        if ai_match:
            try:
                action_input = json.loads(ai_match.group(1))
            except json.JSONDecodeError:
                # Try to extract key-value from non-JSON format
                raw = ai_match.group(1)
                kv  = re.findall(r'"(\w+)":\s*"([^"]*)"', raw)
                action_input = dict(kv) if kv else {}

        return thought, action, action_input, final_answer

    def _build_system_prompt(self) -> str:
        """Default system prompt — subclasses should override."""
        tools_spec = tool_registry.list_tools_prompt()
        return f"""You are {self.agent_name}, a medical AI assistant.

You have access to the following tools:

{tools_spec}

Use the ReAct format to answer questions:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: {{"param_name": "param_value"}}

After seeing the Observation, continue reasoning. When you have enough information:

Thought: <final reasoning>
Final Answer: <complete, well-structured answer with citations>

Rules:
- Always cite sources as [FileName.pdf, p. X] inline
- Never make up information not in the documents
- If information is not available, say so clearly
- Keep answers medically accurate and clear
"""
