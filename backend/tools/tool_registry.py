"""
tools/tool_registry.py — Central registry of all callable tools for the agents.

Each tool is a plain Python callable that takes typed inputs and returns a dict.
Agents call tools by name through ToolRegistry.call(tool_name, **kwargs).
"""
from __future__ import annotations
import inspect
from typing import Callable, Any


class Tool:
    """Describes a single callable tool."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: dict,
    ):
        self.name        = name
        self.description = description
        self.func        = func
        self.parameters  = parameters   # {param_name: description}

    def to_prompt_spec(self) -> str:
        """Return a text description the agent can read to decide when to use this tool."""
        params_str = "\n".join(
            f"  - {k}: {v}" for k, v in self.parameters.items()
        )
        return (
            f"Tool: {self.name}\n"
            f"Description: {self.description}\n"
            f"Parameters:\n{params_str}"
        )


class ToolRegistry:
    """Central registry — register tools, look them up, call them."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        print(f"[ToolRegistry] Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, name: str, **kwargs) -> dict:
        """Call a tool by name, return its dict output."""
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: '{name}'. Available: {list(self._tools.keys())}"}
        try:
            inspect.signature(tool.func).bind(**kwargs)
        except TypeError as e:
            return {"error": f"Invalid arguments for tool '{name}': {e}"}
        try:
            result = tool.func(**kwargs)
            if not isinstance(result, dict):
                result = {"result": result}
            return result
        except Exception as e:
            return {"error": f"Tool '{name}' raised an error: {str(e)}"}

    def list_tools_prompt(self) -> str:
        """Return a formatted string listing all tools (used in agent system prompt)."""
        return "\n\n".join(t.to_prompt_spec() for t in self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())


# ── Global registry instance ──────────────────────────────────────────
tool_registry = ToolRegistry()
