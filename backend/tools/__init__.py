# tools/__init__.py
from tools.tool_registry import tool_registry
from tools.search_tool import register_search_tool
from tools.summarize_tool import register_summarize_tool
from tools.symptom_checker_tool import register_symptom_checker_tool

# Register all tools into the registry at import time
register_search_tool(tool_registry)
register_summarize_tool(tool_registry)
register_symptom_checker_tool(tool_registry)
