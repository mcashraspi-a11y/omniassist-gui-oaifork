"""Task routing between the primary agent loop and registered MCP tools.

Also translates registered Python callables into the OpenAI tool schema
(``{"type": "function", "function": {...}}``) that the model expects.
"""

import inspect
import types
import typing


class TaskRouter:
    """Routes tasks between internal sub-agents and registered MCP tools."""

    _JSON_TYPES = {
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
    }

    def __init__(self, tool_registry):
        self.registry = tool_registry

    def route(self, task: str) -> str:
        return f"[TaskRouter] Analyzing task routing parameters for: '{task}'"

    @classmethod
    def _json_schema_for(cls, annotation) -> dict:
        """Maps a Python annotation onto a JSON-schema fragment."""
        if annotation is inspect.Parameter.empty:
            return {"type": "string"}
        if annotation in cls._JSON_TYPES:
            return {"type": cls._JSON_TYPES[annotation]}
        if annotation is dict:
            return {"type": "object"}
        if annotation in (list, tuple, set, frozenset):
            return {"type": "array", "items": {"type": "string"}}

        origin = typing.get_origin(annotation)
        if origin is typing.Literal:
            return {"type": "string", "enum": [str(a) for a in typing.get_args(annotation)]}
        if origin in (list, tuple, set, frozenset):
            args = typing.get_args(annotation)
            item = cls._json_schema_for(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item}
        if origin is dict:
            return {"type": "object"}
        # `X | None` produces types.UnionType, which typing.get_origin reports as
        # the UnionType itself rather than typing.Union, so both must be checked.
        if origin is typing.Union or origin is types.UnionType:
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                return cls._json_schema_for(args[0])
            return {"type": "string"}
        return {"type": "string"}

    def tool_schemas(self) -> list[dict]:
        """Builds OpenAI function tool schemas from the registry's callables."""
        schemas = []
        for name, func in sorted(self.registry.tools.items()):
            doc = inspect.getdoc(func) or ""
            properties = {}
            required = []
            try:
                signature = inspect.signature(func)
            except (TypeError, ValueError):
                signature = None
            for pname, param in (signature.parameters.items() if signature else []):
                if pname in ("self", "cls"):
                    continue
                properties[pname] = self._json_schema_for(param.annotation)
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": doc.split("\n\n")[0].strip() or name,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schemas

    def execute(self, tool_name: str, args: dict | None = None) -> str:
        """Invokes a registered tool by name and returns its result as text."""
        return str(self.registry.execute_tool(tool_name, **(args or {})))
