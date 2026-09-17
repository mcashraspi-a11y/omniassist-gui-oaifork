import unittest

from mcp_tools.registry import MCPToolRegistry


class TestMCPTools(unittest.TestCase):
    def setUp(self):
        self.registry = MCPToolRegistry()

    def test_registry_loading(self):
        self.assertGreater(len(self.registry.tools), 0)

    def test_only_functions_are_registered(self):
        """Classes imported by a tool module must not leak into the registry."""
        import inspect

        for name, tool in self.registry.tools.items():
            self.assertTrue(inspect.isfunction(tool), f"{name} is not a function")

    def test_missing_dependency_does_not_break_discovery(self):
        """A failing module is reported but the other tools still load."""
        self.assertNotIn("registry", self.registry.load_errors)
        for name in ("run_shell", "write_file", "basic_calculate"):
            self.assertIn(name, self.registry.tools)

    def test_describe_exposes_parameters(self):
        described = {t["name"]: t for t in self.registry.describe()}
        self.assertIn("run_shell", described)
        self.assertIn("command", described["run_shell"]["parameters"])
        self.assertTrue(described["run_shell"]["parameters"]["command"]["required"])

    def test_execute_tool_runs_the_function(self):
        result = self.registry.execute_tool("basic_calculate", expression="2 + 2")
        self.assertEqual(result, "4")

    def test_execute_unknown_tool_returns_error(self):
        self.assertIn("not found", self.registry.execute_tool("nope"))


class TestOpenAIToolSchemas(unittest.TestCase):
    """The router must emit valid OpenAI function tool schemas."""

    def setUp(self):
        from core.router import TaskRouter

        self.schemas = TaskRouter(self.registry_of_tools()).tool_schemas()

    @staticmethod
    def registry_of_tools():
        from mcp_tools.registry import MCPToolRegistry

        return MCPToolRegistry()

    def test_schema_shape(self):
        for schema in self.schemas:
            self.assertEqual(schema["type"], "function")
            self.assertIn("function", schema)
            self.assertIn("name", schema["function"])
            self.assertIn("description", schema["function"])
            params = schema["function"]["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertIsInstance(params["properties"], dict)
            self.assertIsInstance(params["required"], list)

    def test_annotations_map_to_json_types(self):
        by_name = {s["function"]["name"]: s["function"] for s in self.schemas}
        props = by_name["web_search"]["parameters"]["properties"]
        self.assertEqual(props["query"], {"type": "string"})
        self.assertEqual(props["max_results"], {"type": "integer"})
        # Not listed as required, because the tool declares a default.
        self.assertNotIn("max_results", by_name["web_search"]["parameters"]["required"])

    def test_required_reflects_defaults(self):
        by_name = {s["function"]["name"]: s["function"] for s in self.schemas}
        self.assertEqual(by_name["run_shell"]["parameters"]["required"], ["command"])
        self.assertEqual(by_name["basic_calculate"]["parameters"]["required"], ["expression"])

    def test_every_registered_tool_gets_a_schema(self):
        names = {s["function"]["name"] for s in self.schemas}
        self.assertEqual(names, set(MCPToolRegistry().tools))

    def test_descriptions_come_from_docstrings(self):
        by_name = {s["function"]["name"]: s["function"] for s in self.schemas}
        self.assertIn("shell command", by_name["run_shell"]["description"])

    def test_schemas_are_json_serializable(self):
        import json

        json.dumps(self.schemas)

    def test_complex_annotations_fall_back_to_string(self):
        from core.router import TaskRouter

        self.assertEqual(
            TaskRouter._json_schema_for(list[str]), {"type": "array", "items": {"type": "string"}}
        )
        self.assertEqual(TaskRouter._json_schema_for(dict), {"type": "object"})
        self.assertEqual(TaskRouter._json_schema_for(str | None), {"type": "string"})
        self.assertEqual(TaskRouter._json_schema_for(int | None), {"type": "integer"})


if __name__ == "__main__":
    unittest.main()
