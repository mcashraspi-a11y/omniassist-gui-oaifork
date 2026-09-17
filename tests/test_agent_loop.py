"""Tests for the agent loop: tool execution and tool-result feedback."""

import os
import tempfile
import unittest
from unittest.mock import patch

from core.agent import OmniAssist
from core.events import AgentEvent
from core.llm import AssistantMessage, ChatResult, ModelTarget, ToolCall


class _ScriptedChain:
    """Returns queued responses, recording the messages it was given."""

    def __init__(self, responses, offline=False):
        self._responses = list(responses)
        self.seen = []
        self.offline = offline

    def complete(self, messages, tools=None, temperature=0.2, tool_choice=None):
        self.seen.append({"messages": messages, "tools": tools})
        if not self._responses:
            raise RuntimeError("scripted chain exhausted")
        return ChatResult(
            target=ModelTarget("test", "test-model", "http://x/v1", "k"),
            message=self._responses.pop(0),
        )


def _make_agent(responses, offline=False, max_iterations=10):
    agent = object.__new__(OmniAssist)
    agent.model_id = "test-model"
    agent.provider = "test"
    agent.fallback_models = []
    agent.targets = []
    agent.config_errors = []
    agent.max_iterations = max_iterations
    agent.temperature = 0.2
    agent.offline = offline
    agent.chain = _ScriptedChain(responses, offline=offline)
    agent.client = agent.chain  # compatibility with older call sites

    from core.reasoning import CognitiveEngine
    from core.router import TaskRouter
    from core.state import ConversationState
    from mcp_tools.registry import MCPToolRegistry

    agent.state = ConversationState()
    agent.reasoning = CognitiveEngine()
    agent.tool_registry = MCPToolRegistry()
    agent.router = TaskRouter(agent.tool_registry)
    return agent


def _call(name, args, call_id="call_1"):
    import json

    return ToolCall(id=call_id, name=name, arguments=json.dumps(args))


class TestAgentLoop(unittest.TestCase):
    def test_plain_response_needs_one_iteration(self):
        agent = _make_agent([AssistantMessage(content="hello there")])
        events = list(agent.run_stream("hi"))
        self.assertEqual([e.type for e in events], ["plan", "final"])
        self.assertEqual(events[-1].content, "hello there")

    def test_tool_call_is_actually_executed(self):
        agent = _make_agent([
            AssistantMessage(tool_calls=[_call("basic_calculate", {"expression": "6 * 7"})]),
            AssistantMessage(content="The answer is 42."),
        ])
        events = list(agent.run_stream("what is 6*7"))

        self.assertEqual(
            [e.type for e in events], ["plan", "tool_call", "tool_result", "final"]
        )
        tool_result = next(e for e in events if e.type == "tool_result")
        # The real tool ran: basic_calculate returns the computed value.
        self.assertEqual(tool_result.content, "42")
        self.assertEqual(tool_result.tool, "basic_calculate")

    def test_tool_result_is_fed_back_as_a_tool_message(self):
        agent = _make_agent([
            AssistantMessage(tool_calls=[_call("basic_calculate", {"expression": "1 + 1"})]),
            AssistantMessage(content="done"),
        ])
        list(agent.run_stream("add one and one"))

        second = agent.chain.seen[1]["messages"]
        assistant_turn = next(m for m in second if m.get("tool_calls"))
        self.assertEqual(assistant_turn["tool_calls"][0]["function"]["name"], "basic_calculate")
        self.assertEqual(assistant_turn["tool_calls"][0]["id"], "call_1")

        tool_turn = next(m for m in second if m["role"] == "tool")
        self.assertEqual(tool_turn["tool_call_id"], "call_1")
        self.assertEqual(tool_turn["content"], "2")

    def test_tool_schemas_are_always_sent(self):
        agent = _make_agent([AssistantMessage(content="ok")])
        list(agent.run_stream("hi"))
        schemas = agent.chain.seen[0]["tools"]
        names = {s["function"]["name"] for s in schemas}
        self.assertIn("run_shell", names)
        self.assertEqual(schemas[0]["type"], "function")

    def test_system_prompt_is_the_first_message(self):
        agent = _make_agent([AssistantMessage(content="ok")])
        list(agent.run_stream("hi"))
        first = agent.chain.seen[0]["messages"][0]
        self.assertEqual(first["role"], "system")
        self.assertIn("OmniAssist", first["content"])

    def test_loop_stops_at_max_iterations(self):
        agent = _make_agent(
            [
                AssistantMessage(tool_calls=[_call("basic_calculate", {"expression": "1"})]),
                AssistantMessage(tool_calls=[_call("basic_calculate", {"expression": "2"})]),
            ],
            max_iterations=2,
        )
        events = list(agent.run_stream("loop forever"))
        self.assertEqual(events[-1].type, "final")
        self.assertIn("maximum number of iterations", events[-1].content)

    def test_final_iteration_forces_a_text_answer(self):
        agent = _make_agent(
            [
                AssistantMessage(tool_calls=[_call("basic_calculate", {"expression": "1"})]),
                AssistantMessage(content="stopped and summarized"),
            ],
            max_iterations=2,
        )
        events = list(agent.run_stream("do work"))
        self.assertEqual(events[-1].type, "final")
        self.assertEqual(events[-1].content, "stopped and summarized")

        final_turn = agent.chain.seen[-1]["messages"]
        self.assertIn("Do not call any more tools", str(final_turn[-1]))
        self.assertNotIn("Do not call any more tools", str(agent.chain.seen[0]["messages"]))

    def test_unknown_tool_is_reported_to_the_model_not_raised(self):
        agent = _make_agent([
            AssistantMessage(tool_calls=[_call("does_not_exist", {})]),
            AssistantMessage(content="recovered"),
        ])
        events = list(agent.run_stream("use a bad tool"))
        result = next(e for e in events if e.type == "tool_result")
        self.assertIn("not found", result.content)
        self.assertEqual(events[-1].content, "recovered")

    def test_multiple_tool_calls_in_one_turn_are_all_executed(self):
        agent = _make_agent([
            AssistantMessage(tool_calls=[
                _call("basic_calculate", {"expression": "2 * 3"}, "call_a"),
                _call("basic_calculate", {"expression": "4 * 5"}, "call_b"),
            ]),
            AssistantMessage(content="both done"),
        ])
        results = [e.content for e in agent.run_stream("two things") if e.type == "tool_result"]
        self.assertEqual(results, ["6", "20"])

        tool_turns = [m for m in agent.chain.seen[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_turns], ["call_a", "call_b"])

    def test_model_failure_is_reported_not_raised(self):
        agent = _make_agent([])
        events = list(agent.run_stream("hi"))
        self.assertEqual(events[-1].type, "error")
        self.assertIn("Agent Runtime Exception Handled", events[-1].content)

    def test_offline_agent_runs_end_to_end(self):
        agent = _make_agent([AssistantMessage(content="Offline mode: stub")], offline=True)
        self.assertTrue(agent.offline)
        events = list(agent.run_stream("hello"))
        self.assertEqual(events[-1].type, "final")


class TestConversationMemory(unittest.TestCase):
    def test_second_turn_replays_the_first(self):
        agent = _make_agent([
            AssistantMessage(content="first answer"),
            AssistantMessage(content="second answer"),
        ])
        list(agent.run_stream("first question"))
        list(agent.run_stream("second question"))

        second_call = agent.chain.seen[1]["messages"]
        roles = [m["role"] for m in second_call]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(second_call[1]["content"], "first question")
        self.assertEqual(second_call[2]["content"], "first answer")
        self.assertEqual(second_call[3]["content"], "second question")

    def test_completed_tool_exchange_is_replayed(self):
        agent = _make_agent([
            AssistantMessage(tool_calls=[_call("basic_calculate", {"expression": "2 + 2"})]),
            AssistantMessage(content="4"),
            AssistantMessage(content="still 4"),
        ])
        list(agent.run_stream("what is 2+2"))
        list(agent.run_stream("are you sure"))

        # Run one used two model calls (the tool turn, then the answer), so the
        # second turn is seen[2].
        self.assertEqual(len(agent.chain.seen), 3)
        roles = [m["role"] for m in agent.chain.seen[2]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "user"])
        # The earlier tool exchange is replayed verbatim, so the model can see it.
        self.assertEqual(agent.chain.seen[2]["messages"][3]["content"], "4")

    def test_state_trimming_never_orphans_a_tool_message(self):
        from core.state import ConversationState

        state = ConversationState(max_history=3)
        state.add_message("user", "u")
        state.add_message("assistant", "", tool_calls=[{"id": "1"}])
        state.add_message("tool", "result", tool_call_id="1")
        state.add_message("assistant", "final")

        self.assertNotEqual(state.history[0]["role"], "tool")

    def test_no_message_is_mutated_after_being_stored(self):
        from core.state import ConversationState

        state = ConversationState()
        state.add_message("assistant", "", tool_calls=[{"id": "1"}])
        snapshot = state.messages()
        snapshot[0]["tool_calls"].append({"id": "injected"})
        self.assertEqual(len(state.history[0]["tool_calls"]), 1)


class TestAgentEvents(unittest.TestCase):
    def test_serialization(self):
        event = AgentEvent(
            "tool_call", "run_shell({'command': 'ls'})", tool="run_shell", args={"command": "ls"}
        )
        payload = event.to_dict()
        self.assertEqual(payload["type"], "tool_call")
        self.assertEqual(payload["tool"], "run_shell")
        self.assertEqual(payload["args"], {"command": "ls"})


class TestAgentConstruction(unittest.TestCase):
    def _write_config(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_offline_when_no_keys_are_set(self):
        with patch.dict(os.environ, {}, clear=True):
            agent = OmniAssist(config_path=self._write_config(
                "provider: openai\nmodels:\n  primary: {provider: openai, model: gpt-4o}\n"
            ))
        self.assertTrue(agent.offline)

    def test_online_when_the_primary_key_is_set(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            agent = OmniAssist(config_path=self._write_config(
                "provider: openai\nmodels:\n  primary: {provider: openai, model: gpt-4o}\n"
            ))
        self.assertFalse(agent.offline)
        self.assertEqual(agent.model_id, "gpt-4o")
        self.assertEqual(agent.provider, "openai")

    def test_shipped_config_builds_a_chain_for_openai_only(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            agent = OmniAssist()
        self.assertEqual(agent.provider, "openai")
        self.assertEqual(agent.model_id, "gpt-4o-mini")
        # Every other provider in config.yml lacks a key, so the chain is
        # trimmed to the two OpenAI entries and the rest are reported.
        self.assertEqual([t.provider for t in agent.targets], ["openai", "openai"])
        self.assertEqual([t.model for t in agent.targets], ["gpt-4o-mini", "gpt-4o"])
        self.assertTrue(any("google" in e for e in agent.config_errors))

    def test_shipped_config_with_every_key_uses_the_full_chain(self):
        env = {
            "OPENAI_API_KEY": "k",
            "GEMINI_API_KEY": "k",
            "GROQ_API_KEY": "k",
            "MISTRAL_API_KEY": "k",
            "DEEPSEEK_API_KEY": "k",
            "OPENROUTER_API_KEY": "k",
        }
        with patch.dict(os.environ, env, clear=True):
            agent = OmniAssist()
        self.assertEqual(
            [t.provider for t in agent.targets],
            ["openai", "openai", "google", "groq", "mistral", "deepseek", "openrouter"],
        )
        self.assertEqual(agent.config_errors, [])

    def test_explicit_model_id_overrides_the_config(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            agent = OmniAssist(model_id="gpt-4o")
        self.assertEqual(agent.model_id, "gpt-4o")

    def test_config_loads_agent_settings(self):
        path = self._write_config(
            "models:\n  primary: {provider: openai, model: gpt-4o}\n"
            "agent:\n  max_iterations: 3\n  temperature: 0.75\n"
        )
        with patch.dict(os.environ, {}, clear=True):
            agent = OmniAssist(config_path=path)
        self.assertEqual(agent.max_iterations, 3)
        self.assertAlmostEqual(agent.temperature, 0.75)

    def test_broken_config_does_not_raise(self):
        path = self._write_config("this: [is: not: valid: yaml")
        with patch.dict(os.environ, {}, clear=True):
            agent = OmniAssist(config_path=path)
        self.assertTrue(agent.offline)

    def test_missing_config_falls_back_to_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            agent = OmniAssist(config_path="/nonexistent/config.yml")
        self.assertTrue(agent.offline)
        self.assertEqual(agent.model_id, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
