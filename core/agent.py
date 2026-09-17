"""OmniAssist primary agent: lifecycle, reasoning loop, and tool execution.

The agent drives a real multi-step loop against any OpenAI-compatible
``/chat/completions`` endpoint. Each iteration sends the transcript and the
tool schemas; if the model asks for tool calls they are executed and their
results are appended as ``role: "tool"`` messages, so the model can observe
them and decide what to do next. The loop ends when the model replies with
plain text or when ``max_iterations`` is reached.

Conversation history is kept in :class:`ConversationState` and replayed on every
turn, so follow-up questions retain context.
"""

import copy
import os

import yaml

from core.llm import ModelChain, ModelTarget
from core.reasoning import CognitiveEngine
from core.router import TaskRouter
from core.selfmodify import SelfModifier
from core.state import ConversationState
from mcp_tools.registry import MCPToolRegistry

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"
CONFIG_RELPATH = ("config", "config.yml")


class OmniAssist:
    """Main OmniAssist primary agent class coordinating lifecycle, reasoning, and tools."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        config_path: str | None = None,
    ):
        self.state = ConversationState()
        self.reasoning = CognitiveEngine()
        self.self_modifier = SelfModifier()
        self.tool_registry = MCPToolRegistry()
        self.router = TaskRouter(self.tool_registry)

        self.max_iterations = 10
        self.temperature = 0.2
        self.provider = provider or DEFAULT_PROVIDER
        self.model_id = model_id or DEFAULT_MODEL
        self.fallback_models: list[str] = []
        self.config_errors: list[str] = []
        self._config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), *CONFIG_RELPATH
        )
        self._load_config(model_id, provider, api_key)

        self.chain = ModelChain(self.targets)
        self.offline = self.chain.offline

    # ------------------------------------------------------------------ config

    def _load_config(self, model_id: str | None, provider: str | None, api_key: str | None):
        from core.llm import build_targets

        config_data = {}
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as exc:  # noqa: BLE001 - a broken config must not block startup
                self.config_errors.append(f"config unreadable: {exc}")
                config_data = {}

        models_cfg = dict(config_data.get("models") or {})
        providers_cfg = config_data.get("providers") or {}
        default_provider = (
            provider or config_data.get("provider") or models_cfg.pop("provider", None) or DEFAULT_PROVIDER
        )

        agent_cfg = config_data.get("agent") or {}
        self.max_iterations = int(agent_cfg.get("max_iterations", self.max_iterations))
        self.temperature = float(agent_cfg.get("temperature", self.temperature))

        if model_id:
            models_cfg["primary"] = {"provider": default_provider, "model": model_id}

        targets, problems = build_targets(
            models_cfg,
            providers_cfg,
            default_provider=default_provider,
            api_key_override=api_key,
        )
        self.targets: list[ModelTarget] = targets
        self.config_errors.extend(problems)

        if targets:
            self.provider = targets[0].provider
            self.model_id = targets[0].model
        elif model_id:
            self.model_id = model_id
        self.fallback_models = [t.model for t in targets[1:]]

    # ------------------------------------------------------------------ prompt

    def _build_system_prompt(self, plan: str) -> str:
        """Generates an explicit system prompt directing multi-step autonomous behavior."""
        return f"""You are OmniAssist, an autonomous, highly capable AI operational agent.

### CORE INSTRUCTIONS & AUTONOMY:
1. **Multi-Step Execution**: You are equipped with direct tool access. When given a complex goal, break it down into sequential, logical sub-steps. Do not ask the user for permission to execute sub-steps—take initiative using available tools.
2. **Tool Feedback Loops**: Execute tools sequentially. Inspect the results of each tool execution, evaluate if your sub-goal was achieved, adapt if errors occur, and trigger the next step until the overall task is fully resolved.
3. **Problem-Solving**: If a tool returns an error or incomplete data, retry with modified parameters or try an alternative tool before giving up.
4. **Final Response**: Once all tool calls and multi-step actions are complete, synthesize a clean, concise, and structured final summary for the user.

### CURRENT CONTEXT & STRATEGIC PLAN:
{plan}
"""

    def _plan(self, prompt: str) -> str:
        return self.reasoning.evaluate_plan(prompt)

    def _tool_schemas(self) -> list[dict]:
        return self.router.tool_schemas()

    # -------------------------------------------------------------------- loop

    def run(self, prompt: str) -> str:
        """Runs the full agent loop and returns the final assistant text."""
        text = ""
        for event in self.run_stream(prompt):
            if event.type in ("final", "error"):
                text = event.content
        return text

    def run_stream(self, prompt: str):
        """Generator yielding agent events.

        Event types: ``plan``, ``tool_call``, ``tool_result``, ``final``, ``error``.
        Consuming this generator is what actually advances the agent loop.
        """
        from core.events import AgentEvent

        plan = self._plan(prompt)
        yield AgentEvent("plan", plan)

        tools = self._tool_schemas()
        messages = [
            {"role": "system", "content": self._build_system_prompt(plan)},
            *self.state.messages(),
            {"role": "user", "content": prompt},
        ]
        self.state.add_message("user", prompt)

        new_messages: list[dict] = []
        max_iterations = max(1, self.max_iterations)
        for iteration in range(max_iterations):
            turn = messages + new_messages
            if iteration == max_iterations - 1:
                turn = [
                    *turn,
                    {
                        "role": "user",
                        "content": (
                            "This is your final step. Do not call any more tools. "
                            "Summarize what you found and answer the user now."
                        ),
                    },
                ]
            try:
                result = self.chain.complete(
                    turn, tools=tools, temperature=self.temperature
                )
            except Exception as exc:  # noqa: BLE001 - surface, never crash the UI
                text = f"Agent Runtime Exception Handled: {exc}"
                self.state.add_message("assistant", text)
                yield AgentEvent("error", text)
                return

            message = result.message
            calls = copy.deepcopy(message.tool_calls)
            if not calls:
                text = message.content or "Execution completed."
                self.state.add_message("assistant", text)
                yield AgentEvent("final", text)
                return

            assistant_turn = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ],
            }
            new_messages.append(assistant_turn)
            self.state.add_message("assistant", message.content or "", tool_calls=assistant_turn["tool_calls"])

            for call in calls:
                args = call.parsed_args()
                yield AgentEvent("tool_call", f"{call.name}({args})", tool=call.name, args=args)
                tool_result = self.router.execute(call.name, args)
                yield AgentEvent("tool_result", tool_result, tool=call.name)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": tool_result,
                }
                new_messages.append(tool_message)
                self.state.add_message("tool", tool_result, tool_call_id=call.id, name=call.name)

        text = "Reached the maximum number of iterations before completing the task."
        self.state.add_message("assistant", text)
        yield AgentEvent("final", text)
