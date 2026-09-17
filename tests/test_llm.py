"""Tests for the OpenAI-compatible LLM layer: config, chain, and fallback."""

import os
import unittest
from unittest.mock import patch

import yaml

from core.llm import (
    DEFAULT_BASE_URL,
    ModelChain,
    ModelTarget,
    build_targets,
    resolve_api_key,
)


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeCompletion:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, model, messages, **kwargs):
        self._owner.calls.append({"model": model, "messages": messages, **kwargs})
        if model in self._owner.fail_on:
            raise RuntimeError(f"model {model} unavailable")
        return _FakeCompletion(self._owner.responses.get(model, _FakeMessage(content="ok")))


class _FakeChat:
    def __init__(self, owner):
        self.completions = _FakeCompletions(owner)


class _FakeClient:
    """Stands in for openai.OpenAI without touching the network."""

    def __init__(self, owner, options):
        self.owner = owner
        self.options = options
        self.chat = _FakeChat(owner)


def _chain(targets, fail_on=(), responses=None):
    holder = type("Holder", (), {})()
    holder.calls = []
    holder.fail_on = set(fail_on)
    holder.responses = responses or {}
    chain = ModelChain(targets, client_factory=lambda target: _FakeClient(holder, target))
    return chain, holder


class TestTargetResolution(unittest.TestCase):
    """Every test here pins the environment: ambient provider keys would
    otherwise leak in and silently change the chain under test."""

    def setUp(self):
        self._env = patch.dict(os.environ, {}, clear=True)
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_shorthand_string_uses_default_provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            targets, problems = build_targets({"primary": "gpt-4o-mini"}, {}, "openai")
        self.assertEqual(problems, [])
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].provider, "openai")
        self.assertEqual(targets[0].model, "gpt-4o-mini")

    def test_provider_slash_model_shorthand(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}, clear=True):
            targets, _ = build_targets({"primary": "groq/llama-3.3-70b"}, {}, "openai")
        self.assertEqual(targets[0].provider, "groq")
        self.assertEqual(targets[0].model, "llama-3.3-70b")

    def test_shorthand_splits_on_the_first_slash_only(self):
        """Groq nests vendor names in the model ID (openai/gpt-oss-120b), so
        only the first slash separates the provider."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}, clear=True):
            targets, problems = build_targets(
                {"primary": "groq/openai/gpt-oss-120b"}, {}, "openai"
            )
        self.assertEqual(problems, [])
        self.assertEqual(targets[0].provider, "groq")
        self.assertEqual(targets[0].model, "openai/gpt-oss-120b")
        self.assertEqual(targets[0].base_url, "https://api.groq.com/openai/v1")

    def test_cross_provider_and_cross_key_chain(self):
        env = {"OPENAI_API_KEY": "k-openai", "GROQ_API_KEY": "k-groq"}
        with patch.dict(os.environ, env, clear=True):
            targets, problems = build_targets(
                {
                    "primary": {"provider": "openai", "model": "gpt-4o-mini"},
                    "fallbacks": [
                        {"provider": "groq", "model": "llama-3.3-70b"},
                        {"provider": "mistral", "model": "mistral-large-latest"},
                    ],
                },
                {},
                "openai",
            )
        self.assertEqual([t.provider for t in targets], ["openai", "groq"])
        self.assertEqual([t.api_key for t in targets], ["k-openai", "k-groq"])
        # mistral has no key, so it is dropped from the chain and reported
        # rather than silently attempted.
        self.assertEqual(len(problems), 1)
        self.assertIn("mistral", problems[0])

    def test_primary_without_a_key_is_dropped_and_reported(self):
        """No key means offline, transparently, rather than a failed request."""
        with patch.dict(os.environ, {}, clear=True):
            targets, problems = build_targets(
                {"primary": {"provider": "openai", "model": "gpt-4o"}}, {}, "openai"
            )
        self.assertEqual(targets, [])
        self.assertIn("no key found", problems[0])
        self.assertIn("OPENAI_API_KEY", problems[0])

    def test_keyless_local_provider_is_kept(self):
        """Local servers take no credential, so they stay in the chain."""
        with patch.dict(os.environ, {}, clear=True):
            targets, problems = build_targets(
                {"primary": {"provider": "ollama", "model": "llama3.1"}}, {}, "openai"
            )
        self.assertEqual(problems, [])
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].base_url, "http://localhost:11434/v1")
        self.assertEqual(targets[0].api_key, "")

    def test_api_key_optional_marks_any_provider_keyless(self):
        with patch.dict(os.environ, {}, clear=True):
            targets, problems = build_targets(
                {"primary": {"provider": "acme", "model": "m"}},
                {"acme": {"base_url": "http://acme/v1", "api_key_optional": True}},
                "openai",
            )
        self.assertEqual(problems, [])
        self.assertEqual(targets[0].provider, "acme")

    def test_keyless_provider_can_fall_back_to_a_cloud_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            targets, problems = build_targets(
                {
                    "primary": {"provider": "ollama", "model": "llama3.1"},
                    "fallbacks": [{"provider": "openai", "model": "gpt-4o"}],
                },
                {},
                "openai",
            )
        self.assertEqual(problems, [])
        self.assertEqual([t.provider for t in targets], ["ollama", "openai"])

    def test_unkeyed_primary_with_no_key_anywhere_leaves_no_targets(self):
        with patch.dict(os.environ, {}, clear=True):
            targets, problems = build_targets(
                {
                    "primary": {"provider": "openai", "model": "gpt-4o-mini"},
                    "fallbacks": [{"provider": "groq", "model": "llama"}],
                },
                {},
                "openai",
            )
        self.assertEqual(targets, [])
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("no key found" in p for p in problems))

    def test_no_models_configured_is_reported(self):
        with patch.dict(os.environ, {}, clear=True):
            targets, problems = build_targets({}, {}, "openai")
        self.assertEqual(targets, [])
        self.assertEqual(problems, ["no models configured: add models.primary to config.yml"])

    def test_models_key_fans_out(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            targets, _ = build_targets(
                {"primary": {"provider": "openai", "models": ["gpt-4o", "gpt-4o-mini"]}},
                {},
                "openai",
            )
        self.assertEqual([t.model for t in targets], ["gpt-4o", "gpt-4o-mini"])

    def test_provider_base_url_override_in_config(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            targets, _ = build_targets(
                {"primary": "gpt-4o"},
                {"openai": {"base_url": "https://gateway.internal/v1"}},
                "openai",
            )
        self.assertEqual(targets[0].base_url, "https://gateway.internal/v1")

    def test_env_base_url_override_beats_config(self):
        env = {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "http://localhost:1234/v1"}
        with patch.dict(os.environ, env, clear=True):
            targets, _ = build_targets(
                {"primary": "gpt-4o"},
                {"openai": {"base_url": "https://api.openai.com/v1"}},
                "openai",
            )
        self.assertEqual(targets[0].base_url, "http://localhost:1234/v1")

    def test_entry_level_base_url_override(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            targets, _ = build_targets(
                {"primary": {"provider": "openai", "model": "gpt-4o", "base_url": "http://x/v1"}},
                {},
                "openai",
            )
        self.assertEqual(targets[0].base_url, "http://x/v1")

    def test_unknown_provider_name_without_config_falls_back_to_openai_url(self):
        with patch.dict(os.environ, {}, clear=True):
            targets, _ = build_targets(
                {"primary": {"provider": "unknown", "model": "m"}},
                {"unknown": {"api_key_optional": True}},
                "openai",
            )
        self.assertEqual(targets[0].base_url, DEFAULT_BASE_URL)

    def test_missing_model_name_is_reported_not_crashed(self):
        targets, problems = build_targets({"primary": {"provider": "openai"}}, {}, "openai")
        self.assertEqual(targets, [])
        self.assertIn("missing model name", problems[0])

    def test_api_key_override_applies_to_every_target(self):
        with patch.dict(os.environ, {}, clear=True):
            targets, _ = build_targets(
                {
                    "primary": {"provider": "openai", "model": "gpt-4o"},
                    "fallbacks": [{"provider": "groq", "model": "llama"}],
                },
                {},
                "openai",
                api_key_override="explicit-key",
            )
        self.assertEqual([t.api_key for t in targets], ["explicit-key", "explicit-key"])

    def test_describe_never_leaks_the_key(self):
        described = ModelTarget(
            provider="openai", model="gpt-4o", base_url="https://api.openai.com/v1",
            api_key="super-secret", api_key_env="OPENAI_API_KEY",
        ).describe()
        self.assertTrue(described["api_key_set"])
        self.assertNotIn("super-secret", str(described))


class TestKeyResolution(unittest.TestCase):
    def test_google_accepts_gemini_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "g-key"}, clear=True):
            key, env = resolve_api_key("google", None)
        self.assertEqual((key, env), ("g-key", "GEMINI_API_KEY"))

    def test_google_also_accepts_google_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "g-key"}, clear=True):
            key, env = resolve_api_key("google", None)
        self.assertEqual((key, env), ("g-key", "GOOGLE_API_KEY"))

    def test_explicit_env_name_wins(self):
        env = {"MY_CUSTOM_KEY": "custom", "OPENAI_API_KEY": "native"}
        with patch.dict(os.environ, env, clear=True):
            key, name = resolve_api_key("openai", "MY_CUSTOM_KEY")
        self.assertEqual((key, name), ("custom", "MY_CUSTOM_KEY"))

    def test_unknown_provider_derives_conventional_name(self):
        with patch.dict(os.environ, {"ACME_API_KEY": "a"}, clear=True):
            key, _ = resolve_api_key("acme", None)
        self.assertEqual(key, "a")

    def test_no_key_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            key, env = resolve_api_key("openai", None)
        self.assertEqual(key, "")
        self.assertEqual(env, "OPENAI_API_KEY")


class TestModelChain(unittest.TestCase):
    def test_primary_is_used_first(self):
        chain, holder = _chain([ModelTarget("openai", "gpt-4o", "http://x/v1", "k")])
        result = chain.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result.target.model, "gpt-4o")
        self.assertEqual([c["model"] for c in holder.calls], ["gpt-4o"])

    def test_falls_through_to_the_next_target(self):
        targets = [
            ModelTarget("openai", "gpt-4o", "http://x/v1", "k"),
            ModelTarget("groq", "llama-3.3-70b", "http://y/v1", "k2"),
        ]
        chain, holder = _chain(targets, fail_on=["gpt-4o"])
        result = chain.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result.target.provider, "groq")
        self.assertEqual([c["model"] for c in holder.calls], ["gpt-4o", "llama-3.3-70b"])

    def test_fallback_can_cross_base_urls(self):
        """A provider fallback means a different client and endpoint."""
        targets = [
            ModelTarget("openai", "gpt-4o", "https://api.openai.com/v1", "k"),
            ModelTarget("ollama", "llama3.1", "http://localhost:11434/v1", ""),
        ]
        seen = []

        def factory(target):
            seen.append(target.base_url)
            holder = type("H", (), {})()
            holder.calls, holder.responses = [], {}
            holder.fail_on = {"gpt-4o"} if "openai.com" in target.base_url else set()
            return _FakeClient(holder, target)

        chain = ModelChain(targets, client_factory=factory)
        result = chain.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result.target.model, "llama3.1")
        self.assertEqual(seen, ["https://api.openai.com/v1", "http://localhost:11434/v1"])

    def test_all_targets_failing_raises_with_every_error(self):
        targets = [
            ModelTarget("openai", "gpt-4o", "http://x/v1", "k"),
            ModelTarget("groq", "llama", "http://y/v1", "k"),
        ]
        chain, _ = _chain(targets, fail_on=["gpt-4o", "llama"])
        with self.assertRaises(RuntimeError) as ctx:
            chain.complete([{"role": "user", "content": "hi"}])
        self.assertIn("gpt-4o", str(ctx.exception))
        self.assertIn("llama", str(ctx.exception))

    def test_tools_are_sent_when_provided(self):
        chain, holder = _chain([ModelTarget("openai", "gpt-4o", "http://x/v1", "k")])
        schemas = [{"type": "function", "function": {"name": "t"}}]
        chain.complete([{"role": "user", "content": "hi"}], tools=schemas)
        self.assertEqual(holder.calls[0]["tools"], schemas)
        self.assertEqual(holder.calls[0]["tool_choice"], "auto")

    def test_tools_omitted_when_empty(self):
        chain, holder = _chain([ModelTarget("openai", "gpt-4o", "http://x/v1", "k")])
        chain.complete([{"role": "user", "content": "hi"}], tools=[])
        self.assertNotIn("tools", holder.calls[0])

    def test_per_target_temperature_overrides_the_default(self):
        targets = [ModelTarget("openai", "gpt-4o", "http://x/v1", "k", temperature=0.9)]
        chain, holder = _chain(targets)
        chain.complete([{"role": "user", "content": "hi"}], temperature=0.1)
        self.assertEqual(holder.calls[0]["temperature"], 0.9)

    def test_client_is_reused_for_the_same_endpoint_and_key(self):
        created = []
        target = ModelTarget("openai", "gpt-4o", "http://x/v1", "k")

        def factory(t):
            created.append(t)
            holder = type("H", (), {})()
            holder.calls, holder.responses, holder.fail_on = [], {}, set()
            return _FakeClient(holder, t)

        chain = ModelChain([target], client_factory=factory)
        chain.complete([{"role": "user", "content": "a"}])
        chain.complete([{"role": "user", "content": "b"}])
        self.assertEqual(len(created), 1)

    def test_tool_calls_are_normalized_from_sdk_objects(self):
        targets = [ModelTarget("openai", "gpt-4o", "http://x/v1", "k")]
        msg = _FakeMessage(
            content=None,
            tool_calls=[_FakeToolCall("call_1", "basic_calculate", '{"expression": "6*7"}')],
        )
        chain, _ = _chain(targets, responses={"gpt-4o": msg})
        result = chain.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(len(result.message.tool_calls), 1)
        call = result.message.tool_calls[0]
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.name, "basic_calculate")
        self.assertEqual(call.parsed_args(), {"expression": "6*7"})

    def test_malformed_tool_arguments_do_not_crash(self):
        from core.llm import ToolCall

        self.assertEqual(ToolCall("i", "t", "not json").parsed_args(), {})
        self.assertEqual(ToolCall("i", "t", "[1,2]").parsed_args(), {})
        self.assertEqual(ToolCall("i", "t", "").parsed_args(), {})

    def test_empty_chain_is_offline_and_calls_no_model(self):
        chain = ModelChain([])
        self.assertTrue(chain.offline)
        result = chain.complete([{"role": "user", "content": "hello"}])
        self.assertIn("Offline mode", result.message.content)
        self.assertIn("hello", result.message.content)


class TestShippedConfigs(unittest.TestCase):
    """The committed config files must stay loadable and correctly ordered."""

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _models(self, name):
        with open(os.path.join(self.ROOT, "config", name), encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _chain(self, name, env):
        cfg = self._models(name)
        with patch.dict(os.environ, env, clear=True):
            targets, _ = build_targets(
                cfg.get("models") or {}, cfg.get("providers") or {},
                cfg.get("provider") or "openai",
            )
        return [t.label for t in targets]

    def test_both_configs_parse_and_declare_a_primary(self):
        for name in ("config.yml", "config.yaml.example"):
            cfg = self._models(name)
            self.assertIn("models", cfg, name)
            self.assertIn("primary", cfg["models"], name)

    def test_shipped_config_uses_the_verified_groq_model(self):
        # With only a Groq key, the shipped chain trims to Groq's entry.
        self.assertEqual(
            self._chain("config.yml", {"GROQ_API_KEY": "k"}),
            ["groq/openai/gpt-oss-120b"],
        )

    def test_example_config_chain_with_only_a_groq_key(self):
        # Groq's group carries the run; OpenRouter is skipped for want of a key,
        # and the keyless local server stays as the last resort.
        self.assertEqual(
            self._chain("config.yaml.example", {"GROQ_API_KEY": "k"}),
            [
                "groq/openai/gpt-oss-120b",
                "groq/openai/gpt-oss-20b",
                "groq/qwen/qwen3.8-27b",
                "groq/openai/gpt-oss-safeguard-20b",
                "ollama/llama3.1",
            ],
        )

    def test_shipped_config_stays_offline_with_no_keys(self):
        # The local fallback is deliberately commented out in the shipped file,
        # so a fresh clone with no .env does not try to reach localhost.
        self.assertEqual(self._chain("config.yml", {}), [])

    def test_fallback_provider_group_orders_after_the_primary_providers(self):
        # Groq's group, then OpenRouter's group, then the keyless local server.
        chain = self._chain(
            "config.yaml.example",
            {"GROQ_API_KEY": "k", "OPENROUTER_API_KEY": "k"},
        )
        providers = [label.split("/", 1)[0] for label in chain]
        self.assertEqual(providers, ["groq"] * 4 + ["openrouter"] * 2 + ["ollama"])

    def test_local_fallback_survives_with_no_cloud_keys(self):
        self.assertEqual(
            self._chain("config.yaml.example", {}),
            ["ollama/llama3.1"],
        )


if __name__ == "__main__":
    unittest.main()
