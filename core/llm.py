"""OpenAI-compatible LLM layer: provider/model/key resolution and fallback.

OmniAssist talks to any OpenAI-compatible ``/chat/completions`` endpoint. A
provider is just a ``base_url`` plus the environment variable holding its API
key, so OpenAI, Google's OpenAI-compatible endpoint, Azure-style gateways, and
local servers (Ollama, vLLM, LM Studio, llama.cpp) all work the same way.

Configuration lives in ``config/config.yml``; secrets never do. Each config
entry names a provider and a model, and may override the provider's
``base_url`` or ``api_key_env``. The resulting target chain is tried in order:

    primary -> fallbacks[0] -> fallbacks[1] -> ...

Falling through can therefore cross providers and cross API keys, not just
models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Known endpoints, so a provider works from its name alone even if the config
# only names it. config/config.yml overrides these.
_BUILTIN_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "azure": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
}

# Providers whose env var name does not follow the "<PROVIDER>_API_KEY" rule.
_WELL_KNOWN_KEY_ENVS = {
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY",),
    "ollama": (),
    "local": (),
    "vllm": (),
    "lmstudio": (),
}

# Providers that serve models without requiring a credential.
_KEYLESS_PROVIDERS = {"ollama", "local", "vllm", "lmstudio"}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if str(v).strip()]


@dataclass(frozen=True)
class ModelTarget:
    """One attempt in the fallback chain: a provider, a model, and a key."""

    provider: str
    model: str
    base_url: str
    api_key: str = ""
    api_key_env: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    def describe(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_set": bool(self.api_key),
        }


@dataclass
class ToolCall:
    """A normalized tool call from an assistant message."""

    id: str
    name: str
    arguments: str = "{}"

    def parsed_args(self) -> dict:
        """Best-effort decode of the JSON argument string."""
        try:
            args = json.loads(self.arguments or "{}")
        except (ValueError, TypeError):
            return {}
        return args if isinstance(args, dict) else {}


@dataclass
class AssistantMessage:
    """The subset of an OpenAI ``message`` object the agent loop needs."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @classmethod
    def from_sdk(cls, message) -> AssistantMessage:
        raw_calls = getattr(message, "tool_calls", None) or []
        calls = []
        for index, call in enumerate(raw_calls):
            function = getattr(call, "function", None)
            if function is None:
                continue
            calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{index}",
                    name=getattr(function, "name", "") or "",
                    arguments=getattr(function, "arguments", None) or "{}",
                )
            )
        return cls(content=getattr(message, "content", None), tool_calls=calls)


@dataclass
class ChatResult:
    """A completion plus the target that produced it."""

    target: ModelTarget
    message: AssistantMessage


class OfflineMessage(AssistantMessage):
    """Deterministic stand-in used when no credentials are configured."""

    def __init__(self, prompt: str):
        super().__init__(
            content=(
                "Offline mode: no API key is configured for any model in the "
                "chain, so no model was called.\n\n"
                f"Received prompt: {prompt[:400]}"
            )
        )


def _key_env_candidates(provider: str) -> list[str]:
    known = _WELL_KNOWN_KEY_ENVS.get(provider)
    if known is not None:
        return list(known)
    return [f"{provider.upper().replace('-', '_')}_API_KEY"]


def resolve_base_url(provider: str, configured: str | None) -> str:
    """Provider base URL, with an env override for deployment-time changes."""
    env_name = f"{provider.upper().replace('-', '_')}_BASE_URL"
    return (
        os.environ.get(env_name)
        or configured
        or _BUILTIN_BASE_URLS.get(provider, DEFAULT_BASE_URL)
    )


def resolve_api_key(provider: str, configured_env) -> tuple[str, str | None]:
    """Finds the first set key for a provider.

    Returns ``(key, env_var_name)``. An explicit ``api_key_env`` in the config
    wins; otherwise the provider's well-known variable is checked, and finally
    ``<PROVIDER>_API_KEY``.
    """
    candidates = _as_list(configured_env) or _key_env_candidates(provider)
    # A deployment may also inject the key under the provider's canonical name.
    for name in _key_env_candidates(provider):
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value, name
    return "", candidates[0] if candidates else None


def _entry_to_specs(entry) -> list[dict]:
    """Accepts a shorthand or longhand config entry. Returns one or more specs.

    Shorthand:  ``gemini-2.5-flash``
    Longhand:   ``{provider: google, model: gemini-2.5-flash}``
    Fan-out:    ``{provider: openai, models: [gpt-4o, gpt-4o-mini]}``
    """
    if isinstance(entry, str):
        name = entry.strip()
        if not name:
            return []
        if "/" in name:
            provider, model = name.split("/", 1)
            return [{"provider": provider.strip(), "model": model.strip()}]
        return [{"provider": "", "model": name}]
    if isinstance(entry, dict):
        models = entry.get("models")
        if models:
            return [
                {**entry, "model": model, "models": None}
                for model in _as_list(models)
            ]
        return [dict(entry)]
    return []


def build_targets(
    models_cfg: dict,
    providers_cfg: dict | None = None,
    default_provider: str = "openai",
    api_key_override: str | None = None,
) -> tuple[list[ModelTarget], list[str]]:
    """Resolves the configured chain into concrete targets.

    A target is kept when it has a usable key, or when its provider is known not
    to need one (local servers, or ``api_key_optional: true``). Everything else
    is dropped and reported, so a partially-credentialed config still runs and
    the caller can see exactly what was skipped. With no targets left the agent
    runs in offline mode.

    ``api_key_override`` forces one key across every target, which is what a
    caller passing ``OmniAssist(api_key=...)`` expects.

    Returns ``(targets, problems)``.
    """
    providers_cfg = providers_cfg or {}
    entries: list = []
    primary = models_cfg.get("primary")
    if primary:
        entries.extend(_entry_to_specs(primary))
    for entry in models_cfg.get("fallbacks") or []:
        entries.extend(_entry_to_specs(entry))

    targets: list[ModelTarget] = []
    problems: list[str] = []
    for spec in entries:
        model = (spec.get("model") or "").strip()
        provider = (spec.get("provider") or "").strip() or default_provider
        if not model:
            problems.append("entry: missing model name")
            continue

        provider_cfg = providers_cfg.get(provider) or {}
        base_url = resolve_base_url(provider, spec.get("base_url") or provider_cfg.get("base_url"))
        configured_env = spec.get("api_key_env", provider_cfg.get("api_key_env"))
        api_key, key_env = resolve_api_key(provider, configured_env)
        if api_key_override:
            api_key, key_env = api_key_override, "api_key argument"
        key_required = provider not in _KEYLESS_PROVIDERS and not provider_cfg.get("api_key_optional")

        if key_required and not api_key:
            problems.append(f"{provider}/{model}: no key found in {key_env or 'env'}")
            continue

        targets.append(
            ModelTarget(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                api_key_env=key_env,
                temperature=spec.get("temperature"),
                max_tokens=spec.get("max_tokens"),
            )
        )

    if not entries:
        problems.append("no models configured: add models.primary to config.yml")
    return targets, problems


class ModelChain:
    """Tries each target in order until one returns a completion."""

    def __init__(
        self,
        targets: list[ModelTarget],
        offline_message=OfflineMessage,
        client_factory=None,
    ):
        self.targets = targets or []
        self.offline = not self.targets
        self.errors: list[str] = []
        self._offline_message = offline_message
        self._client_factory = client_factory or self._default_client_factory
        self._clients: dict[str, object] = {}

    @staticmethod
    def _default_client_factory(target: ModelTarget):
        from openai import OpenAI

        kwargs = {"base_url": target.base_url, "timeout": 120.0, "max_retries": 1}
        # Local servers accept any placeholder; omitting the header entirely is
        # what some of them choke on.
        kwargs["api_key"] = target.api_key or "not-needed"
        return OpenAI(**kwargs)

    def _client_for(self, target: ModelTarget):
        key = f"{target.base_url}|{target.api_key}"
        if key not in self._clients:
            self._clients[key] = self._client_factory(target)
        return self._clients[key]

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        tool_choice: str | None = None,
    ) -> ChatResult:
        """Runs one completion, walking the chain on failure."""
        if self.offline:
            prompt = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    prompt = str(message.get("content") or "")
                    break
            return ChatResult(
                target=ModelTarget(provider="offline", model="offline", base_url=""),
                message=self._offline_message(prompt),
            )

        errors = []
        for target in self.targets:
            try:
                client = self._client_for(target)
                kwargs = {
                    "model": target.model,
                    "messages": messages,
                    "temperature": (
                        target.temperature if target.temperature is not None else temperature
                    ),
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice or "auto"
                if target.max_tokens:
                    kwargs["max_tokens"] = target.max_tokens
                response = client.chat.completions.create(**kwargs)
                message = AssistantMessage.from_sdk(response.choices[0].message)
                return ChatResult(target=target, message=message)
            except Exception as exc:  # noqa: BLE001 - fall through the chain
                errors.append(f"{target.label}: {exc}")

        self.errors = errors
        raise RuntimeError("; ".join(errors))
