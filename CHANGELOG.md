# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Calendar Versioning](https://calver.org/).

## [v2026.5] - 2026-09-17

_Makes OmniAssist provider-agnostic. The agent now speaks the OpenAI `/chat/completions` protocol instead of being tied to Google Gemini, so OpenAI and any compatible provider work through one code path._

### Changed
- Replaced the Gemini-only integration with an OpenAI-compatible client (`core/llm.py`). Google is still supported through its OpenAI-compatible endpoint.
- `core/agent.py` now drives an OpenAI chat-message loop with native function calling, replacing the Gemini content/tool-call format.
- `core/router.py` emits OpenAI function tool schemas (`{"type": "function", "function": {...}}`) derived from tool signatures and type annotations.
- `core/state.py` stores OpenAI-shaped messages and trims history without orphaning a tool result from its call.
- `config/config.yml` is now the single source for providers, endpoints, and the model fallback chain. It contains no secrets.
- Model entries accept shorthand (`"gpt-4o"`), provider-qualified shorthand (`"groq/llama-3.3-70b"`), longhand dicts, and a `models:` list that fans one entry out into several targets. Any entry can override `base_url`, `api_key_env`, `temperature`, or `max_tokens`.
- Provider endpoints resolve from the config, then a built-in table of known providers, then `OPENAI_BASE_URL`-style environment overrides.
- `interfaces/api/server.py` reports the resolved provider chain and any skipped entries in `GET /api/health`.
- The CLI and web UI display the active `provider/model` and surface models skipped for a missing key.
- Version is now 2026.5 "Cake".

### Fixed
- Corrected shipped model IDs against live provider model lists. The Groq fallback named `llama-3.3-70b-versatile`, which Groq has retired and now rejects with `model_not_found`; the chain now uses models confirmed working against a live Groq key.

### Added
- `config/config.yaml.example`, a fully annotated configuration template. Every model ID in it was checked against a live endpoint and labelled `[LIVE]` (called end to end, chat and tool calling confirmed) or `[CATALOG]` (present in OpenRouter's public model catalog but not called, because no key was available). It demonstrates a fallback provider with its own chain: Groq's four verified models, then OpenRouter's two, then a local Ollama server.
- Recorded the Groq models that chat but reject the `tools` parameter (`allam-2-7b`, `groq/compound`, `groq/compound-mini`), so they are excluded on evidence rather than by assumption.
- Fallback that can cross both providers and API keys: a failed or rate-limited provider steps down to the next entry, which may be a different vendor with a different credential.
- Support for keyless local servers (Ollama, vLLM, LM Studio) via `api_key_optional`, including as a fallback behind cloud models.
- `tests/test_llm.py` covering target resolution, key discovery, endpoint overrides, the fallback chain, and key redaction.
- An end-to-end agent loop test that asserts a real tool executes and that its result is fed back to the model as a `tool` message.

### Removed
- The `google-genai` dependency, replaced by `openai`.
- `tests/test_core.py` and `tests/test_fallback_chain.py`, whose coverage moved to `tests/test_llm.py`, `tests/test_agent_loop.py`, and `tests/test_mcp_tools.py`.

### Security
- No API key is ever read from `config/config.yml`; keys come only from the environment.
- `ModelTarget.describe()` redacts keys so they cannot leak into logs or the health endpoint.

## [v2026.4] - 2026-09-17

_Initial release. Version numbers 2026.1–2026.3 were skipped so the project version aligns with the CLI release 2026.4 "Biscotti"._

### Added
- Added a browser web UI (`interfaces/web/`) served directly by the API server — a single-page app with no build step. Includes streaming chat replies, a session sidebar, a live agent execution trace, and a tools inspector.
- Added a `POST /api/run` endpoint and a `WS /ws/run` WebSocket that streams every agent step as JSON (`session`, `plan`, `tool_call`, `tool_result`, `final`, `error`, `done`).
- Added `GET /api/health`, `GET /api/tools`, `GET /api/sessions`, `GET /api/sessions/{id}`, and `DELETE /api/sessions/{id}`.
- Added optional bearer-token authentication (`OMNIASSIST_API_TOKEN`) covering every endpoint that runs the agent or touches session storage. The UI accepts the token via `?token=`, stores it in `localStorage`, and strips it from the URL.
- Added `core/events.py` (`AgentEvent`) so the CLI and web UI share one streaming interface.
- Added `python main.py --web [--host] [--port]` to launch the web server, plus `OMNIASSIST_HOST` / `OMNIASSIST_PORT` env overrides.
- Added an offline mode: with no API key configured, the CLI, web UI, and test suite still run end-to-end against a deterministic `OfflineClient` stub.
- Added session persistence (`memory/session.py`) with listing, loading, and deletion, exposed over the API.
- Added `tests/test_agent_loop.py` (tool execution, result feedback, iteration limits) and `tests/test_api.py` (endpoints, WebSocket streaming, auth).
- Added `ruff` to the dev requirements so local linting matches CI.
- Added `__init__.py` files for `mcp_tools/`, `memory/`, `subagents/`, `interfaces/`, `interfaces/api/`, `interfaces/adapters/`, and `tests/`.

### Changed
- **Agent loop is now genuinely multi-step.** `OmniAssist.run()` and the new `run_stream()` execute tool calls and feed each result back to the model as structured `function_response` parts, repeating until the model answers in text or `agent.max_iterations` is reached. Previously a single model call was made and tool results were returned to the user without ever reaching the model.
- The final iteration now appends an instruction to stop calling tools, guaranteeing the loop terminates with a summary instead of spinning to the cap.
- Rewrote `core/router.py` to build SDK `FunctionDeclaration`s from registered tools and to expose a real `execute()` method. The agent previously called `router.execute` behind a `hasattr` guard against a class that only defined `route()`, silently skipping every tool call.
- Tool discovery is now fault-tolerant and type-accurate: a module that fails to import is recorded in `registry.load_errors` instead of aborting discovery, and only true module-level functions are registered (the `DDGS` class no longer leaks in as a "tool").
- `mcp_tools/ddgs_search.py` now imports `ddgs` rather than the deprecated `duckduckgo_search` package.
- Replaced the two placeholder model fallback lists with real, verified model IDs: `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → `gemini-3.5-flash` → `gemini-3-flash-preview` → `gemini-2.5-flash` → `gemini-3.8-flash` → `gemma-4-31b-it` → `gemma-4-26b-a4b-it`.
- `interfaces/api/server.py` no longer instantiates an agent at import time, so the module can be imported without credentials. Agents are now created per session.
- `main.py` is now a real entry point with argument parsing instead of a bare `from interfaces.cli import main`.
- The CLI renders the plan and each tool call/result inline as the agent works, instead of only showing the final answer.
- Session IDs are validated against `^[A-Za-z0-9_-]{1,64}$` before any filesystem access.
- `requirements.txt` now declares the dependencies the code actually imports (`fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `ddgs`).
- CI and lint workflows now trigger on `main` (the default branch) as well as `master`; previously they only watched `master`, so they never ran.
- Rewrote the README with a Web UI & API section, the streaming protocol, an endpoint table, and an explicit Security section documenting the known limitations of the shell denylist and `eval`-based calculators.
- `.gitignore` now covers `data/` (session storage) and virtualenv directories; `.env.example` documents the new web and auth settings.

### Fixed
- Fixed `tests/test_core.py::test_agent_initialization`, which raised `ValueError` without an API key. It now asserts offline mode and uses an explicit key when one is needed.
- Fixed the `ruff` line-length failure risk in `interfaces/api/server.py` by following the Google function-calling guidance in the emitted system prompt.

### Removed
- Removed the duplicate `duckduckgo-search` dependency (superseded by `ddgs`).
- Removed `pytest` duplication across `requirements-dev.txt` and `requirements-dev-2.txt`.

### Security
- Documented that `mcp_tools/shell_tool.py` is a denylist and not a sandbox. It is bypassable via path traversal (`rm -rf /opt/../etc`), trailing slashes, unexpanded variables (`$HOME`), long flags (`--recursive --force`), command chaining, and `find / -delete`. The README and Security section now state this plainly rather than describing the tooling as sandboxed.
- Added optional token auth to the web API, which exposes shell-executing tools over HTTP for the first time.

[v2026.5]: https://github.com/AfifaM9/omniassist/tree/v2026.5
[v2026.4]: https://github.com/AfifaM9/omniassist/tree/v2026.4
