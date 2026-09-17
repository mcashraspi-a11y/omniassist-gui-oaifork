# OmniAssist 🚀

![CI](https://github.com/AfifaM9/omniassist-gui/actions/workflows/ci.yml/badge.svg)
![Lint](https://github.com/AfifaM9/omniassist-gui/actions/workflows/lint.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-2026.5-blue)

Operationalized Multi-Agent Networked Intelligence & Autonomous System Services Integration Toolkit (2026.5 "Cake")

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture & Directory Structure](#architecture--directory-structure)
4. [Prerequisites & Requirements](#prerequisites--requirements)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [Web UI & API](#web-ui--api)
9. [Tool Ecosystem & MCP Integration](#tool-ecosystem--mcp-integration)
10. [Security](#security)
11. [Development & Contributing](#development--contributing)
12. [License](#license)

---

## Overview

**OmniAssist** is a lightweight, modular, and extensible AI operational agent framework designed to bridge the gap between large language models and local machine execution. By combining an OpenAI-compatible chat and function-calling layer, a flexible Model Context Protocol (MCP) tool registry, and a configurable model/provider fallback chain, OmniAssist operates directly within your terminal environment or in the browser as a fully autonomous assistant.

The agent runs a real multi-step loop: it plans, calls tools, observes the results, and keeps iterating until the task is done. Two front-ends are included — an interactive terminal interface built on Rich, and a browser UI with a live execution trace.

---

## Key Features

- **Autonomous Tool Execution:** Dynamically interprets user intent, matches prompts against registered tools, and executes native Python or system commands.
- **True Multi-Step Loop:** Tool results are fed back to the model as structured `function_response` turns, so the agent can chain calls, react to errors, and adapt before answering.
- **Web UI with Live Trace:** Watch each planning step, tool call, and tool result stream into the browser over a WebSocket as it happens.
- **Interactive CLI:** Powered by Python's `rich` and `readline` libraries to provide a fluid, uninterrupted input loop with arrow-key history and clean block rendering.
- **Robust Error Handling:** Intercepts runtime exceptions, LLM formatting issues, and tool-registry mismatches gracefully to keep the session alive.
- **Model Fallback Chain:** Tries the primary model first, then automatically steps down a configured list of fallbacks when a model is unavailable.
- **Offline Mode:** With no API key configured, the CLI, Web UI, and test suite still run end-to-end against a deterministic stub.
- **Session Persistence:** Conversations are stored on disk and can be listed, reopened, and deleted from the Web UI.
- **Secure Environment Management:** Strictly isolates configuration credentials via `.env` files and `.gitignore` safety guards, with optional bearer-token auth on the web API.

---

## Architecture & Directory Structure

```text
omniassist
├── .env.example             # Environment variable template/defaults (committed to git)
├── .gitignore               # Git exclusion rules (env files, venvs, cache, logs)
├── .github/                 # GitHub repository metadata & community guidelines
│   ├── CODE_OF_CONDUCT.md   # Community behavior standards and enforcement guidelines
│   ├── CONTRIBUTING.md      # Guidelines for submitting pull requests, issues, and code
│   ├── SECURITY.md          # Vulnerability reporting process and security policies
│   ├── PULL_REQUEST_TEMPLATE.md # Standard PR checklist, scope, and testing template
│   ├── ISSUE_TEMPLATE/      # GitHub issue creation templates
│   │   ├── bug_report.md    # Template for reporting bugs, errors, and reproducible crashes
│   │   └── feature_request.md # Template for proposing new sub-agents, tools, or enhancements
│   └── workflows/          # GitHub Actions workflows
│       ├── ci.yml           # Continuous integration pipeline (pytest on main)
│       └── lint.yml         # Code linting workflow (ruff on main)
├── CHANGELOG.md             # Version history and release notes
├── LICENSE.txt              # MIT License terms and copyright notice
├── README.md                # Project documentation, architecture overview, and setup guide
├── main.py                  # Entry point (CLI by default, web server with --web)
├── config/
│   ├── config.yml           # Unified YAML configuration file for models, paths, and options
│   └── config.yaml.example  # Annotated template: verified model/provider fallback chains
├── core/                    # Core agent orchestrator & execution loop
│   ├── __init__.py
│   ├── agent.py             # Main OmniAssist class (lifecycle, multi-step agent loop)
│   ├── events.py            # AgentEvent objects streamed to the CLI and Web UI
│   ├── llm.py               # OpenAI-compatible client, providers, and fallback chain
│   ├── reasoning.py         # Cognitive engine (ReAct, Plan-and-Solve, Self-Reflection)
│   ├── router.py            # OpenAI function tool schemas + execution bridge to the registry
│   ├── selfmodify.py        # Code self-rewriting, agent update, and runtime patch logic
│   └── state.py             # Active conversation state & runtime context tracking
├── interfaces/              # I/O adapters & communication channels
│   ├── adapters/            # External messaging adapters
│   │   └── terminal_adapter.py
│   ├── api/                 # FastAPI server (REST endpoints & WebSocket streams)
│   │   └── server.py
│   ├── web/                 # Single-page web UI (no build step)
│   │   ├── index.html       # Chat shell, session sidebar, inspector panel
│   │   ├── styles.css       # Dark theme styling
│   │   └── app.js           # Streaming client, trace rendering, session management
│   └── cli.py               # Interactive terminal interface for OmniAssist
├── memory/                  # Multi-tiered memory architecture
│   ├── conversation.py      # Working memory & short-term message buffer
│   ├── session.py           # Session persistence across agent reboots
│   └── vector_store.py      # Long-term semantic memory (RAG vector index)
├── mcp_tools/               # Model Context Protocol (MCP) tool integrations
│   ├── basic_calc.py        # Arithmetic and standard math tool helper
│   ├── code_runner.py       # Sandboxed code execution environment
│   ├── ddgs_search.py       # DuckDuckGo web search integration wrapper via `ddgs`
│   ├── file_tools.py        # Filesystem interface (I/O, directory traversal, file editing)
│   ├── python_tool.py       # Tool interface for evaluating Python snippets
│   ├── registry.py          # Fault-tolerant tool discovery and loader
│   ├── robocalc.py          # Complex calculator (cmath)
│   ├── search_tools.py      # Search tools (web scraping, API access, knowledge search)
│   ├── shell_tool.py        # Tool interface for executing system shell commands
│   └── web_fetch.py         # Web content fetching and parsing
├── subagents/               # Specialized sub-agents supervised by OmniAssist
│   ├── base.py              # Abstract base class for specialized sub-agents
│   ├── code_agent.py        # Code generation, execution, and debugging sub-agent
│   ├── planner_agent.py     # Complex task breakdown & multi-step planning sub-agent
│   └── research_agent.py    # Information retrieval & document synthesis sub-agent
├── tests/                   # Automated test suite
│   ├── test_agent_loop.py   # Agent loop, tool execution, and result feedback tests
│   ├── test_api.py          # FastAPI endpoints, WebSocket streaming, and auth tests
│   ├── test_cli.py          # Tests for CLI slash commands
│   ├── test_llm.py          # Target resolution, key handling, and fallback chain tests
│   ├── test_mcp_tools.py    # Tool registry, execution, and OpenAI schema tests
│   └── test_shell_tool.py   # Security tests for shell command blocking
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Dev dependencies (includes -r requirements.txt)
└── requirements-dev-2.txt   # Standalone dev dependencies (pytest + ruff)
```

---

## Prerequisites & Requirements

- **Python:** Version 3.11 or higher.
- **API Key:** One for whichever provider you use, supplied through the environment (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, ...). Optional — without any key the app runs in offline mode.

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AfifaM9/omniassist-gui.git
   cd omniassist-gui
   ```

2. **Install Dependencies** (choose one method):

   **Method A:** Install `requirements.txt`
   ```bash
   python -m pip install -r requirements.txt
   ```

   **Method B:** Or, Install `requirements-dev.txt` (includes `requirements.txt`, adds pytest + ruff)
   ```bash
   python -m pip install -r requirements-dev.txt
   ```

   **Method C:** Concatenate requirements files
   ```bash
   python -m pip install -r requirements.txt -r requirements-dev-2.txt
   ```

---

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file and set the key for each provider you want to use. Only the ones you set are used; the rest are skipped:

   ```env
   GROQ_API_KEY=gsk_...
   # OPENAI_API_KEY=sk-...
   # OPENROUTER_API_KEY=sk-or-...
   ```

   With a single key set the app runs on that provider's chain alone; with none it falls back to offline mode.

3. `config/config.yml` ships ready to run and already selects a working chain. To start from a fully annotated template instead — including a fallback provider with its own chain, and notes on which models were verified against live endpoints — copy the example:

   ```bash
   cp config/config.yaml.example config/config.yml
   ```

   The chain below is tried in order. Group each provider's models together to give it its own chain, so Groq runs its models first, then OpenRouter, then a local server:

   ```yaml
   provider: "groq"

   models:
     primary:
       provider: "groq"
       model: "openai/gpt-oss-120b"
     fallbacks:
       # Groq's own chain.
       - provider: "groq"
         model: "openai/gpt-oss-20b"
       - provider: "groq"
         model: "qwen/qwen3.8-27b"
       # Fallback provider, with its own chain.
       - provider: "openrouter"
         model: "meta-llama/llama-3.3-70b-instruct"
       - provider: "openrouter"
         model: "deepseek/deepseek-chat"
       # Last resort: local, needs no key.
       - provider: "ollama"
         model: "llama3.1"
   ```

   Any entry may also override `base_url` or `api_key_env`, and a `models:` list fans one entry out into several targets. Deployment-specific endpoints can be changed without editing the file by setting `<PROVIDER>_BASE_URL`, such as `GROQ_BASE_URL`.

   Keys are never read from this file — only from the environment.

---

## Running the Application

**Option 1 — Terminal interface (default):**
```bash
python main.py
```

Once loaded, you can chat with OmniAssist or issue direct commands:
- Type `/help` to see available slash commands.
- Type `exit`, `quit`, or `q` to terminate the session (case insensitive).
- Tool calls are printed inline (`=> tool(args)`) as the agent works.

**Option 2 — Web UI:**
```bash
python main.py --web
```
Then open <http://127.0.0.1:8000>.

---

## Web UI & API

The web UI is a single-page app with no build step, served directly by the FastAPI backend.

**Starting the server:**
```bash
python main.py --web --host 0.0.0.0 --port 12000
```

You can also configure it through the environment:
```env
OMNIASSIST_HOST=0.0.0.0
OMNIASSIST_PORT=12000
```

**Interface features:**
- Streaming replies over a WebSocket, with a live trace of every plan, tool call, and tool result.
- A session sidebar listing stored conversations (click to reopen, `×` to delete).
- A tools inspector showing every registered tool, its description, and its parameters.
- A status indicator showing the active model and whether the app is in offline mode.

**Endpoints:**

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | The single-page web UI |
| `GET` | `/api/health` | Status, active model, fallback chain, tool count |
| `GET` | `/api/tools` | Registered tools with parameter metadata |
| `GET` | `/api/sessions` | Stored conversations, newest first |
| `GET` | `/api/sessions/{id}` | One stored conversation with its event trace |
| `DELETE` | `/api/sessions/{id}` | Delete a stored conversation |
| `POST` | `/api/run` | Run the agent and return the final answer plus all events |
| `WS` | `/ws/run` | Run the agent and stream every step as JSON |

**Streaming protocol.** Connect to `/ws/run` and send `{"prompt": "...", "session_id": "..."}`
(`session_id` is optional; one is generated and returned in an initial `session` frame).
You then receive one JSON frame per step, ending with a `done` frame:

```
{"type": "session",     "content": "<session id>"}
{"type": "plan",        "content": "[CognitiveEngine] Strategy: ReAct | ..."}
{"type": "tool_call",   "content": "basic_calculate({'expression': '99 * 3'})", "tool": "basic_calculate", "args": {...}}
{"type": "tool_result", "content": "297", "tool": "basic_calculate"}
{"type": "final",       "content": "The result of 99 * 3 is 297."}
{"type": "done",        "session_id": "<session id>"}
```

An `error` frame replaces `final` if the run fails.

**Example:**
```bash
curl -s http://127.0.0.1:8000/api/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is 99 * 3? Use the basic_calculate tool."}'
```

---

## Tool Ecosystem & MCP Integration

OmniAssist discovers callable Python functions in `mcp_tools/` at startup. For each function it builds a `FunctionDeclaration` (name, docstring summary, parameters) and hands it to the model, so the model chooses which tools to call and with what arguments.

Discovery is fault-tolerant: if one tool module fails to import — for example because an optional dependency is missing — the module is reported in `GET /api/health` under `tool_load_errors` and every other tool still loads normally.

Only true module-level functions are registered; classes and imported names are skipped.

---

## Security

> **OmniAssist executes code on the host it runs on.** It is not sandboxed.

`run_shell` executes commands with `shell=True`, `run_python` uses Python's `exec`, and `write_file` accepts arbitrary paths. These are deliberate design choices for a local automation tool, but they mean the Web UI must not be exposed to an untrusted network.

**If you bind the server to anything other than localhost, set a token:**
```env
OMNIASSIST_API_TOKEN=some-long-random-value
```
Every endpoint that runs the agent or reads/writes sessions then requires a bearer token. Open the UI with the token so the browser can store it:
```
https://your-host/?token=some-long-random-value
```
It is saved to `localStorage` and stripped from the URL immediately. Programmatic clients can use `Authorization: Bearer <token>` or `?token=`.

Notes and known limitations:
- `_is_blocked_command` in `mcp_tools/shell_tool.py` is a **denylist**, not a sandbox. It blocks fork bombs and `rm -rf` against root/sensitive directories, but it is trivially bypassed by path traversal (`rm -rf /opt/../etc`), trailing slashes, unexpanded variables (`$HOME`), long flags (`--recursive --force`), command chaining, and `find / -delete`. Treat it as a guard against accidents, not attacks.
- `basic_calculate` and `robo_calculate` use `eval` with `__builtins__` stripped. That is not a security boundary.
- The model chooses tool arguments, so a prompt-injected model can reach any file or URL the process can.
- Never commit `.env`. It is gitignored by default.

For vulnerability reporting, see [`.github/SECURITY.md`](.github/SECURITY.md).

---

## Development & Contributing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run the linter:
```bash
python -m ruff check .
```

Contributions are welcome! Please check the `.github/` directory for code of conduct, contribution guidelines, and pull request templates.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes using casual, descriptive commit messages.
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.
   * Add a **Before vs. After** if needed

---

## License

Distributed under the terms specified in [`LICENSE.txt`](LICENSE.txt).

