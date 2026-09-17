"""
==============================================================================
 OMNIASSIST ENTRY POINT SCRIPT (2026.5 "Cake")
==============================================================================
 MODEL PROVIDERS:
   OmniAssist speaks the OpenAI /chat/completions protocol, so it works with
   OpenAI and any compatible provider (Google, Groq, Mistral, OpenRouter,
   Ollama, vLLM, LM Studio, ...).

   Providers, endpoints, and the model fallback chain live in
   config/config.yml. API keys are read from environment variables only --
   never from the config file. See .env.example.

   Rule: try the primary model first, then step down the fallback chain. A
   step may cross to a different provider and therefore a different API key.

 USAGE:
   python main.py                 # interactive terminal interface
   python main.py --web           # web UI + API (default http://127.0.0.1:8000)
   python main.py --web --port 12000 --host 0.0.0.0
==============================================================================
"""

import argparse
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass


def run_web(host: str, port: int, reload: bool):
    import uvicorn

    uvicorn.run(
        "interfaces.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(prog="omniassist", description="OmniAssist agent framework")
    parser.add_argument("--web", action="store_true", help="Launch the web UI and REST/WebSocket API")
    parser.add_argument("--host", default=os.environ.get("OMNIASSIST_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OMNIASSIST_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Auto-reload the web server on file changes")
    args = parser.parse_args()

    if args.web:
        run_web(args.host, args.port, args.reload)
        return

    from interfaces.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    sys.exit(main())
