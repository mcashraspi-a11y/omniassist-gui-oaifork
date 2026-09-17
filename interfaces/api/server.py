"""FastAPI backend for the OmniAssist web UI.

Endpoints:
  GET  /api/health            runtime status and model chain
  GET  /api/tools             registered tools and their parameters
  GET  /api/sessions          stored conversations
  GET  /api/sessions/{id}     one stored conversation
  DEL  /api/sessions/{id}     delete a stored conversation
  POST /api/run               run the agent, return the final answer
  WS   /ws/run                run the agent, stream every step as JSON
  GET  /                      the single-page web UI
"""

import asyncio
import hmac
import logging
import os
import uuid

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.agent import OmniAssist
from memory.session import SessionPersistence

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
VERSION = '2026.5 "Cake"'

# The agent can run shell commands and write files, so the API is not safe to
# expose unauthenticated. Set OMNIASSIST_API_TOKEN to require a bearer token.
API_TOKEN = os.environ.get("OMNIASSIST_API_TOKEN")

app = FastAPI(title="OmniAssist API", version=VERSION)
sessions = SessionPersistence()
_agents: dict[str, OmniAssist] = {}


def get_agent(session_id: str) -> OmniAssist:
    """Returns a per-session agent so conversations stay isolated."""
    if session_id not in _agents:
        _agents[session_id] = OmniAssist()
    return _agents[session_id]


def _authorized(token: str | None) -> bool:
    if not API_TOKEN:
        return True
    return bool(token) and hmac.compare_digest(token, API_TOKEN)


async def require_token(request: Request):
    """Dependency guarding every state-changing or tool-running endpoint."""
    if not API_TOKEN:
        return
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else request.query_params.get("token")
    if not _authorized(token):
        raise HTTPException(status_code=401, detail="Missing or invalid API token")


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str | None = None


def _record(session_id: str, prompt: str, answer: str, events: list[dict]) -> None:
    stored = sessions.load_session(session_id)
    messages = stored.get("messages", []) if "error" not in stored else []
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": answer, "events": events})
    sessions.save_session(session_id, {"messages": messages})


@app.get("/api/health")
def health():
    agent = get_agent("__health__")
    return {
        "status": "ok",
        "version": VERSION,
        "offline": agent.offline,
        "provider": agent.provider,
        "model": agent.model_id,
        "fallbacks": agent.fallback_models,
        "chain": [t.describe() for t in agent.targets],
        "config_errors": agent.config_errors,
        "max_iterations": agent.max_iterations,
        "tools": len(agent.tool_registry.tools),
        "tool_load_errors": agent.tool_registry.load_errors,
    }


@app.get("/api/tools")
def list_tools():
    return {"tools": get_agent("__health__").tool_registry.describe()}


@app.get("/api/sessions", dependencies=[Depends(require_token)])
def list_sessions():
    return {"sessions": sessions.list_sessions()}


@app.get("/api/sessions/{session_id}", dependencies=[Depends(require_token)])
def get_session(session_id: str):
    data = sessions.load_session(session_id)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


@app.delete("/api/sessions/{session_id}", dependencies=[Depends(require_token)])
def delete_session(session_id: str):
    if not sessions.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    _agents.pop(session_id, None)
    return {"deleted": session_id}


@app.post("/api/run", dependencies=[Depends(require_token)])
def run_agent_endpoint(req: PromptRequest):
    session_id = req.session_id or str(uuid.uuid4())
    agent = get_agent(session_id)
    events = []
    answer = ""
    for event in agent.run_stream(req.prompt):
        payload = event.to_dict()
        events.append(payload)
        if event.type in ("final", "error"):
            answer = event.content
    _record(session_id, req.prompt, answer, events)
    return {"session_id": session_id, "response": answer, "events": events}


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket):
    """Streams agent events over a WebSocket.

    Client sends ``{"prompt": "...", "session_id": "..."}`` and receives one JSON
    frame per event, followed by a ``done`` frame.
    """
    if not _authorized(websocket.query_params.get("token")):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            prompt = (payload.get("prompt") or "").strip()
            if not prompt:
                await websocket.send_json({"type": "error", "content": "Empty prompt."})
                continue

            session_id = payload.get("session_id") or str(uuid.uuid4())
            await websocket.send_json({"type": "session", "content": session_id})

            agent = get_agent(session_id)
            events = []
            answer = ""
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def produce(agent=agent, loop=loop, queue=queue, prompt=prompt):
                # The agent loop is synchronous; drain it on a worker thread and
                # hand events to the event loop so the socket keeps streaming.
                try:
                    for event in agent.run_stream(prompt):
                        loop.call_soon_threadsafe(queue.put_nowait, event.to_dict())
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, produce)

            while True:
                item = await queue.get()
                if item is None:
                    break
                events.append(item)
                if item["type"] in ("final", "error"):
                    answer = item["content"]
                await websocket.send_json(item)

            _record(session_id, prompt, answer, events)
            await websocket.send_json({"type": "done", "session_id": session_id})
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001 - keep the socket usable
        try:
            await websocket.send_json({"type": "error", "content": f"Server error: {e}"})
        except Exception:
            logging.getLogger(__name__).warning("Could not report error to websocket", exc_info=True)


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
