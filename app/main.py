from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .agents.agent import Agent, AgentInputError, AgentOutputError
from .providers.deepseek import (
    DeepSeekProvider,
    LlmConfigurationError,
    LlmRequestError,
)
from .schemas import ChatSendRequest, ChatSendResponse, ChatSession, ChatSessionSummary
from .services.chat_sessions import (
    ChatSessionNotFound,
    ChatSessionService,
    DEFAULT_CHAT_DB_PATH,
    SQLiteChatSessionRepository,
)


def _allowed_origins() -> list[str]:
    configured_origins = os.getenv("FRONTEND_ORIGINS")
    if not configured_origins:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


app = FastAPI(title="Rubik Agent Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "POST"],
    allow_headers=["Content-Type"],
)

@lru_cache(maxsize=1)
def get_chat_repository() -> SQLiteChatSessionRepository:
    database_path = os.getenv("CHAT_DB_PATH") or str(DEFAULT_CHAT_DB_PATH)
    return SQLiteChatSessionRepository(database_path)


def get_chat_session_service() -> ChatSessionService:
    agent = Agent(DeepSeekProvider())
    return ChatSessionService(get_chat_repository(), agent)


@app.get("/api/health")
def health() -> dict[str, bool | str]:
    return {
        "status": "ok",
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


@app.post("/api/chat/sessions", response_model=ChatSession, status_code=201)
def create_chat_session(
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    return service.create()


@app.get("/api/chat/sessions", response_model=list[ChatSessionSummary])
def list_chat_sessions(
    service: ChatSessionService = Depends(get_chat_session_service),
) -> list[ChatSessionSummary]:
    return service.list()


@app.delete("/api/chat/sessions", status_code=204)
def clear_chat_sessions(
    service: ChatSessionService = Depends(get_chat_session_service),
) -> Response:
    service.clear()
    return Response(status_code=204)


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSession)
def get_chat_session(
    session_id: str,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    try:
        return service.get(session_id)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None


@app.post(
    "/api/chat/sessions/{session_id}/messages",
    response_model=ChatSendResponse,
)
def send_chat_message(
    session_id: str,
    request: ChatSendRequest,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSendResponse:
    try:
        return service.send(session_id, request.content)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except AgentInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except (AgentOutputError, LlmRequestError):
        raise HTTPException(status_code=502, detail="Запрос к модели завершился ошибкой") from None
    except LlmConfigurationError:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY не задан") from None


STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"
app.mount("/", StaticFiles(directory=STATIC_ROOT, html=True), name="static")
