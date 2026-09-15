from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
    )


class ChatSendRequest(StrictModel):
    content: Annotated[str, Field(min_length=1, max_length=12_000)]


class ChatMessage(StrictModel):
    id: str
    role: Literal["user", "assistant"]
    kind: Literal["message", "command"] = "message"
    content: str
    created_at: datetime


class ChatSessionSummary(StrictModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatSession(ChatSessionSummary):
    messages: list[ChatMessage]


class ChatSendResponse(StrictModel):
    session: ChatSessionSummary
    user_message: ChatMessage
    assistant_message: ChatMessage


MemoryLayer = Literal["working", "long_term"]


class MemoryCreateRequest(StrictModel):
    layer: MemoryLayer
    category: Annotated[str, Field(min_length=1, max_length=40)]
    content: Annotated[str, Field(min_length=1, max_length=4_000)]
    session_id: str | None = None
    source_session_id: str | None = None
    source_text: Annotated[str, Field(min_length=1, max_length=4_000)] | None = None


class MemoryEntry(StrictModel):
    id: str
    layer: MemoryLayer
    category: str
    content: str
    session_id: str | None
    created_at: datetime
    updated_at: datetime


class MemorySnapshot(StrictModel):
    working: list[MemoryEntry]
    long_term: list[MemoryEntry]
