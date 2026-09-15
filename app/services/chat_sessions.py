from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Protocol
from uuid import uuid4

from ..agents.agent import Agent, AgentMessage, ProfileContext
from ..agents.memory_extractor import MemoryExtractionError, MemoryExtractor
from ..agents.profile_interviewer import ProfileInterviewError, ProfileInterviewer
from ..schemas import ChatMessage, ChatSendResponse, ChatSession, ChatSessionSummary
from .memory import MemoryRepository
from .profiles import (
    DEFAULT_PROFILE_ID,
    ProfileRepository,
    StoredProfile,
    ensure_profile_schema,
)


DEFAULT_CHAT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat.sqlite3"
DEFAULT_DB_PATH = DEFAULT_CHAT_DB_PATH
logger = logging.getLogger(__name__)


class ChatSessionNotFound(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    role: str
    kind: str
    content: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredSession:
    id: str
    profile_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[StoredMessage, ...] = ()


class ChatSessionRepository(Protocol):
    def create(self, profile_id: str = DEFAULT_PROFILE_ID) -> StoredSession: ...

    def list(self, profile_id: str | None = None) -> list[StoredSession]: ...

    def get(self, session_id: str) -> StoredSession: ...

    def clear(self, profile_id: str | None = None) -> None: ...

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> StoredSession: ...

    def append_command(self, session_id: str, command_text: str) -> StoredSession: ...


class SQLiteChatSessionRepository:
    """Durable chat history isolated behind a repository boundary."""

    def __init__(self, database_path: str | Path = DEFAULT_CHAT_DB_PATH) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            ensure_profile_schema(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL DEFAULT 'default'
                        REFERENCES user_profiles(id),
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    kind TEXT NOT NULL DEFAULT 'message'
                        CHECK (kind IN ('message', 'command')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id, position);
                """,
            )
            session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_sessions)")
            }
            if "profile_id" not in session_columns:
                connection.execute(
                    """
                    ALTER TABLE chat_sessions
                    ADD COLUMN profile_id TEXT REFERENCES user_profiles(id)
                    """,
                )
                connection.execute(
                    "UPDATE chat_sessions SET profile_id = ? WHERE profile_id IS NULL",
                    (DEFAULT_PROFILE_ID,),
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_profile
                ON chat_sessions(profile_id, updated_at)
                """,
            )

            message_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_messages)")
            }
            if "kind" not in message_columns:
                connection.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN kind TEXT NOT NULL DEFAULT 'message'
                        CHECK (kind IN ('message', 'command'))
                    """,
                )

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def _load_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> StoredSession:
        row = connection.execute(
            """
            SELECT id, profile_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise ChatSessionNotFound(session_id)

        message_rows = connection.execute(
            """
            SELECT id, role, kind, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        messages = tuple(
            StoredMessage(
                id=message["id"],
                role=message["role"],
                kind=message["kind"],
                content=message["content"],
                created_at=self._datetime(message["created_at"]),
            )
            for message in message_rows
        )
        return StoredSession(
            id=row["id"],
            profile_id=row["profile_id"] or DEFAULT_PROFILE_ID,
            title=row["title"],
            created_at=self._datetime(row["created_at"]),
            updated_at=self._datetime(row["updated_at"]),
            messages=messages,
        )

    def create(self, profile_id: str = DEFAULT_PROFILE_ID) -> StoredSession:
        session_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions
                    (id, profile_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    profile_id,
                    "Новый чат",
                    self._timestamp(now),
                    self._timestamp(now),
                ),
            )
        return StoredSession(session_id, profile_id, "Новый чат", now, now)

    def list(self, profile_id: str | None = None) -> list[StoredSession]:
        with self._connection() as connection:
            if profile_id is None:
                rows = connection.execute(
                    """
                    SELECT id FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT 100
                    """,
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id FROM chat_sessions
                    WHERE profile_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 100
                    """,
                    (profile_id,),
                ).fetchall()
            return [self._load_session(connection, row["id"]) for row in rows]

    def get(self, session_id: str) -> StoredSession:
        with self._connection() as connection:
            return self._load_session(connection, session_id)

    def clear(self, profile_id: str | None = None) -> None:
        with self._connection() as connection:
            if profile_id is None:
                connection.execute("DELETE FROM chat_sessions")
            else:
                connection.execute(
                    "DELETE FROM chat_sessions WHERE profile_id = ?",
                    (profile_id,),
                )

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> StoredSession:
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            position = len(session.messages)
            title = session.title
            if not session.messages:
                title = user_content.replace("\n", " ").strip()[:60] or "Новый чат"

            connection.executemany(
                """
                INSERT INTO chat_messages
                    (id, session_id, position, role, kind, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(uuid4()), session_id, position, "user", "message",
                        user_content, timestamp,
                    ),
                    (
                        str(uuid4()), session_id, position + 1, "assistant", "message",
                        assistant_content, timestamp,
                    ),
                ],
            )
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, session_id),
            )

        return self.get(session_id)

    def append_command(self, session_id: str, command_text: str) -> StoredSession:
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            position = len(session.messages)
            title = session.title
            if not session.messages:
                title = command_text.replace("\n", " ").strip()[:60] or "Новый чат"
            connection.execute(
                """
                INSERT INTO chat_messages
                    (id, session_id, position, role, kind, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()), session_id, position, "user", "command",
                    command_text, timestamp,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, session_id),
            )
        return self.get(session_id)


class ChatSessionService:
    """Adapt persistent sessions to the storage-agnostic Agent."""

    def __init__(
        self,
        repository: ChatSessionRepository,
        agent: Agent,
        memory_repository: MemoryRepository | None = None,
        memory_extractor: MemoryExtractor | None = None,
        profile_repository: ProfileRepository | None = None,
        profile_interviewer: ProfileInterviewer | None = None,
    ) -> None:
        self._repository = repository
        self._agent = agent
        self._memory_repository = memory_repository
        self._memory_extractor = memory_extractor
        self._profile_repository = profile_repository
        self._profile_interviewer = profile_interviewer

    @staticmethod
    def _summary(session: StoredSession) -> ChatSessionSummary:
        return ChatSessionSummary(
            id=session.id,
            profile_id=session.profile_id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    @staticmethod
    def _message(message: StoredMessage) -> ChatMessage:
        return ChatMessage(
            id=message.id,
            role=message.role,
            kind=message.kind,
            content=message.content,
            created_at=message.created_at,
        )

    @staticmethod
    def _profile_context(profile: StoredProfile) -> ProfileContext:
        return {
            "name": profile.name,
            "description": profile.description,
            "language": profile.language,
            "tone": profile.tone,
            "detail_level": profile.detail_level,
            "response_format": profile.response_format,
            "constraints": list(profile.constraints),
        }

    @classmethod
    def _response(cls, updated: StoredSession) -> ChatSendResponse:
        return ChatSendResponse(
            session=cls._summary(updated),
            user_message=cls._message(updated.messages[-2]),
            assistant_message=cls._message(updated.messages[-1]),
        )

    def _memory_context(
        self,
        session_id: str,
        profile_id: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        if self._memory_repository is None:
            return [], []
        return (
            [
                {"category": memory.category, "content": memory.content}
                for memory in self._memory_repository.list_working(session_id)
            ],
            [
                {"category": memory.category, "content": memory.content}
                for memory in self._memory_repository.list_long_term(profile_id)
            ],
        )

    @staticmethod
    def _pending_request(session: StoredSession) -> str | None:
        return next(
            (
                message.content
                for message in session.messages
                if message.role == "user" and message.kind == "message"
            ),
            None,
        )

    def _finish_onboarding(
        self,
        *,
        session: StoredSession,
        content: str,
        profile: StoredProfile,
    ) -> ChatSendResponse:
        pending_request = self._pending_request(session)
        if pending_request is None:
            answer = (
                "Профиль готов. Теперь напишите задачу — дальше я буду учитывать "
                "эти настройки автоматически."
            )
            return self._response(
                self._repository.append_exchange(session.id, content.strip(), answer),
            )

        working_memory, long_term_memory = self._memory_context(
            session.id,
            session.profile_id,
        )
        profile_context = self._profile_context(profile)
        answer = self._agent.respond(
            [],
            pending_request,
            profile=profile_context,
            working_memory=working_memory,
            long_term_memory=long_term_memory,
        )
        updated = self._repository.append_exchange(session.id, content.strip(), answer)
        self._remember(
            session_id=session.id,
            profile_id=session.profile_id,
            profile=profile_context,
            context=[],
            content=pending_request,
            working_memory=working_memory,
            long_term_memory=long_term_memory,
        )
        return self._response(updated)

    def _onboard(
        self,
        *,
        session: StoredSession,
        content: str,
        profile: StoredProfile,
    ) -> ChatSendResponse:
        assert self._profile_repository is not None
        assert self._profile_interviewer is not None

        normalized = content.strip().casefold()
        if normalized in {"/skip", "skip", "пропустить", "пропусти", "по умолчанию"}:
            completed = self._profile_repository.apply_interview_update(
                profile.id,
                {},
                next_step=3,
                complete=True,
            )
            return self._finish_onboarding(
                session=session,
                content=content,
                profile=completed,
            )

        if profile.onboarding_step == 0:
            self._profile_repository.apply_interview_update(
                profile.id,
                {},
                next_step=1,
                complete=False,
            )
            updated = self._repository.append_exchange(
                session.id,
                content.strip(),
                self._profile_interviewer.questions[1],
            )
            return self._response(updated)

        step = profile.onboarding_step
        try:
            inference = self._profile_interviewer.extract(
                step=step,
                answer=content,
                profile=self._profile_context(profile),
            )
        except ProfileInterviewError:
            logger.warning("Automatic profile extraction failed", exc_info=True)
            answer = (
                "Не получилось надёжно разобрать ответ. Попробуйте сформулировать "
                "ещё раз.\n\n" + self._profile_interviewer.questions[step]
            )
            return self._response(
                self._repository.append_exchange(session.id, content.strip(), answer),
            )

        next_step = min(step + 1, 3)
        complete = step == 3
        updated_profile = self._profile_repository.apply_interview_update(
            profile.id,
            inference.values(),
            next_step=next_step,
            complete=complete,
        )
        if complete:
            return self._finish_onboarding(
                session=session,
                content=content,
                profile=updated_profile,
            )

        answer = self._profile_interviewer.questions[next_step]
        return self._response(
            self._repository.append_exchange(session.id, content.strip(), answer),
        )

    def create(self, profile_id: str = DEFAULT_PROFILE_ID) -> ChatSession:
        if self._profile_repository is not None:
            self._profile_repository.get(profile_id)
        session = self._repository.create(profile_id)
        return ChatSession(**self._summary(session).model_dump(), messages=[])

    def list(self, profile_id: str | None = None) -> list[ChatSessionSummary]:
        if profile_id is not None and self._profile_repository is not None:
            self._profile_repository.get(profile_id)
        return [
            self._summary(session)
            for session in self._repository.list(profile_id)
        ]

    def get(self, session_id: str) -> ChatSession:
        session = self._repository.get(session_id)
        return ChatSession(
            **self._summary(session).model_dump(),
            messages=[self._message(message) for message in session.messages],
        )

    def clear(self, profile_id: str | None = None) -> None:
        if profile_id is not None and self._profile_repository is not None:
            self._profile_repository.get(profile_id)
        self._repository.clear(profile_id)

    @staticmethod
    def _memory_key(
        layer: str,
        category: str,
        content: str,
    ) -> tuple[str, str, str]:
        normalized_content = " ".join(content.split()).casefold()
        return layer, category, normalized_content

    def _remember(
        self,
        *,
        session_id: str,
        profile_id: str,
        profile: ProfileContext | None,
        context: list[AgentMessage],
        content: str,
        working_memory: list[dict[str, str]],
        long_term_memory: list[dict[str, str]],
    ) -> None:
        if self._memory_repository is None or self._memory_extractor is None:
            return

        try:
            candidates = self._memory_extractor.extract(
                context=context,
                current_message=content,
                profile=profile,
                working_memory=working_memory,
                long_term_memory=long_term_memory,
            )
            known = {
                self._memory_key(
                    "working",
                    memory["category"],
                    memory["content"],
                )
                for memory in working_memory
            }
            known.update(
                self._memory_key(
                    "long_term",
                    memory["category"],
                    memory["content"],
                )
                for memory in long_term_memory
            )
            for candidate in candidates:
                key = self._memory_key(
                    candidate.layer,
                    candidate.category,
                    candidate.content,
                )
                if key in known:
                    continue
                self._memory_repository.add(
                    layer=candidate.layer,
                    category=candidate.category,
                    content=candidate.content,
                    session_id=session_id if candidate.layer == "working" else None,
                    profile_id=profile_id if candidate.layer == "long_term" else None,
                )
                known.add(key)
        except MemoryExtractionError:
            logger.warning("Automatic memory extraction failed", exc_info=True)
        except Exception:
            logger.exception("Automatic memory persistence failed")

    def send(self, session_id: str, content: str) -> ChatSendResponse:
        session = self._repository.get(session_id)
        profile = None
        if self._profile_repository is not None:
            stored_profile = self._profile_repository.get(session.profile_id)
            if (
                self._profile_interviewer is not None
                and not stored_profile.onboarding_complete
            ):
                return self._onboard(
                    session=session,
                    content=content,
                    profile=stored_profile,
                )
            profile = self._profile_context(stored_profile)
        context: list[AgentMessage] = [
            {"role": message.role, "content": message.content}
            for message in session.messages
            if message.kind == "message"
        ]
        working_memory, long_term_memory = self._memory_context(
            session_id,
            session.profile_id,
        )
        answer = self._agent.respond(
            context,
            content,
            profile=profile,
            working_memory=working_memory,
            long_term_memory=long_term_memory,
        )
        updated = self._repository.append_exchange(session_id, content.strip(), answer)
        self._remember(
            session_id=session_id,
            profile_id=session.profile_id,
            profile=profile,
            context=context,
            content=content,
            working_memory=working_memory,
            long_term_memory=long_term_memory,
        )
        return self._response(updated)
