from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Protocol
from uuid import uuid4

from ..agents.agent import Agent, AgentMessage, ProfileContext
from ..agents.memory_extractor import MemoryExtractionError, MemoryExtractor
from ..agents.profile_interviewer import ProfileInterviewError, ProfileInterviewer
from ..schemas import ChatMessage, ChatSendResponse, ChatSession, ChatSessionSummary, TaskActionRequest, TaskSummary, TaskView
from ..agents.task_state import TaskContext, TaskConflict, TaskState, approve_plan, complete_step, pause, replan, resume, transition
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
class StoredTask:
    context: TaskContext
    revision: int
    progress_revision: int


@dataclass(frozen=True, slots=True)
class StoredSession:
    id: str
    profile_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[StoredMessage, ...] = ()
    task: StoredTask | None = None


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


    def task_operation(self, session_id: str) -> AbstractContextManager[None]: ...

    def create_task(self, session_id: str, task: TaskContext) -> StoredSession: ...

    def update_task(
        self, session_id: str, task: TaskContext,
        user_content: str, assistant_content: str, *,
        expected_revision: int | None = None,
        expected_progress_revision: int | None = None,
        pause_only: bool = False,
    ) -> StoredSession: ...


class SQLiteChatSessionRepository:
    """Durable chat history isolated behind a repository boundary."""

    def __init__(self, database_path: str | Path = DEFAULT_CHAT_DB_PATH) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._task_locks: dict[str, Lock] = {}
        self._task_locks_guard = Lock()
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

                CREATE TABLE IF NOT EXISTS chat_tasks (
                    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    context TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    progress_revision INTEGER NOT NULL DEFAULT 0
                );
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
        task_row = connection.execute(
            "SELECT context, revision, progress_revision FROM chat_tasks WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        task = None if task_row is None else StoredTask(
            context=TaskContext.from_json(task_row["context"]),
            revision=task_row["revision"],
            progress_revision=task_row["progress_revision"],
        )
        return StoredSession(
            id=row["id"],
            profile_id=row["profile_id"] or DEFAULT_PROFILE_ID,
            title=row["title"],
            created_at=self._datetime(row["created_at"]),
            updated_at=self._datetime(row["updated_at"]),
            messages=messages,
            task=task,
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
            connection.execute("BEGIN")
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
            connection.execute("BEGIN")
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
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            self._append_exchange(connection, session, user_content, assistant_content)

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


    def _append_exchange(
        self, connection: sqlite3.Connection, session: StoredSession,
        user_content: str, assistant_content: str,
    ) -> None:
        timestamp = self._timestamp(datetime.now(timezone.utc))
        position = len(session.messages)
        title = session.title
        if not session.messages:
            title = user_content.replace("\n", " ").strip()[:60] or "Новый чат"
        connection.executemany(
            """
            INSERT INTO chat_messages
                (id, session_id, position, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (str(uuid4()), session.id, position, "user", user_content, timestamp),
                (str(uuid4()), session.id, position + 1, "assistant", assistant_content, timestamp),
            ],
        )
        connection.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, timestamp, session.id),
        )

    @contextmanager
    def task_operation(self, session_id: str) -> Iterator[None]:
        # The repository is shared by requests. Pause/resume deliberately bypass
        # this lock; a running generation can finish into a paused snapshot.
        with self._task_locks_guard:
            lock = self._task_locks.setdefault(session_id, Lock())
        if not lock.acquire(blocking=False):
            raise TaskConflict("Шаг уже выполняется. Дождитесь ответа агента")
        try:
            yield
        finally:
            lock.release()

    def create_task(self, session_id: str, task: TaskContext) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            if session.task is not None:
                raise TaskConflict("В этом чате уже есть задача. Создайте новый чат")
            connection.execute(
                "INSERT INTO chat_tasks (session_id, context) VALUES (?, ?)",
                (session_id, task.to_json()),
            )
            self._append_exchange(
                connection, session, "Задача: " + task.task,
                "Задача создана. Нажмите «Сформировать план», затем утвердите его.",
            )
        return self.get(session_id)

    def update_task(
        self, session_id: str, task: TaskContext,
        user_content: str, assistant_content: str, *,
        expected_revision: int | None = None,
        expected_progress_revision: int | None = None,
        pause_only: bool = False,
    ) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            stored = session.task
            if stored is None:
                raise TaskConflict("В чате нет задачи")
            if expected_revision is not None and stored.revision != expected_revision:
                raise TaskConflict("Состояние изменилось. Обновите чат и повторите действие")
            if expected_progress_revision is not None:
                if stored.progress_revision != expected_progress_revision:
                    raise TaskConflict("Состояние задачи изменилось во время выполнения")
                # A pause arriving during successful validation is acknowledged
                # as completed: DONE is terminal and cannot remain paused.
                task = replace(
                    task, paused=stored.context.paused and task.state != TaskState.DONE,
                )
            connection.execute(
                """
                UPDATE chat_tasks
                SET context = ?, revision = revision + 1,
                    progress_revision = progress_revision + ?
                WHERE session_id = ?
                """,
                (task.to_json(), 0 if pause_only else 1, session_id),
            )
            self._append_exchange(connection, session, user_content, assistant_content)
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
            task=None if session.task is None else TaskSummary(
                state=session.task.context.state,
                step=session.task.context.step,
                total=session.task.context.total,
                current=session.task.context.current,
                paused=session.task.context.paused,
                revision=session.task.revision,
            ),
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
            **self._summary(session).model_dump(exclude={"task"}),
            messages=[self._message(message) for message in session.messages],
            task=None if session.task is None else TaskView(
                **session.task.context.to_dict(), revision=session.task.revision,
            ),
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
        if session.task is not None and session.task.context.state != TaskState.DONE:
            raise TaskConflict("Используйте действия задачи или пересмотрите план")
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


    def start_task(self, session_id: str, task: str) -> ChatSession:
        session = self._repository.get(session_id)
        if self._profile_repository is not None and self._profile_interviewer is not None:
            profile = self._profile_repository.get(session.profile_id)
            if not profile.onboarding_complete:
                raise TaskConflict("Сначала завершите интервью профиля или отправьте /skip в обычном чате")
        self._repository.create_task(session_id, TaskContext(task=task.strip()))
        return self.get(session_id)

    def task_action(self, session_id: str, request: TaskActionRequest) -> ChatSession:
        if request.action in {"pause", "resume"}:
            return self._apply_task_action(session_id, request)
        with self._repository.task_operation(session_id):
            return self._apply_task_action(session_id, request)

    def _apply_task_action(self, session_id: str, request: TaskActionRequest) -> ChatSession:
        session = self._repository.get(session_id)
        stored = session.task
        if stored is None:
            raise TaskConflict("В чате нет задачи")
        if request.revision != stored.revision:
            raise TaskConflict("Состояние изменилось. Обновите чат и повторите действие")
        ctx = stored.context
        action = request.action
        if action == "pause":
            updated, answer = pause(ctx), "Задача на паузе. Этап, шаг и результаты сохранены."
        elif action == "resume":
            updated = resume(ctx)
            answer = "Продолжаем с сохранённого состояния. Ожидаемое действие: " + ctx.current
        elif action == "approve":
            updated = approve_plan(ctx)
            answer = "План утверждён. Следующий шаг: " + updated.current
        elif action == "replan":
            updated = replan(ctx, request.content)
            answer = "Требования сохранены. Сформируйте новый план; готовые результаты доступны агенту."
        else:
            if ctx.paused:
                raise TaskConflict("Задача на паузе. Сначала нажмите «Продолжить»")
            profile = None if self._profile_repository is None else self._profile_context(
                self._profile_repository.get(session.profile_id),
            )
            working_memory, long_term_memory = self._memory_context(session_id, session.profile_id)
            updated, answer = self._advance_task(
                ctx, profile=profile,
                working_memory=working_memory, long_term_memory=long_term_memory,
            )
        is_generation = action == "advance"
        self._repository.update_task(
            session_id, updated,
            {"advance": ctx.current, "approve": "Утвердить план", "pause": "Пауза",
             "resume": "Продолжить", "replan": "Пересмотреть план: " + request.content}[action],
            answer,
            expected_revision=None if is_generation else request.revision,
            expected_progress_revision=stored.progress_revision if is_generation else None,
            pause_only=action in {"pause", "resume"},
        )
        return self.get(session_id)

    def _advance_task(self, ctx: TaskContext, **context) -> tuple[TaskContext, str]:
        if ctx.expected_action == "generate_plan":
            output = self._agent.plan_task(ctx, **context)
            updated = replace(ctx, plan=tuple(output.plan), criteria=tuple(output.criteria))
            answer = output.summary + "\n\nПлан:\n" + "\n".join(
                f"{index}. {title}" for index, title in enumerate(output.plan, 1)
            ) + "\n\nКритерии готовности:\n" + "\n".join(
                "• " + criterion for criterion in output.criteria
            ) + "\n\nУтвердите план или укажите изменения."
            return updated, answer
        if ctx.expected_action == "execute_step":
            answer = self._agent.execute_task_step(ctx, **context)
            return complete_step(ctx, answer), answer
        if ctx.expected_action == "validate":
            output = self._agent.validate_task(ctx, **context)
            if output.passed:
                updated = transition(
                    ctx, TaskState.DONE, result=output.report,
                    validation_report=output.report,
                )
            else:
                updated = transition(
                    ctx, TaskState.EXECUTION,
                    plan=(*ctx.plan, *output.repair_steps),
                    validation_report=output.report,
                )
            answer = output.report
            if not output.passed:
                answer += "\n\nШаги исправления:\n" + "\n".join(output.repair_steps)
            return updated, answer
        raise TaskConflict("Сейчас нужно утвердить план или задача уже завершена")
