from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Literal, Protocol
from uuid import uuid4

from ..schemas import MemoryCreateRequest, MemoryEntry, MemorySnapshot


MemoryLayer = Literal["working", "long_term"]
DEFAULT_MEMORY_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat.sqlite3"


class MemoryValidationError(ValueError):
    pass


class MemoryNotFound(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class StoredMemory:
    id: str
    layer: MemoryLayer
    category: str
    content: str
    session_id: str | None
    created_at: datetime
    updated_at: datetime


class MemoryRepository(Protocol):
    def add(
        self,
        *,
        layer: MemoryLayer,
        category: str,
        content: str,
        session_id: str | None,
    ) -> StoredMemory: ...

    def list_working(self, session_id: str) -> list[StoredMemory]: ...

    def list_long_term(self) -> list[StoredMemory]: ...

    def delete(self, layer: MemoryLayer, memory_id: str) -> bool: ...


class SessionLookup(Protocol):
    def get(self, session_id: str) -> object: ...

    def append_command(self, session_id: str, command_text: str) -> object: ...


class SQLiteMemoryRepository:
    """Persist working and long-term memories in physically separate tables."""

    _tables: dict[MemoryLayer, str] = {
        "working": "working_memory",
        "long_term": "long_term_memory",
    }

    def __init__(self, database_path: str | Path = DEFAULT_MEMORY_DB_PATH) -> None:
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS working_memory (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL
                        REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_working_memory_session
                ON working_memory(session_id, created_at);

                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """,
            )

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def _stored(self, row: sqlite3.Row, layer: MemoryLayer) -> StoredMemory:
        return StoredMemory(
            id=row["id"],
            layer=layer,
            category=row["category"],
            content=row["content"],
            session_id=row["session_id"] if layer == "working" else None,
            created_at=self._datetime(row["created_at"]),
            updated_at=self._datetime(row["updated_at"]),
        )

    def add(
        self,
        *,
        layer: MemoryLayer,
        category: str,
        content: str,
        session_id: str | None,
    ) -> StoredMemory:
        memory_id = str(uuid4())
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        table = self._tables[layer]
        with self._connection() as connection:
            if layer == "working":
                connection.execute(
                    f"""
                    INSERT INTO {table}
                        (id, session_id, category, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, session_id, category, content, timestamp, timestamp),
                )
            else:
                connection.execute(
                    f"""
                    INSERT INTO {table}
                        (id, category, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_id, category, content, timestamp, timestamp),
                )
        return StoredMemory(
            id=memory_id,
            layer=layer,
            category=category,
            content=content,
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )

    def _list(self, layer: MemoryLayer, session_id: str | None = None) -> list[StoredMemory]:
        table = self._tables[layer]
        with self._connection() as connection:
            if layer == "working":
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE session_id = ? ORDER BY created_at, id",
                    (session_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"SELECT * FROM {table} ORDER BY created_at, id",
                ).fetchall()
        return [self._stored(row, layer) for row in rows]

    def list_working(self, session_id: str) -> list[StoredMemory]:
        return self._list("working", session_id)

    def list_long_term(self) -> list[StoredMemory]:
        return self._list("long_term")

    def delete(self, layer: MemoryLayer, memory_id: str) -> bool:
        table = self._tables[layer]
        with self._connection() as connection:
            cursor = connection.execute(
                f"DELETE FROM {table} WHERE id = ?",
                (memory_id,),
            )
        return cursor.rowcount > 0


class MemoryService:
    """Validate explicit writes and expose memory snapshots to the API."""

    def __init__(
        self,
        repository: MemoryRepository,
        sessions: SessionLookup,
    ) -> None:
        self._repository = repository
        self._sessions = sessions

    @staticmethod
    def _entry(memory: StoredMemory) -> MemoryEntry:
        return MemoryEntry(
            id=memory.id,
            layer=memory.layer,
            category=memory.category,
            content=memory.content,
            session_id=memory.session_id,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )

    def create(self, request: MemoryCreateRequest) -> MemoryEntry:
        if request.layer == "working":
            if request.session_id is None:
                raise MemoryValidationError(
                    "Для рабочей памяти нужно явно указать session_id",
                )
            self._sessions.get(request.session_id)
        elif request.session_id is not None:
            raise MemoryValidationError(
                "Долговременная память не должна быть привязана к чат-сессии",
            )

        if (request.source_session_id is None) != (request.source_text is None):
            raise MemoryValidationError(
                "source_session_id и source_text нужно передавать вместе",
            )
        if request.source_session_id is not None:
            self._sessions.get(request.source_session_id)
            if (
                request.layer == "working"
                and request.source_session_id != request.session_id
            ):
                raise MemoryValidationError(
                    "Рабочую команду можно сохранить только в текущем чате",
                )

        stored = self._repository.add(
            layer=request.layer,
            category=request.category,
            content=request.content,
            session_id=request.session_id,
        )
        if request.source_session_id is not None and request.source_text is not None:
            self._sessions.append_command(
                request.source_session_id,
                request.source_text,
            )
        return self._entry(stored)

    def snapshot(self, session_id: str | None = None) -> MemorySnapshot:
        working: list[StoredMemory] = []
        if session_id is not None:
            self._sessions.get(session_id)
            working = self._repository.list_working(session_id)
        return MemorySnapshot(
            working=[self._entry(memory) for memory in working],
            long_term=[
                self._entry(memory) for memory in self._repository.list_long_term()
            ],
        )

    def delete(self, layer: MemoryLayer, memory_id: str) -> None:
        if not self._repository.delete(layer, memory_id):
            raise MemoryNotFound(memory_id)
