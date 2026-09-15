# Rubik Agent Chat

Минимальный standalone-чат с агентом, выделенный из ветки `main` проекта
RubikStudyHarness. В приложении остался только чатовый контур: создание сессий,
отправка сообщений и сохранение независимой истории каждого диалога.

Агент изолирован от кода на уровне архитектуры: ему передаются только системная
инструкция и сообщения текущей сессии. У провайдера нет tools/function calling,
доступа к файлам, репозиторию, shell или окружению процесса.

## Что внутри

- FastAPI backend и компактный web-интерфейс;
- отдельные чат-сессии с независимым контекстом;
- SQLite-хранилище истории, переживающее перезапуск;
- input/output policy для ограничения пользовательского ввода и ответа;
- изолированный DeepSeek provider;
- тесты backend, provider и frontend syntax check в CI.

Benchmark, runtime debug-настройки, multi-agent логика, workflow-интеграции и
инструменты работы с кодом в проект не входят.

## Запуск

Требуется Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Заполните `DEEPSEEK_API_KEY` в `.env`, затем запустите сервер:

```bash
set -a
source .env
set +a
uvicorn app.main:app --reload --port 8000
```

Откройте <http://127.0.0.1:8000>. История по умолчанию хранится в
`data/chat.sqlite3`; путь можно изменить через `CHAT_DB_PATH`.

## API

- `GET /api/health` — состояние backend и наличие ключа DeepSeek;
- `POST /api/chat/sessions` — создать чат;
- `GET /api/chat/sessions` — список чатов;
- `DELETE /api/chat/sessions` — удалить все чаты и сообщения;
- `GET /api/chat/sessions/{id}` — получить историю чата;
- `POST /api/chat/sessions/{id}/messages` — отправить сообщение агенту.

```text
HTTP API → ChatSessionService → ChatSessionRepository → SQLite
                         └──→ Agent → LanguageModel → DeepSeek API
```

## Проверки

```bash
python -m pytest -q
python -m compileall -q app tests
node --check static/app.js
```
