from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import json


class TaskState(StrEnum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


class TaskConflict(ValueError):
    """The requested action cannot be applied to the current snapshot."""


@dataclass(frozen=True, slots=True)
class StepResult:
    title: str
    output: str


@dataclass(frozen=True, slots=True)
class TaskContext:
    task: str
    state: TaskState = TaskState.PLANNING
    step: int = 0  # Number of completed steps in the current plan.
    plan: tuple[str, ...] = ()
    done: tuple[StepResult, ...] = ()
    criteria: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    previous_results: tuple[StepResult, ...] = ()
    validation_report: str = ""
    result: str = ""
    paused: bool = False

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("Задача не должна быть пустой")
        if self.step != len(self.done) or not 0 <= self.step <= self.total:
            raise ValueError("Прогресс не соответствует сохранённым результатам")
        if self.state != TaskState.PLANNING and (not self.plan or not self.criteria):
            raise ValueError("Для выполнения нужен план и критерии готовности")
        if self.state in {TaskState.VALIDATION, TaskState.DONE} and self.step != self.total:
            raise ValueError("Сначала нужно завершить все шаги")
        if self.state == TaskState.EXECUTION and self.step == self.total:
            raise ValueError("Все шаги выполнены — нужна проверка")
        if self.state == TaskState.DONE and (not self.result or self.paused):
            raise ValueError("Завершённая задача должна содержать итог")

    @property
    def total(self) -> int:
        return len(self.plan)

    @property
    def expected_action(self) -> str:
        if self.state == TaskState.PLANNING:
            return "approve_plan" if self.plan else "generate_plan"
        return {
            TaskState.EXECUTION: "execute_step",
            TaskState.VALIDATION: "validate",
            TaskState.DONE: "none",
        }[self.state]

    @property
    def current(self) -> str:
        if self.state == TaskState.EXECUTION:
            return self.plan[self.step]
        return {
            "generate_plan": "Сформировать план",
            "approve_plan": "Утвердить план",
            "validate": "Проверить результат по критериям",
            "none": "Задача завершена",
        }[self.expected_action]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "total": self.total,
            "current": self.current,
            "expected_action": self.expected_action,
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str) -> TaskContext:
        data = json.loads(value)
        data["state"] = TaskState(data["state"])
        for field in ("plan", "criteria", "notes"):
            data[field] = tuple(data[field])
        for field in ("done", "previous_results"):
            data[field] = tuple(StepResult(**item) for item in data[field])
        return cls(**data)


TRANSITIONS = {
    TaskState.PLANNING: frozenset({TaskState.EXECUTION}),
    TaskState.EXECUTION: frozenset({TaskState.VALIDATION, TaskState.PLANNING}),
    TaskState.VALIDATION: frozenset({TaskState.DONE, TaskState.EXECUTION}),
    TaskState.DONE: frozenset(),
}


def transition(ctx: TaskContext, target: TaskState, **changes) -> TaskContext:
    if ctx.paused:
        raise TaskConflict("Задача на паузе. Сначала нажмите «Продолжить»")
    if target not in TRANSITIONS[ctx.state]:
        raise TaskConflict(f"Переход {ctx.state} → {target} запрещён")
    return replace(ctx, **changes, state=target)


def approve_plan(ctx: TaskContext) -> TaskContext:
    if ctx.state != TaskState.PLANNING or not ctx.plan:
        raise TaskConflict("Нет плана для утверждения")
    return transition(ctx, TaskState.EXECUTION)


def complete_step(ctx: TaskContext, output: str) -> TaskContext:
    if ctx.paused or ctx.state != TaskState.EXECUTION:
        raise TaskConflict("Сейчас нельзя выполнить шаг")
    done = (*ctx.done, StepResult(ctx.current, output))
    changes = {"step": ctx.step + 1, "done": done}
    if ctx.step + 1 == ctx.total:
        return transition(ctx, TaskState.VALIDATION, **changes)
    return replace(ctx, **changes)


def pause(ctx: TaskContext) -> TaskContext:
    if ctx.state == TaskState.DONE:
        raise TaskConflict("Задача уже завершена")
    return replace(ctx, paused=True)


def resume(ctx: TaskContext) -> TaskContext:
    if not ctx.paused:
        raise TaskConflict("Задача не находится на паузе")
    return replace(ctx, paused=False)


def replan(ctx: TaskContext, note: str) -> TaskContext:
    if ctx.paused or ctx.state not in {TaskState.PLANNING, TaskState.EXECUTION}:
        raise TaskConflict("Сейчас нельзя пересмотреть план")
    changes = {
        "plan": (), "criteria": (), "step": 0, "done": (),
        "notes": (*ctx.notes, note),
        "previous_results": (*ctx.previous_results, *ctx.done),
        "validation_report": "", "result": "",
    }
    if ctx.state == TaskState.EXECUTION:
        return transition(ctx, TaskState.PLANNING, **changes)
    return replace(ctx, **changes)


@dataclass(frozen=True, slots=True)
class StoredTask:
    context: TaskContext
    revision: int
    progress_revision: int
