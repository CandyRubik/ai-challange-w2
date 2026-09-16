from dataclasses import replace

import pytest

from app.agents.task_state import (
    TaskContext, TaskConflict, TaskState, approve_plan, complete_step,
    pause, replan, resume, transition,
)


def planned_task() -> TaskContext:
    return TaskContext(task="Подготовить курс", plan=("Темы", "Практика"), criteria=("Есть упражнения",))


@pytest.mark.parametrize("target", [TaskState.VALIDATION, TaskState.DONE, TaskState.PLANNING])
def test_planning_rejects_forbidden_transitions(target: TaskState) -> None:
    with pytest.raises(TaskConflict, match="запрещён"):
        transition(planned_task(), target)


def test_plan_and_completed_steps_are_required() -> None:
    with pytest.raises(TaskConflict):
        approve_plan(TaskContext(task="Курс"))
    with pytest.raises(ValueError, match="завершить"):
        transition(approve_plan(planned_task()), TaskState.VALIDATION)


def test_progress_and_expected_action_follow_saved_results() -> None:
    ctx = approve_plan(planned_task())
    assert ctx.current == "Темы"
    first = complete_step(ctx, "Готовые темы")
    assert first.step == 1 and first.total == 2
    assert first.current == "Практика"
    assert ctx.step == 0  # Old snapshots remain immutable.
    validated = complete_step(first, "Готовые упражнения")
    assert validated.state == TaskState.VALIDATION
    assert validated.expected_action == "validate"
    assert TaskContext.from_json(validated.to_json()) == validated


def test_pause_preserves_the_stage_step_and_expected_action() -> None:
    ctx = complete_step(approve_plan(planned_task()), "Готовые темы")
    paused = pause(ctx)
    assert paused == replace(ctx, paused=True)
    with pytest.raises(TaskConflict):
        complete_step(paused, "Нельзя выполнять")
    assert resume(paused) == ctx


def test_replanning_archives_results_and_records_new_requirements() -> None:
    ctx = complete_step(approve_plan(planned_task()), "Готовые темы")
    updated = replan(ctx, "Добавить итоговый проект")
    assert updated.state == TaskState.PLANNING
    assert updated.expected_action == "generate_plan"
    assert updated.step == 0 and updated.plan == () and updated.done == ()
    assert updated.previous_results == ctx.done
    assert updated.notes == ("Добавить итоговый проект",)


def test_done_is_terminal() -> None:
    ctx = complete_step(complete_step(approve_plan(planned_task()), "Темы"), "Практика")
    done = transition(ctx, TaskState.DONE, result="Итоговый курс")
    for target in TaskState:
        with pytest.raises(TaskConflict):
            transition(done, target)
    with pytest.raises(TaskConflict):
        pause(done)
