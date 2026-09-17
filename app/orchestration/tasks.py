from __future__ import annotations

from dataclasses import replace

from ..agents.agent import Agent
from ..memory.context import MemoryContext
from ..invariants import InvariantPolicy
from ..state.task import TaskContext, TaskConflict, TaskState, complete_step, transition
from .context import OrchestrationContext


class TaskOrchestrator:
    """Choose the next model stage and apply explicit domain transitions."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def advance(
        self, ctx: TaskContext, *, orchestration: OrchestrationContext,
        memory: MemoryContext,
    ) -> tuple[TaskContext, str]:
        policy = InvariantPolicy(orchestration.invariants)
        upper = orchestration.invariants is not None and orchestration.invariants.uppercase_enabled
        if ctx.expected_action == "generate_plan":
            output = self._agent.plan_task(ctx, orchestration=orchestration, memory=memory)
            updated = replace(
                ctx, plan=tuple(text.upper() if upper else text for text in output.plan),
                criteria=tuple(text.upper() if upper else text for text in output.criteria),
            )
            answer = output.summary + "\n\nПлан:\n" + "\n".join(
                f"{index}. {title}" for index, title in enumerate(output.plan, 1)
            ) + "\n\nКритерии готовности:\n" + "\n".join(
                "• " + criterion for criterion in output.criteria
            ) + "\n\nУтвердите план или укажите изменения."
            return updated, answer
        if ctx.expected_action == "execute_step":
            answer = self._agent.execute_task_step(ctx, orchestration=orchestration, memory=memory)
            answer = policy.apply(answer)
            return complete_step(ctx, answer), answer
        if ctx.expected_action == "validate":
            output = self._agent.validate_task(ctx, orchestration=orchestration, memory=memory)
            report = policy.apply(output.report)
            if output.passed:
                updated = transition(
                    ctx, TaskState.DONE, result=report,
                    validation_report=report,
                )
            else:
                updated = transition(
                    ctx, TaskState.EXECUTION,
                    plan=(*ctx.plan, *(text.upper() if upper else text for text in output.repair_steps)),
                    validation_report=report,
                )
            answer = report
            if not output.passed:
                answer += "\n\nШаги исправления:\n" + "\n".join(output.repair_steps)
            return updated, answer
        raise TaskConflict("Сейчас нужно утвердить план или задача уже завершена")
