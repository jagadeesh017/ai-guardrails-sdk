from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from ..core.context import ExecutionContext
from ..core.decision import Action, Violation
from ..core.policy import RuleConfig


class RuleResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class RuleResult:
    rule_name: str
    rule_type: str
    status: RuleResult = RuleResult.PASS
    violation: Violation = None
    action: Action = None
    latency_ms: float = 0.0
    error: str = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRule(ABC):
    def __init__(self, config: RuleConfig):
        self.config = config
        self.name = config.type
        self.enabled = config.enabled
        self.action_type = config.action
        self.fail_mode = config.fail_mode
        self.timeout_ms = config.timeout_ms

    @abstractmethod
    def check(self, ctx: ExecutionContext) -> RuleResult:
        pass

    async def check_async(self, ctx: ExecutionContext) -> RuleResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, ctx)

    def _create_violation(self, message: str, severity: str = "high", **metadata) -> Violation:
        return Violation(
            rule_name=self.name,
            rule_type=self.config.type,
            message=message,
            severity=severity,
            metadata=metadata,
        )

    def _create_action(self, action_type: str, reason: str, modified_data: Any = None, **metadata) -> Action:
        from ..core.decision import ActionType
        at = ActionType(action_type)
        if at == ActionType.REDACT:
            return Action.redact(self.name, reason, modified_data, **metadata)
        elif at == ActionType.WARN:
            return Action.warn(self.name, reason, **metadata)
        elif at == ActionType.REQUIRE_APPROVAL:
            return Action.require_approval(self.name, reason, **metadata)
        elif at == ActionType.THROTTLE:
            return Action.throttle(self.name, reason, **metadata)
        elif at == ActionType.LOG:
            return Action.log(self.name, reason, **metadata)
        return Action.block(self.name, reason, **metadata)


class Detector(ABC):
    @abstractmethod
    def detect(self, ctx: ExecutionContext) -> list[Violation]:
        pass


T = TypeVar("T")


class Evaluator[T](ABC):
    @abstractmethod
    def evaluate(self, input_data: T, ctx: ExecutionContext) -> list[Violation]:
        pass

    async def evaluate_async(self, input_data: T, ctx: ExecutionContext) -> list[Violation]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, input_data, ctx)