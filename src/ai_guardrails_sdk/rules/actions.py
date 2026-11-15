import re
from collections.abc import Callable
from typing import Any

from ..core.context import ExecutionContext
from ..core.decision import Action, ActionType, Violation
from .base import BaseRule, RuleConfig, RuleResult


class ActionExecutor:
    def __init__(self):
        self._handlers: dict[ActionType, Callable] = {
            ActionType.BLOCK: self._block,
            ActionType.WARN: self._warn,
            ActionType.REDACT: self._redact,
            ActionType.REQUIRE_APPROVAL: self._require_approval,
            ActionType.THROTTLE: self._throttle,
            ActionType.LOG: self._log,
        }

    def execute(self, violation: Violation, action_type: ActionType, ctx: ExecutionContext, **kwargs) -> Action:
        handler = self._handlers.get(action_type, self._block)
        return handler(violation, ctx, **kwargs)

    def _block(self, violation: Violation, ctx: ExecutionContext, **kwargs) -> Action:
        return Action.block(violation.rule_name, violation.message, **kwargs)

    def _warn(self, violation: Violation, ctx: ExecutionContext, **kwargs) -> Action:
        return Action.warn(violation.rule_name, violation.message, **kwargs)

    def _redact(self, violation: Violation, ctx: ExecutionContext, **kwargs) -> Action:
        modified = self._apply_redaction(ctx, violation)
        return Action.redact(violation.rule_name, violation.message, modified, **kwargs)

    def _require_approval(self, violation: Violation, ctx: ExecutionContext, **kwargs) -> Action:
        return Action.require_approval(violation.rule_name, violation.message, **kwargs)

    def _throttle(self, violation: Violation, ctx: ExecutionContext, **kwargs) -> Action:
        return Action.throttle(violation.rule_name, violation.message, **kwargs)

    def _log(self, violation: Violation, ctx: ExecutionContext, **kwargs) -> Action:
        return Action.log(violation.rule_name, violation.message, **kwargs)

    def _apply_redaction(self, ctx: ExecutionContext, violation: Violation) -> Any:
        text = ctx.text or ctx.response
        if not text:
            return None

        metadata = violation.metadata or {}
        if "redaction_fn" in metadata:
            return metadata["redaction_fn"](text)
        if "patterns" in metadata:
            result = text
            for pattern, replacement in metadata["patterns"]:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            return result
        return text


class BaseRuleWithExecutor(BaseRule):
    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.executor = ActionExecutor()

    def apply_action(self, violation: Violation, ctx: ExecutionContext, modified_data: Any = None) -> Action:
        action_type = ActionType(self.action_type)
        if action_type == ActionType.REDACT and modified_data is not None:
            return Action.redact(violation.rule_name, violation.message, modified_data)
        return self.executor.execute(violation, action_type, ctx)

    def create_result(
        self, 
        ctx: ExecutionContext, 
        violation: Violation = None, 
        modified_data: Any = None,
        error: str = None
    ) -> RuleResult:
        if error:
            return RuleResult(
                rule_name=self.name,
                rule_type=self.config.type,
                status="error",
                error=error,
            )

        if violation:
            action = self.apply_action(violation, ctx, modified_data)

            if action.type == ActionType.BLOCK or action.type == ActionType.REDACT:
                status = "fail"
            elif action.type == ActionType.WARN:
                status = "pass"
            else:
                status = "fail"

            return RuleResult(
                rule_name=self.name,
                rule_type=self.config.type,
                status=status,
                violation=violation,
                action=action,
            )

        return RuleResult(
            rule_name=self.name,
            rule_type=self.config.type,
            status="pass",
        )