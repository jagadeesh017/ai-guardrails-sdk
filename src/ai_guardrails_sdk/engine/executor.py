import asyncio
from abc import ABC, abstractmethod

from ..core.context import ExecutionContext
from ..core.decision import Decision, DecisionStatus
from ..core.policy import EngineConfig, ExecutionMode, FailMode
from ..rules.base import BaseRule, RuleResult


class Executor(ABC):
    @abstractmethod
    async def execute(self, rules: list[BaseRule], ctx: ExecutionContext, config: EngineConfig) -> Decision:
        pass


class SequentialExecutor(Executor):
    async def execute(self, rules: list[BaseRule], ctx: ExecutionContext, config: EngineConfig) -> Decision:
        all_violations = []
        all_actions = []
        modified_data = None
        final_status = DecisionStatus.ALLOW

        for rule in rules:
            if not rule.enabled:
                continue

            try:
                result = await asyncio.wait_for(
                    rule.check_async(ctx),
                    timeout=config.timeout_ms / 1000,
                )
            except TimeoutError:
                if config.fail_mode == FailMode.FAIL_CLOSED:
                    violation = Violation(rule.name, rule.config.type, f"Rule timed out after {config.timeout_ms}ms")
                    action = rule._create_action("block", f"Rule timed out: {rule.name}")
                    return Decision(status=DecisionStatus.DENY, violations=[violation], actions=[action])
                continue
            except Exception as e:
                if config.fail_mode == FailMode.FAIL_CLOSED:
                    violation = Violation(rule.name, rule.config.type, f"Rule error: {e}")
                    action = rule._create_action("block", f"Rule error: {e}")
                    return Decision(status=DecisionStatus.DENY, violations=[violation], actions=[action])
                continue

            if result.error:
                if config.fail_mode == FailMode.FAIL_CLOSED:
                    violation = Violation(rule.name, rule.config.type, f"Rule error: {result.error}")
                    action = rule._create_action("block", f"Rule error: {result.error}")
                    return Decision(status=DecisionStatus.DENY, violations=[violation], actions=[action])
                continue

            if result.violation:
                all_violations.append(result.violation)
                if result.action:
                    all_actions.append(result.action)

                    if result.action.type.value == "block":
                        final_status = DecisionStatus.DENY
                        if config.fail_fast:
                            break
                    elif result.action.type.value == "redact":
                        final_status = DecisionStatus.MODIFY
                        modified_data = result.action.modified_data
                    elif result.action.type.value == "retry":
                        final_status = DecisionStatus.RETRY

        return Decision(
            status=final_status,
            violations=all_violations,
            actions=all_actions,
            modified_data=modified_data,
        )


class ParallelExecutor(Executor):
    async def execute(self, rules: list[BaseRule], ctx: ExecutionContext, config: EngineConfig) -> Decision:
        enabled_rules = [r for r in rules if r.enabled]
        if not enabled_rules:
            return Decision.allow()

        async def run_rule(rule: BaseRule) -> RuleResult:
            try:
                return await asyncio.wait_for(
                    rule.check_async(ctx),
                    timeout=config.timeout_ms / 1000,
                )
            except TimeoutError:
                if config.fail_mode == FailMode.FAIL_CLOSED:
                    return RuleResult(
                        rule_name=rule.name,
                        rule_type=rule.config.type,
                        status="fail",
                        violation=Violation(rule.name, rule.config.type, f"Rule timed out after {config.timeout_ms}ms"),
                        action=rule._create_action("block", f"Rule timed out: {rule.name}"),
                        error="timeout",
                    )
                return RuleResult(rule_name=rule.name, rule_type=rule.config.type, status="pass")
            except Exception as e:
                if config.fail_mode == FailMode.FAIL_CLOSED:
                    return RuleResult(
                        rule_name=rule.name,
                        rule_type=rule.config.type,
                        status="fail",
                        violation=Violation(rule.name, rule.config.type, f"Rule error: {e}"),
                        action=rule._create_action("block", f"Rule error: {e}"),
                        error=str(e),
                    )
                return RuleResult(rule_name=rule.name, rule_type=rule.config.type, status="pass")

        results = await asyncio.gather(*[run_rule(r) for r in enabled_rules])

        all_violations = []
        all_actions = []
        modified_data = None
        final_status = DecisionStatus.ALLOW

        for result in results:
            if result.error and config.fail_mode == FailMode.FAIL_CLOSED:
                return Decision(
                    status=DecisionStatus.DENY,
                    violations=[Violation(r.rule_name, r.rule_type, f"Error: {r.error}") for r in results if r.error],
                    actions=[],
                )

            if result.violation:
                all_violations.append(result.violation)
                if result.action:
                    all_actions.append(result.action)
                    if result.action.type.value == "block":
                        final_status = DecisionStatus.DENY
                    elif result.action.type.value == "redact":
                        final_status = DecisionStatus.MODIFY
                        modified_data = result.action.modified_data

        return Decision(
            status=final_status,
            violations=all_violations,
            actions=all_actions,
            modified_data=modified_data,
        )


def create_executor(mode: ExecutionMode) -> Executor:
    if mode == ExecutionMode.PARALLEL:
        return ParallelExecutor()
    return SequentialExecutor()