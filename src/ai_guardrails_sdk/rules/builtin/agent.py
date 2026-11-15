
from ai_guardrails_sdk.core.context import ExecutionContext
from ai_guardrails_sdk.core.decision import Violation
from ai_guardrails_sdk.core.policy import RuleConfig
from ai_guardrails_sdk.rules.actions import BaseRuleWithExecutor
from ai_guardrails_sdk.rules.base import RuleResult


class ToolAllowlistRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        if ctx.stage.value != "agent" or not ctx.tool_call:
            return self.create_result(ctx)

        allowed = set(self.config.params.get("allowed_tools", []))
        denied = set(self.config.params.get("denied_tools", []))

        tool_name = ctx.tool_call.name
        if denied and tool_name in denied:
            violation = self._create_violation(
                f"Denied tool: {tool_name}",
                metadata={"tool": tool_name},
            )
            return self.create_result(ctx, violation)

        if allowed and tool_name not in allowed:
            violation = self._create_violation(
                f"Tool not in allowlist: {tool_name}",
                metadata={"tool": tool_name, "allowed": list(allowed)},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


class ToolSchemaRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        if ctx.stage.value != "agent" or not ctx.tool_call:
            return self.create_result(ctx)

        tool_name = ctx.tool_call.name
        schemas = self.config.params.get("schemas", {})

        if tool_name not in schemas:
            return self.create_result(ctx)

        schema = schemas[tool_name]
        args = ctx.tool_call.arguments or {}

        violations = []

        required = schema.get("required", [])
        for req in required:
            if req not in args:
                violations.append(Violation(
                    rule_name="tool_schema",
                    rule_type="missing_required",
                    message=f"Missing required argument: {req}",
                    metadata={"tool": ctx.tool_call.name, "field": req},
                ))

        properties = schema.get("properties", {})
        for key, value in args.items():
            if key in properties:
                prop_schema = properties[key]
                expected_type = prop_schema.get("type")
                if expected_type:
                    type_map = {"string": str, "number": (int, float), "integer": int,
                                "boolean": bool, "array": list, "object": dict}
                    expected = type_map.get(expected_type)
                    if expected and not isinstance(value, expected):
                        violations.append(Violation(
                            rule_name="tool_schema",
                            rule_type="type_mismatch",
                            message=f"Type mismatch for {key}: expected {expected_type}, got {type(value).__name__}",
                            metadata={"tool": ctx.tool_call.name, "field": key, "expected": expected_type},
                        ))

        if violations:
            violation = violations[0]
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


class HITLRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        if ctx.stage.value != "agent" or not ctx.tool_call:
            return self.create_result(ctx)

        required_for = set(self.config.params.get("required_for", []))
        approval_callback = self.config.params.get("approval_callback")

        if ctx.tool_call.name not in required_for:
            return self.create_result(ctx)

        if not approval_callback:
            violation = self._create_violation(
                f"HITL approval required for {ctx.tool_call.name}, but no callback configured",
                metadata={"tool": ctx.tool_call.name},
            )
            return self.create_result(ctx, violation)

        approved = approval_callback(ctx.tool_call.name, ctx.tool_call.arguments)
        if not approved:
            violation = self._create_violation(
                f"HITL approval denied for {ctx.tool_call.name}",
                metadata={"tool": ctx.tool_call.name, "arguments": ctx.tool_call.arguments},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


class LoopCounterRule(BaseRuleWithExecutor):
    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self._step_count = 0
        self._tool_call_count = 0

    def check(self, ctx: ExecutionContext) -> RuleResult:
        max_steps = self.config.params.get("max_steps", 10)
        max_tool_calls = self.config.params.get("max_tool_calls", 20)

        if ctx.tool_call:
            self._tool_call_count += 1
            if self._tool_call_count > max_tool_calls:
                violation = self._create_violation(
                    f"Max tool calls exceeded: {max_tool_calls}",
                    metadata={"tool_calls": self._tool_call_count, "max": max_tool_calls},
                )
                return self.create_result(ctx, violation)

        self._step_count += 1
        if self._step_count > max_steps:
            violation = self._create_violation(
                f"Max steps exceeded: {max_steps}",
                metadata={"steps": self._step_count, "max": max_steps},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)

    def reset(self):
        self._step_count = 0
        self._tool_call_count = 0


def register_agent_rules(registry):
    registry.register("tool_allowlist", ToolAllowlistRule, stages=["agent"])
    registry.register("tool_schema", ToolSchemaRule, stages=["agent"])
    registry.register("hitl", HITLRule, stages=["agent"])
    registry.register("loop_counter", LoopCounterRule, stages=["agent"])