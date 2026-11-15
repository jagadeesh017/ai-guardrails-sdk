
from ai_guardrails_sdk.core.context import ExecutionContext
from ai_guardrails_sdk.core.policy import RuleConfig
from ai_guardrails_sdk.evaluators.base import Evaluator
from ai_guardrails_sdk.rules.actions import BaseRuleWithExecutor
from ai_guardrails_sdk.rules.base import RuleResult


class ModelEvaluatorRule(BaseRuleWithExecutor):
    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.evaluator: Evaluator = None

    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def check(self, ctx: ExecutionContext) -> RuleResult:
        if not self.evaluator:
            violation = self._create_violation(
                "No evaluator configured for model rule",
                severity="low",
            )
            return self.create_result(ctx, violation)

        text = ctx.text or ctx.response
        if not text:
            return self.create_result(ctx)

        try:
            result = self.evaluator.evaluate(text, ctx)
        except Exception as e:
            violation = self._create_violation(
                f"Evaluator error: {e}",
                severity="high",
            )
            return self.create_result(ctx, violation)

        if not result.violations:
            return self.create_result(ctx)

        violation = result.violations[0]
        return self.create_result(ctx, violation)


def register_model_rules(registry):
    registry.register("model_evaluator", ModelEvaluatorRule, stages=["input", "output", "rag", "agent"])