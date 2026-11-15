
from ai_guardrails_sdk.core.context import ExecutionContext
from ai_guardrails_sdk.rules.actions import BaseRuleWithExecutor
from ai_guardrails_sdk.rules.base import RuleResult


class DocumentRBACRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        if ctx.stage.value != "rag":
            return self.create_result(ctx)

        user_roles = ctx.request.metadata.get("roles", [])
        required_roles = self.config.params.get("required_roles", [])

        if required_roles and not any(role in user_roles for role in required_roles):
            violation = self._create_violation(
                f"Access denied: required roles {required_roles}",
                metadata={"required_roles": required_roles, "user_roles": user_roles},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


class IndirectInjectionRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        if ctx.stage.value != "rag":
            return self.create_result(ctx)

        INJECTION_PATTERNS = [
            r"ignore (previous|above|all) instructions",
            r"disregard (previous|above|all) (instructions|prompts)",
            r"you are now (a|an) ",
            r"pretend (to be|you are)",
            r"system prompt",
            r"<\|.*\|>",
        ]

        for chunk in ctx.rag_chunks:
            chunk_lower = chunk.lower()
            for pattern in INJECTION_PATTERNS:
                import re
                if re.search(pattern, chunk_lower):
                    violation = self._create_violation(
                        f"Indirect injection in retrieved chunk: {pattern}",
                        metadata={"pattern": pattern, "chunk_preview": chunk[:100]},
                    )
                    return self.create_result(ctx, violation)

        return self.create_result(ctx)


class TokenTruncationRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        if ctx.stage.value != "rag":
            return self.create_result(ctx)

        max_tokens = self.config.params.get("max_tokens", 4000)
        total_tokens = sum(len(c.split()) for c in ctx.rag_chunks) * 1.3

        if total_tokens > max_tokens:
            violation = self._create_violation(
                f"Context exceeds token limit: {int(total_tokens)} > {max_tokens}",
                metadata={"estimated_tokens": int(total_tokens), "max_tokens": max_tokens},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


def register_rag_rules(registry):
    registry.register("rbac", DocumentRBACRule, stages=["rag"])
    registry.register("indirect_injection", IndirectInjectionRule, stages=["rag"])
    registry.register("token_truncation", TokenTruncationRule, stages=["rag"])