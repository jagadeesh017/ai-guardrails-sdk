import json
import threading
import time
from collections import defaultdict

from ai_guardrails_sdk.core.context import ExecutionContext
from ai_guardrails_sdk.core.policy import RuleConfig
from ai_guardrails_sdk.rules.actions import BaseRuleWithExecutor
from ai_guardrails_sdk.rules.base import RuleResult


class RateLimiterRule(BaseRuleWithExecutor):
    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.rpm = config.params.get("requests_per_minute", 60)
        self.rph = config.params.get("requests_per_hour", 1000)
        self._minute_counts = defaultdict(list)
        self._hour_counts = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, ctx: ExecutionContext) -> RuleResult:
        user_id = ctx.request.user_id or "anonymous"

        with self._lock:
            now = time.time()
            self._cleanup_old(now, user_id)

            minute_count = len(self._minute_counts[user_id])
            hour_count = len(self._hour_counts[user_id])

            if minute_count >= self.rpm:
                violation = self._create_violation(
                    f"Rate limit exceeded: {self.rpm} req/min",
                    metadata={"user_id": user_id, "current": minute_count, "limit": self.rpm},
                )
                return self.create_result(ctx, violation)

            if hour_count >= self.rph:
                violation = self._create_violation(
                    f"Rate limit exceeded: {self.rph} req/hour",
                    metadata={"user_id": user_id, "current": hour_count, "limit": self.rph},
                )
                return self.create_result(ctx, violation)

            self._minute_counts[user_id].append(now)
            self._hour_counts[user_id].append(now)

        return self.create_result(ctx)

    def _cleanup_old(self, now: float, user_id: str):
        self._minute_counts[user_id] = [t for t in self._minute_counts[user_id] if now - t < 60]
        self._hour_counts[user_id] = [t for t in self._hour_counts[user_id] if now - t < 3600]


class BudgetGuardrail(BaseRuleWithExecutor):
    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.daily_token_limit = config.params.get("daily_token_limit", 100000)
        self.daily_cost_limit_usd = config.params.get("daily_cost_limit_usd", 50.0)
        self.cost_per_1k = config.params.get("cost_per_1k_tokens", 0.01)
        self._tokens_used = 0
        self._cost_used = 0.0
        self._last_reset = time.time()
        self._lock = threading.Lock()

    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.text or ctx.response or ""
        try:
            data = json.loads(text)
            tokens = data.get("tokens", len(text) // 4)
        except json.JSONDecodeError:
            tokens = len(text) // 4

        cost = (tokens / 1000) * self.cost_per_1k

        with self._lock:
            self._maybe_reset()
            if self._tokens_used + tokens > self.daily_token_limit:
                violation = self._create_violation(
                    f"Daily token budget exceeded: {self._tokens_used + tokens}/{self.daily_token_limit}",
                    metadata={"tokens_used": self._tokens_used + tokens, "limit": self.daily_token_limit},
                )
                return self.create_result(ctx, violation)

            if self._cost_used + cost > self.daily_cost_limit_usd:
                violation = self._create_violation(
                    f"Daily cost budget exceeded: ${self._cost_used + cost:.2f}/${self.daily_cost_limit_usd}",
                    metadata={"cost_used": self._cost_used + cost, "limit": self.daily_cost_limit_usd},
                )
                return self.create_result(ctx, violation)

            self._tokens_used += tokens
            self._cost_used += cost

        return self.create_result(ctx)

    def _maybe_reset(self):
        if time.time() - self._last_reset > 86400:
            self._tokens_used = 0
            self._cost_used = 0.0
            self._last_reset = time.time()

    def reset(self):
        with self._lock:
            self._tokens_used = 0
            self._cost_used = 0.0
            self._last_reset = time.time()


class AuditLoggerRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        log_level = self.config.params.get("log_level", "info")
        include_payloads = self.config.params.get("include_payloads", False)

        log_entry = {
            "timestamp": time.time(),
            "trace_id": ctx.request.trace_id,
            "user_id": ctx.request.user_id,
            "tenant_id": ctx.request.tenant_id,
            "stage": ctx.stage.value,
            "action": "guardrail_check",
        }

        if include_payloads:
            log_entry["text"] = ctx.text or ctx.response

        import logging
        logger = logging.getLogger("ai_guardrails.audit")
        if log_level == "debug":
            logger.debug(f"Audit: {log_entry}")
        elif log_level == "info":
            logger.info(f"Audit: trace={ctx.request.trace_id} stage={ctx.stage.value} user={ctx.request.user_id}")

        return self.create_result(ctx)


def register_ops_rules(registry):
    registry.register("rate_limiter", RateLimiterRule, stages=["ops"])
    registry.register("budget", BudgetGuardrail, stages=["ops"])
    registry.register("audit_logger", AuditLoggerRule, stages=["ops"])