import json
import re

from ai_guardrails_sdk.core.context import ExecutionContext
from ai_guardrails_sdk.core.decision import ActionType
from ai_guardrails_sdk.core.policy import RuleConfig
from ai_guardrails_sdk.rules.actions import BaseRuleWithExecutor
from ai_guardrails_sdk.rules.base import RuleResult


class JSONSchemaRule(BaseRuleWithExecutor):
    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self._retry_count = 0

    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.response or ""
        schema = self.config.params.get("schema")
        max_retries = self.config.params.get("max_retries", 3)

        if not schema:
            return self.create_result(ctx)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            self._retry_count += 1
            if self._retry_count <= max_retries and self.config.action == ActionType.RETRY:
                violation = self._create_violation(
                    f"Invalid JSON: {e}",
                    metadata={"error": str(e), "retry": self._retry_count},
                )
                return self.create_result(ctx, violation)
            violation = self._create_violation(
                f"Invalid JSON after {self._retry_count} retries: {e}",
                metadata={"error": str(e)},
            )
            return self.create_result(ctx, violation)

        errors = self._validate(data, schema)
        if errors:
            self._retry_count += 1
            if self._retry_count <= max_retries and self.config.action == ActionType.RETRY:
                violation = self._create_violation(
                    f"Schema validation failed: {errors}",
                    metadata={"errors": errors, "retry": self._retry_count},
                )
                return self.create_result(ctx, violation)
            violation = self._create_violation(
                f"Schema validation failed after {self._retry_count} retries: {errors}",
                metadata={"errors": errors},
            )
            return self.create_result(ctx, violation)

        self._retry_count = 0
        return self.create_result(ctx)

    def _validate(self, data: dict, schema: dict, path: str = "") -> list[str]:
        errors = []
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"{path}{field}: required field missing")

        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                prop_schema = properties[key]
                expected_type = prop_schema.get("type")
                if expected_type:
                    type_map = {"string": str, "number": (int, float), "integer": int,
                                "boolean": bool, "array": list, "object": dict}
                    expected = type_map.get(expected_type)
                    if expected and not isinstance(value, expected):
                        errors.append(f"{path}{key}: expected {expected_type}, got {type(value).__name__}")
                if expected_type == "object" and isinstance(value, dict):
                    errors.extend(self._validate(value, prop_schema, f"{path}{key}."))
        return errors

    def reset(self):
        self._retry_count = 0


class HallucinationRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.response or ""
        context = ctx.context or self.config.params.get("context")
        threshold = self.config.params.get("threshold", 0.7)

        if not context:
            violation = self._create_violation(
                "No context provided for grounding check",
                severity="low",
            )
            return self.create_result(ctx, violation)

        text_lower = text.lower()
        context_lower = context.lower()
        text_sentences = [s.strip() for s in text_lower.split('.') if s.strip()]

        ungrounded = []
        for sent in text_sentences:
            sent_words = set(sent.split())
            context_words = set(context_lower.split())
            overlap = sent_words & context_words
            score = len(overlap) / len(sent_words) if sent_words else 0
            if score <= threshold:
                ungrounded.append(sent)

        if ungrounded:
            violation = self._create_violation(
                f"Ungrounded sentences: {'; '.join(ungrounded)}",
                metadata={"ungrounded": ungrounded, "threshold": threshold},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "api_key": r"\bsk-[A-Za-z0-9]{32,}\b",
}


class PIILeakRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.response or ""
        entities = self.config.params.get("entities", ["email", "phone", "ssn", "api_key"])

        found = []
        for entity in entities:
            pattern = PII_PATTERNS.get(entity)
            if pattern and re.search(pattern, text):
                found.append(entity)

        if found:
            violation = self._create_violation(
                f"PII leak detected: {', '.join(found)}",
                metadata={"entities": found},
            )
            if self.config.action == ActionType.REDACT:
                modified = self._redact_text(text, found)
                return self.create_result(ctx, violation, modified_data=modified)
            return self.create_result(ctx, violation)

        return self.create_result(ctx)

    def _redact_text(self, text: str, entities: list[str]) -> str:
        result = text
        for entity in entities:
            pattern = PII_PATTERNS.get(entity)
            if pattern:
                result = re.sub(pattern, f"[REDACTED_{entity.upper()}]", result)
        return result


class BrandFilterRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.response or ""
        blocked = [t.lower() for t in self.config.params.get("blocked_terms", [])]
        replacement = self.config.params.get("replacement", "[REDACTED]")

        text_lower = text.lower()
        found = [term for term in blocked if term in text_lower]

        if found:
            violation = self._create_violation(
                f"Brand/competitor term detected: {', '.join(found)}",
                metadata={"terms": found},
            )
            if self.config.action == ActionType.REDACT:
                modified = self._filter_text(text, found, replacement)
                return self.create_result(ctx, violation, modified_data=modified)
            return self.create_result(ctx, violation)

        return self.create_result(ctx)

    def _filter_text(self, text: str, terms: list[str], replacement: str) -> str:
        result = text
        for term in terms:
            result = re.sub(re.escape(term), replacement, result, flags=re.IGNORECASE)
        return result


def register_output_rules(registry):
    registry.register("json_schema", JSONSchemaRule, stages=["output"])
    registry.register("hallucination", HallucinationRule, stages=["output"])
    registry.register("pii_leak", PIILeakRule, stages=["output"])
    registry.register("brand_filter", BrandFilterRule, stages=["output"])