import re

from ai_guardrails_sdk.core.context import ExecutionContext
from ai_guardrails_sdk.core.decision import ActionType
from ai_guardrails_sdk.core.policy import RuleConfig
from ai_guardrails_sdk.rules.actions import BaseRuleWithExecutor
from ai_guardrails_sdk.rules.base import RuleResult


class MaxLengthRule(BaseRuleWithExecutor):
    def __init__(self, config: RuleConfig):
        super().__init__(config)
        maximum = config.params.get("maximum", 1000)
        if maximum < 0:
            raise ValueError("maximum must be zero or greater")

    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.text or ""
        maximum = self.config.params.get("maximum", 1000)

        if len(text) > maximum:
            violation = self._create_violation(
                f"Text exceeds maximum length of {maximum} characters",
                metadata={"length": len(text), "maximum": maximum},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


class BlockedWordsRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.text or ctx.response or ""
        words = [w.lower() for w in self.config.params.get("words", [])]
        text_lower = text.lower()

        for word in words:
            if word in text_lower:
                violation = self._create_violation(
                    f"Blocked word detected: {word}",
                    metadata={"word": word},
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


class PIIRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.text or ""
        entities = self.config.params.get("entities", list(PII_PATTERNS.keys()))

        found = []
        for entity in entities:
            pattern = PII_PATTERNS.get(entity)
            if pattern and re.search(pattern, text):
                found.append(entity)

        if found:
            violation = self._create_violation(
                f"PII detected: {', '.join(found)}",
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


TOXIC_PATTERNS = [
    r"\b(hate|stupid|idiot|moron|dumb)\b",
    r"\b(kill|die|death|murder)\b",
]


class ToxicityRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.text or ""
        threshold = self.config.params.get("threshold", 0.5)
        text_lower = text.lower()

        matches = sum(1 for p in TOXIC_PATTERNS if re.search(p, text_lower))
        score = matches / len(TOXIC_PATTERNS)

        if score >= threshold:
            violation = self._create_violation(
                f"Toxic content detected (score: {score:.2f})",
                metadata={"score": score, "threshold": threshold},
            )
            return self.create_result(ctx, violation)

        return self.create_result(ctx)


INJECTION_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"disregard (previous|above|all) (instructions|prompts)",
    r"you are now (a|an) ",
    r"pretend (to be|you are)",
    r"roleplay as",
    r"system prompt",
    r"<\|.*\|>",
    r"\[INST\].*\[/INST\]",
    r"### (instruction|system|user|assistant)",
    r"jailbreak",
    r"bypass (safety|filter|guard)",
    r"override (safety|instructions)",
]


class PromptInjectionRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.text or ""
        text_lower = text.lower()

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                violation = self._create_violation(
                    f"Prompt injection detected: {pattern}",
                    metadata={"pattern": pattern},
                )
                return self.create_result(ctx, violation)

        return self.create_result(ctx)


TOPIC_KEYWORDS = {
    "violence": ["bomb", "weapon", "gun", "knife", "kill", "murder", "attack", "fight", "violence", "hurt", "harm"],
    "illegal": ["illegal", "crime", "steal", "drug", "hack", "fraud", "launder", "blackmail"],
    "self_harm": ["suicide", "kill myself", "hurt myself", "self harm", "cut myself", "end my life"],
    "hate_speech": ["hate", "nazi", "racist", "slur", "supremacy", "genocide"],
    "sexual": ["porn", "sex", "nude", "explicit", "erotic"],
}


class TopicRule(BaseRuleWithExecutor):
    def check(self, ctx: ExecutionContext) -> RuleResult:
        text = ctx.text or ""
        denied_topics = self.config.params.get("denied_topics", [])
        text_lower = text.lower()

        for topic in denied_topics:
            keywords = TOPIC_KEYWORDS.get(topic, [topic])
            for kw in keywords:
                if kw in text_lower:
                    violation = self._create_violation(
                        f"Denied topic detected: {topic} (matched: {kw})",
                        metadata={"topic": topic, "keyword": kw},
                    )
                    return self.create_result(ctx, violation)

        return self.create_result(ctx)


def register_input_rules(registry):
    registry.register("max_length", MaxLengthRule, stages=["input"])
    registry.register("blocked_words", BlockedWordsRule, stages=["input"])
    registry.register("pii", PIIRule, stages=["input"])
    registry.register("toxicity", ToxicityRule, stages=["input"])
    registry.register("prompt_injection", PromptInjectionRule, stages=["input"])
    registry.register("topic", TopicRule, stages=["input"])