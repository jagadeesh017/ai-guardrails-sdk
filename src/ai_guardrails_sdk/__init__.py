import asyncio
import re
from typing import Callable, Awaitable
import yaml


class Decision:
    def __init__(self, allowed: bool, reason: str, action: str = "block", modified_text: str = None):
        self.allowed = allowed
        self.reason = reason
        self.action = action  # "block", "redact", "warn"
        self.modified_text = modified_text

    def __repr__(self):
        return f"Decision(allowed={self.allowed}, action={self.action}, reason='{self.reason}')"


class BaseGuardrail:
    def __init__(self, action: str = "block"):
        self.action = action  # "block", "warn", "redact"

    def check(self, text: str) -> Decision:
        raise NotImplementedError

    def _make_decision(self, allowed: bool, reason: str, modified_text: str = None) -> Decision:
        if not allowed:
            if self.action == "redact" and modified_text:
                return Decision(False, reason, action="redact", modified_text=modified_text)
            elif self.action == "warn":
                return Decision(True, reason, action="warn")
            return Decision(False, reason, action="block")
        return Decision(True, reason, modified_text=modified_text)

    async def check_async(self, text: str) -> Decision:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, text)


class MaxLengthGuardrail(BaseGuardrail):
    def __init__(self, maximum: int, action: str = "block"):
        super().__init__(action)
        if maximum < 0:
            raise ValueError("maximum must be zero or greater")
        self.maximum = maximum

    def check(self, text: str) -> Decision:
        if len(text) <= self.maximum:
            return self._make_decision(True, f"Text is within {self.maximum} characters")
        return self._make_decision(False, f"Text exceeds {self.maximum} characters")


class BlockedWordsGuardrail(BaseGuardrail):
    def __init__(self, words: list[str], action: str = "block"):
        super().__init__(action)
        self.words = [w.lower() for w in words]

    def check(self, text: str) -> Decision:
        text_lower = text.lower()
        for word in self.words:
            if word in text_lower:
                return self._make_decision(False, f"Blocked word detected: {word}")
        return self._make_decision(True, "No blocked words found")


PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "api_key": r"\bsk-[A-Za-z0-9]{32,}\b",
}


class PIIGuardrail(BaseGuardrail):
    def __init__(self, entities: list[str] = None, action: str = "block"):
        super().__init__(action)
        self.entities = entities or list(PII_PATTERNS.keys())

    def check(self, text: str) -> Decision:
        found = []
        for entity in self.entities:
            pattern = PII_PATTERNS.get(entity)
            if pattern and re.search(pattern, text):
                found.append(entity)
        if found:
            reason = f"PII detected: {', '.join(found)}"
            modified = self.redact(text) if self.action == "redact" else None
            return self._make_decision(False, reason, modified)
        return self._make_decision(True, "No PII detected")

    def redact(self, text: str) -> str:
        result = text
        for entity in self.entities:
            pattern = PII_PATTERNS.get(entity)
            if pattern:
                result = re.sub(pattern, f"[REDACTED_{entity.upper()}]", result)
        return result


TOXIC_PATTERNS = [
    r"\b(hate|stupid|idiot|moron|dumb)\b",
    r"\b(kill|die|death|murder)\b",
]


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


class PromptInjectionGuardrail(BaseGuardrail):
    def __init__(self, action: str = "block"):
        super().__init__(action)

    def check(self, text: str) -> Decision:
        text_lower = text.lower()
        for p in INJECTION_PATTERNS:
            if re.search(p, text_lower):
                return self._make_decision(False, f"Prompt injection detected: {p}")
        return self._make_decision(True, "No prompt injection detected")


TOPIC_KEYWORDS = {
    "violence": ["bomb", "weapon", "gun", "knife", "kill", "murder", "attack", "fight", "violence", "hurt", "harm"],
    "illegal": ["illegal", "crime", "steal", "drug", "hack", "fraud", "launder", "blackmail"],
    "self_harm": ["suicide", "kill myself", "hurt myself", "self harm", "cut myself", "end my life"],
    "hate_speech": ["hate", "nazi", "racist", "slur", "supremacy", "genocide"],
    "sexual": ["porn", "sex", "nude", "explicit", "erotic"],
}


class TopicGuardrail(BaseGuardrail):
    def __init__(
        self,
        denied_topics: list[str] = None,
        action: str = "block",
    ):
        super().__init__(action)
        self.denied_topics = [t.lower() for t in (denied_topics or [])]

    def check(self, text: str) -> Decision:
        text_lower = text.lower()

        if self.denied_topics:
            for topic in self.denied_topics:
                keywords = TOPIC_KEYWORDS.get(topic, [topic])
                for kw in keywords:
                    if kw in text_lower:
                        return self._make_decision(False, f"Denied topic detected: {topic} (matched: {kw})")

        return self._make_decision(True, "Topic check passed")


class ToxicityGuardrail(BaseGuardrail):
    def __init__(self, threshold: float = 0.5, action: str = "block"):
        super().__init__(action)
        self.threshold = threshold

    def check(self, text: str) -> Decision:
        text_lower = text.lower()
        matches = sum(1 for p in TOXIC_PATTERNS if re.search(p, text_lower))
        score = matches / len(TOXIC_PATTERNS)
        if score >= self.threshold:
            return self._make_decision(False, f"Toxic content detected (score: {score:.2f})")
        return self._make_decision(True, "No toxicity detected")


async def check_async(text: str, guardrails: list[BaseGuardrail]) -> Decision:
    current_text = text
    for g in guardrails:
        decision = await g.check_async(current_text)
        if not decision.allowed:
            if decision.action == "redact" and decision.modified_text:
                current_text = decision.modified_text
                continue
            elif decision.action == "warn":
                continue
            return decision
        if decision.modified_text:
            current_text = decision.modified_text
    return Decision(True, "All checks passed", modified_text=current_text)


def check(text: str, guardrails: list[BaseGuardrail]) -> Decision:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(check_async(text, guardrails))


class Pipeline:
    def __init__(
        self,
        input_guardrails: list[BaseGuardrail] = None,
        output_guardrails: list[BaseGuardrail] = None,
    ):
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []

    def run(self, prompt: str, llm_fn: Callable[[str], str]) -> Decision:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.run_async(prompt, llm_fn))

    async def run_async(
        self, prompt: str, llm_fn: Callable[[str], Awaitable[str] | str]
    ) -> Decision:
        input_result = await check_async(prompt, self.input_guardrails)
        if not input_result.allowed and input_result.action == "block":
            return Decision(False, f"Input blocked: {input_result.reason}")

        cleaned_prompt = input_result.modified_text or prompt

        if asyncio.iscoroutinefunction(llm_fn):
            response = await llm_fn(cleaned_prompt)
        else:
            response = llm_fn(cleaned_prompt)

        output_result = await check_async(response, self.output_guardrails)
        if not output_result.allowed and output_result.action == "block":
            return Decision(False, f"Output blocked: {output_result.reason}")

        final_response = output_result.modified_text or response
        return Decision(True, f"Success: {final_response}", modified_text=final_response)


GUARDRAIL_REGISTRY = {
    "max_length": MaxLengthGuardrail,
    "blocked_words": BlockedWordsGuardrail,
    "pii": PIIGuardrail,
    "toxicity": ToxicityGuardrail,
    "prompt_injection": PromptInjectionGuardrail,
    "topic": TopicGuardrail,
}


def load_guardrail(config: dict) -> BaseGuardrail:
    gtype = config.pop("type")
    cls = GUARDRAIL_REGISTRY.get(gtype)
    if not cls:
        raise ValueError(f"Unknown guardrail type: {gtype}")
    return cls(**config)


def load_pipeline_from_yaml(path: str) -> Pipeline:
    with open(path) as f:
        data = yaml.safe_load(f)
    pipe_config = data.get("pipeline", {})
    input_guards = [load_guardrail(c) for c in pipe_config.get("input_guardrails", [])]
    output_guards = [load_guardrail(c) for c in pipe_config.get("output_guardrails", [])]
    return Pipeline(input_guards, output_guards)