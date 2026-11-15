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


class ToolAllowlistGuardrail(BaseGuardrail):
    def __init__(
        self,
        allowed_tools: list[str] = None,
        denied_tools: list[str] = None,
        action: str = "block",
    ):
        super().__init__(action)
        self.allowed_tools = set(allowed_tools or [])
        self.denied_tools = set(denied_tools or [])

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            tool_name = data.get("name") or data.get("tool")
            if tool_name:
                if self.denied_tools and tool_name in self.denied_tools:
                    return self._make_decision(False, f"Denied tool: {tool_name}")
                if self.allowed_tools and tool_name not in self.allowed_tools:
                    return self._make_decision(False, f"Tool not in allowlist: {tool_name}")
        except json.JSONDecodeError:
            pass
        return self._make_decision(True, "Tool allowlist check passed")


class ToolSchemaGuardrail(BaseGuardrail):
    def __init__(self, schemas: dict[str, dict] = None, action: str = "block"):
        super().__init__(action)
        self.schemas = schemas or {}

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            tool_name = data.get("name") or data.get("tool")
            arguments = data.get("arguments") or data.get("args") or {}
            if tool_name and tool_name in self.schemas:
                schema = self.schemas[tool_name]
                required = schema.get("required", [])
                properties = schema.get("properties", {})
                for req in required:
                    if req not in arguments:
                        return self._make_decision(False, f"Missing required arg: {req}")
                for key, value in arguments.items():
                    if key in properties:
                        expected_type = properties[key].get("type")
                        if expected_type and not self._check_type(value, expected_type):
                            return self._make_decision(False, f"Type mismatch for {key}: expected {expected_type}")
        except json.JSONDecodeError:
            pass
        return self._make_decision(True, "Tool schema validation passed")

    def _check_type(self, value, expected_type: str) -> bool:
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}
        expected = type_map.get(expected_type)
        return expected is None or isinstance(value, expected)


class HITLGuardrail(BaseGuardrail):
    def __init__(
        self,
        required_for: list[str] = None,
        approvers: list[str] = None,
        approval_callback=None,
        action: str = "require_approval",
    ):
        super().__init__(action)
        self.required_for = set(required_for or [])
        self.approvers = set(approvers or [])
        self.approval_callback = approval_callback

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            tool_name = data.get("name") or data.get("tool")
            if tool_name and tool_name in self.required_for:
                if self.approval_callback:
                    approved = self.approval_callback(tool_name, data.get("arguments", {}))
                    if not approved:
                        return self._make_decision(False, f"HITL approval required for: {tool_name}")
        except json.JSONDecodeError:
            pass
        return self._make_decision(True, "HITL check passed")


class LoopCounterGuardrail(BaseGuardrail):
    def __init__(self, max_steps: int = 10, max_tool_calls: int = 20, action: str = "block"):
        super().__init__(action)
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.step_count = 0
        self.tool_call_count = 0

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            if "tool" in data or "name" in data:
                self.tool_call_count += 1
                if self.tool_call_count > self.max_tool_calls:
                    return self._make_decision(False, f"Max tool calls exceeded: {self.max_tool_calls}")
        except json.JSONDecodeError:
            pass
        self.step_count += 1
        if self.step_count > self.max_steps:
            return self._make_decision(False, f"Max steps exceeded: {self.max_steps}")
        return self._make_decision(True, "Loop counter check passed")

    def reset(self):
        self.step_count = 0
        self.tool_call_count = 0


class JSONSchemaGuardrail(BaseGuardrail):
    def __init__(self, schema: dict = None, action: str = "retry", max_retries: int = 3):
        super().__init__(action)
        self.schema = schema
        self.max_retries = max_retries
        self.retry_count = 0

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            if self.schema:
                errors = self._validate(data, self.schema)
                if errors:
                    self.retry_count += 1
                    if self.action == "retry" and self.retry_count <= self.max_retries:
                        return self._make_decision(False, f"Schema validation failed: {errors}", modified_text=None)
                    return self._make_decision(False, f"Schema validation failed after {self.retry_count} retries: {errors}")
        except json.JSONDecodeError as e:
            self.retry_count += 1
            if self.action == "retry" and self.retry_count <= self.max_retries:
                return self._make_decision(False, f"Invalid JSON: {e}", modified_text=None)
            return self._make_decision(False, f"Invalid JSON after {self.retry_count} retries: {e}")
        return self._make_decision(True, "JSON schema validation passed")

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
                    type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}
                    expected = type_map.get(expected_type)
                    if expected and not isinstance(value, expected):
                        errors.append(f"{path}{key}: expected {expected_type}, got {type(value).__name__}")
                if expected_type == "object" and isinstance(value, dict):
                    errors.extend(self._validate(value, prop_schema, f"{path}{key}."))
        return errors

    def reset(self):
        self.retry_count = 0


class HallucinationGuardrail(BaseGuardrail):
    def __init__(self, context: str = None, threshold: float = 0.7, action: str = "warn"):
        super().__init__(action)
        self.context = context
        self.threshold = threshold

    def check(self, text: str) -> Decision:
        if not self.context:
            return self._make_decision(True, "No context provided for grounding check")
        text_lower = text.lower()
        context_lower = self.context.lower()
        text_sentences = [s.strip() for s in text_lower.split('.') if s.strip()]
        ungrounded = []
        for sent in text_sentences:
            sent_words = set(sent.split())
            context_words = set(context_lower.split())
            overlap = sent_words & context_words
            score = len(overlap) / len(sent_words) if sent_words else 0
            if score <= self.threshold:
                ungrounded.append(sent)
        if ungrounded:
            return self._make_decision(False, f"Ungrounded sentences: {'; '.join(ungrounded)}")
        return self._make_decision(True, "All sentences grounded")


class PIILeakGuardrail(BaseGuardrail):
    def __init__(self, entities: list[str] = None, action: str = "redact"):
        super().__init__(action)
        self.entities = entities or ["email", "phone", "ssn", "api_key"]

    def check(self, text: str) -> Decision:
        import re
        found = []
        for entity in self.entities:
            pattern = PII_PATTERNS.get(entity)
            if pattern and re.search(pattern, text):
                found.append(entity)
        if found:
            reason = f"PII leak detected: {', '.join(found)}"
            modified = self._redact(text) if self.action == "redact" else None
            return self._make_decision(False, reason, modified)
        return self._make_decision(True, "No PII leak detected")

    def _redact(self, text: str) -> str:
        import re
        result = text
        for entity in self.entities:
            pattern = PII_PATTERNS.get(entity)
            if pattern:
                result = re.sub(pattern, f"[REDACTED_{entity.upper()}]", result)
        return result


class BrandFilterGuardrail(BaseGuardrail):
    def __init__(self, blocked_terms: list[str] = None, replacement: str = "[REDACTED]", action: str = "redact"):
        super().__init__(action)
        self.blocked_terms = [t.lower() for t in (blocked_terms or [])]
        self.replacement = replacement

    def check(self, text: str) -> Decision:
        text_lower = text.lower()
        found = []
        for term in self.blocked_terms:
            if term in text_lower:
                found.append(term)
        if found:
            reason = f"Brand/competitor term detected: {', '.join(found)}"
            modified = self._filter(text) if self.action == "redact" else None
            return self._make_decision(False, reason, modified)
        return self._make_decision(True, "No brand terms detected")

    def _filter(self, text: str) -> str:
        import re
        result = text
        for term in self.blocked_terms:
            result = re.sub(re.escape(term), self.replacement, result, flags=re.IGNORECASE)
        return result


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
    "tool_allowlist": ToolAllowlistGuardrail,
    "tool_schema": ToolSchemaGuardrail,
    "hitl": HITLGuardrail,
    "loop_counter": LoopCounterGuardrail,
    "json_schema": JSONSchemaGuardrail,
    "hallucination": HallucinationGuardrail,
    "pii_leak": PIILeakGuardrail,
    "brand_filter": BrandFilterGuardrail,
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