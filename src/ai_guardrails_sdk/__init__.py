import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Awaitable, Any
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


# ============================================================
# Model Integration System - Type-safe adapters for ML classifiers
# ============================================================

@dataclass
class ClassificationResult:
    """Standard output from any semantic classifier."""
    label: str
    confidence: float
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModelAdapter(ABC):
    """Abstract base for all ML model adapters. Implement for your model type."""

    @abstractmethod
    def predict(self, text: str) -> list[ClassificationResult]:
        """Sync prediction - override for simple models."""
        pass

    async def predict_async(self, text: str) -> list[ClassificationResult]:
        """Async prediction - runs sync in executor by default."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, text)

    @abstractmethod
    def get_labels(self) -> list[str]:
        """Return all possible labels this model can predict."""
        pass


class HFClassifierAdapter(BaseModelAdapter):
    """Adapter for HuggingFace transformers classifiers (JEV, DeBERTa, etc.)."""

    def __init__(self, model_name: str, label_map: dict[int, str] = None, threshold: float = 0.5, **kwargs):
        self.model_name = model_name
        self.label_map = label_map
        self.threshold = threshold
        self._pipeline = None
        self._kwargs = kwargs

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("text-classification", model=self.model_name, **self._kwargs)

    def predict(self, text: str) -> list[ClassificationResult]:
        self._load()
        results = self._pipeline(text, top_k=None)
        outputs = []
        for r in results:
            label = self.label_map.get(r["label"], r["label"]) if self.label_map else r["label"]
            if r["score"] >= self.threshold:
                outputs.append(ClassificationResult(label=label, confidence=r["score"]))
        return outputs

    async def predict_async(self, text: str) -> list[ClassificationResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, text)

    def get_labels(self) -> list[str]:
        if self.label_map:
            return list(self.label_map.values())
        return []


class PresidioPIIAdapter(BaseModelAdapter):
    """Adapter for Microsoft Presidio PII detection."""

    def __init__(self, entities: list[str] = None, language: str = "en", threshold: float = 0.5):
        self.entities = entities
        self.language = language
        self.threshold = threshold
        self._analyzer = None

    def _load(self):
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()

    def predict(self, text: str) -> list[ClassificationResult]:
        self._load()
        results = self._analyzer.analyze(text=text, entities=self.entities, language=self.language)
        outputs = []
        for r in results:
            if r.score >= self.threshold:
                outputs.append(ClassificationResult(
                    label=r.entity_type,
                    confidence=r.score,
                    metadata={"start": r.start, "end": r.end, "text": text[r.start:r.end]}
                ))
        return outputs

    def get_labels(self) -> list[str]:
        return self.entities or ["PERSON", "EMAIL", "PHONE_NUMBER", "CREDIT_CARD", "API_KEY"]


class ONNXClassifierAdapter(BaseModelAdapter):
    """Adapter for ONNX runtime models (fast, no PyTorch dependency)."""

    def __init__(self, model_path: str, label_map: dict[int, str], threshold: float = 0.5):
        self.model_path = model_path
        self.label_map = label_map
        self.threshold = threshold
        self._session = None

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path)

    def predict(self, text: str) -> list[ClassificationResult]:
        self._load()
        # Tokenize - implement based on your model's tokenizer
        import numpy as np
        inputs = self._tokenize(text)
        outputs = self._session.run(None, inputs)
        logits = outputs[0]
        probs = self._softmax(logits)
        results = []
        for idx, prob in enumerate(probs[0]):
            if prob >= self.threshold:
                label = self.label_map.get(idx, str(idx))
                results.append(ClassificationResult(label=label, confidence=float(prob)))
        return results

    def _tokenize(self, text: str):
        # Implement based on your model (e.g., using transformers tokenizer)
        raise NotImplementedError("Implement _tokenize for your ONNX model")

    def _softmax(self, x):
        import numpy as np
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def get_labels(self) -> list[str]:
        return list(self.label_map.values())


class ModelGuardrail(BaseGuardrail):
    """Generic guardrail that uses any BaseModelAdapter."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        target_labels: list[str] = None,
        action: str = "block",
    ):
        super().__init__(action)
        self.adapter = adapter
        self.target_labels = set(target_labels or [])

    def check(self, text: str) -> Decision:
        results = self.adapter.predict(text)
        for r in results:
            if not self.target_labels or r.label in self.target_labels:
                if r.confidence >= self.adapter.threshold:
                    return self._make_decision(
                        False,
                        f"{r.label} detected (confidence: {r.confidence:.2f})",
                    )
        return self._make_decision(True, "No target labels detected")

    async def check_async(self, text: str) -> Decision:
        results = await self.adapter.predict_async(text)
        for r in results:
            if not self.target_labels or r.label in self.target_labels:
                if r.confidence >= self.adapter.threshold:
                    return self._make_decision(
                        False,
                        f"{r.label} detected (confidence: {r.confidence:.2f})",
                    )
        return self._make_decision(True, "No target labels detected")


# Register model-based guardrails
def register_model_guardrail(name: str, adapter: BaseModelAdapter, target_labels: list[str] = None, action: str = "block"):
    """Factory to create and register a model-based guardrail."""
    guardrail = ModelGuardrail(adapter, target_labels, action)
    GUARDRAIL_REGISTRY[name] = lambda **kwargs: guardrail
    return guardrail


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


import time
from collections import defaultdict


class RateLimiterGuardrail(BaseGuardrail):
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        action: str = "throttle",
    ):
        super().__init__(action)
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.minute_counts = defaultdict(list)
        self.hour_counts = defaultdict(list)

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            user_id = data.get("user_id", "anonymous")
        except json.JSONDecodeError:
            user_id = "anonymous"

        now = time.time()
        self._cleanup_old(now)

        minute_count = len(self.minute_counts[user_id])
        hour_count = len(self.hour_counts[user_id])

        if minute_count >= self.rpm:
            return self._make_decision(False, f"Rate limit exceeded: {self.rpm} req/min")
        if hour_count >= self.rph:
            return self._make_decision(False, f"Rate limit exceeded: {self.rph} req/hour")

        self.minute_counts[user_id].append(now)
        self.hour_counts[user_id].append(now)
        return self._make_decision(True, f"Rate limit OK ({minute_count}/{self.rpm} min, {hour_count}/{self.rph} hour)")

    def _cleanup_old(self, now: float):
        for uid in list(self.minute_counts.keys()):
            self.minute_counts[uid] = [t for t in self.minute_counts[uid] if now - t < 60]
            if not self.minute_counts[uid]:
                del self.minute_counts[uid]
        for uid in list(self.hour_counts.keys()):
            self.hour_counts[uid] = [t for t in self.hour_counts[uid] if now - t < 3600]
            if not self.hour_counts[uid]:
                del self.hour_counts[uid]


class BudgetGuardrail(BaseGuardrail):
    def __init__(
        self,
        daily_token_limit: int = 100000,
        daily_cost_limit_usd: float = 50.0,
        cost_per_1k_tokens: float = 0.01,
        action: str = "block",
    ):
        super().__init__(action)
        self.daily_token_limit = daily_token_limit
        self.daily_cost_limit_usd = daily_cost_limit_usd
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.tokens_used = 0
        self.cost_used = 0.0
        self.last_reset = time.time()

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            tokens = data.get("tokens", len(text) // 4)
            cost = (tokens / 1000) * self.cost_per_1k_tokens
        except json.JSONDecodeError:
            tokens = len(text) // 4
            cost = (tokens / 1000) * self.cost_per_1k_tokens

        self._maybe_reset()

        if self.tokens_used + tokens > self.daily_token_limit:
            return self._make_decision(False, f"Daily token budget exceeded: {self.tokens_used + tokens}/{self.daily_token_limit}")
        if self.cost_used + cost > self.daily_cost_limit_usd:
            return self._make_decision(False, f"Daily cost budget exceeded: ${self.cost_used + cost:.2f}/${self.daily_cost_limit_usd}")

        self.tokens_used += tokens
        self.cost_used += cost
        return self._make_decision(True, f"Budget OK (tokens: {self.tokens_used}/{self.daily_token_limit}, cost: ${self.cost_used:.2f}/${self.daily_cost_limit_usd})")

    def _maybe_reset(self):
        if time.time() - self.last_reset > 86400:
            self.tokens_used = 0
            self.cost_used = 0.0
            self.last_reset = time.time()

    def reset(self):
        self.tokens_used = 0
        self.cost_used = 0.0
        self.last_reset = time.time()


class AuditLoggerGuardrail(BaseGuardrail):
    def __init__(self, log_level: str = "info", include_payloads: bool = False, action: str = "log"):
        super().__init__(action)
        self.log_level = log_level
        self.include_payloads = include_payloads
        self.logs = []

    def check(self, text: str) -> Decision:
        import json
        try:
            data = json.loads(text)
            user_id = data.get("user_id", "anonymous")
            trace_id = data.get("trace_id", "unknown")
        except json.JSONDecodeError:
            user_id = "anonymous"
            trace_id = "unknown"

        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "trace_id": trace_id,
            "action": "guardrail_check",
        }
        if self.include_payloads:
            log_entry["payload"] = text

        self.logs.append(log_entry)

        import logging
        logger = logging.getLogger("ai_guardrails")
        if self.log_level == "info":
            logger.info(f"Audit: user={user_id} trace={trace_id}")
        elif self.log_level == "debug":
            logger.debug(f"Audit: {log_entry}")

        return self._make_decision(True, "Audit logged")


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
    "rate_limiter": RateLimiterGuardrail,
    "budget": BudgetGuardrail,
    "audit_logger": AuditLoggerGuardrail,
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