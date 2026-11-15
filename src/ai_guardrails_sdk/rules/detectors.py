import re

from ..core.context import ExecutionContext
from ..core.decision import Violation
from .base import Detector


class TextDetector(Detector):
    def __init__(self, patterns: dict[str, str] = None, keywords: dict[str, list[str]] = None):
        self.patterns = patterns or {}
        self.keywords = keywords or {}

    def detect(self, ctx: ExecutionContext) -> list[Violation]:
        if not ctx.text:
            return []

        violations = []
        text_lower = ctx.text.lower()

        for name, pattern in self.patterns.items():
            if re.search(pattern, ctx.text, re.IGNORECASE):
                violations.append(Violation(
                    rule_name="text_detector",
                    rule_type="pattern",
                    message=f"Pattern '{name}' matched",
                    metadata={"pattern": pattern, "matched_text": ctx.text[:100]},
                ))

        for topic, words in self.keywords.items():
            for word in words:
                if word.lower() in text_lower:
                    violations.append(Violation(
                        rule_name="text_detector",
                        rule_type="keyword",
                        message=f"Keyword '{word}' in topic '{topic}' matched",
                        metadata={"topic": topic, "keyword": word},
                    ))

        return violations


class ToolDetector(Detector):
    def detect(self, ctx: ExecutionContext) -> list[Violation]:
        if not ctx.tool_call:
            return []

        violations = []
        tc = ctx.tool_call

        if not tc.name:
            violations.append(Violation(
                rule_name="tool_detector",
                rule_type="missing_name",
                message="Tool call missing name",
                metadata={"tool_call": tc.arguments},
            ))

        return violations


class ContextDetector(Detector):
    def __init__(self, required_fields: list[str] = None):
        self.required_fields = required_fields or []

    def detect(self, ctx: ExecutionContext) -> list[Violation]:
        violations = []

        if ctx.stage.value == "rag" and not ctx.rag_chunks:
            violations.append(Violation(
                rule_name="context_detector",
                rule_type="missing_chunks",
                message="RAG stage requires retrieved chunks",
            ))

        if ctx.stage.value == "output" and not ctx.response:
            violations.append(Violation(
                rule_name="context_detector",
                rule_type="missing_response",
                message="Output stage requires response",
            ))

        if ctx.stage.value == "output" and not ctx.context:
            violations.append(Violation(
                rule_name="context_detector",
                rule_type="missing_context",
                message="Grounding check requires context",
            ))

        return violations


class JSONDetector(Detector):
    def __init__(self, schema: dict = None):
        self.schema = schema

    def detect(self, ctx: ExecutionContext) -> list[Violation]:
        text = ctx.text or ctx.response
        if not text:
            return []

        import json
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return [Violation(
                rule_name="json_detector",
                rule_type="invalid_json",
                message=f"Invalid JSON: {e}",
                metadata={"error": str(e)},
            )]

        violations = []
        if self.schema:
            violations.extend(self._validate(data, self.schema))
        return violations

    def _validate(self, data: dict, schema: dict, path: str = "") -> list[Violation]:
        violations = []
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                violations.append(Violation(
                    rule_name="json_detector",
                    rule_type="missing_field",
                    message=f"Required field '{path}{field}' missing",
                    metadata={"field": f"{path}{field}"},
                ))

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
                        violations.append(Violation(
                            rule_name="json_detector",
                            rule_type="type_mismatch",
                            message=f"Field '{path}{key}': expected {expected_type}, got {type(value).__name__}",
                            metadata={"field": f"{path}{key}", "expected": expected_type, "got": type(value).__name__},
                        ))
                if expected_type == "object" and isinstance(value, dict):
                    violations.extend(self._validate(value, prop_schema, f"{path}{key}."))
        return violations