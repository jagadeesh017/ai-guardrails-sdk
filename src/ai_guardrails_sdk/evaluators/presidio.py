from ..core.context import ExecutionContext
from ..core.decision import Violation
from .base import EvaluationResult, Evaluator


class PresidioPIIEvaluator(Evaluator[str]):
    def __init__(
        self,
        entities: list[str] = None,
        language: str = "en",
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(threshold, **kwargs)
        self.entities = entities
        self.language = language
        self._analyzer = None

    def _load(self):
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError:
                raise ImportError("presidio-analyzer required: pip install presidio-analyzer")
            self._analyzer = AnalyzerEngine()

    def evaluate(self, text: str, ctx: ExecutionContext) -> EvaluationResult:
        self._load()
        results = self._analyzer.analyze(text=text, entities=self.entities, language=self.language)
        violations = []

        for r in results:
            if r.score >= self.threshold:
                violations.append(Violation(
                    rule_name="presidio_pii",
                    rule_type="model",
                    message=f"{r.entity_type} detected (confidence: {r.score:.2f})",
                    metadata={
                        "entity_type": r.entity_type,
                        "confidence": r.score,
                        "start": r.start,
                        "end": r.end,
                        "text": text[r.start:r.end],
                    },
                ))

        return EvaluationResult(violations=violations)

    def get_labels(self) -> list[str]:
        return self.entities or ["PERSON", "EMAIL", "PHONE_NUMBER", "CREDIT_CARD", "API_KEY"]