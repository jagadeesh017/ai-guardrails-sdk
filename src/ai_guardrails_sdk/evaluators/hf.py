from ..core.context import ExecutionContext
from ..core.decision import Violation
from .base import EvaluationResult, Evaluator


class HFClassifierEvaluator(Evaluator[str]):
    def __init__(
        self,
        model_name: str,
        label_map: dict[int, str] = None,
        threshold: float = 0.5,
        target_labels: list[str] = None,
        **kwargs,
    ):
        super().__init__(threshold, **kwargs)
        self.model_name = model_name
        self.label_map = label_map or {}
        self.target_labels = set(target_labels or [])
        self._pipeline = None
        self._kwargs = kwargs

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("transformers required: pip install transformers torch")
            self._pipeline = pipeline("text-classification", model=self.model_name, **self._kwargs)

    def evaluate(self, text: str, ctx: ExecutionContext) -> EvaluationResult:
        self._load()
        results = self._pipeline(text, top_k=None)
        violations = []

        for r in results:
            label = self.label_map.get(r["label"], r["label"])
            if (not self.target_labels or label in self.target_labels) and r["score"] >= self.threshold:
                violations.append(Violation(
                    rule_name="hf_classifier",
                    rule_type="model",
                    message=f"{label} detected (confidence: {r['score']:.2f})",
                    metadata={"label": label, "confidence": r["score"], "model": self.model_name},
                ))

        return EvaluationResult(violations=violations)

    def get_labels(self) -> list[str]:
        if self.target_labels:
            return list(self.target_labels)
        if self.label_map:
            return list(self.label_map.values())
        return []


class HFZeroShotEvaluator(Evaluator[str]):
    def __init__(
        self,
        model_name: str,
        candidate_labels: list[str],
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(threshold, **kwargs)
        self.model_name = model_name
        self.candidate_labels = candidate_labels
        self._pipeline = None
        self._kwargs = kwargs

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("transformers required: pip install transformers torch")
            self._pipeline = pipeline("zero-shot-classification", model=self.model_name, **self._kwargs)

    def evaluate(self, text: str, ctx: ExecutionContext) -> EvaluationResult:
        self._load()
        results = self._pipeline(text, self.candidate_labels)
        violations = []

        for label, score in zip(results["labels"], results["scores"], strict=False):
            if score >= self.threshold:
                violations.append(Violation(
                    rule_name="hf_zero_shot",
                    rule_type="model",
                    message=f"{label} detected (confidence: {score:.2f})",
                    metadata={"label": label, "confidence": score, "model": self.model_name},
                ))

        return EvaluationResult(violations=violations)

    def get_labels(self) -> list[str]:
        return self.candidate_labels