import numpy as np

from ..core.context import ExecutionContext
from ..core.decision import Violation
from .base import EvaluationResult, Evaluator


class ONNXClassifierEvaluator(Evaluator[str]):
    def __init__(
        self,
        model_path: str,
        label_map: dict[int, str],
        threshold: float = 0.5,
        tokenizer_name: str | None = None,
        **kwargs,
    ):
        super().__init__(threshold, **kwargs)
        self.model_path = model_path
        self.label_map = label_map
        self.tokenizer_name = tokenizer_name
        self._session = None
        self._tokenizer = None

    def _load(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("onnxruntime required: pip install onnxruntime")
            self._session = ort.InferenceSession(self.model_path)

        if self.tokenizer_name and self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError("transformers required for tokenizer: pip install transformers")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def evaluate(self, text: str, ctx: ExecutionContext) -> EvaluationResult:
        self._load()
        inputs = self._tokenize(text)
        outputs = self._session.run(None, inputs)
        logits = outputs[0]
        probs = self._softmax(logits)
        violations = []

        for idx, prob in enumerate(probs[0]):
            if prob >= self.threshold:
                label = self.label_map.get(idx, str(idx))
                violations.append(Violation(
                    rule_name="onnx_classifier",
                    rule_type="model",
                    message=f"{label} detected (confidence: {prob:.2f})",
                    metadata={"label": label, "confidence": float(prob), "model": self.model_path},
                ))

        return EvaluationResult(violations=violations)

    def _tokenize(self, text: str):
        if self._tokenizer:
            return self._tokenizer(text, return_tensors="np", padding=True, truncation=True)
        raise NotImplementedError("Tokenizer required for ONNX model. Provide tokenizer_name or implement _tokenize.")

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def get_labels(self) -> list[str]:
        return list(self.label_map.values())