from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from ..core.context import ExecutionContext
from ..core.decision import Violation

T = TypeVar("T")


@dataclass
class EvaluationResult:
    violations: list[Violation]
    metadata: dict[str, Any] = None


class Evaluator[T](ABC):
    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold
        self.config = kwargs

    @abstractmethod
    def evaluate(self, input_data: T, ctx: ExecutionContext) -> EvaluationResult:
        pass

    async def evaluate_async(self, input_data: T, ctx: ExecutionContext) -> EvaluationResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, input_data, ctx)

    def get_labels(self) -> list[str]:
        return []