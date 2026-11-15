from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .policy import RuleConfig


class RuleRegistry:
    def __init__(self):
        self._rules: dict[str, Callable[..., Any]] = {}
        self._stage_rules: dict[str, set[str]] = defaultdict(set)

    def register(self, rule_type: str, factory: Callable[..., Any], stages: list[str] = None) -> None:
        if rule_type in self._rules:
            raise ValueError(f"Rule type '{rule_type}' already registered")
        self._rules[rule_type] = factory
        if stages:
            for stage in stages:
                self._stage_rules[stage].add(rule_type)

    def unregister(self, rule_type: str) -> None:
        self._rules.pop(rule_type, None)
        for stage_rules in self._stage_rules.values():
            stage_rules.discard(rule_type)

    def get(self, rule_type: str) -> Callable[..., Any] | None:
        return self._rules.get(rule_type)

    def get_stage_rules(self, stage: str) -> list[str]:
        return list(self._stage_rules.get(stage, set()))

    def create_rule(self, rule_config: RuleConfig, **extra_kwargs) -> Any:
        factory = self.get(rule_config.type)
        if not factory:
            raise ValueError(f"Unknown rule type: {rule_config.type}")
        params = rule_config.params.copy()
        params.update(extra_kwargs)
        return factory(**params)

    def list_rules(self) -> list[str]:
        return list(self._rules.keys())


class EvaluatorRegistry:
    def __init__(self):
        self._evaluators: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        self._evaluators[name] = factory

    def get(self, name: str) -> Callable[..., Any] | None:
        return self._evaluators.get(name)

    def create(self, name: str, **kwargs) -> Any:
        factory = self.get(name)
        if not factory:
            raise ValueError(f"Unknown evaluator: {name}")
        return factory(**kwargs)


rule_registry = RuleRegistry()
evaluator_registry = EvaluatorRegistry()