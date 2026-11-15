"""
AI Guardrails SDK - A reusable, framework-independent Python SDK for AI guardrails.
"""

from .config import PolicyLoader
from .core import (
    Action,
    ActionType,
    Decision,
    DecisionStatus,
    EngineConfig,
    ExecutionContext,
    ExecutionMode,
    FailMode,
    Policy,
    RequestState,
    RuleConfig,
    Stage,
    StageConfig,
    ToolCall,
    UserState,
    Violation,
)
from .core.registry import evaluator_registry, rule_registry
from .engine import GuardrailPipeline
from .rules.builtin import register_all_rules

# Register all built-in rules on import
register_all_rules(rule_registry)

__version__ = "0.2.0"

__all__ = [
    # Core types
    "ExecutionContext",
    "RequestState",
    "UserState",
    "ToolCall",
    "Stage",
    "Decision",
    "DecisionStatus",
    "Violation",
    "Action",
    "ActionType",
    "Policy",
    "RuleConfig",
    "StageConfig",
    "EngineConfig",
    "FailMode",
    "ExecutionMode",
    # Config
    "PolicyLoader",
    # Engine
    "GuardrailPipeline",
    # Registry
    "rule_registry",
    "evaluator_registry",
]