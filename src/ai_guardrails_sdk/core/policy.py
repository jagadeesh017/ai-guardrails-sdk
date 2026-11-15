from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionType(StrEnum):
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    REQUIRE_APPROVAL = "require_approval"
    THROTTLE = "throttle"
    LOG = "log"
    RETRY = "retry"


class FailMode(StrEnum):
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"


class ExecutionMode(StrEnum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class RuleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    enabled: bool = True
    action: ActionType = ActionType.BLOCK
    fail_mode: FailMode = FailMode.FAIL_OPEN
    timeout_ms: int = 5000
    params: dict[str, Any] = Field(default_factory=dict)


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    fail_mode: FailMode = FailMode.FAIL_OPEN
    timeout_ms: int = 10000
    rules: list[RuleConfig] = Field(default_factory=list)


class Policy(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str = "1.0"
    name: str = "default"
    description: str = ""
    fail_mode: FailMode = FailMode.FAIL_OPEN
    global_timeout_ms: int = 30000
    input: StageConfig = Field(default_factory=StageConfig)
    rag: StageConfig = Field(default_factory=StageConfig)
    agent: StageConfig = Field(default_factory=StageConfig)
    output: StageConfig = Field(default_factory=StageConfig)
    ops: StageConfig = Field(default_factory=StageConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_stage(self, stage_name: str) -> StageConfig | None:
        return getattr(self, stage_name, None)

    def get_rule(self, stage_name: str, rule_type: str) -> RuleConfig | None:
        stage = self.get_stage(stage_name)
        if stage:
            for rule in stage.rules:
                if rule.type == rule_type:
                    return rule
        return None


@dataclass
class EngineConfig:
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    fail_mode: FailMode = FailMode.FAIL_OPEN
    timeout_ms: int = 5000
    max_retries: int = 0
    fail_fast: bool = True