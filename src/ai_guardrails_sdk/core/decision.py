from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DecisionStatus(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    RETRY = "retry"
    ESCALATE = "escalate"


class ActionType(StrEnum):
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    REQUIRE_APPROVAL = "require_approval"
    THROTTLE = "throttle"
    LOG = "log"
    RETRY = "retry"


@dataclass(frozen=True)
class Violation:
    rule_name: str
    rule_type: str
    message: str
    severity: str = "high"
    metadata: dict[str, Any] = field(default_factory=dict)
    location: str | None = None


@dataclass(frozen=True)
class Action:
    type: ActionType
    reason: str
    rule_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    modified_data: Any = None

    @classmethod
    def block(cls, rule_name: str, reason: str, **metadata) -> "Action":
        return cls(ActionType.BLOCK, reason, rule_name, metadata)

    @classmethod
    def warn(cls, rule_name: str, reason: str, **metadata) -> "Action":
        return cls(ActionType.WARN, reason, rule_name, metadata)

    @classmethod
    def redact(cls, rule_name: str, reason: str, modified_data: Any, **metadata) -> "Action":
        return cls(ActionType.REDACT, reason, rule_name, metadata, modified_data)

    @classmethod
    def require_approval(cls, rule_name: str, reason: str, **metadata) -> "Action":
        return cls(ActionType.REQUIRE_APPROVAL, reason, rule_name, metadata)

    @classmethod
    def throttle(cls, rule_name: str, reason: str, **metadata) -> "Action":
        return cls(ActionType.THROTTLE, reason, rule_name, metadata)

    @classmethod
    def log(cls, rule_name: str, reason: str, **metadata) -> "Action":
        return cls(ActionType.LOG, reason, rule_name, metadata)


@dataclass
class Decision:
    status: DecisionStatus
    violations: list[Violation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    modified_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return self.status in (DecisionStatus.DENY, DecisionStatus.ESCALATE)

    @property
    def is_modified(self) -> bool:
        return self.status == DecisionStatus.MODIFY

    @classmethod
    def allow(cls, modified_data: Any = None, metadata: dict = None) -> "Decision":
        return cls(
            status=DecisionStatus.ALLOW,
            modified_data=modified_data,
            metadata=metadata or {},
        )

    @classmethod
    def deny(cls, violation: Violation, action: Action = None, metadata: dict = None) -> "Decision":
        return cls(
            status=DecisionStatus.DENY,
            violations=[violation],
            actions=[action] if action else [Action.block(violation.rule_name, violation.message)],
            metadata=metadata or {},
        )

    @classmethod
    def modify(cls, violation: Violation, action: Action, modified_data: Any, metadata: dict = None) -> "Decision":
        return cls(
            status=DecisionStatus.MODIFY,
            violations=[violation],
            actions=[action],
            modified_data=modified_data,
            metadata=metadata or {},
        )

    @classmethod
    def retry(cls, violation: Violation, action: Action = None, metadata: dict = None) -> "Decision":
        return cls(
            status=DecisionStatus.RETRY,
            violations=[violation],
            actions=[action] if action else [Action.block(violation.rule_name, violation.message)],
            metadata=metadata or {},
        )

    @classmethod
    def escalate(cls, violation: Violation, action: Action = None, metadata: dict = None) -> "Decision":
        return cls(
            status=DecisionStatus.ESCALATE,
            violations=[violation],
            actions=[action] if action else [Action.block(violation.rule_name, violation.message)],
            metadata=metadata or {},
        )

    def add_violation(self, violation: Violation, action: Action) -> "Decision":
        new_violations = self.violations + [violation]
        new_actions = self.actions + [action]
        new_status = self.status
        if action.type == ActionType.BLOCK:
            new_status = DecisionStatus.DENY
        elif action.type == ActionType.REDACT:
            new_status = DecisionStatus.MODIFY
        return Decision(
            status=new_status,
            violations=new_violations,
            actions=new_actions,
            modified_data=self.modified_data,
            metadata=self.metadata,
        )