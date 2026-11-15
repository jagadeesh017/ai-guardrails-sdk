import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Stage(StrEnum):
    INPUT = "input"
    RAG = "rag"
    AGENT = "agent"
    OUTPUT = "output"
    OPS = "ops"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class RequestState:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value


@dataclass
class UserState:
    user_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value


@dataclass
class ExecutionContext:
    stage: Stage
    request: RequestState
    user: UserState | None = None
    text: str | None = None
    tool_call: ToolCall | None = None
    rag_chunks: list[str] = field(default_factory=list)
    context: str | None = None
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_input(cls, text: str, user_id: str = None, trace_id: str = None, **kwargs) -> "ExecutionContext":
        request = RequestState(trace_id=trace_id or str(uuid.uuid4()), user_id=user_id, **kwargs)
        user = UserState(user_id=user_id) if user_id else None
        return cls(stage=Stage.INPUT, request=request, user=user, text=text)

    @classmethod
    def create_rag(cls, chunks: list[str], text: str = None, user_id: str = None, trace_id: str = None, context: str = None, **kwargs) -> "ExecutionContext":
        request = RequestState(trace_id=trace_id or str(uuid.uuid4()), user_id=user_id, **kwargs)
        user = UserState(user_id=user_id) if user_id else None
        return cls(stage=Stage.RAG, request=request, user=user, text=text, rag_chunks=chunks, context=context)

    @classmethod
    def create_agent(cls, tool_call: ToolCall, user_id: str = None, trace_id: str = None, **kwargs) -> "ExecutionContext":
        request = RequestState(trace_id=trace_id or str(uuid.uuid4()), user_id=user_id, **kwargs)
        user = UserState(user_id=user_id) if user_id else None
        return cls(stage=Stage.AGENT, request=request, user=user, tool_call=tool_call)

    @classmethod
    def create_output(cls, response: str, user_id: str = None, trace_id: str = None, context: str = None, **kwargs) -> "ExecutionContext":
        request = RequestState(trace_id=trace_id or str(uuid.uuid4()), user_id=user_id, **kwargs)
        user = UserState(user_id=user_id) if user_id else None
        return cls(stage=Stage.OUTPUT, request=request, user=user, response=response, context=context)

    @classmethod
    def create_ops(cls, user_id: str = None, trace_id: str = None, **kwargs) -> "ExecutionContext":
        request = RequestState(trace_id=trace_id or str(uuid.uuid4()), user_id=user_id, **kwargs)
        user = UserState(user_id=user_id) if user_id else None
        return cls(stage=Stage.OPS, request=request, user=user)

    def with_stage(self, stage: Stage) -> "ExecutionContext":
        return ExecutionContext(
            stage=stage,
            request=self.request,
            user=self.user,
            text=self.text,
            tool_call=self.tool_call,
            rag_chunks=self.rag_chunks,
            context=self.context,
            response=self.response,
            metadata=self.metadata.copy(),
        )

    def with_text(self, text: str) -> "ExecutionContext":
        return ExecutionContext(
            stage=self.stage,
            request=self.request,
            user=self.user,
            text=text,
            tool_call=self.tool_call,
            rag_chunks=self.rag_chunks,
            context=self.context,
            response=self.response,
            metadata=self.metadata.copy(),
        )

    def with_response(self, response: str) -> "ExecutionContext":
        return ExecutionContext(
            stage=self.stage,
            request=self.request,
            user=self.user,
            text=self.text,
            tool_call=self.tool_call,
            rag_chunks=self.rag_chunks,
            context=self.context,
            response=response,
            metadata=self.metadata.copy(),
        )