from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..core.context import RequestState, UserState

_request_state: ContextVar[RequestState | None] = ContextVar("request_state", default=None)
_user_states: dict[str, UserState] = {}
_user_states_lock = Lock()


def get_request_state() -> RequestState | None:
    return _request_state.get()


def set_request_state(state: RequestState) -> None:
    _request_state.set(state)


def clear_request_state() -> None:
    _request_state.set(None)


def get_user_state(user_id: str) -> UserState:
    with _user_states_lock:
        if user_id not in _user_states:
            _user_states[user_id] = UserState(user_id=user_id)
        return _user_states[user_id]


def clear_user_state(user_id: str) -> None:
    with _user_states_lock:
        _user_states.pop(user_id, None)


def clear_all_user_states() -> None:
    with _user_states_lock:
        _user_states.clear()


@dataclass
class PipelineState:
    trace_id: str
    current_stage: str = ""
    stage_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)