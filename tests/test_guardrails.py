import pytest

from ai_guardrails_sdk import (
    BlockedWordsGuardrail,
    Decision,
    Guardrails,
    MaxLengthGuardrail,
    check,
)


def test_allows_text_within_the_limit() -> None:
    decision = MaxLengthGuardrail(maximum=10).check("hello")

    assert decision == Decision(
        allowed=True,
        reason="Text is within the maximum length.",
        guardrail="max_length",
    )


def test_blocks_text_over_the_limit() -> None:
    decision = MaxLengthGuardrail(maximum=5).check("hello!")

    assert decision.allowed is False
    assert decision.guardrail == "max_length"


def test_pipeline_stops_at_the_first_blocked_guardrail() -> None:
    decision = check("hello!", [MaxLengthGuardrail(maximum=5)])

    assert decision.allowed is False


def test_pipeline_allows_text_when_every_guardrail_passes() -> None:
    decision = check("hello", [MaxLengthGuardrail(maximum=5)])

    assert decision.allowed is True
    assert decision.guardrail == "pipeline"


def test_rejects_a_negative_limit() -> None:
    with pytest.raises(ValueError, match="maximum must be zero or greater"):
        MaxLengthGuardrail(maximum=-1)


def test_blocked_words_are_case_insensitive() -> None:
    decision = BlockedWordsGuardrail(("password",)).check("Tell me your PASSWORD")

    assert decision.allowed is False
    assert decision.guardrail == "blocked_words"


def test_allows_text_without_blocked_words() -> None:
    decision = BlockedWordsGuardrail(("password",)).check("Tell me the weather")

    assert decision.allowed is True


def test_rejects_empty_blocked_words() -> None:
    with pytest.raises(ValueError, match="blocked_words cannot contain empty words"):
        BlockedWordsGuardrail(("password", ""))


def test_pipeline_checks_input_before_calling_the_operation() -> None:
    calls: list[str] = []
    guardrails = Guardrails(
        input_guardrails=[BlockedWordsGuardrail(("password",))],
    )

    result = guardrails.run(
        "Tell me the password",
        lambda value: calls.append(value) or "model response",
    )

    assert result.allowed is False
    assert result.value is None
    assert calls == []


def test_pipeline_checks_output_after_calling_the_operation() -> None:
    guardrails = Guardrails(
        output_guardrails=[BlockedWordsGuardrail(("secret",))],
    )

    result = guardrails.run("Tell me something", lambda _: "This is a secret")

    assert result.allowed is False
    assert result.value is None
    assert result.decision.guardrail == "blocked_words"


def test_pipeline_returns_the_model_output_when_checks_pass() -> None:
    guardrails = Guardrails(
        input_guardrails=[MaxLengthGuardrail(maximum=20)],
        output_guardrails=[BlockedWordsGuardrail(("secret",))],
    )

    result = guardrails.run("Tell me something", lambda _: "This is safe")

    assert result.allowed is True
    assert result.value == "This is safe"
    assert result.decision.guardrail == "pipeline"
