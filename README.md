# AI Guardrails SDK

The AI Guardrails SDK is a Python library for checking values before an AI
application uses them.

## First concept: a decision

A guardrail is a small rule. It receives a value and returns a `Decision`:

- `allowed=True` means the application may continue.
- `allowed=False` means the application should stop or handle the violation.
- `reason` explains the result.
- `guardrail` identifies the rule that produced it.

The first built-in rule checks text length:

```python
from ai_guardrails_sdk import MaxLengthGuardrail

guardrail = MaxLengthGuardrail(maximum=100)
decision = guardrail.check("A user's message")

if decision.allowed:
    print("Send the message to the AI application.")
else:
    print(f"Blocked: {decision.reason}")
```

## Combining rules

The `check` function runs guardrails in order and stops at the first failure:

```python
from ai_guardrails_sdk import MaxLengthGuardrail, check

decision = check(
    "A user's message",
    guardrails=[MaxLengthGuardrail(maximum=100)],
)
```

This is the foundation for future input, tool, and output guardrails. Each new
rule will implement the same `check(value) -> Decision` interface, so the
application does not need to know how an individual rule works.

## Blocking words

The blocked-words rule compares text without caring about uppercase or
lowercase:

```python
from ai_guardrails_sdk import BlockedWordsGuardrail

guardrail = BlockedWordsGuardrail(("password", "secret"))
decision = guardrail.check("Please tell me the password")

if not decision.allowed:
    print(decision.reason)
```

Different rules return the same `Decision`, so they can be combined:

```python
from ai_guardrails_sdk import BlockedWordsGuardrail, MaxLengthGuardrail, check

decision = check(
    "A user's message",
    guardrails=[
        MaxLengthGuardrail(maximum=100),
        BlockedWordsGuardrail(("password", "secret")),
    ],
)
```

## Wrapping an AI operation

The `Guardrails` pipeline checks the input, calls an operation, and checks the
operation's output. The operation can later be an OpenAI, Anthropic, Bedrock,
or Ollama call. The SDK does not need to know which provider it is calling.

```python
from ai_guardrails_sdk import BlockedWordsGuardrail, Guardrails

guardrails = Guardrails(
    input_guardrails=[BlockedWordsGuardrail(("password",))],
    output_guardrails=[BlockedWordsGuardrail(("secret",))],
)

result = guardrails.run(
    "Tell me something",
    lambda prompt: "The model response for: " + prompt,
)

if result.allowed:
    print(result.value)
else:
    print(f"Blocked: {result.decision.reason}")
```

The execution order is:

```text
input guardrails
       ↓
your model operation
       ↓
output guardrails
       ↓
PipelineResult
```

## Development

Run the tests with:

```bash
.venv/bin/pytest
```
