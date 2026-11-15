# AI Guardrails SDK Architecture Plan

## Goal

Build one Python SDK that sits between an application and any AI provider.
The application sends a request to the SDK. The SDK checks the request, calls
the provider, checks the response, and returns a clear result.

The SDK will not be an AI provider or an agent framework. It will be an
enforcement layer that works with OpenAI, Anthropic, Bedrock, Ollama, or a
plain Python function.

## The central idea: a pipeline

Every protected operation follows this flow:

```text
Application
    |
    v
Guardrails pipeline
    |
    +-- 1. Input checks
    |
    +-- 2. Optional RAG checks
    |
    +-- 3. Optional agent/tool checks
    |
    +-- 4. AI provider call
    |
    +-- 5. Output checks
    |
    +-- 6. Audit event
    |
    v
Application receives a result
```

The application should not need to know how each individual guardrail works.
It only needs to provide the input and the operation that calls the model.

## Main SDK concepts

### 1. Decision

The result of one guardrail check.

```text
Decision
    allowed: yes or no
    reason: why
    guardrail: which rule decided
    action: block, warn, redact, retry, or allow
```

### 2. Guardrail

A small, focused rule that checks one thing.

Examples:

- maximum text length,
- blocked words,
- PII detection,
- tool allowlist,
- output schema validation.

Every guardrail uses the same interface so rules can be combined.

### 3. Stage

The point in the AI request lifecycle where a guardrail runs:

- `input`
- `rag`
- `agent`
- `output`
- `ops`

### 4. Policy

A named collection of configured guardrails.

Example:

```text
strict_customer_support
    input: maximum length + blocked words
    output: PII detection + schema validation
    failure mode: fail closed
```

### 5. Provider operation

A function or adapter that performs the actual AI call.

The SDK will not directly depend on a specific provider. This keeps the core
small and allows provider adapters to be added separately.

## Target package structure

```text
src/ai_guardrails_sdk/
    __init__.py          Public API
    models.py            Decision, pipeline result, request, response
    guardrails.py        Guardrail interface and simple built-in rules
    pipeline.py          Synchronous pipeline
    async_pipeline.py    Asynchronous pipeline
    policy.py            Policy definitions and configuration
    providers/
        base.py           Provider adapter interface
        openai.py         OpenAI adapter
        anthropic.py      Anthropic adapter
    shields/
        input_shield.py   User input checks
        rag_shield.py     Retrieved context checks
        agent_shield.py   Tool and agent action checks
        output_shield.py  Model response checks
        ops_shield.py     Limits, budgets, and audit events
    telemetry.py         Structured audit events
    errors.py             Explicit SDK exceptions
```

We will not create all of these files at once. Each file will be added only
when its behavior is understood and tested.

## Five shield modules

### Input shield

Runs before the provider call.

First versions:

- maximum length,
- blocked words,
- simple secret and PII patterns.

Actions:

- `allow`,
- `block`,
- `warn`,
- `redact`.

### RAG shield

Runs on retrieved documents before they are added to the model context.

First versions:

- context length limit,
- document filtering callback,
- metadata-based access check.

Later:

- indirect prompt-injection detection.

### Agent shield

Runs before a tool or agent action.

First versions:

- tool allowlist,
- tool denylist,
- argument validation.

Later:

- human approval callback,
- maximum step count.

### Output shield

Runs after the provider returns a response.

First versions:

- blocked words,
- required text format,
- Pydantic response validation.

Later:

- retry after validation failure,
- grounding checks,
- output PII redaction.

### Operations shield

Controls operational behavior around the request.

First versions:

- timeout handling,
- fail-open and fail-closed behavior,
- structured audit event.

Later:

- rate limits,
- token and cost budgets,
- OpenTelemetry integration.

## Public developer experience

The first simple API will be function-based:

```python
from ai_guardrails_sdk import Guardrails, MaxLengthGuardrail

guard = Guardrails(
    input_guardrails=[MaxLengthGuardrail(maximum=1000)],
)

result = guard.run(
    "user message",
    operation=call_my_model,
)
```

Later, we can add equivalent convenience APIs:

```python
guard.wrap(provider_client)
```

and:

```python
@guard.protect(policy="strict")
def ask_model(prompt: str) -> str:
    ...
```

These are convenience layers. They must use the same pipeline underneath.

## Configuration

After the Python API is stable, support a configuration file:

```yaml
policies:
  strict:
    input:
      max_length: 1000
      blocked_words:
        - password
    failure_mode: fail_closed
```

Configuration will create the same Python policy objects as direct code.
There must be one behavior, not separate YAML and Python implementations.

## Execution modes

### Synchronous mode

The first implementation runs checks in a predictable order. This is easiest
to understand and debug.

### Asynchronous mode

After the synchronous behavior is stable, independent checks can run with
`asyncio.gather`. Each check must still produce the same `Decision` format.

Async execution will not be added merely to claim speed. We will benchmark it
and document the actual latency.

## Failure behavior

Every policy will explicitly define what happens when a check fails or an
external evaluator is unavailable:

- `fail_closed`: block the operation,
- `fail_open`: allow it but record a warning.

The default will be conservative and explicit. The SDK will not silently
ignore evaluator errors.

## Implementation phases

### Phase 1: Core decisions and local rules

Build and understand:

- `Decision`,
- guardrail interface,
- maximum length rule,
- blocked words rule,
- ordered guardrail execution.

### Phase 2: Unified input/output pipeline

Build:

- input checks before the model call,
- output checks after the model call,
- `PipelineResult`,
- tests proving blocked input never calls the provider.

### Phase 3: Clean policies

Build:

- named policies,
- stages,
- action modes,
- explicit failure modes,
- validation for invalid policy configuration.

### Phase 4: Tool and agent protection

Build:

- tool allowlists,
- tool denylist,
- argument validation,
- step counter.

### Phase 5: Output validation

Build:

- structured output validation,
- Pydantic integration,
- safe fallback behavior,
- controlled retry behavior.

### Phase 6: Configuration files

Build:

- YAML loading,
- JSON loading,
- configuration validation,
- conversion into the same policy objects used by Python.

### Phase 7: Provider adapters and convenience APIs

Build:

- provider adapter protocol,
- one provider adapter as an example,
- `wrap()` support,
- decorator support.

### Phase 8: Async execution and telemetry

Build:

- async guardrail protocol,
- parallel independent checks,
- timeout handling,
- structured audit events,
- optional OpenTelemetry integration.

### Phase 9: Advanced shields

Consider only after the core is reliable:

- RAG injection scanning,
- PII and secret redaction,
- semantic evaluators,
- rate limiting,
- cost budgets,
- human approval workflows.

## What we will not build first

The first release will not include:

- a dashboard,
- a hosted control plane,
- every provider integration,
- automatic LLM-based evaluation for every request,
- a marketplace,
- complex agent-framework integrations.

The core must remain useful without any external AI evaluator.

## Definition of done for each phase

Before moving to the next phase:

1. The behavior is documented with a small example.
2. The implementation has focused tests.
3. Invalid input produces an explicit error.
4. The public API is understandable without reading internals.
5. The existing tests and linter pass.
6. We can explain why every new file and class exists.
