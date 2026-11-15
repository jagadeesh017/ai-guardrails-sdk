"""Test YAML config with PII and toxicity."""

import asyncio
import time
import json
from ai_guardrails_sdk import (
    MaxLengthGuardrail,
    BlockedWordsGuardrail,
    PIIGuardrail,
    ToxicityGuardrail,
    Pipeline,
    Decision,
    check_async,
    check,
    load_pipeline_from_yaml,
)


async def fake_llm_async(prompt: str) -> str:
    await asyncio.sleep(0.1)
    responses = {
        "hello": json.dumps({"response": "Hi there!"}),
        "my email is john@example.com": json.dumps({"response": "Got it, your email is john@example.com"}),
        "you are stupid": json.dumps({"response": "I don't like that tone"}),
        "tell me a secret": json.dumps({"response": "The secret is 12345"}),
        "write a very long response": json.dumps({"response": "This is a very long response that exceeds the output limit easily"}),
    }
    return responses.get(prompt, json.dumps({"response": "Default response"}))


print("=== 1. Test Individual Guardrails ===")

pii = PIIGuardrail(["email", "phone", "ssn"])
print(f"  Email test: {pii.check('contact me at john@example.com')}")
print(f"  Clean test: {pii.check('hello world')}")
print(f"  Redact: {pii.redact('email: john@example.com, phone: 555-123-4567')}")

tox = ToxicityGuardrail(0.5)
print(f"  Toxic: {tox.check('you are stupid')}")
print(f"  Clean: {tox.check('hello there')}")


print("\n=== 2. Pipeline from YAML ===")
pipeline = load_pipeline_from_yaml("guardrails.yaml")

async def test_pipeline():
    tests = [
        "hello",
        "my email is john@example.com",
        "you are stupid",
        "tell me a secret",
    ]
    for prompt in tests:
        result = await pipeline.run_async(prompt, fake_llm_async)
        print(f"  '{prompt}' -> {result}")

asyncio.run(test_pipeline())