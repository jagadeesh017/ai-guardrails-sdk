import asyncio
from collections.abc import Awaitable, Callable

from ..core.context import ExecutionContext, Stage
from ..core.decision import Decision
from ..core.policy import EngineConfig, Policy
from ..core.registry import rule_registry
from ..rules.base import BaseRule
from .executor import create_executor


class GuardrailPipeline:
    def __init__(self, policy: Policy, engine_config: EngineConfig = None):
        self.policy = policy
        self.engine_config = engine_config or EngineConfig()
        self.executor = create_executor(self.engine_config.mode)

    def _get_rules_for_stage(self, stage: Stage) -> list[BaseRule]:
        stage_config = getattr(self.policy, stage.value, None)
        if not stage_config or not stage_config.enabled:
            return []

        rules = []
        for rule_config in stage_config.rules:
            if not rule_config.enabled:
                continue
            factory = rule_registry.get(rule_config.type)
            if factory:
                try:
                    rule = factory(rule_config)
                    rules.append(rule)
                except Exception as e:
                    print(f"Warning: Failed to create rule {rule_config.type}: {e}")
        return rules

    async def execute_stage(self, stage: Stage, ctx: ExecutionContext) -> Decision:
        rules = self._get_rules_for_stage(stage)
        if not rules:
            return Decision.allow()

        ctx = ctx.with_stage(stage)
        return await self.executor.execute(rules, ctx, self.engine_config)

    async def run(self, ctx: ExecutionContext, llm_fn: Callable[[str], Awaitable[str] | str] = None) -> Decision:
        # Stage 1: INPUT
        input_decision = await self.execute_stage(Stage.INPUT, ctx)
        if input_decision.is_blocked:
            return input_decision

        cleaned_text = input_decision.modified_data or ctx.text

        # Stage 2: RAG (optional)
        if self.policy.rag.enabled:
            rag_ctx = ctx.with_stage(Stage.RAG).with_text(cleaned_text)
            rag_decision = await self.execute_stage(Stage.RAG, rag_ctx)
            if rag_decision.is_blocked:
                return rag_decision
            if rag_decision.is_modified:
                cleaned_text = rag_decision.modified_data

        # Stage 3: AGENT (optional, for tool calls)
        if self.policy.agent.enabled and llm_fn:
            # This would need tool call handling logic
            pass

        # Stage 4: Execute LLM if provided
        response = None
        if llm_fn:
            if asyncio.iscoroutinefunction(llm_fn):
                response = await llm_fn(cleaned_text)
            else:
                response = llm_fn(cleaned_text)

        # Stage 5: OUTPUT
        if self.policy.output.enabled:
            output_ctx = ctx.with_stage(Stage.OUTPUT).with_response(response or "")
            output_ctx = output_ctx.with_text("")  # Clear text for output stage
            output_decision = await self.execute_stage(Stage.OUTPUT, output_ctx)
            if output_decision.is_blocked:
                return output_decision
            if output_decision.is_modified:
                response = output_decision.modified_data

        # Stage 6: OPS (always runs for logging/metrics)
        if self.policy.ops.enabled:
            ops_ctx = ctx.with_stage(Stage.OPS)
            await self.execute_stage(Stage.OPS, ops_ctx)

        return Decision.allow(modified_data=response)

    def run_sync(self, ctx: ExecutionContext, llm_fn: Callable[[str], str] = None) -> Decision:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.run(ctx, llm_fn))