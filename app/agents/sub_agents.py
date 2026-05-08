from app.agents.base import BaseAgent
from app.core.llm import llm_client
from app.core.context_manager import ContextOverflowError
from app.schemas.context import SharedContext
from app.schemas.agents import (
    DecompositionOutput,
    RAGOutput,
    CritiqueOutput,
    SynthesisOutput
)
from app.agents.tool_executor import ToolExecutor
from app.tools.implementations import WebSearchStub, SelfReflectionTool


async def _handle_overflow(context: SharedContext) -> None:
    """
    Shared helper that triggers the CompressionAgent when a ContextOverflowError occurs.
    Imported here (not at module level) to avoid circular imports.
    """
    # Local import to break circular dependency chain
    from app.agents.compression import CompressionAgent
    compressor = CompressionAgent()
    await compressor.execute(context)


class DecompositionAgent(BaseAgent):
    """
    Breaks complex queries into a dependency graph of sub-tasks.
    """
    agent_max_budget: int = 4000

    SYSTEM_PROMPT = """
    You are a Strategic Planner. Your job is to take a complex user query and decompose it
    into a logical dependency graph of sub-tasks.
    Ensure that each task is atomic and that dependencies are explicitly defined.
    Output must follow the DecompositionOutput schema.
    """

    async def execute(self, context: SharedContext) -> None:
        prompt = f"{self.SYSTEM_PROMPT}\n\nQUERY: {context.original_query}"
        output = await llm_client.generate_structured_output(prompt, DecompositionOutput)
        try:
            await self.save_to_scratchpad(context, "decomposition", output.model_dump())
        except ContextOverflowError:
            await _handle_overflow(context)
            # Retry after compression
            await self.save_to_scratchpad(context, "decomposition", output.model_dump())


class RAGAgent(BaseAgent):
    """
    Performs multi-hop reasoning over retrieved information, actively using tools.
    """
    agent_max_budget: int = 4000

    SYSTEM_PROMPT = """
    You are a Knowledge Specialist. Perform multi-hop reasoning over the information
    provided to answer the current tasks.
    Connect the dots logically and always cite your sources using the Citation schema.
    """

    async def execute(self, context: SharedContext) -> None:
        tasks = context.sub_agent_scratchpad.get("decomposition", "No tasks defined yet.")

        web_search = WebSearchStub()
        search_result = await ToolExecutor.execute_with_retries(
            agent_id="rag",
            tool=web_search,
            initial_input={"query": context.original_query},
            job_id=context.job_id
        )
        search_data = search_result.data if search_result.is_success else search_result.failure.model_dump()

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"TASKS & CONTEXT:\n{tasks}\n\n"
            f"SEARCH RESULTS (from Tool):\n{search_data}"
        )
        output = await llm_client.generate_structured_output(prompt, RAGOutput)
        try:
            await self.save_to_scratchpad(context, "rag", output.model_dump())
        except ContextOverflowError:
            await _handle_overflow(context)
            await self.save_to_scratchpad(context, "rag", output.model_dump())


class CritiqueAgent(BaseAgent):
    """
    Evaluates reasoning and claims in the scratchpad for accuracy and consistency.
    """
    agent_max_budget: int = 4000

    SYSTEM_PROMPT = """
    You are a Rigorous Auditor. Examine all claims and reasoning currently in the scratchpad.
    Look for contradictions, hallucinations, or weak logic.
    Assign a confidence score (0-100) and flag specific problematic text spans.
    Output must follow the CritiqueOutput schema.
    """

    async def execute(self, context: SharedContext) -> None:
        current_state = context.sub_agent_scratchpad
        prompt = f"{self.SYSTEM_PROMPT}\n\nCURRENT SCRATCHPAD STATE:\n{current_state}"
        output = await llm_client.generate_structured_output(prompt, CritiqueOutput)
        try:
            await self.save_to_scratchpad(context, "critique", output.model_dump())
        except ContextOverflowError:
            await _handle_overflow(context)
            await self.save_to_scratchpad(context, "critique", output.model_dump())


class SynthesisAgent(BaseAgent):
    """
    Combines all findings into a final coherent answer with provenance tracking.
    """
    agent_max_budget: int = 4000

    SYSTEM_PROMPT = """
    You are a Master Synthesizer. Your goal is to produce the final answer for the user.
    1. Read the entire scratchpad.
    2. Resolve any contradictions flagged by the CritiqueAgent or Reflection Tool.
    3. Generate a high-quality, professional answer.
    4. Provide a provenance map linking parts of your answer back to the source agents or data chunks.
    Output must follow the SynthesisOutput schema.
    """

    async def execute(self, context: SharedContext) -> None:
        current_state = context.sub_agent_scratchpad

        reflection_tool = SelfReflectionTool()
        reflection_result = await ToolExecutor.execute_with_retries(
            agent_id="synthesis",
            tool=reflection_tool,
            initial_input={"context_dump": str(current_state)},
            job_id=context.job_id
        )
        reflection_data = (
            reflection_result.data if reflection_result.is_success
            else reflection_result.failure.model_dump()
        )

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"FULL SCRATCHPAD DATA:\n{current_state}\n\n"
            f"SELF REFLECTION FEEDBACK (from Tool):\n{reflection_data}"
        )
        output = await llm_client.generate_structured_output(prompt, SynthesisOutput)
        try:
            await self.save_to_scratchpad(context, "synthesis", output.model_dump())
        except ContextOverflowError:
            await _handle_overflow(context)
            await self.save_to_scratchpad(context, "synthesis", output.model_dump())

        context.is_completed = True
