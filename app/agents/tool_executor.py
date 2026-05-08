import time
import json
from uuid import UUID
from pydantic import BaseModel, Field
from app.tools.base import BaseTool, ToolResult
from app.core.database import AsyncSessionLocal
from app.models.logging import ExecutionLog
from app.core.llm import llm_client

class ToolRetryModification(BaseModel):
    modified_input: dict = Field(..., description="The modified dictionary of inputs to pass to the tool to fix the error.")

class ToolOutputEvaluation(BaseModel):
    is_sufficient: bool = Field(..., description="True if the tool output is useful and answers the intent, False otherwise.")
    feedback: str = Field(..., description="Feedback on why it is insufficient or what needs to change.")

class ToolExecutor:
    """
    Utility class that handles agent's interaction with tools, 
    including logging, evaluation, and strict fallback loop.
    """
    @staticmethod
    async def _log_execution(job_id: UUID, agent_id: str, tool_name: str, input_data: dict, output_data: dict, latency_ms: float, event_type: str):
        async with AsyncSessionLocal() as session:
            try:
                input_hash = str(hash(json.dumps(input_data, sort_keys=True)))
            except TypeError:
                input_hash = "unhashable_input"
            
            try:
                output_hash = str(hash(json.dumps(output_data, sort_keys=True)))
            except TypeError:
                output_hash = "unhashable_output"

            log = ExecutionLog(
                job_id=job_id,
                agent_id=agent_id,
                event_type=event_type,
                input_hash=input_hash,
                output_hash=output_hash,
                latency_ms=latency_ms
            )
            session.add(log)
            await session.commit()

    @staticmethod
    async def execute_with_retries(agent_id: str, tool: BaseTool, initial_input: dict, job_id: UUID) -> ToolResult:
        max_retries = 2
        current_input = initial_input

        for attempt in range(max_retries + 1):
            start_time = time.time()
            
            # Execute tool and catch failures strictly in the Python flow
            result = await tool.execute(current_input)
            latency_ms = (time.time() - start_time) * 1000

            event_type = "tool_call_success" if result.is_success else f"tool_call_failure_attempt_{attempt}"
            output_data = result.data if result.is_success else result.failure.model_dump()
            
            # Log raw execution
            await ToolExecutor._log_execution(job_id, agent_id, tool.name, current_input, output_data, latency_ms, event_type)

            if result.is_success:
                # LLM explicitly evaluates if the successful output is "sufficient"
                eval_prompt = f"Evaluate the output of tool '{tool.name}'.\nInput: {current_input}\nOutput: {result.data}\nIs this output sufficient and useful?"
                try:
                    evaluation = await llm_client.generate_structured_output(eval_prompt, ToolOutputEvaluation)
                    if evaluation.is_sufficient:
                        # Log acceptance
                        await ToolExecutor._log_execution(job_id, agent_id, tool.name, current_input, {"status": "accepted"}, 0.0, "tool_output_accepted")
                        return result
                    else:
                        # Log rejection and prepare for retry
                        await ToolExecutor._log_execution(job_id, agent_id, tool.name, current_input, {"status": "rejected", "reason": evaluation.feedback}, 0.0, "tool_output_rejected")
                        if attempt < max_retries:
                            retry_prompt = f"Tool '{tool.name}' succeeded but output was insufficient.\nFeedback: {evaluation.feedback}\nOriginal input: {current_input}\nProvide a modified input dictionary."
                            modification = await llm_client.generate_structured_output(retry_prompt, ToolRetryModification)
                            current_input = modification.modified_input
                            continue
                except Exception:
                    # In case of eval failure, default to accepted
                    await ToolExecutor._log_execution(job_id, agent_id, tool.name, current_input, {"status": "accepted_unverified"}, 0.0, "tool_output_accepted")
                    return result

            # Tool failed. Agent implicitly rejects due to failure.
            if result.failure is not None:
                failure_reason = result.failure.reason
                failure_msg = result.failure.message
            else:
                failure_reason = "unknown_error"
                failure_msg = "Tool returned failure state with no FailureContract."

            await ToolExecutor._log_execution(job_id, agent_id, tool.name, current_input, {"status": "rejected", "reason": failure_reason}, 0.0, "tool_output_rejected")

            if attempt < max_retries:
                # Modify input using LLM based on FailureContract message
                prompt = f"Tool '{tool.name}' failed with this error:\n{failure_msg}\nOriginal input: {current_input}\nProvide a modified input dictionary to fix the error."
                try:
                    modification = await llm_client.generate_structured_output(prompt, ToolRetryModification)
                    current_input = modification.modified_input
                except Exception:
                    # Fail gracefully if retry modification fails
                    break

        return result
