import json
from uuid import UUID
from app.core.token_counter import count_tokens
from app.schemas.context import SharedContext
from app.core.database import AsyncSessionLocal
from app.models.logging import ExecutionLog

class ContextOverflowError(Exception):
    """Exception raised when an agent attempts to exceed its allocated context budget."""
    pass

class ContextBudgetManager:
    """
    Manages token budgets dynamically to prevent API errors and enforce agent limits.
    """
    @staticmethod
    async def log_policy_violation(job_id: UUID, agent_id: str, overflow_amount: int):
        """
        Logs a strict policy violation when an agent attempts to write more tokens than allowed.
        """
        async with AsyncSessionLocal() as session:
            log = ExecutionLog(
                job_id=job_id,
                agent_id=agent_id,
                event_type="policy_violation",
                input_hash="context_overflow",
                output_hash=str(overflow_amount),
                latency_ms=0.0,
                token_count=overflow_amount
            )
            session.add(log)
            await session.commit()

    @staticmethod
    async def check_remaining_budget(agent_id: str, context: SharedContext, proposed_addition: str, agent_max_budget: int) -> int:
        """
        Checks if adding the proposed string will exceed the agent's max budget.
        Raises ContextOverflowError if it does, initiating the compression fallback.
        """
        # Calculate current context footprint
        current_context_str = json.dumps(context.model_dump(), default=str)
        current_tokens = count_tokens(current_context_str)
        proposed_tokens = count_tokens(proposed_addition)
        
        total_tokens = current_tokens + proposed_tokens
        
        if total_tokens > agent_max_budget:
            overflow = total_tokens - agent_max_budget
            await ContextBudgetManager.log_policy_violation(context.job_id, agent_id, overflow)
            raise ContextOverflowError(f"Agent '{agent_id}' exceeded budget by {overflow} tokens.")
            
        return agent_max_budget - total_tokens
