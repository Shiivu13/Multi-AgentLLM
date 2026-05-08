from abc import ABC, abstractmethod
import json
from typing import Any
from app.schemas.context import SharedContext
from app.core.context_manager import ContextBudgetManager, ContextOverflowError

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the orchestration system.
    Enforces reading from and writing to the SharedContext, governed by strict token budgets.
    """
    agent_max_budget: int = 4000  # Default budget threshold before triggering compression

    @abstractmethod
    async def execute(self, context: SharedContext) -> None:
        """
        Executes the agent logic. Agents MUST use `save_to_scratchpad` to mutate context.
        """
        pass

    async def save_to_scratchpad(self, context: SharedContext, key: str, value: Any) -> None:
        """
        Safely appends data to the scratchpad after verifying the token budget.
        Raises ContextOverflowError if the agent exceeds its allowed token capacity.
        """
        proposed_addition = json.dumps({key: value}, default=str)
        await ContextBudgetManager.check_remaining_budget(
            agent_id=self.__class__.__name__, 
            context=context, 
            proposed_addition=proposed_addition, 
            agent_max_budget=self.agent_max_budget
        )
        context.sub_agent_scratchpad[key] = value
