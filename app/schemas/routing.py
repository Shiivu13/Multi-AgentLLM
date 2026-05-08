from enum import Enum
from pydantic import BaseModel, Field

class AgentType(str, Enum):
    DECOMPOSITION = "decomposition"
    RAG = "rag"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    COMPLETED = "completed"

class OrchestratorDecision(BaseModel):
    """
    Schema for the Orchestrator's routing decision.
    """
    reasoning: str = Field(..., description="Step-by-step logic justifying the routing decision.")
    next_agent: AgentType = Field(..., description="The type of agent to call next.")
    context_budget_allocated: int = Field(..., description="Token budget assigned for this turn.")
