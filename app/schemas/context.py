from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID

class Message(BaseModel):
    """
    Represents a single turn in a conversational or agentic sequence.
    """
    role: str = Field(..., description="Role of the sender: user, assistant, system, or tool")
    content: str = Field(..., description="The content of the message")
    agent_source: Optional[str] = Field(None, description="The specific agent that generated this message, if applicable")

class SharedContext(BaseModel):
    """
    The master state object shared across the entire orchestration layer and sub-agents.
    """
    job_id: UUID = Field(..., description="The unique identifier for the current job (database reference)")
    original_query: str = Field(..., description="The initial user query")
    chat_history: List[Message] = Field(default_factory=list, description="List of messages exchanged so far")
    current_budget_remaining: int = Field(default=8000, description="Token or cost budget remaining for this job execution")
    sub_agent_scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Intermediate outputs from sub-agents (e.g., citations, thought graphs)")
    is_completed: bool = Field(default=False, description="Flag indicating if the job has successfully reached a terminal state")
