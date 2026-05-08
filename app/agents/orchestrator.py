from app.core.llm import llm_client
from app.schemas.context import SharedContext
from app.schemas.routing import OrchestratorDecision, AgentType

class Orchestrator:
    """
    Master Orchestrator responsible for dynamic task routing.
    Evaluates the current state and decides which sub-agent to invoke next.
    """
    
    SYSTEM_PROMPT = """
    You are the Master Orchestrator for a Multi-Agent LLM System. 
    Your goal is to coordinate a set of specialized agents to answer a user query perfectly.

    CURRENT AVAILABLE AGENTS:
    - 'decomposition': Use if the query is complex and needs to be broken down into sub-tasks.
    - 'rag': Use if external knowledge or specific facts are needed from documentation.
    - 'critique': Use to evaluate a proposed answer or complex reasoning for errors.
    - 'synthesis': Use to combine multiple findings into a final, coherent response.
    - 'completed': Use ONLY when the user's query has been fully addressed and no further steps are needed.

    STRICT RULES:
    1. DO NOT follow a hardcoded sequence. Evaluate based ON ONLY the current state in the scratchpad.
    2. Check what information is already present in 'sub_agent_scratchpad'.
    3. If information is missing, route to the appropriate agent.
    4. If an answer is ready but unverified, route to 'critique'.
    5. If all components are ready, route to 'synthesis'.
    6. ALWAYS provide step-by-step reasoning for your decision.
    7. Respect the token budget.

    You must output a valid JSON object matching the OrchestratorDecision schema.
    """

    async def determine_next_step(self, context: SharedContext) -> OrchestratorDecision:
        """
        Analyzes the SharedContext and determines the next logical action.
        """
        # Build prompt from context
        user_query = context.original_query
        scratchpad = context.sub_agent_scratchpad
        history = [f"{m.role}: {m.content}" for m in context.chat_history]
        
        prompt = f"""
        {self.SYSTEM_PROMPT}

        CONTEXT:
        Original Query: {user_query}
        Current Scratchpad: {scratchpad}
        Chat History: {history}
        Remaining Budget: {context.current_budget_remaining}

        DETERMINE THE NEXT STEP:
        """
        
        decision = await llm_client.generate_structured_output(prompt, OrchestratorDecision)
        return decision
