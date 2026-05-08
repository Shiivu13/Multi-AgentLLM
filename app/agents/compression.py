import json
from typing import Dict, Any
from pydantic import BaseModel, Field
from app.agents.base import BaseAgent
from app.schemas.context import SharedContext
from app.core.llm import llm_client

class SummarizedContent(BaseModel):
    compressed_text: str = Field(..., description="Highly condensed lossy summarization of the original text.")

class CompressionAgent(BaseAgent):
    """
    Reduces the token footprint of the SharedContext when a ContextOverflowError occurs.
    Enforces strict separation of Lossless (structured) and Lossy (natural language) data.
    """
    agent_max_budget: int = 16000  # Give it an elevated budget to process the overflow

    SYSTEM_PROMPT = """
    You are an elite Context Compression AI.
    Your task is to summarize the provided natural language text to save tokens (Lossy Compression).
    Retain key facts, entities, and intent, but ruthlessly remove conversational filler.
    """

    async def execute(self, context: SharedContext) -> None:
        lossless_scratchpad: Dict[str, Any] = {}
        lossy_scratchpad: Dict[str, Any] = {}

        # 1. PURE PYTHON LOSSLESS VS LOSSY SEPARATION
        for key, value in context.sub_agent_scratchpad.items():
            if isinstance(value, (dict, list)):
                # Dictionaries and lists (tool outputs, citations, schemas) are structured data.
                lossless_scratchpad[key] = value
            elif isinstance(value, str):
                try:
                    # If it parses as JSON, it's structured. Keep it lossless.
                    parsed = json.loads(value)
                    lossless_scratchpad[key] = parsed
                except json.JSONDecodeError:
                    # Plain text string. Target for lossy compression.
                    lossy_scratchpad[key] = value
            else:
                # Ints, bools, etc.
                lossless_scratchpad[key] = value

        # 2. LOSSY COMPRESSION ON NATURAL LANGUAGE (Chat History)
        compressed_history = []
        for msg in context.chat_history:
            if len(msg.content) > 100:  # Only compress long messages
                prompt = f"{self.SYSTEM_PROMPT}\n\nORIGINAL TEXT:\n{msg.content}"
                try:
                    summary = await llm_client.generate_structured_output(prompt, SummarizedContent)
                    msg.content = summary.compressed_text
                except Exception:
                    pass
            compressed_history.append(msg)

        context.chat_history = compressed_history

        # 3. LOSSY COMPRESSION ON UNSTRUCTURED SCRATCHPAD NOTES
        for key, text_value in lossy_scratchpad.items():
            if len(text_value) > 100:
                prompt = f"{self.SYSTEM_PROMPT}\n\nORIGINAL NOTES:\n{text_value}"
                try:
                    summary = await llm_client.generate_structured_output(prompt, SummarizedContent)
                    lossless_scratchpad[key] = summary.compressed_text
                except Exception:
                    lossless_scratchpad[key] = text_value
            else:
                lossless_scratchpad[key] = text_value

        # Rebuild the scratchpad with the preserved lossless data + new lossy summaries
        context.sub_agent_scratchpad = lossless_scratchpad
        
        # Reset the token budget for the remainder of the job
        context.current_budget_remaining = 8000
