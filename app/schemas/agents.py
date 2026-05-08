from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SubTask(BaseModel):
    """
    A single node in the decomposition dependency graph.
    """
    id: str = Field(..., description="Unique identifier for the sub-task (e.g., T1, T2).")
    description: str = Field(..., description="Clear explanation of what needs to be solved.")
    dependencies: List[str] = Field(default_factory=list, description="List of IDs this task depends on.")

class DecompositionOutput(BaseModel):
    """
    Output of the DecompositionAgent.
    """
    tasks: List[SubTask] = Field(..., description="The complete list of sub-tasks forming a dependency graph.")

class Citation(BaseModel):
    """
    Mapping a specific claim to its source material.
    """
    claim: str = Field(..., description="The specific factual claim made.")
    source_chunk: str = Field(..., description="The identifier or snippet of the source chunk.")

class RAGOutput(BaseModel):
    """
    Output of the RAGAgent.
    """
    reasoning: str = Field(..., description="The multi-hop logic used to connect facts.")
    citations: List[Citation] = Field(..., description="List of citations supporting the reasoning.")

class ClaimReview(BaseModel):
    """
    A specific evaluation of a claim made by another agent.
    """
    claim: str = Field(..., description="The claim being reviewed.")
    confidence_score: int = Field(..., ge=0, le=100, description="Confidence score from 0 to 100.")
    flagged_text_spans: List[str] = Field(default_factory=list, description="Specific text spans that contain errors or contradictions.")
    critique_note: str = Field(..., description="Explanation of why the claim was flagged or its score.")

class CritiqueOutput(BaseModel):
    """
    Output of the CritiqueAgent.
    """
    reviews: List[ClaimReview] = Field(..., description="List of reviews for the current scratchpad state.")

class SynthesisOutput(BaseModel):
    """
    Final output of the SynthesisAgent.
    """
    final_answer: str = Field(..., description="The comprehensive final answer for the user.")
    provenance_map: Dict[str, str] = Field(..., description="Mapping of sentences/sections back to source agents or RAG chunks.")
