import uuid
from sqlalchemy import Column, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class EvalRun(Base):
    """
    Represents one full execution of the evaluation suite.
    """
    __tablename__ = "eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    overall_score = Column(Float, nullable=True)

    # Relationships
    test_case_results = relationship("TestCaseResult", back_populates="run", cascade="all, delete-orphan")
    prompt_proposals = relationship("PromptRewriteProposal", back_populates="run", cascade="all, delete-orphan")


class TestCaseResult(Base):
    """
    Stores results for a single test case in an EvalRun.
    """
    __tablename__ = "test_case_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    test_category = Column(String, nullable=False)  # baseline, ambiguous, adversarial
    input_prompt = Column(String, nullable=False)
    exact_tool_calls = Column(JSON, nullable=True)       # Exact sequence of tool calls made
    final_output = Column(String, nullable=True)
    dimension_scores = Column(JSON, nullable=True)        # Dict of score_name -> {score, justification}
    exact_prompts_used = Column(JSON, nullable=True)      # Dict of agent_id -> system_prompt snapshot

    # Relationships
    run = relationship("EvalRun", back_populates="test_case_results")


class PromptRewriteProposal(Base):
    """
    A MetaAgent-generated proposal to improve a specific agent's system prompt.
    Must be manually approved before application.
    """
    __tablename__ = "prompt_rewrite_proposals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    target_agent = Column(String, nullable=False)         # e.g. "RAGAgent", "DecompositionAgent"
    original_prompt = Column(String, nullable=False)
    proposed_prompt = Column(String, nullable=False)
    structured_diff = Column(JSON, nullable=False)        # Line-by-line diff object
    justification = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, approved, rejected

    # Relationships
    run = relationship("EvalRun", back_populates="prompt_proposals")
