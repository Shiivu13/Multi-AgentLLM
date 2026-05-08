import uuid
from sqlalchemy import Column, String, Float, Integer, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base

class Job(Base):
    """
    Represents a single user request or task orchestration.
    """
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    query = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    execution_logs = relationship("ExecutionLog", back_populates="job", cascade="all, delete-orphan")


class ExecutionLog(Base):
    """
    Represents a single agent action, tool call, or granular event within a Job.
    """
    __tablename__ = "execution_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    agent_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)  # routing, tool_call, agent_output, policy_violation
    input_hash = Column(String, nullable=True)
    output_hash = Column(String, nullable=True)
    latency_ms = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)

    # Relationships
    job = relationship("Job", back_populates="execution_logs")
