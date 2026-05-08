import uuid
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, asc, desc

from app.core.database import AsyncSessionLocal
from app.schemas.api import (
    QueryRequest, TraceResponse, EvalSummaryResponse,
    RewriteApprovalRequest, ReEvalResponse
)
from app.schemas.context import SharedContext, Message
from app.schemas.routing import AgentType
from app.agents.orchestrator import Orchestrator
from app.evals.harness import AGENT_REGISTRY, EvaluationHarness
from app.models.logging import Job, ExecutionLog
from app.models.evals import EvalRun, TestCaseResult, PromptRewriteProposal
from app.core.streaming import sse_stream

router = APIRouter(prefix="/api/v1")

async def query_generator(prompt: str, job_id: uuid.UUID):
    context = SharedContext(
        job_id=job_id,
        original_query=prompt,
        chat_history=[Message(role="user", content=prompt)]
    )
    orchestrator = Orchestrator()
    
    # Save job to DB
    async with AsyncSessionLocal() as session:
        job = Job(id=job_id, query=prompt, status="running")
        session.add(job)
        await session.commit()
    
    first_event = True
    for _ in range(10):
        if context.is_completed:
            break
        decision = await orchestrator.determine_next_step(context)
        
        event_payload = {
            "type": "agent_activity",
            "agent": decision.next_agent.value,
            "status": "starting",
            "reasoning": decision.reasoning,
        }
        # Embed job_id in the very first event so clients can track the trace
        if first_event:
            event_payload["job_id"] = str(job_id)
            first_event = False

        yield event_payload
        
        if decision.next_agent == AgentType.COMPLETED:
            context.is_completed = True
            break
            
        agent = AGENT_REGISTRY.get(decision.next_agent)
        if agent:
            await agent.execute(context)
            
            # Log execution
            async with AsyncSessionLocal() as session:
                log = ExecutionLog(
                    job_id=job_id,
                    agent_id=decision.next_agent.value,
                    event_type="agent_output",
                    output_hash=str(context.sub_agent_scratchpad.get(decision.next_agent.value))
                )
                session.add(log)
                await session.commit()
            
            yield {
                "type": "agent_activity",
                "agent": decision.next_agent.value,
                "status": "completed",
                "budget_remaining": context.current_budget_remaining
            }

    # Final answer token-by-token
    final_answer = context.sub_agent_scratchpad.get("synthesis", {}).get("final_answer", "")
    for word in final_answer.split():
        yield {"type": "token", "content": word + " "}
        
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if job:
            job.status = "completed"
            await session.commit()

@router.post("/query")
async def process_query(request: QueryRequest):
    job_id = uuid.uuid4()
    return StreamingResponse(
        sse_stream(query_generator(request.query, job_id)),
        media_type="text/event-stream"
    )

@router.get("/jobs/{job_id}/trace", response_model=TraceResponse)
async def get_job_trace(job_id: uuid.UUID):
    async with AsyncSessionLocal() as session:
        stmt = select(ExecutionLog).where(ExecutionLog.job_id == job_id).order_by(asc(ExecutionLog.timestamp))
        result = await session.execute(stmt)
        logs = result.scalars().all()
        
        if not logs:
            raise HTTPException(status_code=404, detail="Job not found")
            
        traces = [
            {
                "id": str(log.id),
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "agent_id": log.agent_id,
                "event_type": log.event_type,
                "output_hash": log.output_hash,
                "latency_ms": log.latency_ms,
                "token_count": log.token_count
            }
            for log in logs
        ]
        return TraceResponse(job_id=job_id, traces=traces)

@router.get("/evals/latest", response_model=EvalSummaryResponse)
async def get_latest_eval():
    async with AsyncSessionLocal() as session:
        stmt = select(EvalRun).order_by(desc(EvalRun.timestamp)).limit(1)
        result = await session.execute(stmt)
        latest_run = result.scalars().first()
        
        if not latest_run:
            raise HTTPException(status_code=404, detail="No evaluation runs found")
            
        stmt_cases = select(TestCaseResult).where(TestCaseResult.run_id == latest_run.id)
        result_cases = await session.execute(stmt_cases)
        cases = result_cases.scalars().all()
        
        category_scores = {}
        category_counts = {}
        dimension_scores = {}
        dimension_counts = {}
        
        for case in cases:
            cat = case.test_category
            if case.dimension_scores and "correctness" in case.dimension_scores:
                score = case.dimension_scores["correctness"].get("score", 0.0)
                category_scores[cat] = category_scores.get(cat, 0.0) + score
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
                for dim, dim_data in case.dimension_scores.items():
                    if isinstance(dim_data, dict) and "score" in dim_data:
                        dimension_scores[dim] = dimension_scores.get(dim, 0.0) + dim_data["score"]
                        dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        for cat in category_scores:
            category_scores[cat] /= max(1, category_counts[cat])
            
        for dim in dimension_scores:
            dimension_scores[dim] /= max(1, dimension_counts[dim])
            
        return EvalSummaryResponse(
            run_id=latest_run.id,
            overall_score=latest_run.overall_score or 0.0,
            category_scores=category_scores,
            dimension_scores=dimension_scores
        )

@router.post("/evals/rewrites/{rewrite_id}/approve")
async def approve_rewrite(rewrite_id: uuid.UUID, payload: RewriteApprovalRequest):
    async with AsyncSessionLocal() as session:
        proposal = await session.get(PromptRewriteProposal, rewrite_id)
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
            
        proposal.status = "approved" if payload.approved else "rejected"
        await session.commit()
        return {"status": "success", "new_status": proposal.status}

@router.post("/evals/re_eval", response_model=ReEvalResponse)
async def trigger_reeval():
    # Placeholder for re-evaluating only failed test cases
    harness = EvaluationHarness()
    new_run_id = await harness.run_eval_suite()
    return ReEvalResponse(new_run_id=new_run_id)
