import difflib
import uuid
from pydantic import BaseModel, Field
from sqlalchemy import select, asc
from app.core.database import AsyncSessionLocal
from app.core.llm import llm_client
from app.models.evals import TestCaseResult, PromptRewriteProposal
from app.agents.sub_agents import (
    DecompositionAgent,
    RAGAgent,
    CritiqueAgent,
    SynthesisAgent,
)

# Registry of mutable agent prompts that MetaAgent can propose changes for
MUTABLE_PROMPTS: dict[str, str] = {
    "DecompositionAgent": DecompositionAgent.SYSTEM_PROMPT,
    "RAGAgent": RAGAgent.SYSTEM_PROMPT,
    "CritiqueAgent": CritiqueAgent.SYSTEM_PROMPT,
    "SynthesisAgent": SynthesisAgent.SYSTEM_PROMPT,
}


class PromptImprovementProposal(BaseModel):
    target_agent: str = Field(..., description="The agent whose system prompt should be rewritten.")
    proposed_prompt: str = Field(..., description="The full improved system prompt text.")
    justification: str = Field(..., description="Step-by-step reasoning for why this rewrite improves performance on failing cases.")


META_AGENT_PROMPT = """
You are a Meta-Evaluator AI. Your role is to improve the performance of a Multi-Agent LLM System.

You are given:
1. The worst-performing test cases from an evaluation run.
2. The current system prompts for each agent.

Your task is to:
1. Identify which agent's prompt is most responsible for the failures.
2. Propose a specific, improved rewrite of that agent's system prompt.
3. Provide a detailed justification linking the failures to specific weaknesses in the current prompt.

RULES:
- Only propose ONE rewrite per call.
- Do not make the prompt too permissive for adversarial cases.
- Preserve the output schema requirements.

Output must follow the PromptImprovementProposal schema.
"""


def _compute_structured_diff(original: str, proposed: str) -> list[dict]:
    """
    Computes a line-by-line structured diff between two prompt strings.
    Returns a list of diff objects for database persistence.
    """
    original_lines = original.splitlines(keepends=True)
    proposed_lines = proposed.splitlines(keepends=True)
    diff = difflib.unified_diff(original_lines, proposed_lines, fromfile="original", tofile="proposed")
    structured: list[dict] = []
    for line in diff:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            structured.append({"type": "header", "content": line.strip()})
        elif line.startswith("+"):
            structured.append({"type": "addition", "content": line[1:].strip()})
        elif line.startswith("-"):
            structured.append({"type": "removal", "content": line[1:].strip()})
        else:
            structured.append({"type": "context", "content": line.strip()})
    return structured


class MetaAgent:
    """
    Analyses EvalRun results, identifies the worst-performing agent prompts,
    and proposes structured improvements WITHOUT applying them automatically.
    """

    async def propose_improvements(self, run_id: uuid.UUID) -> uuid.UUID | None:
        """
        Queries the worst test cases from a run and generates a PromptRewriteProposal.
        Returns the proposal ID, or None if no improvement could be generated.
        """
        async with AsyncSessionLocal() as session:
            # ── Fetch the 3 lowest-scoring test cases from this run ──────────
            stmt = (
                select(TestCaseResult)
                .where(TestCaseResult.run_id == run_id)
                .order_by(asc(TestCaseResult.id))  # Will refine ordering by score below
            )
            result = await session.execute(stmt)
            all_cases: list[TestCaseResult] = list(result.scalars().all())

            # Sort in Python by the correctness score (lowest first)
            def _correctness_score(case: TestCaseResult) -> float:
                if not case.dimension_scores:
                    return 0.0
                return case.dimension_scores.get("correctness", {}).get("score", 0.0)

            worst_cases = sorted(all_cases, key=_correctness_score)[:3]

            if not worst_cases:
                return None

            # ── Format context for the MetaAgent prompt ──────────────────────
            formatted_cases = "\n\n".join([
                f"Category: {c.test_category}\n"
                f"Input: {c.input_prompt}\n"
                f"Output: {c.final_output}\n"
                f"Scores: {c.dimension_scores}"
                for c in worst_cases
            ])
            formatted_prompts = "\n\n".join([
                f"[{agent}]\n{prompt}"
                for agent, prompt in MUTABLE_PROMPTS.items()
            ])

            meta_prompt = (
                f"{META_AGENT_PROMPT}\n\n"
                f"WORST-PERFORMING TEST CASES:\n{formatted_cases}\n\n"
                f"CURRENT SYSTEM PROMPTS:\n{formatted_prompts}"
            )

            # ── Generate improvement proposal ─────────────────────────────────
            proposal = await llm_client.generate_structured_output(
                meta_prompt, PromptImprovementProposal
            )

            original_prompt = MUTABLE_PROMPTS.get(proposal.target_agent, "")
            structured_diff = _compute_structured_diff(original_prompt, proposal.proposed_prompt)

            # ── Persist proposal with status='pending' — never auto-apply ────
            db_proposal = PromptRewriteProposal(
                run_id=run_id,
                target_agent=proposal.target_agent,
                original_prompt=original_prompt,
                proposed_prompt=proposal.proposed_prompt,
                structured_diff=structured_diff,
                justification=proposal.justification,
                status="pending",
            )
            session.add(db_proposal)
            await session.commit()
            await session.refresh(db_proposal)
            return db_proposal.id
