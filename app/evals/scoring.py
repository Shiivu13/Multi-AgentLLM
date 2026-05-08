from typing import Any
from pydantic import BaseModel, Field
from app.core.llm import llm_client


class ScoringResult(BaseModel):
    score: float = Field(..., ge=0.0, le=10.0, description="Numeric score from 0-10.")
    justification: str = Field(..., description="Written rationale for the score.")


# ──────────────────────────────────────────────────────────────────────────────
# 1. CORRECTNESS  (LLM-as-a-Judge)
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """
You are a strict but fair judge evaluating AI-generated answers.
Compare the expected answer to the actual answer and return a score from 0 to 10
along with a concise justification.
Score 10 = factually identical, 0 = completely wrong or irrelevant.
Output must follow the ScoringResult schema.
"""


async def score_correctness(expected: str, actual: str) -> dict[str, Any]:
    """LLM-as-a-judge that grades the final answer against the expected answer."""
    prompt = (
        f"{JUDGE_PROMPT}\n\n"
        f"EXPECTED ANSWER:\n{expected}\n\n"
        f"ACTUAL ANSWER:\n{actual}"
    )
    result = await llm_client.generate_structured_output(prompt, ScoringResult)
    return result.model_dump()


# ──────────────────────────────────────────────────────────────────────────────
# 2. CITATION ACCURACY  (Pure Python)
# ──────────────────────────────────────────────────────────────────────────────

def score_citation_accuracy(
    output_citations: list[dict[str, str]],
    retrieved_chunks: list[str]
) -> dict[str, Any]:
    """
    Pure Python check: verifies that each cited claim appears in one of the
    retrieved chunks. No LLM call involved.
    """
    if not output_citations:
        return {"score": 0.0, "justification": "No citations were produced."}

    verified = 0
    failed_claims: list[str] = []

    for citation in output_citations:
        claim = citation.get("claim", "").lower()
        matched = any(
            claim in chunk.lower() or any(word in chunk.lower() for word in claim.split() if len(word) > 4)
            for chunk in retrieved_chunks
        )
        if matched:
            verified += 1
        else:
            failed_claims.append(citation.get("claim", ""))

    score = round((verified / len(output_citations)) * 10, 2)
    justification = (
        f"{verified}/{len(output_citations)} citations verified in retrieved chunks."
        + (f" Unverified: {failed_claims}" if failed_claims else "")
    )
    return {"score": score, "justification": justification}


# ──────────────────────────────────────────────────────────────────────────────
# 3. TOOL EFFICIENCY  (Pure Python)
# ──────────────────────────────────────────────────────────────────────────────

# Max acceptable calls per tool type before penalties kick in
TOOL_CALL_THRESHOLDS: dict[str, int] = {
    "web_search": 2,
    "python_sandbox": 3,
    "nl_to_sql": 2,
    "self_reflection": 1,
}


def score_tool_efficiency(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Penalises redundant tool usage. Each tool type has a max acceptable call count.
    Excess calls reduce the score proportionally.
    """
    if not tool_calls:
        return {"score": 10.0, "justification": "No tools were used — acceptable for simple queries."}

    call_counts: dict[str, int] = {}
    for call in tool_calls:
        tool_name = call.get("tool_name", "unknown")
        call_counts[tool_name] = call_counts.get(tool_name, 0) + 1

    penalties: list[str] = []
    total_penalty = 0.0

    for tool, count in call_counts.items():
        threshold = TOOL_CALL_THRESHOLDS.get(tool, 1)
        if count > threshold:
            excess = count - threshold
            penalty = excess * 1.5
            total_penalty += penalty
            penalties.append(f"{tool} called {count}x (threshold {threshold}x, -{penalty:.1f}pts)")

    score = max(0.0, round(10.0 - total_penalty, 2))
    justification = (
        "All tool calls within acceptable thresholds." if not penalties
        else "Penalty applied: " + "; ".join(penalties)
    )
    return {"score": score, "justification": justification}


# ──────────────────────────────────────────────────────────────────────────────
# 4. CRITIQUE AGREEMENT  (Pure Python)
# ──────────────────────────────────────────────────────────────────────────────

def score_critique_agreement(
    critique_flags: list[dict[str, Any]],
    final_synthesis: str
) -> dict[str, Any]:
    """
    Pure Python check: verifies the SynthesisAgent addressed the contradictions
    that CritiqueAgent explicitly flagged.
    """
    if not critique_flags:
        return {"score": 10.0, "justification": "No contradictions were flagged by CritiqueAgent."}

    synthesis_lower = final_synthesis.lower()
    resolved = 0
    unresolved_spans: list[str] = []

    for review in critique_flags:
        for span in review.get("flagged_text_spans", []):
            # If the flagged span is absent from the final synthesis, it was resolved
            if span.lower() not in synthesis_lower:
                resolved += 1
            else:
                unresolved_spans.append(span)

    total_spans = sum(len(r.get("flagged_text_spans", [])) for r in critique_flags)
    if total_spans == 0:
        return {"score": 10.0, "justification": "No specific text spans were flagged."}

    score = round((resolved / total_spans) * 10, 2)
    justification = (
        f"{resolved}/{total_spans} flagged spans resolved in final synthesis."
        + (f" Still present: {unresolved_spans}" if unresolved_spans else "")
    )
    return {"score": score, "justification": justification}
