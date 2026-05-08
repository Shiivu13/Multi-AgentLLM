# Real-Time Multi-Agent LLM Orchestration System

> A production-grade, asynchronous backend orchestrating specialized LLM sub-agents to solve complex queries through dynamic routing, shared context, and self-improving evaluation loops.

## Five-Minute Quickstart

This system is built for zero-touch deployment. You do not need to install local dependencies or configure local databases.

1.  **Configure Environment**: Copy `.env.example` to `.env` and insert your Gemini API key.
    ```bash
    cp .env.example .env
    # Edit .env to add your GEMINI_API_KEY
    ```
2.  **Start Services**: Launch the entire stack (FastAPI backend + PostgreSQL + pgvector) using Docker Compose.
    ```bash
    docker-compose up --build
    ```

Once running, the interactive API documentation is automatically generated and accessible at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## Architecture Diagram

```text
                                 +-------------------------+
                                 |  Human / API Client     |
                                 +-----------+-------------+
                                             | (HTTP SSE / REST)
                                             v
                                 +-------------------------+
                                 |      FastAPI Router     |
                                 +-----------+-------------+
                                             | 
                                             v
+------------------+             +-------------------------+             +------------------+
|  Self-Improving  |             |  MASTER ORCHESTRATOR    |             |  Shared Context  |
|   Meta-Agent     |             |  (Dynamic Routing)      |             |  (State Memory)  |
|                  |             +-------+--------+--------+             +--------+---------+
| - Analyzes Evals |                     |        |                               |
| - Proposes Prompt|      +--------------+        +--------------+                |
|   Rewrites       |      |                                      |                |
+--------+---------+      v                                      v                v
         |          +------------+                        +-------------+         |
         |          | DECOMP.    |                        |    RAG      |         |
         |          | AGENT      |                        |   AGENT     |         |
         |          +------------+                        +-------------+         |
         |                |                                      |                |
         |          +------------+                        +-------------+         |
         |          | CRITIQUE   |                        |  SYNTHESIS  |         |
         |          | AGENT      |                        |    AGENT    |         |
         |          +------------+                        +-------------+         |
         |                                                                        |
         |                                                                        |
         +------------------------------------------------------------------------+
                                             |
                                             v
                                 +-------------------------+
                                 | PostgreSQL Database     |
                                 | (Jobs, Logs, Evals,     |
                                 |  Proposals, pgvector)   |
                                 +-------------------------+
```

---

## Agent Decision Boundaries

The system avoids monolithic "do-everything" prompts by strictly enforcing boundaries around what each agent can and cannot do.

*   **Master Orchestrator**: 
    *   *DOES*: Evaluates the current state in the `SharedContext` and decides the next step. Routes control to sub-agents.
    *   *BOUNDARY*: Does not answer the user's query itself. It only routes and manages the budget.
*   **Decomposition Agent**:
    *   *DOES*: Breaks complex queries into independent, manageable sub-tasks.
    *   *BOUNDARY*: Does not execute the sub-tasks or search for answers.
*   **RAG Agent**:
    *   *DOES*: Uses tools (like web search or vector DB lookups) to find factual information and cite sources.
    *   *BOUNDARY*: Does not synthesize the final answer; it only populates the scratchpad with raw, cited data.
*   **Critique Agent**:
    *   *DOES*: Reviews proposed answers or logic against the retrieved facts, flagging contradictions, logical gaps, or tone issues.
    *   *BOUNDARY*: Does not rewrite the answer. It only produces structured flags/reviews for the Synthesis agent to address.
*   **Synthesis Agent**:
    *   *DOES*: Combines retrieved facts, resolves critique flags, and generates the final, coherent response to the user.
    *   *BOUNDARY*: Does not look up new information. If information is missing, it must wait for the Orchestrator to route back to RAG.
*   **Compression Agent** (Implicit/Background):
    *   *DOES*: Summarizes older chat history or scratchpad data when the token budget nears its limit.
    *   *BOUNDARY*: Operates transparently; does not interact with the user or influence reasoning direction directly.
*   **Meta-Agent (Evals)**:
    *   *DOES*: Analyzes failed test cases in the `EvalRun` database and proposes structured line-by-line prompt rewrites for underperforming sub-agents.
    *   *BOUNDARY*: Does not automatically deploy changes.

---

## Honest Limitations

To evaluate this system accurately in a production setting, one must understand its failure modes:

1.  **API Rate Limits & Hallucinations**: Despite using `tenacity` for robust retries, the system is fundamentally bound by upstream LLM providers. Free-tier API rate limits (HTTP 429) will cause latency spikes. Furthermore, while the RAG and Critique agents reduce hallucinations, they do not eliminate them, especially on highly ambiguous "baseline" queries where the Orchestrator might skip RAG entirely.
2.  **Lossy Context Compression**: The dynamic context compression mechanism is a trade-off. When the token budget is heavily constrained, the system summarizes the `SharedContext`. This summarization is inherently lossy; highly nuanced logical steps or specific sub-task constraints from early in the sequence might be dropped, leading to degraded Synthesis performance.
3.  **The Critique-Synthesis Loop**: In adversarial edge cases or instances of conflicting retrieved facts, the Critique agent may flag an issue that the Synthesis agent cannot resolve without new data. If the Orchestrator fails to route back to RAG, the system risks entering a stalled "disagreement loop" until the maximum turn count (10) is hit, resulting in a suboptimal or truncated response.

---

## The Self-Improving Loop

The Evaluation Harness and Meta-Agent form a closed-loop system for continuous improvement, completely avoiding black-box third-party frameworks.

**What it DOES:**
*   Runs a deterministic suite of 15 test cases (baseline, ambiguous, adversarial) through the full orchestration pipeline.
*   Scores outputs across 4 custom dimensions: Correctness (LLM-as-a-judge), Citation Accuracy (pure Python), Tool Efficiency, and Critique Agreement.
*   Analyzes the worst-performing test cases via the `MetaAgent`.
*   Generates highly specific, line-by-line `PromptRewriteProposal` diffs targeting the specific sub-agent responsible for the failure.
*   Queues these proposals in the database with a `pending` status.

**What it DOES NOT do:**
*   **Does NOT auto-apply**: No prompt is ever modified without human approval via the `/api/v1/evals/rewrites/{id}/approve` endpoint.
*   **Does NOT rewrite Python code**: The Meta-Agent only mutates the `SYSTEM_PROMPT` strings, never the execution logic.
*   **Does NOT invent tools**: It cannot grant itself new capabilities or alter schemas.

---

## What's Next (Pragmatic Scaling)

To transition this from a robust prototype to a high-scale production environment, the following steps are recommended:

1.  **Fine-Tuned Classification for Orchestration**: Replace the expensive, high-latency LLM call in the Master Orchestrator with a fast, fine-tuned embedding classifier (e.g., a lightweight BERT model) trained on historical routing logs. This would drastically reduce latency and cost for the routing step.
2.  **Semantic Caching**: Implement a Redis-backed semantic cache. If a user asks a query conceptually identical to a previously solved job (measured by vector similarity), the system should bypass orchestration entirely and serve the cached Synthesis output, saving thousands of tokens.
3.  **Asynchronous Task Queues (Celery/Redis)**: Currently, the background tasks and evaluation harnesses are tied to the FastAPI event loop. Moving heavy processing (like the full 15-case Eval Suite) to a dedicated Celery worker pool will prevent the API from blocking under high concurrent load.
