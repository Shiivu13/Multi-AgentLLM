"""
Server-Sent Events (SSE) streaming utility.

Wraps any async generator of dicts into properly formatted SSE lines:
    data: {"type": "agent_activity", "agent": "rag", "status": "searching"}\n\n
    data: {"type": "token", "content": "hello"}\n\n
    data: {"type": "done"}\n\n

The [done] type sentinel tells the client the stream has ended cleanly.
"""

import json

import asyncio
from typing import AsyncGenerator, Any

async def sse_stream(orchestrator_coro: AsyncGenerator[dict[str, Any], None]) -> AsyncGenerator[str, None]:
    """Yield Server‑Sent Events (SSE) strings from an async generator.
    The caller provides a coroutine that yields dict events. Each dict is JSON‑encoded and
    prefixed with ``data: `` per SSE spec. When the upstream generator finishes we emit a
    final ``event: done`` sentinel so the client can cleanly close the connection.
    Errors inside the orchestrator are caught and transformed into a ``type: error``
    SSE payload, ensuring the stream never crashes unexpectedly.
    """
    try:
        async for event in orchestrator_coro:
            data_str = json.dumps(event)
            yield f"data: {data_str}\n\n"
    except Exception as e:
        error_event = {"type": "error", "content": str(e)}
        yield f"data: {json.dumps(error_event)}\n\n"
    finally:
        # Emit a terminal event indicating stream completion.
        done_event = {"type": "done"}
        yield f"data: {json.dumps(done_event)}\n\n"
