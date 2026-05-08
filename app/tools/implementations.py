import asyncio
import subprocess
from pydantic import BaseModel
from typing import List, Dict, Any
from app.tools.base import BaseTool

class WebSearchInput(BaseModel):
    query: str

class WebSearchStub(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def input_schema(self) -> type[BaseModel]:
        return WebSearchInput

    async def _run(self, input_data: WebSearchInput) -> List[Dict[str, Any]]:
        # Explicitly fail if input query is empty
        if not input_data.query.strip():
            raise ValueError("Search query cannot be empty.")
        
        # Simulating structured results
        return [
            {
                "url": "https://example.com/result1",
                "title": "Example Mocked Result",
                "snippet": "This is a simulated web search snippet for the given query.",
                "relevance_score": 0.95
            }
        ]


class PythonCodeInput(BaseModel):
    code: str

class PythonCodeSandbox(BaseTool):
    @property
    def name(self) -> str:
        return "python_sandbox"

    @property
    def input_schema(self) -> type[BaseModel]:
        return PythonCodeInput

    async def _run(self, input_data: PythonCodeInput) -> Dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            "python", "-c", input_data.code,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            # Enforce strict 3-second timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3.0)
            return {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "exit_code": process.returncode
            }
        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError("Python code execution exceeded the strict 3-second limit.")


class NLtoSQLInput(BaseModel):
    query: str

class NLtoSQLDataLookup(BaseTool):
    @property
    def name(self) -> str:
        return "nl_to_sql"

    @property
    def input_schema(self) -> type[BaseModel]:
        return NLtoSQLInput

    async def _run(self, input_data: NLtoSQLInput) -> Dict[str, Any]:
        if not input_data.query.strip():
            raise ValueError("Natural language query cannot be empty.")
        
        return {
            "sql_generated": "SELECT * FROM mock_db WHERE query = 'mock';",
            "results": [{"id": 1, "value": "Local DB mock value"}]
        }


class SelfReflectionInput(BaseModel):
    context_dump: str

class SelfReflectionTool(BaseTool):
    @property
    def name(self) -> str:
        return "self_reflection"

    @property
    def input_schema(self) -> type[BaseModel]:
        return SelfReflectionInput

    async def _run(self, input_data: SelfReflectionInput) -> Dict[str, Any]:
        if not input_data.context_dump.strip():
            raise ValueError("Context dump cannot be empty.")
        
        return {
            "contradictions_found": [],
            "gaps_in_logic": ["Consider providing more concrete numerical evidence for claims."],
            "overall_assessment": "The reasoning is generally sound but lacks depth in certain areas."
        }
