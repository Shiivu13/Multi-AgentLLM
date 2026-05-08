import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, TypeVar
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings

T = TypeVar("T", bound=BaseModel)

# Model: gemini-3.1-flash-lite (High quota 1500 RPD - the 2026 equivalent of 1.5 Flash)
MODEL_NAME = "gemini-3.1-flash-lite"


def _inline_refs(schema: dict) -> dict:
    """
    Recursively resolves $ref / $defs in a JSON Schema so that
    Gemini's JSON mode receives a flat, reference-free schema.
    """
    defs = schema.pop("$defs", {})

    def resolve(node: dict) -> dict:
        if "$ref" in node:
            ref_name = node["$ref"].split("/")[-1]
            return resolve(dict(defs.get(ref_name, {})))
        result = {}
        for k, v in node.items():
            if isinstance(v, dict):
                result[k] = resolve(v)
            elif isinstance(v, list):
                result[k] = [resolve(i) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        return result

    return resolve(schema)


class LLMClient:
    """
    Async client wrapping the new google-genai SDK.
    Uses JSON mime-type mode + Pydantic validation.
    Schema is embedded as prompt text to avoid $defs rejection.
    """

    def __init__(self):
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    async def generate_structured_output(self, prompt: str, schema: Type[T]) -> T:
        """
        Calls Gemini with JSON mode. Schema is embedded in the prompt
        to avoid $defs issues. Response is parsed and validated with Pydantic.
        """
        flat_schema = _inline_refs(schema.model_json_schema())
        schema_hint = json.dumps(flat_schema, indent=2)

        augmented_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with a valid JSON object that matches this schema:\n"
            f"```json\n{schema_hint}\n```"
        )

        response = await self._client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=augmented_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        try:
            content = response.text.strip()
            # Strip markdown code fences if the model wraps output in them
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else content
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            return schema.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse LLM response into {schema.__name__}: "
                f"{exc} | Raw: {response.text[:300]}"
            )


llm_client = LLMClient()
