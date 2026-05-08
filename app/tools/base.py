from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, Any, Optional, Dict
from pydantic import BaseModel, Field

class FailureReason(str, Enum):
    TIMEOUT = "timeout"
    EMPTY_RESULT = "empty_result"
    MALFORMED_INPUT = "malformed_input"
    UNKNOWN_ERROR = "unknown_error"

class FailureContract(BaseModel):
    is_failure: bool = True
    reason: FailureReason = Field(..., description="The category of failure.")
    message: str = Field(..., description="Detailed error message.")
    original_input: Dict[str, Any] = Field(..., description="The input that caused the failure.")

class ToolResult(BaseModel):
    is_success: bool
    data: Optional[Any] = None
    failure: Optional[FailureContract] = None

class BaseTool(ABC):
    """
    Abstract base class for all tools with strict failure contracts.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        """Pydantic model representing the expected input."""
        pass

    @abstractmethod
    async def _run(self, input_data: BaseModel) -> Any:
        """Internal execution logic to be implemented by subclasses."""
        pass

    async def execute(self, input_data: dict) -> ToolResult:
        """
        Public execution wrapper. Enforces the input schema and the FailureContract.
        Catches all exceptions to strictly return structured error states instead of crashing.
        """
        try:
            parsed_input = self.input_schema(**input_data)
        except Exception as e:
            return ToolResult(
                is_success=False,
                failure=FailureContract(
                    reason=FailureReason.MALFORMED_INPUT,
                    message=f"Input validation failed against schema: {str(e)}",
                    original_input=input_data
                )
            )

        try:
            result = await self._run(parsed_input)
            
            # Explicit check for empty result
            if result is None or (isinstance(result, (list, dict, str)) and len(result) == 0):
                return ToolResult(
                    is_success=False,
                    failure=FailureContract(
                        reason=FailureReason.EMPTY_RESULT,
                        message="Tool executed successfully but returned empty results.",
                        original_input=input_data
                    )
                )
                
            return ToolResult(is_success=True, data=result)
            
        except TimeoutError as e:
            return ToolResult(
                is_success=False,
                failure=FailureContract(
                    reason=FailureReason.TIMEOUT,
                    message=f"Tool execution timed out: {str(e)}",
                    original_input=input_data
                )
            )
        except Exception as e:
            return ToolResult(
                is_success=False,
                failure=FailureContract(
                    reason=FailureReason.UNKNOWN_ERROR,
                    message=f"Tool execution encountered an unknown error: {str(e)}",
                    original_input=input_data
                )
            )
