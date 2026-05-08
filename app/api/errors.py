import uuid
from fastapi import Request, status
from fastapi.responses import JSONResponse

class AppError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int = 400, job_id: uuid.UUID | None = None):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.job_id = job_id

async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "job_id": str(exc.job_id) if exc.job_id else None
        }
    )

async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred.",
            "job_id": None
        }
    )
