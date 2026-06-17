from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import settings
from app.core.database import engine, Base
# Import models to ensure they are registered with the metadata before creation
from app.models import logging, evals
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers.main_router import router as main_router
from app.api.errors import AppError, app_error_handler, global_exception_handler

STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event: create all tables automatically
    async with engine.begin() as conn:
        # In a real-world scenario, we would use Alembic migrations instead.
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown event: cleanly dispose of the engine connections
    await engine.dispose()

app = FastAPI(
    title="Real-Time Multi-Agent LLM Orchestration and Evaluation System",
    description="Backend for multi-agent LLM orchestration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(main_router)

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))
