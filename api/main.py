"""
SalesMind AI — FastAPI application entry point.

Run locally:
    uvicorn api.main:app --reload --port 8000

Production:
    uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, leads, results
from config.settings import settings

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

def _configure_logging() -> None:
    level = getattr(logging, settings.log_level, logging.INFO)

    if settings.log_format == "json":
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(level),
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "duckduckgo_search", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan (startup / shutdown)
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SalesMind AI starting up", extra={"model_fast": settings.model_fast, "model_deep": settings.model_deep})
    # Pre-warm pipeline (validates credentials early)
    try:
        from src.graph.pipeline import get_pipeline
        get_pipeline(use_checkpointing=True)
        logger.info("Pipeline initialised successfully")
    except Exception as exc:
        logger.warning("Pipeline pre-warm failed (will retry on first request)", extra={"error": str(exc)})

    # Run DB migrations on startup if DATABASE_URL is configured
    if settings.database_url:
        try:
            from src.db.database import init_db
            await init_db()
            logger.info("Database tables verified")
        except Exception as exc:
            logger.warning("DB init failed", extra={"error": str(exc)})
    yield
    logger.info("SalesMind AI shutting down")


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="SalesMind AI",
    description=(
        "Multi-agent B2B lead qualification pipeline powered by IBM Granite 4.1 "
        "and LangGraph with Granite Guardian safety guardrails."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 1),
        },
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", extra={"error": str(exc), "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Check server logs."},
    )


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #

app.include_router(health.router)
app.include_router(leads.router)
app.include_router(results.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,
        reload=True,
        log_level=settings.log_level.lower(),
    )
