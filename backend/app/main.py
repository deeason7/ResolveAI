"""
FastAPI application factory.

Using the factory pattern (a function that creates and returns the app)
rather than a module-level app = FastAPI() makes it easy to create
isolated test instances with different settings — just call create_app()
with different config overrides.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis import exceptions as redis_errors
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import analytics, auth, complaints, graph, llmops, resolutions, workspace
from app.config import settings
from app.core.deps import get_default_redis
from app.middleware.rate_limit import limiter
from app.services.graph_store import get_default_graph_store

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks.

    Nothing to warm on startup — Postgres/Redis/Qdrant/Neo4j all connect lazily
    on first use, which is how the rest of the app treats its backing services.
    On shutdown we close the Neo4j driver's connection pool, but only if it was
    ever opened: checking cache_info() avoids instantiating a driver (and a
    socket) purely to tear it down in a process that never touched the graph.
    """
    yield
    if get_default_graph_store.cache_info().currsize:
        await get_default_graph_store().close()
        logger.info("closed Neo4j driver")
    if get_default_redis.cache_info().currsize:
        await get_default_redis().aclose()
        logger.info("closed Redis client")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResolveAI",
        description="Intelligent Complaint Resolution Engine",
        version="1.4.1",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's default error includes submitted input (including passwords).
        # Locations and messages are enough for clients to correct their request.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()
                ]
            },
        )

    # Rate limiting
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    def rate_limit_storage_error(request: Request, exc: Exception) -> JSONResponse:
        # SlowAPI's middleware looks up the exact exception type and requires
        # a synchronous handler. An unavailable limit store must not bypass it.
        logger.warning("Redis request limiting failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503, content={"detail": "Request limiting service temporarily unavailable"}
        )

    # SlowAPI looks up exact types instead of walking exception inheritance.
    # Include Redis subclasses such as authentication/OOM errors as well as
    # timeouts; its fallback assumes RateLimitExceeded and would itself fail.
    for error_type in vars(redis_errors).values():
        if isinstance(error_type, type) and issubclass(error_type, redis_errors.RedisError):
            app.add_exception_handler(error_type, rate_limit_storage_error)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})

    # CORS — tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"] if settings.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(complaints.router, prefix="/api/v1")
    app.include_router(analytics.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(llmops.router, prefix="/api/v1")
    app.include_router(resolutions.router, prefix="/api/v1")
    app.include_router(workspace.router, prefix="/api/v1")

    @app.get("/api/v1/health", tags=["infra"])
    async def health() -> dict:
        return {"status": "ok"}

    logger.info("ResolveAI API started (env=%s)", settings.environment)
    return app


app = create_app()
