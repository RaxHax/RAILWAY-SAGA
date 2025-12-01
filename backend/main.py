"""
Main FastAPI Application - Modular structure for the Media Semantic Search Engine.
This file replaces the monolithic api.py with a clean, organized structure.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.logging_config import setup_logging, LOG_FILE_PATH
from backend.dependencies import initialize_search_engine, search_engine
from backend.routers import health, search, media, configuration

# Setup logging
logger = logging.getLogger(__name__)
logger.info("File logging active: %s", LOG_FILE_PATH)

# Paths
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


# Cache control middleware for read-only API optimization
class CacheControlMiddleware(BaseHTTPMiddleware):
    """Add cache headers to responses for better performance."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add cache headers for GET requests
        if request.method == "GET":
            path = request.url.path

            # Cache static assets aggressively (1 year)
            if path.startswith("/assets/") or path.endswith((".js", ".css", ".png", ".jpg", ".ico")):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

            # Cache API responses for 5 minutes
            elif path.startswith("/api/v1/"):
                # Stats and configuration can be cached longer
                if "stats" in path or "models" in path or "schema" in path:
                    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=600"
                # Media listings can be cached
                elif "/media" in path and "search" not in path:
                    response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=120"
                # Search results cached briefly
                elif "/search" in path:
                    response.headers["Cache-Control"] = "public, max-age=30, stale-while-revalidate=60"

            # Add ETag support for better caching (only for non-streaming responses)
            if isinstance(response, Response) and hasattr(response, 'body') and not isinstance(response, StreamingResponse):
                try:
                    response.headers["ETag"] = f'W/"{hash(str(response.body))}"'
                except (AttributeError, TypeError):
                    # Skip ETag generation if body is not accessible
                    pass

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initialize and cleanup resources."""
    # Startup
    logger.info("Initializing Media Search Engine...")
    initialize_search_engine()
    if search_engine is None:
        logger.warning("Supabase credentials not configured. Onboarding wizard is available.")

    yield

    # Shutdown
    logger.info("Shutting down Media Search Engine...")

    # Cleanup resources
    if search_engine is not None:
        try:
            # Clear embedding model from memory
            if hasattr(search_engine, 'embedding_service') and hasattr(search_engine.embedding_service, '_model'):
                if search_engine.embedding_service._model is not None:
                    logger.info("Clearing embedding model from memory...")
                    # Clear CUDA cache if using GPU
                    import torch
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    search_engine.embedding_service._model = None

            # Close database connections
            if hasattr(search_engine, 'database_service') and hasattr(search_engine.database_service, 'client'):
                logger.info("Closing database connections...")
                # Supabase client cleanup happens automatically, but we log it

            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.error("Error during resource cleanup: %s", e)


# Create FastAPI app
app = FastAPI(
    title="Media Semantic Search API",
    description="""
    A high-performance read-only semantic search API for images and videos.

    Features:
    - Lightning-fast semantic search using natural language
    - Find similar media using image queries
    - Optimized for maximum read performance with caching
    - Support for multiple AI models including multilingual options

    READ-ONLY API: Upload operations are disabled for security and performance.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Performance optimizations for LASER-FAST read operations
# 1. Cache control headers for aggressive caching
app.add_middleware(CacheControlMiddleware)

# 2. GZip compression for faster response times
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. Configure CORS for Webflow integration (optimized for read-only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Read-only: no PUT/DELETE/PATCH
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Include routers (READ-ONLY: ingestion, setup, and admin removed)
app.include_router(health.router)
app.include_router(search.router)
app.include_router(media.router)
app.include_router(configuration.router)

# Serve frontend if available
if FRONTEND_DIST.exists():
    logger.info("Serving built frontend from %s", FRONTEND_DIST)
    app.mount(
        "/app",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="frontend-app",
    )

    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        logger.info("Serving frontend assets from %s", assets_dir)
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir, html=False),
            name="frontend-assets",
        )
    else:
        logger.warning("Frontend assets directory missing: %s", assets_dir)

    @app.get("/app", include_in_schema=False)
    async def serve_frontend_root():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/app/{path:path}", include_in_schema=False)
    async def serve_frontend_path(path: str):
        requested = FRONTEND_DIST / path
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")

    favicon_path = FRONTEND_DIST / "favicon.ico"

    if favicon_path.exists():

        @app.get("/favicon.ico", include_in_schema=False)
        async def serve_favicon():
            return FileResponse(favicon_path)


# Run with: uvicorn backend.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
