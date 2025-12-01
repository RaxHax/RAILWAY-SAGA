"""
Main FastAPI Application - Modular structure for the Media Semantic Search Engine.
This file replaces the monolithic api.py with a clean, organized structure.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.core.logging_config import setup_logging, LOG_FILE_PATH
from backend.dependencies import initialize_search_engine, search_engine
from backend.routers import health, ingestion, search, media, configuration, setup, admin

# Setup logging
logger = logging.getLogger(__name__)
logger.info("File logging active: %s", LOG_FILE_PATH)

# Paths
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


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


# Create FastAPI app
app = FastAPI(
    title="Media Semantic Search API",
    description="""
    A powerful semantic search engine for images and videos.

    Features:
    - Upload and index media files with AI-generated embeddings
    - Search using natural language descriptions
    - Find similar media using image queries
    - Support for multiple AI models including multilingual options

    Designed for easy integration with Webflow and other web platforms.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Webflow integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(media.router)
app.include_router(configuration.router)
app.include_router(setup.router)
app.include_router(admin.router)

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
