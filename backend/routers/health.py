"""Health check endpoints."""

import os
from fastapi import APIRouter

from backend.dependencies import search_engine

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    """Health check endpoint."""
    supabase_configured = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))
    return {
        "status": "healthy",
        "service": "Media Semantic Search API (READ-ONLY)",
        "version": "1.0.0",
        "mode": "read-only",
        "engine_ready": search_engine is not None,
        "demo_mode": not supabase_configured,
        "message": "Set SUPABASE_URL and SUPABASE_KEY env vars to enable full functionality" if not supabase_configured else "Ready for search - optimized for maximum performance"
    }


@router.get("/api/v1/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "mode": "read-only",
        "engine_initialized": search_engine is not None,
        "database_connected": search_engine is not None,
        "demo_mode": search_engine is None,
        "supabase_configured": bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")),
        "optimizations": {
            "gzip_compression": True,
            "cache_headers": True,
            "cors_optimized": True
        },
        "setup_instructions": "Set SUPABASE_URL and SUPABASE_KEY environment variables to connect to your database." if search_engine is None else None
    }
