"""Setup and onboarding endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from backend.dependencies import (
    config_manager,
    search_engine,
    refresh_search_engine,
    _test_supabase_connection
)
from backend.models.schemas import (
    SetupStatusResponse,
    SupabaseCredentialsRequest,
    SetupActionResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Setup"])


@router.get("/api/v1/setup/status", response_model=SetupStatusResponse)
async def get_setup_status():
    """
    Return onboarding status so the desktop wizard knows what to show.
    """
    status = config_manager.get_status(engine_ready=search_engine is not None)
    return status


@router.post("/api/v1/setup/test-connection", response_model=SetupActionResponse)
async def test_supabase_connection(request: SupabaseCredentialsRequest):
    """
    Validate Supabase credentials before saving them.
    """
    try:
        _test_supabase_connection(request.supabase_url, request.supabase_key, request.model_key)
        return SetupActionResponse(
            success=True,
            message="Connection successful! Schema and credentials look good.",
            engine_ready=search_engine is not None,
        )
    except Exception as exc:
        logger.error("Supabase connection test failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Supabase connection failed: {exc}",
        )


@router.post("/api/v1/setup/credentials", response_model=SetupActionResponse)
async def save_supabase_credentials(request: SupabaseCredentialsRequest):
    """
    Persist Supabase credentials provided by the onboarding wizard and reinitialize the engine.
    """
    try:
        # Basic validation before we save anything locally
        _test_supabase_connection(request.supabase_url, request.supabase_key, request.model_key)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Supabase validation failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not validate Supabase credentials: {exc}",
        )

    try:
        config_manager.update_supabase_credentials(
            url=str(request.supabase_url),
            key=request.supabase_key,
            model_key=request.model_key,
        )
        config_manager.mark_step_complete(4)
        engine = refresh_search_engine()
        if engine is None:
            raise RuntimeError("Search engine could not be initialized with the provided credentials.")

        return SetupActionResponse(
            success=True,
            message="Supabase connected and search engine is ready!",
            engine_ready=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to save Supabase credentials: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize the search engine with the provided credentials.",
        )
