"""Search endpoints - text, image, and multimodal search."""

import logging
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, Depends
from starlette.concurrency import run_in_threadpool

from backend.dependencies import get_search_engine
from backend.models.schemas import SearchResponse, TextSearchRequest, MediaItemResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Search"])


@router.post("/api/v1/search/text", response_model=SearchResponse)
async def search_by_text(
    request: TextSearchRequest,
    engine=Depends(get_search_engine)
):
    """
    Search for media using natural language.

    Supports hybrid search that combines:
    - Semantic vector similarity
    - Full-text search on descriptions (weighted 1.1x by default)
    - Filename matching

    Examples:
    - "a sunset over the ocean"
    - "people playing basketball"
    - "red car on a mountain road"

    **Webflow Integration:**
    Call this from your Webflow site with a simple fetch request.
    """
    result = await run_in_threadpool(
        engine.search_by_text,
        request.query,
        request.limit,
        request.min_similarity,
        request.file_type,
        request.use_hybrid,
        request.search_filenames,
        request.description_weight,
    )

    return SearchResponse(
        items=[MediaItemResponse(**item.__dict__) for item in result.items],
        query=result.query,
        query_type=result.query_type,
        total_results=result.total_results,
        processing_time_ms=result.processing_time_ms
    )


@router.get("/api/v1/search/text", response_model=SearchResponse)
async def search_by_text_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    file_type: Optional[str] = Query(None),
    search_filenames: bool = Query(True, description="Include filename matching"),
    description_weight: float = Query(1.1, ge=0.0, le=3.0, description="Weight for description matches"),
    engine=Depends(get_search_engine)
):
    """
    Search for media using natural language (GET method).

    Supports hybrid search with:
    - Semantic vector similarity
    - Full-text description search (weighted 1.1x by default)
    - Filename matching

    Convenient for simple integrations and browser testing.
    """
    result = await run_in_threadpool(
        engine.search_by_text,
        q,
        limit,
        0.0,
        file_type,
        True,
        search_filenames,
        description_weight,
    )

    return SearchResponse(
        items=[MediaItemResponse(**item.__dict__) for item in result.items],
        query=result.query,
        query_type=result.query_type,
        total_results=result.total_results,
        processing_time_ms=result.processing_time_ms
    )


@router.post("/api/v1/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="Query image"),
    limit: int = Form(20, ge=1, le=100),
    file_type: Optional[str] = Form(None),
    engine=Depends(get_search_engine)
):
    """
    Search for visually similar media using an image.

    Upload an image to find similar images and videos in your collection.
    """
    try:
        content = await file.read()

        result = await run_in_threadpool(
            engine.search_by_image,
            content,
            limit,
            0.0,
            file_type,
        )

        return SearchResponse(
            items=[MediaItemResponse(**item.__dict__) for item in result.items],
            query=result.query,
            query_type=result.query_type,
            total_results=result.total_results,
            processing_time_ms=result.processing_time_ms
        )

    except Exception as e:
        logger.error(f"Image search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/search/combined", response_model=SearchResponse)
async def search_combined(
    file: Optional[UploadFile] = File(None, description="Optional query image"),
    text_query: Optional[str] = Form(None, description="Optional text query"),
    text_weight: float = Form(0.3, ge=0, le=1),
    image_weight: float = Form(0.7, ge=0, le=1),
    limit: int = Form(20, ge=1, le=100),
    file_type: Optional[str] = Form(None),
    engine=Depends(get_search_engine)
):
    """
    Search using both text and image (multimodal search).

    Combine the power of visual similarity and semantic text matching.
    """
    try:
        image_data = None
        if file:
            image_data = await file.read()

        if not text_query and not image_data:
            raise HTTPException(
                status_code=400,
                detail="At least one of text_query or image file must be provided"
            )

        result = await run_in_threadpool(
            engine.search_combined,
            text_query,
            image_data,
            text_weight,
            image_weight,
            limit,
            0.0,
            file_type,
        )

        return SearchResponse(
            items=[MediaItemResponse(**item.__dict__) for item in result.items],
            query=result.query,
            query_type=result.query_type,
            total_results=result.total_results,
            processing_time_ms=result.processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combined search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
