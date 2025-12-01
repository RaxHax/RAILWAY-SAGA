"""Media management endpoints - READ-ONLY (list and get only)."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from starlette.concurrency import run_in_threadpool

from backend.dependencies import get_search_engine
from backend.models.schemas import MediaItemResponse

router = APIRouter(tags=["Media"])


@router.get("/api/v1/media")
async def list_media(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    file_type: Optional[str] = Query(None),
    engine=Depends(get_search_engine)
):
    """
    List all media items with pagination.
    """
    items = await run_in_threadpool(
        engine.list_all_media,
        limit,
        offset,
        file_type,
    )

    total = await run_in_threadpool(
        engine.database_service.count_media_items,
        file_type,
    )

    return {
        "items": [MediaItemResponse(**item.__dict__) for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total
    }


@router.get("/api/v1/media/{media_id}", response_model=MediaItemResponse)
async def get_media(
    media_id: str,
    engine=Depends(get_search_engine)
):
    """
    Get a single media item by ID.
    """
    item = await run_in_threadpool(engine.get_media_item, media_id)

    if not item:
        raise HTTPException(status_code=404, detail="Media item not found")

    return MediaItemResponse(**item.__dict__)


# PUT and DELETE operations removed - READ-ONLY API
