"""Media management endpoints - list, get, update, delete."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends

from backend.dependencies import get_search_engine
from backend.models.schemas import MediaItemResponse, UpdateDescriptionRequest

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
    items = engine.list_all_media(
        limit=limit,
        offset=offset,
        file_type=file_type
    )

    total = engine.database_service.count_media_items(file_type=file_type)

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
    item = engine.get_media_item(media_id)

    if not item:
        raise HTTPException(status_code=404, detail="Media item not found")

    return MediaItemResponse(**item.__dict__)


@router.put("/api/v1/media/{media_id}/description")
async def update_description(
    media_id: str,
    request: UpdateDescriptionRequest,
    engine=Depends(get_search_engine)
):
    """
    Update the description of a media item.

    This will regenerate the text embedding and combined embedding.
    """
    success = engine.update_description(media_id, request.description)

    if not success:
        raise HTTPException(status_code=404, detail="Media item not found or update failed")

    return {"success": True, "message": "Description updated successfully"}


@router.delete("/api/v1/media/{media_id}")
async def delete_media(
    media_id: str,
    engine=Depends(get_search_engine)
):
    """
    Delete a media item.

    This will remove the file from storage and the database record.
    """
    success = engine.delete_media_item(media_id)

    if not success:
        raise HTTPException(status_code=404, detail="Media item not found or deletion failed")

    return {"success": True, "message": "Media item deleted successfully"}
