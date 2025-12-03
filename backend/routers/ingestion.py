"""Media ingestion endpoints - upload and index media files."""

import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

from backend.dependencies import get_search_engine
from backend.models.schemas import IngestionResponse, BatchPairUploadRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])


@router.post("/api/v1/media/upload", response_model=IngestionResponse)
async def upload_media(
    file: UploadFile = File(..., description="Image file to upload"),
    description: Optional[str] = Form(None, description="Optional text description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    engine=Depends(get_search_engine)
):
    """
    Upload and index an image file.

    The file will be processed to generate AI embeddings and stored in the database.
    Supports JPG, PNG, GIF, WebP, BMP images.

    **Webflow Integration:**
    Use this endpoint with Webflow's form submissions or custom JavaScript.
    """
    try:
        # Read file content
        content = await file.read()

        # Parse tags
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        # Ingest media
        result = await engine.ingest_media(
            file_data=content,
            filename=file.filename,
            description=description,
            tags=tag_list
        )

        return IngestionResponse(
            success=result.success,
            media_id=result.media_id,
            storage_url=result.storage_url,
            thumbnail_url=result.thumbnail_url,
            error=result.error
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/media/upload/batch")
async def upload_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    engine=Depends(get_search_engine)
):
    """
    Upload multiple image files at once.

    Returns a list of ingestion results for each file.
    """
    results = []

    for file in files:
        try:
            content = await file.read()
            result = await engine.ingest_media(
                file_data=content,
                filename=file.filename
            )
            results.append({
                "filename": file.filename,
                "success": result.success,
                "media_id": result.media_id,
                "error": result.error
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "total": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }


@router.post("/api/v1/media/upload/batch-pairs")
async def upload_batch_pairs(
    files: List[UploadFile] = File(default=[], description="Multiple files (images and their .txt descriptions)"),
    engine=Depends(get_search_engine)
):
    """
    Upload image files with paired text description files.

    This endpoint processes files in pairs where:
    - An image file (e.g., 'photo.png', 'sunset.jpg') is paired with
    - A text file with the same base name (e.g., 'photo.txt', 'sunset.txt')

    The text file content becomes the description for the corresponding image.
    Supported naming patterns:
    - photo.png + photo.txt
    - image.jpg + image.txt

    Files without a matching pair will be uploaded without a description.
    """
    # Validate files were provided
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Please select media files and optionally their .txt description files."
        )

    results = []

    # Organize files by base name
    file_map = {}
    text_contents = {}

    for file in files:
        filename = file.filename
        base_name = Path(filename).stem
        ext = Path(filename).suffix.lower()

        if ext == '.txt':
            # Read text file content
            content = await file.read()
            text_contents[base_name] = content.decode('utf-8', errors='ignore').strip()
        else:
            # Store media file reference
            file_map[base_name] = {
                'file': file,
                'filename': filename,
                'content': await file.read()
            }

    # Process media files with their paired descriptions
    for base_name, media_info in file_map.items():
        try:
            # Get description from paired text file if exists
            description = text_contents.get(base_name, None)

            # Parse metadata from description if present
            tags = None
            if description:
                # Extract any tags from metadata (look for common patterns)
                tags = _extract_tags_from_metadata(description)

            result = await engine.ingest_media(
                file_data=media_info['content'],
                filename=media_info['filename'],
                description=description,
                tags=tags
            )

            results.append({
                "filename": media_info['filename'],
                "success": result.success,
                "media_id": result.media_id,
                "has_description": description is not None,
                "description_preview": description[:100] + "..." if description and len(description) > 100 else description,
                "error": result.error
            })
        except Exception as e:
            results.append({
                "filename": media_info['filename'],
                "success": False,
                "has_description": base_name in text_contents,
                "error": str(e)
            })

    return {
        "total_media_files": len(file_map),
        "total_description_files": len(text_contents),
        "paired_count": sum(1 for r in results if r.get("has_description")),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }


@router.post("/api/v1/media/upload/folder")
async def upload_from_folder_path(
    request: BatchPairUploadRequest,
    engine=Depends(get_search_engine)
):
    """
    Upload all image files from a server-side folder path.

    Processes image+text file pairs where:
    - An image file (e.g., 'photo.png') is paired with 'photo.txt'
    - The text file content becomes the description

    **Note:** This endpoint reads from the server filesystem.
    Use /api/v1/media/upload/batch-pairs for client-side uploads.
    """
    folder = Path(request.folder_path)

    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")

    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.folder_path}")

    results = []
    supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

    # Find all media files
    media_files = [f for f in folder.iterdir() if f.suffix.lower() in supported_extensions]

    for media_file in media_files:
        try:
            # Look for paired text file
            txt_file = media_file.with_suffix('.txt')
            description = None
            tags = None

            if txt_file.exists():
                description = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
                tags = _extract_tags_from_metadata(description)

            # Read media file
            with open(media_file, 'rb') as f:
                file_data = f.read()

            result = await engine.ingest_media(
                file_data=file_data,
                filename=media_file.name,
                description=description,
                tags=tags
            )

            results.append({
                "filename": media_file.name,
                "success": result.success,
                "media_id": result.media_id,
                "has_description": description is not None,
                "description_file": txt_file.name if description else None,
                "error": result.error
            })

        except Exception as e:
            results.append({
                "filename": media_file.name,
                "success": False,
                "error": str(e)
            })

    return {
        "folder": str(folder),
        "total_files": len(media_files),
        "paired_count": sum(1 for r in results if r.get("has_description")),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }


def _extract_tags_from_metadata(description: str) -> Optional[List[str]]:
    """
    Extract tags from metadata-rich descriptions.

    Looks for common metadata patterns like:
    - Tags: tag1, tag2, tag3
    - Keywords: keyword1, keyword2
    - Categories: cat1, cat2
    - Hashtags: #tag1 #tag2
    """
    tags = []

    # Look for explicit tag patterns
    tag_patterns = [
        r'(?:tags?|keywords?|categories?)\s*[:=]\s*([^\n]+)',
        r'#(\w+)'
    ]

    for pattern in tag_patterns:
        matches = re.findall(pattern, description, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str):
                # Split by common delimiters
                parts = re.split(r'[,;|]', match)
                tags.extend([t.strip().lower() for t in parts if t.strip()])

    # Remove duplicates and limit
    tags = list(dict.fromkeys(tags))[:10]

    return tags if tags else None
