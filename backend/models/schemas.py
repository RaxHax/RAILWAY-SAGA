"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

from config.user_config import DEFAULT_MODEL_KEY


class MediaItemResponse(BaseModel):
    """Response model for a media item."""
    id: str
    filename: str
    file_type: str
    storage_url: str
    thumbnail_url: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = []
    similarity: float = 0.0
    metadata: dict = {}
    created_at: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search results."""
    items: List[MediaItemResponse]
    query: str
    query_type: str
    total_results: int
    processing_time_ms: float


class IngestionResponse(BaseModel):
    """Response model for media ingestion."""
    success: bool
    media_id: Optional[str] = None
    storage_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """Response model for collection statistics."""
    total_items: int
    images: int
    videos: int
    embedding_model: str
    embedding_dim: int


class TextSearchRequest(BaseModel):
    """Request model for text-based search."""
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    file_type: Optional[str] = Field(None, description="Filter by 'image' or 'video'")
    use_hybrid: bool = Field(True, description="Use hybrid search (vector + text)")
    search_filenames: bool = Field(True, description="Include filename matching in search")
    description_weight: float = Field(1.1, ge=0.0, le=3.0, description="Weight multiplier for description matches")


class UpdateDescriptionRequest(BaseModel):
    """Request model for updating description."""
    description: str = Field(..., description="New description text")


class AvailableModel(BaseModel):
    """Information about an available embedding model."""
    key: str
    type: str
    name: str
    embedding_dim: int
    multilingual: bool
    description: str
    languages: Optional[List[str]] = None


class ModelsResponse(BaseModel):
    """Response model for available models."""
    current_model: str
    available_models: List[AvailableModel]


class SetupStatusResponse(BaseModel):
    """Setup/onboarding wizard status."""
    configured: bool
    engine_ready: bool
    has_supabase_credentials: bool
    supabase_url: Optional[str] = None
    supabase_url_masked: Optional[str] = None
    supabase_key_masked: Optional[str] = None
    model_key: str
    instructions: List[Dict[str, Any]]
    next_step: int


class SupabaseCredentialsRequest(BaseModel):
    """Payload for supplying Supabase credentials from the UI wizard."""
    supabase_url: HttpUrl
    supabase_key: str = Field(..., min_length=16)
    model_key: str = Field(default=DEFAULT_MODEL_KEY)


class SetupActionResponse(BaseModel):
    """Generic response for setup actions."""
    success: bool
    message: str
    engine_ready: bool


class BatchPairUploadRequest(BaseModel):
    """Request model for batch upload with paired files."""
    folder_path: str = Field(..., description="Path to folder containing image+text file pairs")


class WipeDatabaseRequest(BaseModel):
    """Request model for wiping the database."""
    confirmation: str = Field(
        ...,
        description="Must be exactly 'ég vill eyða fokking öllu' to confirm deletion"
    )


class WipeDatabaseResponse(BaseModel):
    """Response model for database wipe operation."""
    success: bool
    message: str
    deleted_records: int = 0
    deleted_files: int = 0
    errors: Optional[List[str]] = None
