"""Configuration endpoints - models, schema, stats."""

import os
from fastapi import APIRouter, Depends

from backend.dependencies import get_optional_search_engine
from backend.models.schemas import StatsResponse, ModelsResponse, AvailableModel
from config.settings import ModelConfig

router = APIRouter(tags=["Configuration"])


@router.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats(engine=Depends(get_optional_search_engine)):
    """
    Get statistics about the media collection.
    Returns demo data if Supabase is not configured.
    """
    if engine is None:
        # Return demo stats when not connected to Supabase
        return StatsResponse(
            total_items=0,
            images=0,
            videos=0,
            embedding_model="clip-vit-base-patch32 (demo mode)",
            embedding_dim=512
        )

    stats = engine.get_stats()
    return StatsResponse(**stats)


@router.get("/api/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List available AI models for embedding generation.
    """
    models = []
    for key, info in ModelConfig.AVAILABLE_MODELS.items():
        models.append(AvailableModel(
            key=key,
            type=info["type"],
            name=info["name"],
            embedding_dim=info["embedding_dim"],
            multilingual=info.get("multilingual", False),
            description=info["description"],
            languages=info.get("languages")
        ))

    current = os.environ.get("MODEL_KEY", "clip-vit-base-patch32")

    return ModelsResponse(
        current_model=current,
        available_models=models
    )


@router.get("/api/v1/schema")
async def get_database_schema():
    """
    Get the SQL schema for setting up the database.

    Copy this SQL and run it in your Supabase SQL editor.
    """
    from backend.database_service import SupabaseService

    vector_dim = int(os.environ.get("VECTOR_DIMENSION", 512))
    service = SupabaseService(
        url="placeholder",
        key="placeholder",
        vector_dimension=vector_dim
    )

    return {
        "schema_sql": service.get_schema_sql(),
        "instructions": [
            "1. Go to your Supabase project dashboard",
            "2. Navigate to SQL Editor",
            "3. Create a new query and paste the schema SQL",
            "4. Run the query to create all required tables and functions",
            "5. Go to Storage and create a bucket named 'media-files'",
            "6. Set the bucket to public if you want direct URLs"
        ]
    }
