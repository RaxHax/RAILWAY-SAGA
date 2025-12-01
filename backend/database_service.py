"""
Supabase Database Service - Handles vector storage, file uploads, and search operations.
Uses pgvector extension for efficient similarity search.
"""

import io
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from supabase import create_client, Client
from PIL import Image

logger = logging.getLogger(__name__)


class SupabaseService:
    """
    Service for interacting with Supabase for:
    - Vector embeddings storage (using pgvector)
    - Media file storage
    - Metadata management
    """
    
    def __init__(
        self,
        url: str,
        key: str,
        vector_dimension: int = 512,
        media_table: str = "media_items",
        storage_bucket: str = "media-files"
    ):
        self.url = url
        self.key = key
        self.vector_dimension = vector_dimension
        self.media_table = media_table
        self.storage_bucket = storage_bucket
        
        self._client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Lazy initialization of Supabase client."""
        if self._client is None:
            self._client = create_client(self.url, self.key)
            logger.info("Supabase client initialized")
        return self._client
    
    async def initialize_database(self) -> None:
        """
        Initialize the database schema.
        Note: This should be run once during setup.
        The SQL needs to be executed in Supabase SQL editor.
        """
        schema_sql = self.get_schema_sql()
        logger.info("Database schema SQL generated. Please run in Supabase SQL editor.")
        return schema_sql
    
    def get_schema_sql(self) -> str:
        """Generate the SQL schema for the database."""
        return f"""
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Media items table
CREATE TABLE IF NOT EXISTS {self.media_table} (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    original_filename TEXT,
    file_type TEXT NOT NULL,  -- 'image' or 'video'
    mime_type TEXT,
    file_size INTEGER,
    storage_path TEXT NOT NULL,
    thumbnail_path TEXT,
    
    -- Optional text description
    description TEXT,
    tags TEXT[],
    
    -- Embeddings (using pgvector)
    visual_embedding vector({self.vector_dimension}),
    text_embedding vector({self.vector_dimension}),
    combined_embedding vector({self.vector_dimension}),
    
    -- Video-specific fields
    duration_seconds FLOAT,
    frame_count INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for vector similarity search
CREATE INDEX IF NOT EXISTS idx_visual_embedding 
ON {self.media_table} 
USING ivfflat (visual_embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_text_embedding 
ON {self.media_table} 
USING ivfflat (text_embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_combined_embedding 
ON {self.media_table} 
USING ivfflat (combined_embedding vector_cosine_ops)
WITH (lists = 100);

-- Text search index on description
CREATE INDEX IF NOT EXISTS idx_description_search 
ON {self.media_table} 
USING gin(to_tsvector('english', COALESCE(description, '')));

-- Index on tags
CREATE INDEX IF NOT EXISTS idx_tags 
ON {self.media_table} 
USING gin(tags);

-- Index on file_type for filtering
CREATE INDEX IF NOT EXISTS idx_file_type 
ON {self.media_table}(file_type);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for updated_at
DROP TRIGGER IF EXISTS update_media_items_updated_at ON {self.media_table};
CREATE TRIGGER update_media_items_updated_at
    BEFORE UPDATE ON {self.media_table}
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function for vector similarity search
CREATE OR REPLACE FUNCTION search_media_by_embedding(
    query_embedding vector({self.vector_dimension}),
    search_type TEXT DEFAULT 'combined',  -- 'visual', 'text', or 'combined'
    match_threshold FLOAT DEFAULT 0.0,
    match_count INT DEFAULT 20,
    file_type_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    filename TEXT,
    file_type TEXT,
    storage_path TEXT,
    thumbnail_path TEXT,
    description TEXT,
    tags TEXT[],
    similarity FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF search_type = 'visual' THEN
        RETURN QUERY
        SELECT 
            m.id,
            m.filename,
            m.file_type,
            m.storage_path,
            m.thumbnail_path,
            m.description,
            m.tags,
            1 - (m.visual_embedding <=> query_embedding) as similarity,
            m.metadata,
            m.created_at
        FROM {self.media_table} m
        WHERE m.visual_embedding IS NOT NULL
            AND (file_type_filter IS NULL OR m.file_type = file_type_filter)
            AND 1 - (m.visual_embedding <=> query_embedding) >= match_threshold
        ORDER BY m.visual_embedding <=> query_embedding
        LIMIT match_count;
    ELSIF search_type = 'text' THEN
        RETURN QUERY
        SELECT 
            m.id,
            m.filename,
            m.file_type,
            m.storage_path,
            m.thumbnail_path,
            m.description,
            m.tags,
            1 - (m.text_embedding <=> query_embedding) as similarity,
            m.metadata,
            m.created_at
        FROM {self.media_table} m
        WHERE m.text_embedding IS NOT NULL
            AND (file_type_filter IS NULL OR m.file_type = file_type_filter)
            AND 1 - (m.text_embedding <=> query_embedding) >= match_threshold
        ORDER BY m.text_embedding <=> query_embedding
        LIMIT match_count;
    ELSE
        RETURN QUERY
        SELECT 
            m.id,
            m.filename,
            m.file_type,
            m.storage_path,
            m.thumbnail_path,
            m.description,
            m.tags,
            1 - (m.combined_embedding <=> query_embedding) as similarity,
            m.metadata,
            m.created_at
        FROM {self.media_table} m
        WHERE m.combined_embedding IS NOT NULL
            AND (file_type_filter IS NULL OR m.file_type = file_type_filter)
            AND 1 - (m.combined_embedding <=> query_embedding) >= match_threshold
        ORDER BY m.combined_embedding <=> query_embedding
        LIMIT match_count;
    END IF;
END;
$$;

-- Hybrid search function (combines vector + text search) - Legacy version
CREATE OR REPLACE FUNCTION hybrid_search_media(
    query_embedding vector({self.vector_dimension}),
    query_text TEXT DEFAULT NULL,
    vector_weight FLOAT DEFAULT 0.7,
    text_weight FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 20,
    file_type_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    filename TEXT,
    file_type TEXT,
    storage_path TEXT,
    thumbnail_path TEXT,
    description TEXT,
    tags TEXT[],
    similarity FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_scores AS (
        SELECT 
            m.id,
            1 - (m.combined_embedding <=> query_embedding) as v_score
        FROM {self.media_table} m
        WHERE m.combined_embedding IS NOT NULL
    ),
    text_scores AS (
        SELECT 
            m.id,
            CASE 
                WHEN query_text IS NOT NULL AND m.description IS NOT NULL 
                THEN ts_rank(to_tsvector('english', m.description), plainto_tsquery('english', query_text))
                ELSE 0
            END as t_score
        FROM {self.media_table} m
    )
    SELECT 
        m.id,
        m.filename,
        m.file_type,
        m.storage_path,
        m.thumbnail_path,
        m.description,
        m.tags,
        (COALESCE(vs.v_score, 0) * vector_weight + COALESCE(ts.t_score, 0) * text_weight) as similarity,
        m.metadata,
        m.created_at
    FROM {self.media_table} m
    LEFT JOIN vector_scores vs ON m.id = vs.id
    LEFT JOIN text_scores ts ON m.id = ts.id
    WHERE (file_type_filter IS NULL OR m.file_type = file_type_filter)
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Enhanced hybrid search function (combines vector + text + filename search)
-- Supports description weighting for metadata-rich content
CREATE OR REPLACE FUNCTION hybrid_search_media_enhanced(
    query_embedding vector({self.vector_dimension}),
    query_text TEXT DEFAULT NULL,
    vector_weight FLOAT DEFAULT 0.7,
    text_weight FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 20,
    file_type_filter TEXT DEFAULT NULL,
    search_filenames BOOLEAN DEFAULT TRUE,
    description_weight FLOAT DEFAULT 1.1
)
RETURNS TABLE (
    id UUID,
    filename TEXT,
    file_type TEXT,
    storage_path TEXT,
    thumbnail_path TEXT,
    description TEXT,
    tags TEXT[],
    similarity FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
DECLARE
    -- Normalize text weight with description multiplier
    effective_text_weight FLOAT := text_weight * description_weight;
    filename_weight FLOAT := 0.15; -- Base weight for filename matches
BEGIN
    RETURN QUERY
    WITH vector_scores AS (
        -- Semantic vector similarity score
        SELECT 
            m.id,
            1 - (m.combined_embedding <=> query_embedding) as v_score
        FROM {self.media_table} m
        WHERE m.combined_embedding IS NOT NULL
    ),
    text_scores AS (
        -- Full-text search on description with configurable weighting
        -- Handles metadata-rich descriptions by using weighted ranking
        SELECT 
            m.id,
            CASE 
                WHEN query_text IS NOT NULL AND m.description IS NOT NULL THEN
                    ts_rank_cd(
                        setweight(to_tsvector('english', COALESCE(m.description, '')), 'A'),
                        plainto_tsquery('english', query_text),
                        32  -- Normalization: divide by document length
                    )
                ELSE 0
            END as t_score
        FROM {self.media_table} m
    ),
    filename_scores AS (
        -- Filename matching score (for files named descriptively)
        SELECT 
            m.id,
            CASE 
                WHEN search_filenames AND query_text IS NOT NULL THEN
                    CASE
                        -- Exact match (case-insensitive, without extension)
                        WHEN LOWER(regexp_replace(m.original_filename, '\\.[^.]+$', '')) = LOWER(query_text) THEN 1.0
                        -- Filename contains query
                        WHEN LOWER(m.original_filename) LIKE '%' || LOWER(query_text) || '%' THEN 0.7
                        -- Query contains filename stem
                        WHEN LOWER(query_text) LIKE '%' || LOWER(regexp_replace(m.original_filename, '\\.[^.]+$', '')) || '%' THEN 0.5
                        -- Partial word match in filename
                        WHEN EXISTS (
                            SELECT 1 FROM unnest(string_to_array(
                                regexp_replace(LOWER(m.original_filename), '[^a-z0-9]+', ' ', 'g'), ' '
                            )) AS word
                            WHERE word != '' AND LOWER(query_text) LIKE '%' || word || '%'
                        ) THEN 0.3
                        ELSE 0
                    END
                ELSE 0
            END as f_score
        FROM {self.media_table} m
    ),
    tag_scores AS (
        -- Tag matching score
        SELECT 
            m.id,
            CASE 
                WHEN query_text IS NOT NULL AND m.tags IS NOT NULL AND array_length(m.tags, 1) > 0 THEN
                    (SELECT COUNT(*)::FLOAT / array_length(m.tags, 1)
                     FROM unnest(m.tags) AS tag
                     WHERE LOWER(tag) LIKE '%' || LOWER(query_text) || '%'
                        OR LOWER(query_text) LIKE '%' || LOWER(tag) || '%')
                ELSE 0
            END as tag_score
        FROM {self.media_table} m
    )
    SELECT 
        m.id,
        m.filename,
        m.file_type,
        m.storage_path,
        m.thumbnail_path,
        m.description,
        m.tags,
        -- Combined score with configurable weights
        -- Description gets extra weight via description_weight multiplier
        (
            COALESCE(vs.v_score, 0) * vector_weight + 
            COALESCE(ts.t_score, 0) * effective_text_weight +
            COALESCE(fs.f_score, 0) * filename_weight +
            COALESCE(tgs.tag_score, 0) * 0.1
        ) as similarity,
        m.metadata,
        m.created_at
    FROM {self.media_table} m
    LEFT JOIN vector_scores vs ON m.id = vs.id
    LEFT JOIN text_scores ts ON m.id = ts.id
    LEFT JOIN filename_scores fs ON m.id = fs.id
    LEFT JOIN tag_scores tgs ON m.id = tgs.id
    WHERE (file_type_filter IS NULL OR m.file_type = file_type_filter)
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Storage bucket policy (run this separately if needed)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('media-files', 'media-files', true);
"""

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for Supabase Storage.
        
        Supabase storage keys cannot contain spaces or special characters.
        This converts them to safe ASCII equivalents.
        """
        import re
        import unicodedata
        
        # Get the extension
        from pathlib import Path
        path = Path(filename)
        ext = path.suffix.lower()
        stem = path.stem
        
        # Normalize unicode characters (é -> e, í -> i, etc.)
        stem = unicodedata.normalize('NFKD', stem)
        stem = stem.encode('ascii', 'ignore').decode('ascii')
        
        # Replace spaces and special chars with underscores
        stem = re.sub(r'[^\w\-]', '_', stem)
        
        # Collapse multiple underscores
        stem = re.sub(r'_+', '_', stem)
        
        # Remove leading/trailing underscores
        stem = stem.strip('_')
        
        # Ensure we have something left
        if not stem:
            stem = "file"
        
        return f"{stem}{ext}"
    
    def upload_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: str = "image/jpeg"
    ) -> str:
        """
        Upload a file to Supabase Storage.
        
        Args:
            file_data: File content as bytes
            filename: Target filename
            content_type: MIME type of the file
            
        Returns:
            Storage path of the uploaded file
        """
        # Sanitize filename to remove spaces and special characters
        safe_filename = self._sanitize_filename(filename)
        storage_path = f"uploads/{uuid.uuid4()}_{safe_filename}"
        
        self.client.storage.from_(self.storage_bucket).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": content_type}
        )
        
        logger.info(f"File uploaded to: {storage_path}")
        return storage_path
    
    def get_file_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Get a signed URL for a stored file."""
        response = self.client.storage.from_(self.storage_bucket).create_signed_url(
            storage_path, expires_in
        )
        return response["signedURL"]
    
    def get_public_url(self, storage_path: str) -> str:
        """Get a public URL for a stored file (if bucket is public)."""
        return self.client.storage.from_(self.storage_bucket).get_public_url(storage_path)
    
    def delete_file(self, storage_path: str) -> None:
        """Delete a file from storage."""
        self.client.storage.from_(self.storage_bucket).remove([storage_path])
        logger.info(f"File deleted: {storage_path}")
    
    def create_media_item(
        self,
        filename: str,
        file_type: str,
        storage_path: str,
        visual_embedding: np.ndarray,
        text_embedding: Optional[np.ndarray] = None,
        combined_embedding: Optional[np.ndarray] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        thumbnail_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        original_filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        file_size: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        frame_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new media item record in the database.
        
        Args:
            filename: Stored filename
            file_type: 'image' or 'video'
            storage_path: Path in storage bucket
            visual_embedding: Visual embedding vector
            text_embedding: Optional text embedding
            combined_embedding: Optional combined embedding
            description: Optional text description
            tags: Optional list of tags
            thumbnail_path: Optional thumbnail path
            metadata: Optional additional metadata
            
        Returns:
            Created record data
        """
        # Prepare embeddings as lists for Supabase
        data = {
            "filename": filename,
            "original_filename": original_filename or filename,
            "file_type": file_type,
            "mime_type": mime_type,
            "file_size": file_size,
            "storage_path": storage_path,
            "thumbnail_path": thumbnail_path,
            "description": description,
            "tags": tags or [],
            "visual_embedding": visual_embedding.tolist(),
            "metadata": metadata or {}
        }
        
        if text_embedding is not None:
            data["text_embedding"] = text_embedding.tolist()
        
        if combined_embedding is not None:
            data["combined_embedding"] = combined_embedding.tolist()
        else:
            # Use visual embedding as combined if not provided
            data["combined_embedding"] = visual_embedding.tolist()
        
        if duration_seconds is not None:
            data["duration_seconds"] = duration_seconds
        if frame_count is not None:
            data["frame_count"] = frame_count
        
        result = self.client.table(self.media_table).insert(data).execute()
        
        logger.info(f"Media item created: {result.data[0]['id']}")
        return result.data[0]
    
    def get_media_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a media item by ID."""
        result = self.client.table(self.media_table).select("*").eq("id", item_id).execute()
        return result.data[0] if result.data else None
    
    def update_media_item(
        self,
        item_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a media item."""
        # Convert numpy arrays to lists if present
        for key in ["visual_embedding", "text_embedding", "combined_embedding"]:
            if key in updates and isinstance(updates[key], np.ndarray):
                updates[key] = updates[key].tolist()
        
        result = self.client.table(self.media_table).update(updates).eq("id", item_id).execute()
        return result.data[0] if result.data else None
    
    def delete_media_item(self, item_id: str) -> None:
        """Delete a media item and its associated files."""
        # Get the item first to find storage paths
        item = self.get_media_item(item_id)
        if item:
            # Delete files from storage
            if item.get("storage_path"):
                try:
                    self.delete_file(item["storage_path"])
                except Exception as e:
                    logger.warning(f"Could not delete file: {e}")
            
            if item.get("thumbnail_path"):
                try:
                    self.delete_file(item["thumbnail_path"])
                except Exception as e:
                    logger.warning(f"Could not delete thumbnail: {e}")
            
            # Delete database record
            self.client.table(self.media_table).delete().eq("id", item_id).execute()
            logger.info(f"Media item deleted: {item_id}")
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        search_type: str = "combined",
        limit: int = 20,
        min_similarity: float = 0.0,
        file_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar media items using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            search_type: 'visual', 'text', or 'combined'
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            file_type: Optional filter by file type
            
        Returns:
            List of matching media items with similarity scores
        """
        result = self.client.rpc(
            "search_media_by_embedding",
            {
                "query_embedding": query_embedding.tolist(),
                "search_type": search_type,
                "match_threshold": min_similarity,
                "match_count": limit,
                "file_type_filter": file_type
            }
        ).execute()
        
        return result.data
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: Optional[str] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        limit: int = 20,
        file_type: Optional[str] = None,
        search_filenames: bool = True,
        description_weight: float = 1.1
    ) -> List[Dict[str, Any]]:
        """
        Perform enhanced hybrid search combining vector similarity, text search, and filename matching.
        
        This search method is designed to handle metadata-rich descriptions that may contain
        structured information (camera settings, location data, timestamps, etc.) alongside
        natural language descriptions.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Optional text query for full-text search
            vector_weight: Weight for vector similarity (default 0.7)
            text_weight: Weight for text similarity (default 0.3)
            limit: Maximum number of results
            file_type: Optional filter by file type
            search_filenames: Include filename matching in search (default True)
            description_weight: Weight multiplier for description matches (default 1.1)
            
        Returns:
            List of matching media items with combined similarity scores
        """
        result = self.client.rpc(
            "hybrid_search_media_enhanced",
            {
                "query_embedding": query_embedding.tolist(),
                "query_text": query_text,
                "vector_weight": vector_weight,
                "text_weight": text_weight,
                "match_count": limit,
                "file_type_filter": file_type,
                "search_filenames": search_filenames,
                "description_weight": description_weight
            }
        ).execute()
        
        return result.data
    
    def list_media_items(
        self,
        limit: int = 100,
        offset: int = 0,
        file_type: Optional[str] = None,
        order_by: str = "created_at",
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """List media items with pagination."""
        query = self.client.table(self.media_table).select(
            "id, filename, file_type, storage_path, thumbnail_path, "
            "description, tags, metadata, created_at"
        )
        
        if file_type:
            query = query.eq("file_type", file_type)
        
        query = query.order(order_by, desc=not ascending).range(offset, offset + limit - 1)
        
        result = query.execute()
        return result.data
    
    def count_media_items(self, file_type: Optional[str] = None) -> int:
        """Count total media items."""
        query = self.client.table(self.media_table).select("id", count="exact")
        
        if file_type:
            query = query.eq("file_type", file_type)
        
        result = query.execute()
        return result.count
    
    def search_by_tags(
        self,
        tags: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search media items by tags."""
        result = self.client.table(self.media_table).select(
            "id, filename, file_type, storage_path, thumbnail_path, "
            "description, tags, metadata, created_at"
        ).contains("tags", tags).limit(limit).execute()
        
        return result.data
    
    def wipe_all_data(self) -> Dict[str, Any]:
        """
        DANGEROUS: Completely wipe all data from the database and storage.
        
        This will:
        1. Delete all files from the storage bucket
        2. Delete all records from the media_items table
        
        Returns:
            Summary of what was deleted
        """
        deleted_files = 0
        deleted_records = 0
        errors = []
        
        try:
            # First, get all media items to find their storage paths
            all_items = self.client.table(self.media_table).select(
                "id, storage_path, thumbnail_path"
            ).execute()
            
            # Delete all files from storage
            for item in all_items.data:
                # Delete main file
                if item.get("storage_path"):
                    try:
                        self.client.storage.from_(self.storage_bucket).remove([item["storage_path"]])
                        deleted_files += 1
                    except Exception as e:
                        errors.append(f"Failed to delete {item['storage_path']}: {str(e)}")
                
                # Delete thumbnail
                if item.get("thumbnail_path"):
                    try:
                        self.client.storage.from_(self.storage_bucket).remove([item["thumbnail_path"]])
                        deleted_files += 1
                    except Exception as e:
                        errors.append(f"Failed to delete thumbnail {item['thumbnail_path']}: {str(e)}")
            
            # Delete all records from the database
            deleted_records = len(all_items.data)
            if deleted_records > 0:
                # Delete all rows - using a condition that matches everything
                self.client.table(self.media_table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            
            logger.warning(f"DATABASE WIPED: {deleted_records} records, {deleted_files} files deleted")
            
            return {
                "success": True,
                "deleted_records": deleted_records,
                "deleted_files": deleted_files,
                "errors": errors if errors else None
            }
            
        except Exception as e:
            logger.error(f"Error wiping database: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_records": deleted_records,
                "deleted_files": deleted_files
            }


def create_supabase_service(
    url: Optional[str] = None,
    key: Optional[str] = None,
    vector_dimension: int = 512
) -> SupabaseService:
    """
    Factory function to create a SupabaseService instance.
    
    Args:
        url: Supabase project URL (or use SUPABASE_URL env var)
        key: Supabase API key (or use SUPABASE_KEY env var)
        vector_dimension: Dimension of embedding vectors
        
    Returns:
        Configured SupabaseService instance
    """
    # Allow common misspellings while nudging users toward the preferred names
    from backend.core.env import get_supabase_env

    env_url, env_key = get_supabase_env()

    url = url or env_url
    key = key or env_key
    
    if not url or not key:
        raise ValueError(
            "Supabase URL and key must be provided either as arguments "
            "or via SUPABASE_URL and SUPABASE_KEY environment variables"
        )
    
    return SupabaseService(url=url, key=key, vector_dimension=vector_dimension)
