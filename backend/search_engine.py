"""
Media Search Engine - Main service that orchestrates media ingestion and search.
Combines embedding generation with vector storage for semantic search.
"""

import io
import logging
import mimetypes
import tempfile
import uuid
import time
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
from PIL import Image

from backend.embedding_service import EmbeddingService, create_embedding_service
from backend.database_service import SupabaseService, create_supabase_service

logger = logging.getLogger(__name__)


@dataclass
class MediaItem:
    """Represents a media item in the search engine."""
    id: str
    filename: str
    file_type: str  # 'image' or 'video'
    storage_url: str
    thumbnail_url: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    similarity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None


@dataclass
class SearchResult:
    """Represents search results."""
    items: List[MediaItem]
    query: str
    query_type: str  # 'text', 'image', 'video', 'combined'
    total_results: int
    processing_time_ms: float


@dataclass 
class IngestionResult:
    """Represents the result of media ingestion."""
    success: bool
    media_id: Optional[str] = None
    storage_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error: Optional[str] = None


class MediaSearchEngine:
    """
    Main search engine class that handles:
    - Media file ingestion (images and videos)
    - Embedding generation
    - Vector storage
    - Semantic search
    """
    
    SUPPORTED_IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    SUPPORTED_VIDEO_TYPES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        database_service: SupabaseService,
        generate_thumbnails: bool = True,
        thumbnail_size: Tuple[int, int] = (256, 256),
        cache_ttl_seconds: int = 120,
        cache_max_entries: int = 256
    ):
        self.embedding_service = embedding_service
        self.database_service = database_service
        self.generate_thumbnails = generate_thumbnails
        self.thumbnail_size = thumbnail_size
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_max_entries = cache_max_entries

        # Lightweight in-memory caches to avoid recomputing hot queries
        self._text_embedding_cache: OrderedDict[str, Tuple[float, np.ndarray]] = OrderedDict()
        self._text_search_cache: OrderedDict[Tuple[Any, ...], Tuple[float, "SearchResult"]] = OrderedDict()
        self._cache_lock = threading.RLock()
        
        logger.info("MediaSearchEngine initialized")
    
    @classmethod
    def from_config(
        cls,
        model_key: str = "clip-vit-base-patch32",
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ) -> "MediaSearchEngine":
        """
        Create a MediaSearchEngine from configuration.
        
        Args:
            model_key: Key for the embedding model to use
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            
        Returns:
            Configured MediaSearchEngine instance
        """
        from config.settings import ModelConfig
        
        # Get embedding dimension for the model
        model_info = ModelConfig.AVAILABLE_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"Unknown model: {model_key}")
        
        embedding_dim = model_info["embedding_dim"]
        
        # Create services
        embedding_service = create_embedding_service(model_key)
        database_service = create_supabase_service(
            url=supabase_url,
            key=supabase_key,
            vector_dimension=embedding_dim
        )
        
        return cls(embedding_service, database_service)
    
    def _prune_cache(self, cache: OrderedDict) -> None:
        """Trim caches to the configured maximum size."""
        while len(cache) > self.cache_max_entries:
            cache.popitem(last=False)

    def _get_cached_text_embedding(self, query: str, now: float) -> Optional[np.ndarray]:
        """Fetch a cached text embedding if it's still fresh."""
        with self._cache_lock:
            entry = self._text_embedding_cache.get(query)
            if not entry:
                return None
            ts, embedding = entry
            if now - ts > self.cache_ttl_seconds:
                self._text_embedding_cache.pop(query, None)
                return None
            self._text_embedding_cache.move_to_end(query)
            return embedding

    def _store_text_embedding(self, query: str, embedding: np.ndarray, now: float) -> None:
        """Store a text embedding in the cache."""
        with self._cache_lock:
            self._text_embedding_cache[query] = (now, embedding)
            self._prune_cache(self._text_embedding_cache)

    def _get_cached_search_result(self, cache_key: Tuple[Any, ...], now: float) -> Optional["SearchResult"]:
        """Fetch a cached search result if it's still within TTL."""
        with self._cache_lock:
            entry = self._text_search_cache.get(cache_key)
            if not entry:
                return None
            ts, result = entry
            if now - ts > self.cache_ttl_seconds:
                self._text_search_cache.pop(cache_key, None)
                return None
            self._text_search_cache.move_to_end(cache_key)
            return result

    def _store_search_result(self, cache_key: Tuple[Any, ...], result: "SearchResult", now: float) -> None:
        """Store a search result in the cache."""
        with self._cache_lock:
            self._text_search_cache[cache_key] = (now, result)
            self._prune_cache(self._text_search_cache)
    
    def _get_file_type(self, filename: str) -> Optional[str]:
        """Determine if file is image or video based on extension."""
        ext = Path(filename).suffix.lower()
        if ext in self.SUPPORTED_IMAGE_TYPES:
            return "image"
        elif ext in self.SUPPORTED_VIDEO_TYPES:
            return "video"
        return None
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for a file."""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"
    
    def _create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = None
    ) -> bytes:
        """Create a thumbnail from an image."""
        size = size or self.thumbnail_size
        img_copy = image.copy()
        img_copy.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (for PNG with alpha)
        if img_copy.mode in ('RGBA', 'P'):
            img_copy = img_copy.convert('RGB')
        
        buffer = io.BytesIO()
        img_copy.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_video_thumbnail(self, video_path: str) -> Optional[bytes]:
        """Create a thumbnail from a video's first frame."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                return self._create_thumbnail(image)
        except Exception as e:
            logger.warning(f"Could not create video thumbnail: {e}")
        
        return None
    
    async def ingest_image(
        self,
        image_data: Union[bytes, Image.Image],
        filename: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest an image into the search engine.
        
        Args:
            image_data: Image as bytes or PIL Image
            filename: Original filename
            description: Optional text description
            tags: Optional list of tags
            metadata: Optional additional metadata
            
        Returns:
            IngestionResult with status and media ID
        """
        try:
            # Convert to PIL Image if bytes
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                file_bytes = image_data
            else:
                image = image_data.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=95)
                buffer.seek(0)
                file_bytes = buffer.getvalue()
            
            # Generate embeddings
            visual_embedding = self.embedding_service.encode_image(image)
            
            text_embedding = None
            combined_embedding = visual_embedding.copy()
            
            if description:
                text_embedding = self.embedding_service.encode_text(description)
                combined_embedding = self.embedding_service.encode_combined(
                    image=image,
                    text=description
                )
            
            # Upload file to storage
            storage_path = self.database_service.upload_file(
                file_data=file_bytes,
                filename=filename,
                content_type=self._get_mime_type(filename)
            )
            
            # Create and upload thumbnail
            thumbnail_path = None
            if self.generate_thumbnails:
                thumbnail_data = self._create_thumbnail(image)
                thumbnail_filename = f"thumb_{Path(filename).stem}.jpg"
                thumbnail_path = self.database_service.upload_file(
                    file_data=thumbnail_data,
                    filename=thumbnail_filename,
                    content_type="image/jpeg"
                )
            
            # Store in database
            result = self.database_service.create_media_item(
                filename=storage_path.split("/")[-1],
                original_filename=filename,
                file_type="image",
                storage_path=storage_path,
                visual_embedding=visual_embedding,
                text_embedding=text_embedding,
                combined_embedding=combined_embedding,
                description=description,
                tags=tags,
                thumbnail_path=thumbnail_path,
                metadata=metadata,
                mime_type=self._get_mime_type(filename),
                file_size=len(file_bytes)
            )
            
            return IngestionResult(
                success=True,
                media_id=result["id"],
                storage_url=self.database_service.get_public_url(storage_path),
                thumbnail_url=self.database_service.get_public_url(thumbnail_path) if thumbnail_path else None
            )
            
        except Exception as e:
            logger.error(f"Error ingesting image: {e}")
            return IngestionResult(success=False, error=str(e))
    
    async def ingest_video(
        self,
        video_data: bytes,
        filename: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sample_rate: int = 30,
        max_frames: int = 10
    ) -> IngestionResult:
        """
        Ingest a video into the search engine.
        
        Args:
            video_data: Video file as bytes
            filename: Original filename
            description: Optional text description
            tags: Optional list of tags
            metadata: Optional additional metadata
            sample_rate: Sample every N frames
            max_frames: Maximum frames to process
            
        Returns:
            IngestionResult with status and media ID
        """
        try:
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
                tmp.write(video_data)
                tmp_path = tmp.name
            
            try:
                # Generate embeddings from video frames
                avg_embedding, frame_embeddings = self.embedding_service.encode_video(
                    tmp_path,
                    sample_rate=sample_rate,
                    max_frames=max_frames
                )
                
                text_embedding = None
                combined_embedding = avg_embedding.copy()
                
                if description:
                    text_embedding = self.embedding_service.encode_text(description)
                    # Combine video embedding with text
                    combined_embedding = 0.7 * avg_embedding + 0.3 * text_embedding
                    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
                
                # Upload file to storage
                storage_path = self.database_service.upload_file(
                    file_data=video_data,
                    filename=filename,
                    content_type=self._get_mime_type(filename)
                )
                
                # Create and upload thumbnail
                thumbnail_path = None
                if self.generate_thumbnails:
                    thumbnail_data = self._create_video_thumbnail(tmp_path)
                    if thumbnail_data:
                        thumbnail_filename = f"thumb_{Path(filename).stem}.jpg"
                        thumbnail_path = self.database_service.upload_file(
                            file_data=thumbnail_data,
                            filename=thumbnail_filename,
                            content_type="image/jpeg"
                        )
                
                # Get video metadata
                import cv2
                cap = cv2.VideoCapture(tmp_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                # Store in database
                result = self.database_service.create_media_item(
                    filename=storage_path.split("/")[-1],
                    original_filename=filename,
                    file_type="video",
                    storage_path=storage_path,
                    visual_embedding=avg_embedding,
                    text_embedding=text_embedding,
                    combined_embedding=combined_embedding,
                    description=description,
                    tags=tags,
                    thumbnail_path=thumbnail_path,
                    metadata={
                        **(metadata or {}),
                        "fps": fps,
                        "processed_frames": len(frame_embeddings)
                    },
                    mime_type=self._get_mime_type(filename),
                    file_size=len(video_data),
                    duration_seconds=duration,
                    frame_count=frame_count
                )
                
                return IngestionResult(
                    success=True,
                    media_id=result["id"],
                    storage_url=self.database_service.get_public_url(storage_path),
                    thumbnail_url=self.database_service.get_public_url(thumbnail_path) if thumbnail_path else None
                )
                
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error ingesting video: {e}")
            return IngestionResult(success=False, error=str(e))
    
    async def ingest_media(
        self,
        file_data: bytes,
        filename: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest a media file (auto-detect type).
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            description: Optional text description
            tags: Optional list of tags
            metadata: Optional additional metadata
            
        Returns:
            IngestionResult with status and media ID
        """
        file_type = self._get_file_type(filename)
        
        if file_type == "image":
            return await self.ingest_image(
                file_data, filename, description, tags, metadata
            )
        elif file_type == "video":
            return await self.ingest_video(
                file_data, filename, description, tags, metadata
            )
        else:
            return IngestionResult(
                success=False,
                error=f"Unsupported file type: {Path(filename).suffix}"
            )
    
    def search_by_text(
        self,
        query: str,
        limit: int = 20,
        min_similarity: float = 0.0,
        file_type: Optional[str] = None,
        use_hybrid: bool = True,
        search_filenames: bool = True,
        description_weight: float = 1.1
    ) -> SearchResult:
        """
        Search for media using a text query.
        
        Supports enhanced hybrid search that combines:
        - Semantic vector similarity for understanding query meaning
        - Full-text search on descriptions (weighted higher by default)
        - Filename matching for exact/partial matches
        
        The description_weight parameter (default 1.1) gives descriptions
        10% more influence than other search factors, which helps with
        metadata-rich descriptions that may contain structured information.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            file_type: Filter by 'image' or 'video'
            use_hybrid: Use hybrid search (vector + text + filename)
            search_filenames: Include filename matching in search
            description_weight: Weight multiplier for description matches (default 1.1)
            
        Returns:
            SearchResult with matching media items
        """
        start_time = time.time()
        now = time.monotonic()

        cache_key = (query, limit, min_similarity, file_type, use_hybrid, search_filenames, description_weight)
        cached = self._get_cached_search_result(cache_key, now)
        if cached:
            return cached
        
        # Generate query embedding
        query_embedding = self._get_cached_text_embedding(query, now)
        if query_embedding is None:
            query_embedding = self.embedding_service.encode_text(query)
            self._store_text_embedding(query, query_embedding, now)
        
        # Search
        if use_hybrid:
            results = self.database_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                limit=limit,
                file_type=file_type,
                search_filenames=search_filenames,
                description_weight=description_weight
            )
        else:
            results = self.database_service.search_by_embedding(
                query_embedding=query_embedding,
                search_type="combined",
                limit=limit,
                min_similarity=min_similarity,
                file_type=file_type
            )
        
        # Convert to MediaItem objects
        items = []
        for r in results:
            items.append(MediaItem(
                id=r["id"],
                filename=r["filename"],
                file_type=r["file_type"],
                storage_url=self.database_service.get_public_url(r["storage_path"]),
                thumbnail_url=self.database_service.get_public_url(r["thumbnail_path"]) if r.get("thumbnail_path") else None,
                description=r.get("description"),
                tags=r.get("tags", []),
                similarity=r.get("similarity", 0.0),
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at")
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        result = SearchResult(
            items=items,
            query=query,
            query_type="text",
            total_results=len(items),
            processing_time_ms=processing_time
        )

        self._store_search_result(cache_key, result, now)
        return result
    
    def search_by_image(
        self,
        image: Union[bytes, Image.Image],
        limit: int = 20,
        min_similarity: float = 0.0,
        file_type: Optional[str] = None
    ) -> SearchResult:
        """
        Search for similar media using an image.
        
        Args:
            image: Query image (bytes or PIL Image)
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            file_type: Filter by 'image' or 'video'
            
        Returns:
            SearchResult with matching media items
        """
        import time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_service.encode_image(image)
        
        # Search
        results = self.database_service.search_by_embedding(
            query_embedding=query_embedding,
            search_type="visual",
            limit=limit,
            min_similarity=min_similarity,
            file_type=file_type
        )
        
        # Convert to MediaItem objects
        items = []
        for r in results:
            items.append(MediaItem(
                id=r["id"],
                filename=r["filename"],
                file_type=r["file_type"],
                storage_url=self.database_service.get_public_url(r["storage_path"]),
                thumbnail_url=self.database_service.get_public_url(r["thumbnail_path"]) if r.get("thumbnail_path") else None,
                description=r.get("description"),
                tags=r.get("tags", []),
                similarity=r.get("similarity", 0.0),
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at")
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResult(
            items=items,
            query="[image query]",
            query_type="image",
            total_results=len(items),
            processing_time_ms=processing_time
        )
    
    def search_combined(
        self,
        text_query: Optional[str] = None,
        image: Optional[Union[bytes, Image.Image]] = None,
        text_weight: float = 0.3,
        image_weight: float = 0.7,
        limit: int = 20,
        min_similarity: float = 0.0,
        file_type: Optional[str] = None
    ) -> SearchResult:
        """
        Search using both text and image queries.
        
        Args:
            text_query: Optional text query
            image: Optional query image
            text_weight: Weight for text query
            image_weight: Weight for image query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            file_type: Filter by 'image' or 'video'
            
        Returns:
            SearchResult with matching media items
        """
        import time
        start_time = time.time()
        
        if text_query is None and image is None:
            raise ValueError("At least one of text_query or image must be provided")
        
        # Generate combined query embedding
        query_embedding = self.embedding_service.encode_combined(
            image=image,
            text=text_query,
            image_weight=image_weight,
            text_weight=text_weight
        )
        
        # Search
        results = self.database_service.search_by_embedding(
            query_embedding=query_embedding,
            search_type="combined",
            limit=limit,
            min_similarity=min_similarity,
            file_type=file_type
        )
        
        # Convert to MediaItem objects
        items = []
        for r in results:
            items.append(MediaItem(
                id=r["id"],
                filename=r["filename"],
                file_type=r["file_type"],
                storage_url=self.database_service.get_public_url(r["storage_path"]),
                thumbnail_url=self.database_service.get_public_url(r["thumbnail_path"]) if r.get("thumbnail_path") else None,
                description=r.get("description"),
                tags=r.get("tags", []),
                similarity=r.get("similarity", 0.0),
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at")
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        query_str = []
        if text_query:
            query_str.append(text_query)
        if image:
            query_str.append("[image]")
        
        return SearchResult(
            items=items,
            query=" + ".join(query_str),
            query_type="combined",
            total_results=len(items),
            processing_time_ms=processing_time
        )
    
    def get_media_item(self, item_id: str) -> Optional[MediaItem]:
        """Get a single media item by ID."""
        result = self.database_service.get_media_item(item_id)
        if not result:
            return None
        
        return MediaItem(
            id=result["id"],
            filename=result["filename"],
            file_type=result["file_type"],
            storage_url=self.database_service.get_public_url(result["storage_path"]),
            thumbnail_url=self.database_service.get_public_url(result["thumbnail_path"]) if result.get("thumbnail_path") else None,
            description=result.get("description"),
            tags=result.get("tags", []),
            metadata=result.get("metadata", {}),
            created_at=result.get("created_at")
        )
    
    def delete_media_item(self, item_id: str) -> bool:
        """Delete a media item."""
        try:
            self.database_service.delete_media_item(item_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting media item: {e}")
            return False
    
    def update_description(
        self,
        item_id: str,
        description: str
    ) -> bool:
        """Update the description of a media item and regenerate embeddings."""
        try:
            item = self.database_service.get_media_item(item_id)
            if not item:
                return False
            
            # Generate new text embedding
            text_embedding = self.embedding_service.encode_text(description)
            
            # Update combined embedding
            visual_embedding = np.array(item["visual_embedding"])
            combined_embedding = 0.7 * visual_embedding + 0.3 * text_embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            # Update database
            self.database_service.update_media_item(item_id, {
                "description": description,
                "text_embedding": text_embedding,
                "combined_embedding": combined_embedding
            })
            
            return True
        except Exception as e:
            logger.error(f"Error updating description: {e}")
            return False
    
    def list_all_media(
        self,
        limit: int = 100,
        offset: int = 0,
        file_type: Optional[str] = None
    ) -> List[MediaItem]:
        """List all media items with pagination."""
        results = self.database_service.list_media_items(
            limit=limit,
            offset=offset,
            file_type=file_type
        )
        
        items = []
        for r in results:
            items.append(MediaItem(
                id=r["id"],
                filename=r["filename"],
                file_type=r["file_type"],
                storage_url=self.database_service.get_public_url(r["storage_path"]),
                thumbnail_url=self.database_service.get_public_url(r["thumbnail_path"]) if r.get("thumbnail_path") else None,
                description=r.get("description"),
                tags=r.get("tags", []),
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at")
            ))
        
        return items
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the media collection."""
        total = self.database_service.count_media_items()
        images = self.database_service.count_media_items(file_type="image")
        videos = self.database_service.count_media_items(file_type="video")
        
        return {
            "total_items": total,
            "images": images,
            "videos": videos,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dim": self.embedding_service.embedding_dim
        }
