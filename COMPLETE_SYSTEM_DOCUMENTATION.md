# Media Semantic Search Engine - Complete System Documentation

**Version:** 1.0.0 (Read-Only Optimized)
**Last Updated:** 2025-12-01

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Database Design](#database-design)
5. [AI/ML Pipeline](#aiml-pipeline)
6. [Search Algorithms](#search-algorithms)
7. [API Reference](#api-reference)
8. [Performance Optimizations](#performance-optimizations)
9. [Configuration & Setup](#configuration--setup)
10. [Deployment Guide](#deployment-guide)
11. [Integration Examples](#integration-examples)
12. [Troubleshooting](#troubleshooting)

---

## System Overview

### What is This System?

The Media Semantic Search Engine is a **high-performance, AI-powered search system** for images and videos that understands natural language queries. Instead of relying on filenames or tags, it uses deep learning embeddings to understand the **semantic meaning** of both media content and search queries.

### Key Capabilities

- **Semantic Text Search**: "sunset over ocean" finds relevant images even if they're named "IMG_1234.jpg"
- **Visual Similarity Search**: Upload an image to find visually similar media
- **Multimodal Search**: Combine text descriptions with image queries
- **Hybrid Search**: Combines vector similarity, full-text search, and filename matching
- **Multilingual Support**: Works in 100+ languages (with appropriate models)
- **Video Support**: Automatically samples and analyzes video frames

### Technology Stack

```
Frontend/API:    FastAPI (Python)
Database:        PostgreSQL + pgvector (via Supabase)
Storage:         Supabase Storage (S3-compatible)
AI Models:       CLIP, SigLIP, Multilingual-CLIP, OpenCLIP
ML Framework:    PyTorch, Transformers (HuggingFace)
Deployment:      Railway, Docker, or any Python hosting
```

### Use Cases

1. **Media Libraries**: Search large photo/video collections by content
2. **E-commerce**: Find products by visual similarity or descriptions
3. **Stock Photography**: Natural language search for stock media
4. **Content Moderation**: Find similar content for review
5. **Digital Asset Management**: Enterprise media organization
6. **Research**: Academic image/video dataset exploration

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT APPLICATIONS                     │
│  (Web, Mobile, API Clients, Webflow, WordPress, etc.)      │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/HTTPS
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Performance Layer (Read-Only Optimized)              │  │
│  │  • GZip Compression Middleware                        │  │
│  │  • Cache Control Headers (stale-while-revalidate)    │  │
│  │  • CORS Optimization (1hr preflight cache)           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Routers (Read-Only)                             │  │
│  │  • Search Router  → Search endpoints                  │  │
│  │  • Media Router   → List/Get operations              │  │
│  │  • Config Router  → Stats, models, schema            │  │
│  │  • Health Router  → Health checks                    │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Search Engine (MediaSearchEngine)                    │  │
│  │  • Orchestrates embedding generation                  │  │
│  │  • Manages search operations                         │  │
│  │  • Handles result ranking                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────┬─────────────────────────────┬────────────────┘
              │                             │
              ▼                             ▼
┌──────────────────────────┐  ┌─────────────────────────────┐
│   Embedding Service       │  │   Database Service          │
│  ┌────────────────────┐  │  │  (Supabase/PostgreSQL)     │
│  │ CLIP/SigLIP Model  │  │  │  ┌──────────────────────┐  │
│  │ • Image Encoding   │  │  │  │  media_items table   │  │
│  │ • Text Encoding    │  │  │  │  • visual_embedding  │  │
│  │ • Video Sampling   │  │  │  │  • text_embedding    │  │
│  └────────────────────┘  │  │  │  • combined_embedding│  │
│  Device: CPU/CUDA/MPS    │  │  │  • metadata (JSONB)  │  │
└──────────────────────────┘  │  └──────────────────────┘  │
                              │  ┌──────────────────────┐  │
                              │  │  Storage Bucket      │  │
                              │  │  • Media files       │  │
                              │  │  • Thumbnails        │  │
                              │  └──────────────────────┘  │
                              └─────────────────────────────┘
```

### Request Flow

#### Search Request Flow

```
1. Client sends search query
   ↓
2. API Router receives request
   ↓
3. Search Engine generates query embedding
   ↓
4. Database Service performs vector similarity search
   ↓
5. PostgreSQL executes hybrid search function
   • Vector similarity (cosine distance)
   • Full-text search on descriptions
   • Filename matching
   • Tag matching
   ↓
6. Results ranked by combined score
   ↓
7. Public URLs generated for media
   ↓
8. Cache headers added by middleware
   ↓
9. Response compressed by GZip middleware
   ↓
10. JSON response returned to client
```

#### Media Ingestion Flow (Disabled in Read-Only Mode)

```
1. Client uploads media file
   ↓
2. API validates file type and size
   ↓
3. Search Engine processes file
   ↓
4. Embedding Service generates embeddings
   • Visual embedding from image/video
   • Text embedding from description
   • Combined embedding (weighted mix)
   ↓
5. Thumbnail generated (if enabled)
   ↓
6. Files uploaded to Supabase Storage
   ↓
7. Database record created with embeddings
   ↓
8. Media ID and URLs returned
```

---

## Core Components

### 1. FastAPI Application (`backend/main.py`)

**Purpose**: Main application entry point, middleware setup, router registration.

**Key Features**:
```python
# Performance middleware stack (order matters!)
1. CacheControlMiddleware     # Adds cache headers
2. GZipMiddleware            # Compresses responses
3. CORSMiddleware            # Handles CORS with optimization

# Lifespan management
- Initializes search engine on startup
- Manages graceful shutdown
```

**Configuration**:
```python
app = FastAPI(
    title="Media Semantic Search API",
    description="High-performance read-only semantic search",
    version="1.0.0",
    lifespan=lifespan
)
```

### 2. Search Engine (`backend/search_engine.py`)

**Purpose**: Orchestrates media ingestion and search operations.

**Class: `MediaSearchEngine`**

```python
class MediaSearchEngine:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        database_service: SupabaseService,
        generate_thumbnails: bool = True,
        thumbnail_size: Tuple[int, int] = (256, 256)
    )
```

**Key Methods**:

1. **`ingest_media(file_data, filename, description, tags, metadata)`**
   - Auto-detects file type (image/video)
   - Generates embeddings
   - Creates thumbnails
   - Uploads to storage
   - Stores in database

2. **`search_by_text(query, limit, min_similarity, file_type)`**
   - Generates query embedding
   - Performs hybrid search (vector + text + filename)
   - Returns ranked results

3. **`search_by_image(image, limit, min_similarity, file_type)`**
   - Encodes query image
   - Searches by visual similarity
   - Returns similar media

4. **`search_combined(text_query, image, text_weight, image_weight)`**
   - Multimodal search
   - Weighted combination of text + image embeddings

**Data Classes**:

```python
@dataclass
class MediaItem:
    id: str
    filename: str
    file_type: str  # 'image' or 'video'
    storage_url: str
    thumbnail_url: Optional[str]
    description: Optional[str]
    tags: List[str]
    similarity: float
    metadata: Dict[str, Any]
    created_at: Optional[str]

@dataclass
class SearchResult:
    items: List[MediaItem]
    query: str
    query_type: str  # 'text', 'image', 'video', 'combined'
    total_results: int
    processing_time_ms: float
```

### 3. Embedding Service (`backend/embedding_service.py`)

**Purpose**: Generates AI embeddings from images, videos, and text.

**Supported Models**:

| Model | Type | Embedding Dim | Multilingual | Best For |
|-------|------|---------------|--------------|----------|
| CLIP ViT-B/32 | Vision-Language | 512 | No | Balanced speed/quality |
| CLIP ViT-L/14 | Vision-Language | 768 | No | Higher quality |
| SigLIP Base | Vision-Language | 768 | Yes | Multilingual content |
| OpenCLIP | Vision-Language | 512-1024 | No | Large-scale datasets |

**Model Architecture**:

```python
class BaseEmbeddingModel(ABC):
    """Base class for all embedding models"""

    @abstractmethod
    def encode_image(image: Image.Image) -> np.ndarray

    @abstractmethod
    def encode_text(text: str) -> np.ndarray

class CLIPModel(BaseEmbeddingModel):
    """Standard CLIP from OpenAI/HuggingFace"""

class MultilingualCLIPModel(BaseEmbeddingModel):
    """Supports 100+ languages"""

class SigLIPModel(BaseEmbeddingModel):
    """Google's improved CLIP alternative"""
```

**Key Operations**:

1. **Image Encoding**:
   ```python
   image → Preprocessor → Model → Normalized Vector (512-1024 dims)
   ```

2. **Text Encoding**:
   ```python
   text → Tokenizer → Model → Normalized Vector (512-1024 dims)
   ```

3. **Video Encoding**:
   ```python
   video → Frame Sampling → Batch Encoding → Average Vector
   ```

4. **Combined Encoding**:
   ```python
   (image_emb * 0.7) + (text_emb * 0.3) → Normalized
   ```

**Device Selection**:
- Automatic CUDA/MPS/CPU detection
- Graceful fallback to CPU if GPU unavailable
- Configurable via device parameter

### 4. Database Service (`backend/database_service.py`)

**Purpose**: Manages PostgreSQL/Supabase operations with vector search.

**Class: `SupabaseService`**

```python
class SupabaseService:
    def __init__(
        self,
        url: str,
        key: str,
        vector_dimension: int = 512,
        media_table: str = "media_items",
        storage_bucket: str = "media-files"
    )
```

**Key Methods**:

1. **Storage Operations**:
   - `upload_file(file_data, filename, content_type)` → storage_path
   - `get_public_url(storage_path)` → public URL
   - `delete_file(storage_path)`

2. **CRUD Operations**:
   - `create_media_item(...)` → Creates record with embeddings
   - `get_media_item(item_id)` → Retrieves single item
   - `update_media_item(item_id, updates)` → Updates record
   - `delete_media_item(item_id)` → Deletes record + files

3. **Search Operations**:
   - `search_by_embedding(query_embedding, search_type, limit)`
   - `hybrid_search(query_embedding, query_text, weights, ...)`
   - `list_media_items(limit, offset, file_type)`
   - `count_media_items(file_type)`

**Filename Sanitization**:
```python
# Handles special characters, spaces, unicode
"My Photo (2024).jpg" → "My_Photo_2024.jpg"
"café-☕-image.png" → "cafe-image.png"
```

### 5. API Routers

#### Search Router (`backend/routers/search.py`)

**Endpoints**:

1. **POST /api/v1/search/text**
   ```json
   {
     "query": "sunset over ocean",
     "limit": 20,
     "min_similarity": 0.0,
     "file_type": null,
     "use_hybrid": true,
     "search_filenames": true,
     "description_weight": 1.1
   }
   ```

2. **GET /api/v1/search/text?q=sunset&limit=20**
   - Simple GET method for easy integration

3. **POST /api/v1/search/image**
   - Upload image as multipart/form-data
   - Returns visually similar media

4. **POST /api/v1/search/combined**
   - Multimodal search (text + image)
   - Configurable weights

#### Media Router (`backend/routers/media.py`)

**Endpoints** (Read-Only):

1. **GET /api/v1/media**
   ```
   ?limit=50&offset=0&file_type=image
   ```
   Returns paginated list of media items

2. **GET /api/v1/media/{media_id}**
   Returns single media item with full details

#### Configuration Router (`backend/routers/configuration.py`)

**Endpoints**:

1. **GET /api/v1/stats**
   ```json
   {
     "total_items": 1000,
     "images": 800,
     "videos": 200,
     "embedding_model": "clip-vit-base-patch32",
     "embedding_dim": 512
   }
   ```

2. **GET /api/v1/models**
   Lists all available embedding models

3. **GET /api/v1/schema**
   Returns SQL schema for database setup

#### Health Router (`backend/routers/health.py`)

**Endpoints**:

1. **GET /**
   ```json
   {
     "status": "healthy",
     "service": "Media Semantic Search API (READ-ONLY)",
     "version": "1.0.0",
     "mode": "read-only",
     "engine_ready": true,
     "demo_mode": false
   }
   ```

2. **GET /api/v1/health**
   Detailed health check with optimization status

### 6. Dependencies (`backend/dependencies.py`)

**Purpose**: Dependency injection for FastAPI endpoints.

**Global State**:
```python
search_engine = None           # Singleton instance
engine_lock = threading.Lock()  # Thread-safe initialization
config_manager = ConfigManager() # User config management
```

**Key Functions**:

1. **`initialize_search_engine()`**
   - Called on app startup
   - Loads credentials from env vars or config file
   - Initializes model and database connection

2. **`get_search_engine()`**
   - FastAPI dependency
   - Returns search engine or raises 503 error
   - Used in all endpoints requiring search

3. **`refresh_search_engine()`**
   - Rebuilds search engine (thread-safe)
   - Used when credentials change

---

## Database Design

### PostgreSQL Schema

**Extensions Required**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for embeddings
```

### Main Table: `media_items`

```sql
CREATE TABLE media_items (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    original_filename TEXT,

    -- File metadata
    file_type TEXT NOT NULL,  -- 'image' or 'video'
    mime_type TEXT,
    file_size INTEGER,

    -- Storage
    storage_path TEXT NOT NULL,
    thumbnail_path TEXT,

    -- Descriptive data
    description TEXT,
    tags TEXT[],

    -- AI Embeddings (pgvector)
    visual_embedding vector(512),      -- From image/video
    text_embedding vector(512),        -- From description
    combined_embedding vector(512),    -- Weighted combination

    -- Video-specific
    duration_seconds FLOAT,
    frame_count INTEGER,

    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Indexes

**Vector Indexes** (IVFFlat for fast approximate search):
```sql
-- Visual similarity search
CREATE INDEX idx_visual_embedding
ON media_items
USING ivfflat (visual_embedding vector_cosine_ops)
WITH (lists = 100);

-- Text similarity search
CREATE INDEX idx_text_embedding
ON media_items
USING ivfflat (text_embedding vector_cosine_ops)
WITH (lists = 100);

-- Combined similarity search
CREATE INDEX idx_combined_embedding
ON media_items
USING ivfflat (combined_embedding vector_cosine_ops)
WITH (lists = 100);
```

**Full-Text Search Index**:
```sql
CREATE INDEX idx_description_search
ON media_items
USING gin(to_tsvector('english', COALESCE(description, '')));
```

**Other Indexes**:
```sql
-- Tag search
CREATE INDEX idx_tags ON media_items USING gin(tags);

-- File type filtering
CREATE INDEX idx_file_type ON media_items(file_type);
```

### Database Functions

#### 1. Basic Vector Search

```sql
CREATE OR REPLACE FUNCTION search_media_by_embedding(
    query_embedding vector(512),
    search_type TEXT DEFAULT 'combined',
    match_threshold FLOAT DEFAULT 0.0,
    match_count INT DEFAULT 20,
    file_type_filter TEXT DEFAULT NULL
)
RETURNS TABLE (...)
```

**Usage**:
```sql
SELECT * FROM search_media_by_embedding(
    '[0.1, 0.2, ...]'::vector(512),
    'combined',
    0.0,
    20,
    'image'
);
```

#### 2. Enhanced Hybrid Search

```sql
CREATE OR REPLACE FUNCTION hybrid_search_media_enhanced(
    query_embedding vector(512),
    query_text TEXT DEFAULT NULL,
    vector_weight FLOAT DEFAULT 0.7,
    text_weight FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 20,
    file_type_filter TEXT DEFAULT NULL,
    search_filenames BOOLEAN DEFAULT TRUE,
    description_weight FLOAT DEFAULT 1.1
)
RETURNS TABLE (...)
```

**Scoring Algorithm**:
```sql
final_score =
    (vector_similarity * vector_weight) +
    (text_search_rank * text_weight * description_weight) +
    (filename_match_score * 0.15) +
    (tag_match_score * 0.1)
```

**Filename Matching Logic**:
- Exact match (case-insensitive): 1.0
- Contains query: 0.7
- Query contains filename: 0.5
- Partial word match: 0.3

### Storage Bucket Configuration

**Supabase Storage**:
```sql
-- Create bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('media-files', 'media-files', true);

-- File structure
media-files/
  uploads/
    {uuid}_{sanitized_filename}.jpg
    {uuid}_{sanitized_filename}.mp4
    thumb_{sanitized_filename}.jpg
```

---

## AI/ML Pipeline

### Embedding Generation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT DATA                            │
│  • Image File (JPEG, PNG, WebP, etc.)                   │
│  • Video File (MP4, AVI, MOV, etc.)                     │
│  • Text Description (Optional)                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              PREPROCESSING                               │
│  Images:  • Resize to 224x224                           │
│           • Normalize [0,1]                             │
│           • Convert to RGB                              │
│                                                         │
│  Videos:  • Extract frames (every 30th frame)          │
│           • Max 10 frames per video                    │
│           • Convert each frame to RGB                  │
│                                                         │
│  Text:    • Tokenize (BPE/WordPiece)                   │
│           • Truncate to 77 tokens                      │
│           • Add [CLS]/[SEP] tokens                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              MODEL ENCODING                              │
│                                                         │
│  CLIP/SigLIP Architecture:                             │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Image Encoder (Vision Transformer)              │  │
│  │  • Patch Embedding (16x16 patches)              │  │
│  │  • Transformer Layers (12 for ViT-B)            │  │
│  │  • Projection Head                               │  │
│  │  → Output: 512-dim vector                        │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Text Encoder (Transformer)                      │  │
│  │  • Token Embedding                               │  │
│  │  • Transformer Layers (12 for CLIP)             │  │
│  │  • Projection Head                               │  │
│  │  → Output: 512-dim vector                        │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              NORMALIZATION                               │
│  • L2 Normalization: v = v / ||v||                      │
│  • Ensures cosine similarity = dot product             │
│  • Embeddings lie on unit hypersphere                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           COMBINED EMBEDDING (Optional)                  │
│                                                         │
│  If both image and text provided:                      │
│  combined = (0.7 * visual_emb) + (0.3 * text_emb)     │
│  combined = combined / ||combined||                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              STORAGE IN DATABASE                         │
│  • visual_embedding: vector(512)                        │
│  • text_embedding: vector(512) or NULL                  │
│  • combined_embedding: vector(512)                      │
└─────────────────────────────────────────────────────────┘
```

### Model Selection Guide

**When to Use Each Model**:

1. **CLIP ViT-B/32** (Default)
   - Best for: General purpose, balanced performance
   - Speed: Fast (50-100 images/sec on CPU)
   - Quality: Good for most use cases
   - Memory: ~500MB

2. **CLIP ViT-L/14**
   - Best for: Higher quality, fine-grained distinctions
   - Speed: Slower (20-30 images/sec on CPU)
   - Quality: Better semantic understanding
   - Memory: ~1.5GB

3. **SigLIP Base**
   - Best for: Multilingual content, improved accuracy
   - Speed: Similar to CLIP ViT-B/32
   - Quality: Better on edge cases
   - Memory: ~800MB

### Similarity Computation

**Cosine Similarity**:
```python
similarity = np.dot(embedding1, embedding2)
# Range: -1 (opposite) to 1 (identical)
# Typical values: 0.2-0.9 for related content
```

**Distance Metrics in PostgreSQL**:
```sql
-- Cosine distance (used in this system)
embedding1 <=> embedding2

-- L2 distance
embedding1 <-> embedding2

-- Inner product
embedding1 <#> embedding2
```

**Similarity Thresholds**:
- 0.9+: Nearly identical content
- 0.7-0.9: Very similar
- 0.5-0.7: Related content
- 0.3-0.5: Loosely related
- <0.3: Likely unrelated

---

## Search Algorithms

### 1. Pure Vector Search

**Algorithm**:
```python
1. Generate query embedding (512-dim vector)
2. Compute cosine similarity with all embeddings
3. Return top-K results sorted by similarity
```

**SQL Implementation**:
```sql
SELECT * FROM media_items
ORDER BY combined_embedding <=> query_embedding
LIMIT 20;
```

**Performance**: ~10-50ms for 100K items (with IVFFlat index)

### 2. Hybrid Search (Enhanced)

**Algorithm**:
```python
1. Compute vector similarity score
2. Compute text search rank (full-text)
3. Compute filename match score
4. Compute tag match score
5. Combine scores with weights
6. Return top-K by combined score
```

**Scoring Formula**:
```
score = (v_sim * 0.7) +              # Vector similarity
        (text_rank * 0.3 * 1.1) +     # Text search * description_weight
        (filename_match * 0.15) +     # Filename matching
        (tag_match * 0.1)             # Tag matching
```

**Example**:
```
Query: "red sports car"

Item 1: ferrari_f40.jpg
  - Vector similarity: 0.85 (image shows red car)
  - Text rank: 0.95 (description: "red Ferrari F40 sports car")
  - Filename match: 0.7 (contains "ferrari")
  - Tag match: 1.0 (tags: ["red", "car", "sports"])
  → Final score: 0.85*0.7 + 0.95*0.33 + 0.7*0.15 + 1.0*0.1 = 1.113

Item 2: IMG_1234.jpg
  - Vector similarity: 0.82 (image shows red car)
  - Text rank: 0.0 (no description)
  - Filename match: 0.0 (no match)
  - Tag match: 0.0 (no tags)
  → Final score: 0.82*0.7 = 0.574

Item 1 ranks higher due to rich metadata!
```

### 3. Filename Matching Algorithm

**Implemented in SQL**:
```sql
CASE
    -- Exact match (1.0)
    WHEN LOWER(remove_extension(filename)) = LOWER(query)
    THEN 1.0

    -- Filename contains query (0.7)
    WHEN LOWER(filename) LIKE '%' || LOWER(query) || '%'
    THEN 0.7

    -- Query contains filename (0.5)
    WHEN LOWER(query) LIKE '%' || LOWER(remove_extension(filename)) || '%'
    THEN 0.5

    -- Partial word match (0.3)
    WHEN EXISTS (
        SELECT 1 FROM unnest(split_filename_words(filename)) AS word
        WHERE LOWER(query) LIKE '%' || word || '%'
    ) THEN 0.3

    ELSE 0
END
```

### 4. Description Weighting

**Purpose**: Boost text matches in metadata-rich descriptions.

**Configuration**:
```python
description_weight = 1.1  # 10% boost to description matches
```

**Use Case**:
```
Description: "Sunset at Golden Gate Bridge, San Francisco, CA.
              Shot with Canon EOS R5, f/8, ISO 100, 24mm lens."

Query: "golden gate sunset"

Without weighting:
  text_score = 0.85 * 0.3 = 0.255

With weighting (1.1x):
  text_score = 0.85 * 0.3 * 1.1 = 0.2805

Helps match detailed metadata while preserving semantic search dominance.
```

### 5. Multimodal Search

**Algorithm**:
```python
1. Generate image embedding (if image provided)
2. Generate text embedding (if text provided)
3. Combine with configurable weights:
   combined = (image_emb * image_weight) + (text_emb * text_weight)
4. Normalize combined embedding
5. Search with combined embedding
```

**Example Use Case**:
```
Input:
  - Image: Photo of a sunset
  - Text: "beach"
  - image_weight: 0.7
  - text_weight: 0.3

Result: Finds sunset photos on beaches
```

---

## API Reference

### Authentication

**Current Version**: No authentication required (read-only public API)

**Future Considerations**:
- API keys via headers
- JWT tokens for user-specific access
- Rate limiting per API key

### Request/Response Formats

**Content-Type**: `application/json`

**Standard Response**:
```json
{
  "items": [...],
  "query": "search query",
  "query_type": "text",
  "total_results": 15,
  "processing_time_ms": 23.5
}
```

**Error Response**:
```json
{
  "detail": "Error message"
}
```

### Endpoints (Detailed)

#### POST /api/v1/search/text

**Request Body**:
```json
{
  "query": "sunset over ocean",
  "limit": 20,
  "min_similarity": 0.0,
  "file_type": null,  // "image" or "video" or null
  "use_hybrid": true,
  "search_filenames": true,
  "description_weight": 1.1
}
```

**Response**:
```json
{
  "items": [
    {
      "id": "uuid-here",
      "filename": "sunset_beach.jpg",
      "file_type": "image",
      "storage_url": "https://...",
      "thumbnail_url": "https://...",
      "description": "Beautiful sunset on California beach",
      "tags": ["sunset", "beach", "california"],
      "similarity": 0.89,
      "metadata": {},
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "query": "sunset over ocean",
  "query_type": "text",
  "total_results": 15,
  "processing_time_ms": 23.5
}
```

#### GET /api/v1/search/text

**Query Parameters**:
```
?q=sunset ocean
&limit=20
&file_type=image
&search_filenames=true
&description_weight=1.1
```

#### POST /api/v1/search/image

**Request**: multipart/form-data
```
file: [binary image data]
limit: 20
file_type: null
```

**Response**: Same as text search

#### POST /api/v1/search/combined

**Request**: multipart/form-data
```
file: [binary image data] (optional)
text_query: "beach sunset" (optional)
text_weight: 0.3
image_weight: 0.7
limit: 20
file_type: null
```

#### GET /api/v1/media

**Query Parameters**:
```
?limit=50
&offset=0
&file_type=image
```

**Response**:
```json
{
  "items": [...],
  "total": 1000,
  "limit": 50,
  "offset": 0,
  "has_more": true
}
```

#### GET /api/v1/media/{media_id}

**Response**:
```json
{
  "id": "uuid",
  "filename": "image.jpg",
  "file_type": "image",
  "storage_url": "https://...",
  "thumbnail_url": "https://...",
  "description": "...",
  "tags": [...],
  "similarity": 0.0,
  "metadata": {},
  "created_at": "..."
}
```

#### GET /api/v1/stats

**Response**:
```json
{
  "total_items": 1000,
  "images": 800,
  "videos": 200,
  "embedding_model": "clip-vit-base-patch32",
  "embedding_dim": 512
}
```

#### GET /api/v1/models

**Response**:
```json
{
  "current_model": "clip-vit-base-patch32",
  "available_models": [
    {
      "key": "clip-vit-base-patch32",
      "type": "clip",
      "name": "openai/clip-vit-base-patch32",
      "embedding_dim": 512,
      "multilingual": false,
      "description": "Balanced speed/quality baseline",
      "languages": ["en"]
    }
  ]
}
```

---

## Performance Optimizations

### 1. Response Compression (GZip)

**Implementation**:
```python
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

**Impact**:
- JSON responses: 70-90% smaller
- 100KB response → 10-30KB
- Faster transfer over network

**Best For**: Text-heavy responses (JSON, HTML)

### 2. Cache Control Headers

**Strategy**: Stale-While-Revalidate

**Implementation**:
```python
class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        if request.method == "GET":
            # Static assets: 1 year
            if is_static_asset(path):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

            # API stats: 5 minutes
            elif "stats" in path:
                response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=600"

            # Search results: 30 seconds
            elif "search" in path:
                response.headers["Cache-Control"] = "public, max-age=30, stale-while-revalidate=60"
```

**Benefits**:
- Browser caching reduces server load
- Stale content served while revalidating
- Better UX (instant results from cache)

### 3. CORS Optimization

**Configuration**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # No PUT/DELETE/PATCH
    allow_headers=["*"],
    max_age=3600  # Cache preflight for 1 hour
)
```

**Impact**:
- Preflight requests cached for 1 hour
- Reduces OPTIONS requests by 95%+

### 4. Database Indexing

**Vector Indexes (IVFFlat)**:
```sql
CREATE INDEX idx_combined_embedding
ON media_items
USING ivfflat (combined_embedding vector_cosine_ops)
WITH (lists = 100);
```

**Performance**:
- Without index: O(n) scan, ~1000ms for 100K items
- With index: O(log n) approximate, ~10-50ms for 100K items

**Trade-off**:
- IVFFlat: Fast but approximate (99%+ recall)
- HNSW: Faster but more memory (use for <1M items)

### 5. Connection Pooling

**Supabase Client**:
```python
@property
def client(self) -> Client:
    if self._client is None:
        self._client = create_client(self.url, self.key)
    return self._client
```

**Benefits**:
- Reuses connections
- Reduces handshake overhead
- Thread-safe singleton pattern

### 6. Lazy Model Loading

**Implementation**:
```python
@property
def model(self) -> BaseEmbeddingModel:
    if self._model is None:
        self.load_model()
    return self._model
```

**Benefits**:
- Faster startup time
- Model only loaded when needed
- Memory efficient

### 7. CPU-Only PyTorch

**Configuration** (requirements.txt):
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.6.0
```

**Benefits**:
- ~200MB package vs ~2GB with CUDA
- Faster build times on Railway
- Lower memory usage
- Same performance on CPU-only hosts

### Performance Benchmarks

**Search Performance** (100K items, CPU):
- Vector search only: 10-30ms
- Hybrid search: 20-50ms
- With caching: <5ms (cache hit)

**Throughput** (single instance):
- Search requests: 50-100 req/sec
- Database queries: 200-500 req/sec
- Storage bandwidth: Limited by Supabase tier

**Scalability**:
- Read-only = horizontally scalable
- Add more API instances
- Use CDN for media URLs
- Consider read replicas for DB

---

## Configuration & Setup

### Environment Variables

**Required**:
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
```

**Optional**:
```bash
MODEL_KEY=clip-vit-base-patch32  # Default model
VECTOR_DIMENSION=512              # Must match model
PORT=8000                         # Server port
```

### Model Configuration

**File**: `config/settings.py`

```python
class ModelConfig:
    AVAILABLE_MODELS = {
        "clip-vit-base-patch32": {
            "type": "clip",
            "name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "multilingual": False,
            "description": "...",
            "languages": ["en"],
        },
        # Add custom models here
    }
```

**To Add a New Model**:
1. Add entry to `AVAILABLE_MODELS`
2. Ensure embedding dimension matches
3. Update database schema if dimension changes
4. Restart application

### Database Setup

**Step 1: Create Supabase Project**
1. Go to [supabase.com](https://supabase.com)
2. Create new project
3. Note URL and anon key

**Step 2: Apply Schema**
1. Go to SQL Editor in Supabase
2. Get schema from `/api/v1/schema` endpoint
3. Run SQL in editor

**Step 3: Create Storage Bucket**
1. Go to Storage
2. Create bucket named `media-files`
3. Set as public (if you want direct URLs)

**Step 4: Configure Environment**
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
```

### Local Development

**Setup**:
```bash
# Clone repository
git clone <repo-url>
cd railway-saga

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SUPABASE_URL="..."
export SUPABASE_KEY="..."

# Run server
uvicorn backend.main:app --reload --port 8000
```

**Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/

---

## Deployment Guide

### Railway Deployment

**Method 1: GitHub Integration**

1. Push code to GitHub
2. Go to [Railway](https://railway.app)
3. Click "New Project" → "Deploy from GitHub"
4. Select repository
5. Railway auto-detects Python app
6. Add environment variables in Railway dashboard
7. Deploy!

**Method 2: Railway CLI**

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# Add environment variables
railway variables set SUPABASE_URL="..."
railway variables set SUPABASE_KEY="..."
```

**Railway Configuration**:

File: `Procfile`
```
web: uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

File: `runtime.txt`
```
python-3.11
```

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ backend/
COPY config/ config/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run**:
```bash
docker build -t media-search-api .
docker run -p 8000:8000 \
  -e SUPABASE_URL="..." \
  -e SUPABASE_KEY="..." \
  media-search-api
```

### AWS/GCP/Azure Deployment

**Compatible Services**:
- AWS: Elastic Beanstalk, ECS, Lambda (with container)
- GCP: Cloud Run, App Engine
- Azure: App Service, Container Instances

**Requirements**:
- Python 3.11+
- 512MB+ RAM (2GB+ recommended)
- CPU: 1+ cores
- Storage: 1GB+ for models

### CDN Integration

**For Media Files**:
1. Configure Supabase Storage with CDN
2. Or use CloudFront/Cloudflare in front of storage URLs
3. Set long cache TTL for media files

**For API Responses**:
1. Use Cloudflare Workers
2. Cache GET requests at edge
3. Respect Cache-Control headers

---

## Integration Examples

### JavaScript/React Example

```javascript
// Search for media
async function searchMedia(query) {
  const response = await fetch('https://your-api.railway.app/api/v1/search/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      limit: 20,
      use_hybrid: true
    })
  });

  const data = await response.json();
  return data.items;
}

// Display results
function displayResults(items) {
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = items.map(item => `
    <div class="media-item">
      <img src="${item.thumbnail_url || item.storage_url}"
           alt="${item.description || item.filename}" />
      <p>${item.description || item.filename}</p>
      <span>Similarity: ${(item.similarity * 100).toFixed(1)}%</span>
    </div>
  `).join('');
}

// Usage
searchMedia('sunset on beach').then(displayResults);
```

### Python Client Example

```python
import requests

class MediaSearchClient:
    def __init__(self, api_url):
        self.api_url = api_url.rstrip('/')

    def search_text(self, query, limit=20, file_type=None):
        """Search by text query"""
        response = requests.post(
            f'{self.api_url}/api/v1/search/text',
            json={
                'query': query,
                'limit': limit,
                'file_type': file_type,
                'use_hybrid': True
            }
        )
        response.raise_for_status()
        return response.json()

    def search_image(self, image_path, limit=20):
        """Search by image"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f'{self.api_url}/api/v1/search/image',
                files={'file': f},
                data={'limit': limit}
            )
        response.raise_for_status()
        return response.json()

    def get_media(self, media_id):
        """Get media item by ID"""
        response = requests.get(
            f'{self.api_url}/api/v1/media/{media_id}'
        )
        response.raise_for_status()
        return response.json()

# Usage
client = MediaSearchClient('https://your-api.railway.app')
results = client.search_text('sunset on beach')
for item in results['items']:
    print(f"{item['filename']}: {item['similarity']:.2f}")
```

### Webflow Integration

```html
<!-- Add to Webflow Embed element -->
<div id="media-search-widget">
  <input type="text" id="search-input" placeholder="Search media..." />
  <button id="search-button">Search</button>
  <div id="search-results"></div>
</div>

<script>
const API_URL = 'https://your-api.railway.app';

async function searchMedia(query) {
  const response = await fetch(
    `${API_URL}/api/v1/search/text?q=${encodeURIComponent(query)}&limit=20`
  );
  return await response.json();
}

function displayResults(results) {
  const container = document.getElementById('search-results');
  container.innerHTML = results.items.map(item => `
    <div class="result-item">
      <img src="${item.thumbnail_url || item.storage_url}" />
      <p>${item.description || item.filename}</p>
    </div>
  `).join('');
}

document.getElementById('search-button').addEventListener('click', async () => {
  const query = document.getElementById('search-input').value;
  if (query) {
    const results = await searchMedia(query);
    displayResults(results);
  }
});
</script>

<style>
#search-results {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
}
.result-item img {
  width: 100%;
  height: 200px;
  object-fit: cover;
}
</style>
```

### WordPress Integration

```php
<?php
// Add to functions.php

function media_search_widget() {
    $api_url = 'https://your-api.railway.app';
    ?>
    <div class="media-search-widget">
        <input type="text" id="wp-search-input" placeholder="Search..." />
        <button id="wp-search-button">Search</button>
        <div id="wp-search-results"></div>
    </div>

    <script>
    jQuery(document).ready(function($) {
        $('#wp-search-button').click(async function() {
            const query = $('#wp-search-input').val();
            const response = await fetch(
                '<?php echo $api_url; ?>/api/v1/search/text?q=' +
                encodeURIComponent(query) + '&limit=20'
            );
            const data = await response.json();

            let html = '';
            data.items.forEach(item => {
                html += `<div class="search-item">
                    <img src="${item.thumbnail_url}" />
                    <p>${item.description}</p>
                </div>`;
            });
            $('#wp-search-results').html(html);
        });
    });
    </script>
    <?php
}

// Register shortcode
add_shortcode('media_search', 'media_search_widget');

// Usage in posts: [media_search]
?>
```

---

## Troubleshooting

### Common Issues

#### 1. Search Returns 503 Error

**Cause**: Search engine not initialized

**Solution**:
```bash
# Check environment variables
echo $SUPABASE_URL
echo $SUPABASE_KEY

# Verify connection
curl http://localhost:8000/api/v1/health

# Check logs
uvicorn backend.main:app --log-level debug
```

#### 2. Vector Search Returns No Results

**Possible Causes**:
- No media in database
- Embeddings not generated
- min_similarity too high

**Solution**:
```bash
# Check database
curl http://localhost:8000/api/v1/stats

# Try with min_similarity=0
curl "http://localhost:8000/api/v1/search/text?q=test&min_similarity=0"
```

#### 3. Slow Search Performance

**Possible Causes**:
- Missing vector indexes
- Large result set
- CPU-bound model inference

**Solutions**:
```sql
-- Verify indexes exist
SELECT indexname FROM pg_indexes
WHERE tablename = 'media_items';

-- Rebuild index if needed
REINDEX INDEX idx_combined_embedding;
```

```python
# Reduce limit
response = search_text(query, limit=10)  # Instead of 100

# Use hybrid search (it's optimized)
response = search_text(query, use_hybrid=True)
```

#### 4. Out of Memory

**Cause**: Model too large for available RAM

**Solutions**:
```python
# Use smaller model
MODEL_KEY=clip-vit-base-patch32  # Instead of ViT-L/14

# Reduce batch size (if processing many items)
# Upgrade server RAM (2GB+ recommended)
```

#### 5. Supabase Connection Timeout

**Possible Causes**:
- Invalid credentials
- Network issues
- Database not configured

**Solution**:
```python
# Test connection
python3 -c "
from supabase import create_client
client = create_client('YOUR_URL', 'YOUR_KEY')
result = client.table('media_items').select('id').limit(1).execute()
print('Connected!', result.data)
"
```

#### 6. CORS Errors in Browser

**Cause**: CORS not configured for your domain

**Solution**:
```python
# In backend/main.py, update:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Specific domain
    # Or use ["*"] for testing
)
```

### Debug Mode

**Enable Verbose Logging**:
```python
# In backend/core/logging_config.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Logs**:
```bash
# Railway
railway logs

# Docker
docker logs <container-id>

# Local
tail -f logs/app.log
```

### Performance Profiling

**Profile Search Endpoint**:
```python
import time

def search_with_timing(query):
    start = time.time()
    result = engine.search_by_text(query)
    end = time.time()

    print(f"Total time: {(end - start) * 1000:.2f}ms")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    return result
```

**Database Query Analysis**:
```sql
-- Enable timing
\timing

-- Explain query plan
EXPLAIN ANALYZE
SELECT * FROM search_media_by_embedding(
    '[0.1, 0.2, ...]'::vector(512),
    'combined',
    0.0,
    20,
    NULL
);
```

---

## Appendices

### A. File Structure

```
RAILWAY-SAGA/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app + middleware
│   ├── dependencies.py            # Dependency injection
│   ├── search_engine.py           # Main search engine
│   ├── embedding_service.py       # AI model management
│   ├── database_service.py        # Supabase operations
│   ├── core/
│   │   ├── __init__.py
│   │   └── logging_config.py      # Logging setup
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic models
│   └── routers/
│       ├── __init__.py
│       ├── health.py              # Health checks
│       ├── search.py              # Search endpoints
│       ├── media.py               # Media endpoints
│       └── configuration.py       # Config endpoints
├── config/
│   ├── __init__.py
│   ├── settings.py                # Model registry
│   └── user_config.py             # User config management
├── .gitignore
├── Procfile                       # Railway process
├── runtime.txt                    # Python version
├── requirements.txt               # Dependencies
├── README.md                      # Deployment guide
├── READ_ONLY_OPTIMIZATIONS.md     # Optimization details
└── COMPLETE_SYSTEM_DOCUMENTATION.md  # This file
```

### B. Dependencies

```
# Core
fastapi>=0.104.0              # Web framework
uvicorn[standard]>=0.24.0     # ASGI server
python-multipart>=0.0.6       # File upload support
pydantic>=2.5.0               # Data validation

# Database
supabase>=2.0.0               # Supabase client

# AI/ML (CPU-optimized)
torch>=2.6.0                  # PyTorch (CPU-only)
transformers>=4.35.0          # HuggingFace models
Pillow>=10.0.0                # Image processing
numpy>=1.24.0                 # Numerical operations

# Media Processing
opencv-python-headless>=4.8.0 # Video processing (no GUI)

# Utilities
python-dotenv>=1.0.0          # Environment variables
requests>=2.31.0              # HTTP client
```

### C. SQL Schema (Complete)

See `/api/v1/schema` endpoint for the complete, up-to-date schema.

Key components:
- `media_items` table with vector columns
- IVFFlat indexes for fast vector search
- Full-text search indexes
- Hybrid search functions
- Automatic timestamp updates

### D. Model Training (For Custom Models)

This system uses **pre-trained models** by default. To train custom models:

**Option 1: Fine-tune CLIP**
```python
from transformers import CLIPModel, CLIPProcessor, Trainer

# Load base model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Fine-tune on your dataset
# (Requires paired image-text data)

# Export fine-tuned model
model.save_pretrained("./my-custom-clip")
```

**Option 2: Use OpenCLIP**
```bash
# OpenCLIP supports custom training
pip install open-clip-torch

# Train on your data
# See: https://github.com/mlfoundations/open_clip
```

**Integration**:
```python
# Add to config/settings.py
"my-custom-model": {
    "type": "clip",
    "name": "path/to/my-custom-clip",
    "embedding_dim": 512,
    ...
}
```

### E. Scaling Recommendations

**< 10K items**:
- Single API instance
- Basic Supabase tier
- No CDN needed

**10K - 100K items**:
- 2-3 API instances (load balanced)
- Supabase Pro tier
- CDN for media files
- Redis cache for search results

**100K - 1M items**:
- 5+ API instances
- Supabase Team tier or managed PostgreSQL
- CDN + edge caching
- Redis cluster
- Database read replicas
- Consider HNSW index instead of IVFFlat

**> 1M items**:
- Kubernetes cluster
- Separate database + storage
- Elasticsearch/Meilisearch for hybrid search
- Dedicated vector database (Pinecone, Weaviate)
- Multi-region deployment

### F. Security Considerations

**Current State** (Read-Only API):
- No authentication required
- Public read access
- No write operations = minimal attack surface

**For Production**:
1. **Add Authentication**:
   - API keys via headers
   - JWT tokens
   - OAuth integration

2. **Rate Limiting**:
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)

   @app.get("/api/v1/search/text")
   @limiter.limit("100/minute")
   async def search_text(...):
       ...
   ```

3. **HTTPS Only**:
   - Enforce TLS
   - Use secure headers

4. **Input Validation**:
   - Already implemented via Pydantic
   - Sanitize filenames
   - Validate file types

5. **CORS Configuration**:
   ```python
   allow_origins=["https://your-domain.com"]  # Not ["*"]
   ```

### G. Monitoring & Observability

**Key Metrics to Track**:
```python
# API Metrics
- Request rate (req/sec)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- Cache hit rate

# Search Metrics
- Search latency
- Results relevance (user feedback)
- Popular queries
- Empty result rate

# System Metrics
- CPU usage
- Memory usage
- Database connections
- Storage usage
```

**Recommended Tools**:
- **Logging**: Structured JSON logs
- **APM**: New Relic, DataDog, or Sentry
- **Metrics**: Prometheus + Grafana
- **Alerts**: PagerDuty, Opsgenie

**Example Logging**:
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_search(query, results, duration_ms):
    logger.info(json.dumps({
        "event": "search",
        "query": query,
        "result_count": len(results),
        "duration_ms": duration_ms,
        "timestamp": datetime.utcnow().isoformat()
    }))
```

### H. Future Enhancements

**Potential Features**:

1. **Faceted Search**:
   - Filter by date ranges
   - Filter by metadata fields
   - Color-based filtering

2. **Advanced Analytics**:
   - Search analytics dashboard
   - Popular content tracking
   - User behavior analysis

3. **Batch Operations**:
   - Bulk search
   - Batch similarity computation

4. **Real-time Updates**:
   - WebSocket support
   - SSE for live results

5. **AI Improvements**:
   - Custom model fine-tuning
   - Query expansion
   - Result re-ranking with user feedback

6. **Content Features**:
   - Automatic tagging
   - Content categorization
   - Duplicate detection

7. **API Enhancements**:
   - GraphQL endpoint
   - Webhook support
   - SDK libraries (Python, JS, Go)

---

## Conclusion

This Media Semantic Search Engine provides a **production-ready, high-performance system** for intelligent media search. The read-only optimized version is designed for **maximum query speed** with minimal operational overhead.

**Key Takeaways**:

1. **AI-Powered**: Uses state-of-the-art vision-language models (CLIP/SigLIP)
2. **Hybrid Search**: Combines vector similarity, text search, and metadata
3. **High Performance**: Optimized with caching, compression, and database indexes
4. **Scalable**: Horizontally scalable read-only architecture
5. **Easy to Deploy**: Railway, Docker, or any Python hosting
6. **Well-Documented**: Comprehensive API docs and code comments

**Next Steps**:

1. Deploy to Railway/your hosting platform
2. Configure Supabase database
3. Set environment variables
4. Test with sample queries
5. Integrate into your application
6. Monitor performance and scale as needed

**Support & Resources**:

- GitHub Repository: [Your repo URL]
- API Documentation: `https://your-api.railway.app/docs`
- Supabase Docs: https://supabase.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- CLIP Paper: https://arxiv.org/abs/2103.00020

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-01
**Maintained By**: [Your Team Name]
**License**: [Your License]

---

*This documentation is intended to be comprehensive enough to replicate this system in another codebase. For questions or contributions, please refer to the repository.*
