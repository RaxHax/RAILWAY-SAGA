# Railway Saga Agent (token-frugal)

## Purpose
Read-only FastAPI semantic search API using Supabase pgvector + embedding models (CLIP/SigLIP variants). Only search/browse endpoints are live; uploads/admin/setup are present but not mounted.

## Key files
- backend/main.py — FastAPI app, CacheControlMiddleware + GZip + CORS (GET/POST/OPTIONS), mounts health/search/media/configuration routers.
- backend/routers/search.py — POST/GET text search; POST image search; POST combined search (text_weight/image_weight, limit, file_type).
- backend/routers/media.py — GET list (limit/offset/file_type), GET by id.
- backend/routers/configuration.py — GET stats, GET models (ModelConfig), GET schema SQL.
- backend/routers/health.py — health + read-only metadata.
- backend/routers/{ingestion,admin,setup}.py — upload/wipe/setup helpers; unused in read-only deployment unless explicitly re-enabled.
- backend/dependencies.py — global search_engine init from env or config/user_config.json; DEMO mode if missing SUPABASE_URL/SUPABASE_KEY.
- backend/search_engine.py — ingestion/search orchestration; search_by_text/image/combined, list_all_media, get_stats.
- backend/embedding_service.py — model loaders (CLIP, multilingual, SigLIP, OpenCLIP), encode_combined.
- backend/database_service.py — Supabase client, schema SQL + RPC hybrid_search_media_enhanced, media listing/counting.
- config/settings.py — model registry + dims; config/user_config.py — saved Supabase creds for the wizard.
- Procfile/runtime/requirements — uvicorn entrypoint, Python 3.11, CPU-only torch stack.

## Live API quick ref
- GET `/` and `/api/v1/health`
- POST `/api/v1/search/text` (body: query, limit, min_similarity, file_type?, use_hybrid?, search_filenames?, description_weight?); GET variant uses query params.
- POST `/api/v1/search/image` (file, limit, file_type?)
- POST `/api/v1/search/combined` (file?, text_query?, text_weight, image_weight, limit, file_type?)
- GET `/api/v1/media?limit&offset&file_type`
- GET `/api/v1/media/{id}`
- GET `/api/v1/stats` • `/api/v1/models` • `/api/v1/schema`

## Ops
Run: `uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}`. Env: SUPABASE_URL, SUPABASE_KEY, MODEL_KEY (default clip-vit-base-patch32), VECTOR_DIMENSION (matches model). Logs land under /tmp/logs on Railway or logs/ locally.

## Token discipline
- Answer with short bullets + file paths; avoid large code dumps unless essential.
- Remind that write endpoints are intentionally disabled; do not suggest enabling them unless asked.
- When touching search logic, align ModelConfig dims with embeddings/DB and hybrid RPC params.
- Prefer pointing to functions/params over quoting them; cite path:line when possible.
