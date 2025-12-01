"""
Ultra-minimal semantic search API for Webflow.
Loads CLIP once at startup, queries Supabase for semantic search.
"""
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import torch
from transformers import CLIPModel, CLIPProcessor
import numpy as np

# Initialize FastAPI
app = FastAPI(title="Semantic Search API")

# CORS for Webflow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global model (loaded once at startup)
MODEL = None
PROCESSOR = None
SUPABASE = None


@app.on_event("startup")
async def load_model():
    """Load CLIP model once at startup."""
    global MODEL, PROCESSOR, SUPABASE

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    MODEL.eval()

    # Connect to Supabase
    SUPABASE = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"]
    )

    print(f"✅ Model loaded on {device}")


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Semantic search endpoint for Webflow.

    Usage: GET /search?q=sunset+over+ocean&limit=20
    """
    # Generate embedding for query
    with torch.no_grad():
        inputs = PROCESSOR(text=[q], return_tensors="pt", padding=True).to(MODEL.device)
        embedding = MODEL.get_text_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().flatten()

    # Search Supabase
    results = SUPABASE.rpc(
        "search_media_by_embedding",
        {
            "query_embedding": embedding.tolist(),
            "search_type": "combined",
            "match_threshold": 0.0,
            "match_count": limit,
            "file_type_filter": None
        }
    ).execute()

    # Return clean response
    return {
        "query": q,
        "results": [
            {
                "id": item["id"],
                "filename": item["filename"],
                "url": SUPABASE.storage.from_("media-files").get_public_url(item["storage_path"]),
                "thumbnail": SUPABASE.storage.from_("media-files").get_public_url(item["thumbnail_path"]) if item.get("thumbnail_path") else None,
                "description": item.get("description"),
                "similarity": item["similarity"]
            }
            for item in results.data
        ]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
