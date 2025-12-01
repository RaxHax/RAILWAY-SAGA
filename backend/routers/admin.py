"""Admin endpoints - dangerous operations."""

import logging

from fastapi import APIRouter, HTTPException, Depends

from backend.dependencies import get_search_engine
from backend.models.schemas import WipeDatabaseRequest, WipeDatabaseResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Admin"])


@router.post("/api/v1/admin/wipe-database", response_model=WipeDatabaseResponse)
async def wipe_database(
    request: WipeDatabaseRequest,
    engine=Depends(get_search_engine)
):
    """
    ⚠️ DANGEROUS: Completely wipe all data from the database and storage.

    This will permanently delete:
    - All media files from storage
    - All thumbnails
    - All database records

    **This action cannot be undone!**

    To confirm, you must provide the exact confirmation phrase:
    `ég vill eyða fokking öllu`
    """
    REQUIRED_CONFIRMATION = "ég vill eyða fokking öllu"

    if request.confirmation != REQUIRED_CONFIRMATION:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid confirmation phrase",
                "required": REQUIRED_CONFIRMATION,
                "provided": request.confirmation,
                "message": "You must type the exact confirmation phrase to wipe the database"
            }
        )

    logger.warning("🚨 DATABASE WIPE INITIATED 🚨")

    result = engine.database_service.wipe_all_data()

    if result["success"]:
        return WipeDatabaseResponse(
            success=True,
            message="🗑️ Database wiped successfully. All data has been permanently deleted.",
            deleted_records=result["deleted_records"],
            deleted_files=result["deleted_files"],
            errors=result.get("errors")
        )
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to wipe database",
                "message": result.get("error"),
                "partial_deletion": {
                    "deleted_records": result["deleted_records"],
                    "deleted_files": result["deleted_files"]
                }
            }
        )


@router.get("/api/v1/webflow/embed-code", tags=["Webflow"])
async def get_webflow_embed_code():
    """
    Get JavaScript code for embedding the search in Webflow.
    """
    embed_code = """
<!-- Media Search Widget for Webflow -->
<div id="media-search-widget">
  <input type="text" id="search-input" placeholder="Search media..." />
  <button id="search-button">Search</button>
  <div id="search-results"></div>
</div>

<script>
const API_URL = 'YOUR_API_URL_HERE';

async function searchMedia(query) {
  const response = await fetch(`${API_URL}/api/v1/search/text?q=${encodeURIComponent(query)}&limit=20`);
  const data = await response.json();
  return data;
}

function displayResults(results) {
  const container = document.getElementById('search-results');
  container.innerHTML = '';

  results.items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'search-result';
    div.innerHTML = `
      <img src="${item.thumbnail_url || item.storage_url}" alt="${item.filename}" />
      <p>${item.description || item.filename}</p>
      <span>Similarity: ${(item.similarity * 100).toFixed(1)}%</span>
    `;
    container.appendChild(div);
  });
}

document.getElementById('search-button').addEventListener('click', async () => {
  const query = document.getElementById('search-input').value;
  if (query) {
    const results = await searchMedia(query);
    displayResults(results);
  }
});

document.getElementById('search-input').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    document.getElementById('search-button').click();
  }
});
</script>

<style>
#media-search-widget {
  font-family: inherit;
  max-width: 800px;
  margin: 0 auto;
}
#search-input {
  width: 70%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
}
#search-button {
  padding: 12px 24px;
  background: #333;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
#search-results {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
  margin-top: 20px;
}
.search-result {
  border: 1px solid #eee;
  border-radius: 8px;
  overflow: hidden;
}
.search-result img {
  width: 100%;
  height: 150px;
  object-fit: cover;
}
.search-result p {
  padding: 8px;
  margin: 0;
}
</style>
"""

    return {
        "embed_code": embed_code,
        "instructions": [
            "1. Add an Embed element to your Webflow page",
            "2. Paste this code into the embed",
            "3. Replace 'YOUR_API_URL_HERE' with your actual API URL",
            "4. Publish your Webflow site"
        ]
    }
