"""Dependency injection for FastAPI endpoints."""

import logging
import os
import threading
from typing import Optional, Dict, Any, Union

from fastapi import HTTPException
from pydantic import HttpUrl

from config.settings import ModelConfig
from config.user_config import ConfigManager, DEFAULT_MODEL_KEY

logger = logging.getLogger(__name__)

# Global search engine instance + lock
search_engine = None
engine_lock = threading.Lock()
config_manager = ConfigManager()

# Demo mode flag
DEMO_MODE = False


def _active_credentials() -> Optional[Dict[str, Any]]:
    """Return the currently configured Supabase credentials, preferring env vars."""
    env_url = os.environ.get("SUPABASE_URL")
    env_key = os.environ.get("SUPABASE_KEY")
    env_model = os.environ.get("MODEL_KEY")

    if env_url and env_key:
        return {
            "url": env_url,
            "key": env_key,
            "model_key": env_model or DEFAULT_MODEL_KEY,
        }

    stored = config_manager.get_supabase_credentials()
    if stored:
        return stored

    return None


def _build_search_engine() -> Optional["MediaSearchEngine"]:
    """Instantiate the MediaSearchEngine if credentials are available."""
    creds = _active_credentials()
    if not creds:
        return None

    model_key = creds.get("model_key") or DEFAULT_MODEL_KEY

    try:
        from backend.search_engine import MediaSearchEngine

        engine = MediaSearchEngine.from_config(
            model_key=model_key,
            supabase_url=creds["url"],
            supabase_key=creds["key"],
        )
        logger.info("Search engine initialized with model: %s", model_key)
        return engine
    except Exception as exc:
        logger.error("Failed to initialize search engine: %s", exc)
        return None


def refresh_search_engine() -> Optional["MediaSearchEngine"]:
    """Rebuild the global search engine instance (thread-safe)."""
    global search_engine

    with engine_lock:
        search_engine = _build_search_engine()
        return search_engine


def initialize_search_engine():
    """Initialize the search engine on startup."""
    global search_engine
    logger.info("Initializing Media Search Engine...")
    search_engine = _build_search_engine()
    if search_engine is None:
        logger.warning("Supabase credentials not configured. Onboarding wizard is available.")


def get_search_engine():
    """Dependency to get the search engine instance."""
    global DEMO_MODE
    if search_engine is None:
        DEMO_MODE = True
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Set SUPABASE_URL and SUPABASE_KEY environment variables."
        )
    return search_engine


def get_optional_search_engine():
    """Dependency that returns None instead of raising if engine not available."""
    return search_engine


def _test_supabase_connection(url: Union[str, HttpUrl], key: str, model_key: str) -> None:
    """
    Perform a lightweight Supabase connectivity check.
    Verifies credentials and that the media_items table exists.
    """
    from backend.database_service import SupabaseService

    model_info = ModelConfig.AVAILABLE_MODELS.get(model_key, ModelConfig.AVAILABLE_MODELS[DEFAULT_MODEL_KEY])
    url_value = str(url)

    temp_service = SupabaseService(
        url=url_value,
        key=key,
        vector_dimension=model_info["embedding_dim"],
    )

    # Try a harmless query; if the schema isn't applied yet this will raise.
    temp_service.client.table(temp_service.media_table).select("id").limit(1).execute()
