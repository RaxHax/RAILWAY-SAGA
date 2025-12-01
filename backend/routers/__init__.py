"""API route handlers organized by domain."""

from . import health, ingestion, search, media, configuration, setup, admin

__all__ = ["health", "ingestion", "search", "media", "configuration", "setup", "admin"]
