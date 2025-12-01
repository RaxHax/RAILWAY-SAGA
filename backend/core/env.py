"""Environment helpers for backend configuration."""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Alternative variable names sometimes used by mistake or legacy setups
_SUPABASE_URL_ALIASES = ("SPABASE_URL",)
_SUPABASE_KEY_ALIASES = ("SPABASE_KEY",)


def _get_env_value(preferred: str, aliases: Tuple[str, ...]) -> Optional[str]:
    """Return the first available env var value among preferred name and aliases."""
    for name in (preferred, *aliases):
        value = os.environ.get(name)
        if value:
            if name != preferred:
                logger.warning(
                    "Using environment variable %s; please rename to %s for consistency.",
                    name,
                    preferred,
                )
            return value
    return None


def get_supabase_env() -> Tuple[Optional[str], Optional[str]]:
    """Fetch Supabase URL and key from environment, allowing common aliases."""
    url = _get_env_value("SUPABASE_URL", _SUPABASE_URL_ALIASES)
    key = _get_env_value("SUPABASE_KEY", _SUPABASE_KEY_ALIASES)
    return url, key
