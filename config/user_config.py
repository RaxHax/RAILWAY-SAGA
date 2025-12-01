"""Lightweight persistence for local setup wizard state."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

from config.settings import ModelConfig

DEFAULT_MODEL_KEY = ModelConfig.DEFAULT_MODEL_KEY


@dataclass
class ConfigManager:
    """Persist Supabase credentials and onboarding progress to disk."""

    config_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "user_config.json")

    def __post_init__(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self) -> None:
        self.config_path.write_text(json.dumps(self._data, indent=2))

    def get_supabase_credentials(self) -> Optional[Dict[str, str]]:
        credentials = self._data.get("supabase")
        if not credentials:
            return None
        if not credentials.get("url") or not credentials.get("key"):
            return None
        return credentials

    def update_supabase_credentials(self, url: str, key: str, model_key: str = DEFAULT_MODEL_KEY) -> None:
        if not url or not key:
            raise ValueError("Supabase URL and key are required")
        if model_key not in ModelConfig.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model key: {model_key}")

        self._data["supabase"] = {
            "url": url,
            "key": key,
            "model_key": model_key or DEFAULT_MODEL_KEY,
        }
        steps = self._data.setdefault("onboarding_steps", {})
        steps["1"] = True
        self._save()

    def get_status(self, engine_ready: bool = False) -> Dict[str, Any]:
        supabase = self.get_supabase_credentials() or {}
        steps = self._data.get("onboarding_steps", {})

        has_credentials = bool(supabase.get("url") and supabase.get("key"))
        masked_key = None
        if supabase.get("key"):
            key_value = supabase["key"]
            masked_key = f"{key_value[:4]}...{key_value[-4:]}" if len(key_value) >= 8 else "***"

        instructions = [
            {"step": 1, "title": "Add Supabase credentials", "complete": has_credentials},
            {"step": 2, "title": "Apply database schema", "complete": steps.get("2", False)},
            {"step": 3, "title": "Create media-files bucket", "complete": steps.get("3", False)},
            {"step": 4, "title": "Start indexing and search", "complete": engine_ready},
        ]

        next_step = next((item["step"] for item in instructions if not item["complete"]), len(instructions) + 1)

        return {
            "configured": has_credentials and engine_ready,
            "engine_ready": engine_ready,
            "has_supabase_credentials": has_credentials,
            "supabase_url": supabase.get("url"),
            "supabase_url_masked": supabase.get("url"),
            "supabase_key_masked": masked_key,
            "model_key": supabase.get("model_key", DEFAULT_MODEL_KEY),
            "instructions": instructions,
            "next_step": next_step,
        }

    def mark_step_complete(self, step: int) -> None:
        steps = self._data.setdefault("onboarding_steps", {})
        steps[str(step)] = True
        self._save()
