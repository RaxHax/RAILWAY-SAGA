"""Central configuration objects for the Media Semantic Search Engine."""

from dataclasses import dataclass
from typing import Dict


class ModelConfig:
    """Registry of supported embedding models."""

    DEFAULT_MODEL_KEY = "xlm-roberta-large-vit-b-32"

    AVAILABLE_MODELS: Dict[str, Dict[str, object]] = {
        "clip-vit-base-patch32": {
            "type": "clip",
            "name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "multilingual": False,
            "description": "Balanced speed/quality baseline CLIP model.",
            "languages": ["en"],
        },
        "clip-vit-large-patch14": {
            "type": "clip",
            "name": "openai/clip-vit-large-patch14",
            "embedding_dim": 768,
            "multilingual": False,
            "description": "Higher quality CLIP model with larger embeddings.",
            "languages": ["en"],
        },
        "siglip-base": {
            "type": "clip",
            "name": "google/siglip-base-patch16-224",
            "embedding_dim": 768,
            "multilingual": True,
            "description": "SigLIP variant with improved multilingual understanding.",
            "languages": ["en", "multilingual"],
        },
        "xlm-roberta-large-vit-b-32": {
            "type": "multilingual-clip",
            "name": "M-CLIP/XLM-Roberta-Large-Vit-B-32",
            "embedding_dim": 512,
            "multilingual": True,
            "description": "M-CLIP: Multilingual vision-language model with XLM-RoBERTa-Large text encoder and ViT-B-32 vision encoder. Supports 100+ languages.",
            "languages": ["multilingual"],
        },
    }


@dataclass
class APIConfig:
    """API-level defaults that can be reused by the application."""

    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    default_page_size: int = 20
