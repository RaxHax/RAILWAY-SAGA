"""
Embedding Service - Handles generation of embeddings from text and media files.
Supports multiple AI models: CLIP, Multilingual CLIP, SigLIP, OpenCLIP.
"""

import io
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _cuda_is_available() -> bool:
    """Safely check whether CUDA can actually be used."""
    try:
        return torch.cuda.is_available()
    except (AssertionError, RuntimeError) as exc:
        logger.warning("CUDA availability check failed: %s", exc)
        return False


def _mps_is_available() -> bool:
    """Check if Apple's Metal (MPS) backend is available."""
    try:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except AttributeError:
        return False


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._embedding_dim = None
        logger.info("Using device '%s' for model '%s'", self.device, self.model_name)
        
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Generate embedding from an image."""
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding from text."""
        pass

    def _resolve_device(self, requested: Optional[str]) -> str:
        """Determine the best available device."""
        if requested:
            resolved = self._validate_device(requested.lower())
            if resolved:
                return resolved
            logger.warning("Requested device '%s' is not available. Falling back to CPU.", requested)
            return "cpu"
        
        for candidate in ("cuda", "mps"):
            resolved = self._validate_device(candidate)
            if resolved:
                return resolved
        return "cpu"

    def _validate_device(self, candidate: str) -> Optional[str]:
        """Validate whether a specific device is usable."""
        if candidate == "cuda":
            return "cuda" if _cuda_is_available() else None
        if candidate == "mps":
            return "mps" if _mps_is_available() else None
        if candidate == "cpu":
            return "cpu"
        return None

    def _move_to_device(self, module):
        """Safely move a torch module to the configured device."""
        if module is None or not hasattr(module, "to"):
            return module
        
        try:
            return module.to(self.device)
        except (RuntimeError, AssertionError) as exc:
            device_info = str(exc).lower()
            if self.device != "cpu" and ("cuda" in device_info or "mps" in device_info):
                logger.warning(
                    "Failed to load model '%s' on %s (%s). Falling back to CPU.",
                    self.model_name,
                    self.device,
                    exc
                )
                self.device = "cpu"
                return module.to(self.device)
            raise
    
    def encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Generate embeddings for a batch of images."""
        return np.array([self.encode_image(img) for img in images])
    
    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        return np.array([self.encode_text(text) for text in texts])


class CLIPModel(BaseEmbeddingModel):
    """Standard CLIP model from OpenAI/HuggingFace."""
    
    def load_model(self) -> None:
        from transformers import CLIPModel as HFCLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP model: {self.model_name}")
        try:
            model = HFCLIPModel.from_pretrained(self.model_name)
            self.model = self._move_to_device(model)

            # Load processor with explicit use_fast parameter to avoid version issues
            logger.debug(f"Loading CLIP processor for {self.model_name}")
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                use_fast=False  # Explicitly use slow processor for compatibility
            )
            self.model.eval()

            # Determine embedding dimension
            self._embedding_dim = self.model.config.projection_dim
            logger.info(f"CLIP model loaded successfully. Embedding dim: {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model {self.model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load CLIP model: {e}") from e
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()


class MultilingualCLIPModel(BaseEmbeddingModel):
    """Multilingual CLIP model supporting 100+ languages."""
    
    def load_model(self) -> None:
        from transformers import CLIPModel as HFCLIPModel, CLIPProcessor
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading Multilingual CLIP model: {self.model_name}")
        base_clip = "openai/clip-vit-base-patch32"
        
        # For multilingual CLIP, we use sentence-transformers for text
        # and standard CLIP processor for images
        if "sentence-transformers" in self.model_name:
            self.text_model = SentenceTransformer(self.model_name, device=self.device)
        else:
            try:
                from multilingual_clip import pt_multilingual_clip
                self.text_model, self.tokenizer = pt_multilingual_clip.load_model(self.model_name)
            except ImportError:
                logger.warning(
                    "multilingual-clip package not available. Falling back to sentence-transformers "
                    "implementation for multilingual embeddings."
                )
                fallback_model = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
                self.text_model = SentenceTransformer(fallback_model, device=self.device)
                self.tokenizer = None
            except Exception as exc:
                logger.error("Failed to load multilingual CLIP model '%s': %s", self.model_name, exc)
                raise
        
        # Use base CLIP for image encoding
        base_model = HFCLIPModel.from_pretrained(base_clip)
        self.model = self._move_to_device(base_model)
        self.processor = CLIPProcessor.from_pretrained(base_clip)
        
        self.model.eval()
        self._embedding_dim = 512
        logger.info(f"Multilingual CLIP loaded. Embedding dim: {self._embedding_dim}")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def encode_text(self, text: str) -> np.ndarray:
        if hasattr(self, 'text_model') and hasattr(self.text_model, 'encode'):
            # Sentence transformers style
            embedding = self.text_model.encode([text], convert_to_numpy=True)
            embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
            return embedding.flatten()
        else:
            # M-CLIP style
            import multilingual_clip.pt_multilingual_clip as mclip
            with torch.no_grad():
                embedding = mclip.forward(self.text_model, self.tokenizer, [text])
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten()


class SigLIPModel(BaseEmbeddingModel):
    """Google's SigLIP model - improved CLIP alternative."""
    
    def load_model(self) -> None:
        from transformers import AutoModel, AutoProcessor
        
        logger.info(f"Loading SigLIP model: {self.model_name}")
        model = AutoModel.from_pretrained(self.model_name)
        self.model = self._move_to_device(model)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        
        self._embedding_dim = self.model.config.vision_config.hidden_size
        logger.info(f"SigLIP model loaded. Embedding dim: {self._embedding_dim}")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            features = outputs / outputs.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            features = outputs / outputs.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()


class OpenCLIPModel(BaseEmbeddingModel):
    """OpenCLIP models trained on LAION datasets."""
    
    def load_model(self) -> None:
        import open_clip
        
        logger.info(f"Loading OpenCLIP model: {self.model_name}")
        
        # Parse model name to get architecture and pretrained weights
        if "ViT-H-14" in self.model_name:
            model_arch = "ViT-H-14"
            pretrained = "laion2b_s32b_b79k"
        elif "ViT-L-14" in self.model_name:
            model_arch = "ViT-L-14"
            pretrained = "laion2b_s32b_b82k"
        else:
            model_arch = "ViT-B-32"
            pretrained = "laion2b_s34b_b79k"
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_arch, pretrained=pretrained, device=self.device
            )
        except (RuntimeError, AssertionError) as exc:
            if self.device != "cpu" and "cuda" in str(exc).lower():
                logger.warning(
                    "Failed to initialize OpenCLIP model '%s' on %s (%s). Retrying on CPU.",
                    self.model_name,
                    self.device,
                    exc
                )
                self.device = "cpu"
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_arch, pretrained=pretrained, device=self.device
                )
            else:
                raise
        self.tokenizer = open_clip.get_tokenizer(model_arch)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_text = self.tokenizer(["test"]).to(self.device)
            dummy_features = self.model.encode_text(dummy_text)
            self._embedding_dim = dummy_features.shape[-1]
        
        logger.info(f"OpenCLIP model loaded. Embedding dim: {self._embedding_dim}")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
    
    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()


class XLMRobertaModel(BaseEmbeddingModel):
    """XLM-RoBERTa model - multilingual text-only transformer (100+ languages)."""

    def load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading XLM-RoBERTa model: {self.model_name}")
        try:
            model = AutoModel.from_pretrained(self.model_name)
            self.model = self._move_to_device(model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.eval()

            # XLM-RoBERTa-large has 1024 hidden dimensions
            self._embedding_dim = self.model.config.hidden_size
            logger.info(f"XLM-RoBERTa model loaded successfully. Embedding dim: {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load XLM-RoBERTa model {self.model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load XLM-RoBERTa model: {e}") from e

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """XLM-RoBERTa is a text-only model and does not support image encoding."""
        raise NotImplementedError(
            "XLM-RoBERTa is a text-only model. Image encoding is not supported. "
            "Please use a vision-language model like CLIP for image encoding."
        )

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding from text using mean pooling over token embeddings.
        """
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get model outputs
            outputs = self.model(**inputs)

            # Use mean pooling over token embeddings
            # Take the mean of all token embeddings (excluding padding)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Expand attention mask to match token embeddings dimensions
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            # Sum embeddings and divide by number of tokens (mean pooling)
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask

            # Normalize the embedding
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().flatten()


class EmbeddingService:
    """
    Main embedding service that manages model loading and provides
    a unified interface for generating embeddings.
    """

    MODEL_CLASSES = {
        "clip": CLIPModel,
        "multilingual-clip": MultilingualCLIPModel,
        "siglip": SigLIPModel,
        "openclip": OpenCLIPModel,
        "xlm-roberta": XLMRobertaModel,
    }
    
    def __init__(self, model_type: str = "clip", model_name: str = "openai/clip-vit-base-patch32"):
        self.model_type = model_type
        self.model_name = model_name
        self._model: Optional[BaseEmbeddingModel] = None
        logger.debug(f"EmbeddingService initialized with model_type={model_type}, model_name={model_name}")
        
    @property
    def model(self) -> BaseEmbeddingModel:
        if self._model is None:
            logger.debug(f"Lazy loading model: {self.model_name}")
            self.load_model()
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        return self.model.embedding_dim
    
    def load_model(self) -> None:
        """Load the configured embedding model."""
        if self.model_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {self.model_type}")

        try:
            logger.info(f"Loading embedding model: type={self.model_type}, name={self.model_name}")
            model_class = self.MODEL_CLASSES[self.model_type]
            self._model = model_class(self.model_name)
            self._model.load_model()
            logger.info(f"Embedding service ready with {self.model_type} model (dim={self._model.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_type}/{self.model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e
    
    def encode_image(self, image: Union[Image.Image, bytes, str, Path]) -> np.ndarray:
        """
        Generate embedding from an image.
        
        Args:
            image: PIL Image, bytes, file path, or URL
            
        Returns:
            Normalized embedding vector as numpy array
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return self.model.encode_image(image)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding from text.
        
        Args:
            text: Text description to encode
            
        Returns:
            Normalized embedding vector as numpy array
        """
        return self.model.encode_text(text)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2))
    
    def encode_combined(
        self, 
        image: Optional[Union[Image.Image, bytes]] = None,
        text: Optional[str] = None,
        image_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> np.ndarray:
        """
        Generate a combined embedding from both image and text.
        
        Args:
            image: Optional image input
            text: Optional text description
            image_weight: Weight for image embedding (default 0.7)
            text_weight: Weight for text embedding (default 0.3)
            
        Returns:
            Combined normalized embedding
        """
        embeddings = []
        weights = []
        
        if image is not None:
            img_emb = self.encode_image(image)
            embeddings.append(img_emb)
            weights.append(image_weight)
        
        if text is not None:
            txt_emb = self.encode_text(text)
            embeddings.append(txt_emb)
            weights.append(text_weight)
        
        if not embeddings:
            raise ValueError("At least one of image or text must be provided")
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Weighted combination
        combined = sum(w * e for w, e in zip(weights, embeddings))
        combined = combined / np.linalg.norm(combined)
        
        return combined


# Factory function for creating embedding service
def create_embedding_service(model_key: str = "clip-vit-base-patch32") -> EmbeddingService:
    """
    Create an embedding service with the specified model.
    
    Args:
        model_key: Key from ModelConfig.AVAILABLE_MODELS
        
    Returns:
        Configured EmbeddingService instance
    """
    from config.settings import ModelConfig
    
    if model_key not in ModelConfig.AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_info = ModelConfig.AVAILABLE_MODELS[model_key]
    return EmbeddingService(
        model_type=model_info["type"],
        model_name=model_info["name"]
    )
