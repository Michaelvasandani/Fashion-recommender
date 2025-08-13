import logging
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
import hashlib
import os
from pathlib import Path
import asyncio
import aiohttp
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
import time

from app.config import settings

logger = logging.getLogger(__name__)

class VisualEmbeddingService:
    """Service for generating visual embeddings using CLIP."""
    
    def __init__(self):
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "openai/clip-vit-base-patch32"
        self.embedding_dimension = 512
        self.image_cache_dir = Path("./image_cache")
        self.image_cache_dir.mkdir(exist_ok=True)
        logger.info(f"VisualEmbeddingService initialized with device: {self._device}")
    
    @property
    def model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Lazy load the CLIP model and processor on first use."""
        if self._model is None or self._processor is None:
            self._load_model()
        return self._model, self._processor
    
    def _load_model(self):
        """Load the CLIP model and processor."""
        try:
            logger.info(f"Loading CLIP model: {self.model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"CLIP model loaded successfully on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise Exception(f"Failed to load CLIP model: {str(e)}")
    
    def _get_image_cache_path(self, url: str) -> Path:
        """Generate cache path for an image URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.image_cache_dir / f"{url_hash}.jpg"
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL with caching."""
        if not url:
            return None
        
        # Check cache first
        cache_path = self._get_image_cache_path(url)
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load cached image: {e}")
                cache_path.unlink()  # Remove corrupted cache
        
        # Download image
        try:
            response = requests.get(url, timeout=5, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; FashionRecommender/1.0)'
            })
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Save to cache
            try:
                image.save(cache_path, 'JPEG', quality=85)
            except Exception as e:
                logger.warning(f"Failed to cache image: {e}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None
    
    async def _download_image_async(self, session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
        """Async version of image download."""
        if not url:
            return None
        
        # Check cache first
        cache_path = self._get_image_cache_path(url)
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert('RGB')
            except Exception:
                cache_path.unlink()
        
        # Download image
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    content = await response.read()
                    image = Image.open(BytesIO(content)).convert('RGB')
                    
                    # Save to cache
                    try:
                        image.save(cache_path, 'JPEG', quality=85)
                    except Exception:
                        pass
                    
                    return image
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image into an embedding vector.
        
        Args:
            image: PIL Image object
            
        Returns:
            Numpy array of shape (embedding_dimension,)
        """
        try:
            model, processor = self.model
            
            # Process image
            inputs = processor(images=image, return_tensors="pt").to(self._device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().squeeze()
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return np.zeros(self.embedding_dimension)
    
    def encode_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """
        Download and encode an image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            Embedding vector or None if failed
        """
        image = self._download_image(url)
        if image is None:
            return None
        
        return self.encode_image(image)
    
    def encode_image_batch(self, urls: List[str]) -> np.ndarray:
        """
        Encode multiple images from URLs.
        
        Args:
            urls: List of image URLs
            
        Returns:
            Numpy array of shape (n_images, embedding_dimension)
        """
        if not urls:
            return np.array([])
        
        embeddings = []
        
        # Download images
        images = []
        for url in urls:
            image = self._download_image(url)
            images.append(image)
        
        # Process valid images
        valid_images = [img for img in images if img is not None]
        
        if not valid_images:
            logger.warning("No valid images to encode")
            return np.zeros((len(urls), self.embedding_dimension))
        
        try:
            model, processor = self.model
            
            # Process all images at once
            inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(self._device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings_array = image_features.cpu().numpy()
            
            # Handle None images
            full_embeddings = np.zeros((len(urls), self.embedding_dimension))
            valid_idx = 0
            for i, image in enumerate(images):
                if image is not None:
                    full_embeddings[i] = embeddings_array[valid_idx]
                    valid_idx += 1
            
            return full_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch image encoding: {str(e)}")
            return np.zeros((len(urls), self.embedding_dimension))
    
    async def encode_image_batch_async(self, urls: List[str]) -> np.ndarray:
        """
        Async version of batch image encoding.
        
        Args:
            urls: List of image URLs
            
        Returns:
            Numpy array of shape (n_images, embedding_dimension)
        """
        if not urls:
            return np.array([])
        
        # Download images asynchronously
        async with aiohttp.ClientSession() as session:
            tasks = [self._download_image_async(session, url) for url in urls]
            images = await asyncio.gather(*tasks)
        
        # Process valid images
        valid_images = [img for img in images if img is not None]
        
        if not valid_images:
            return np.zeros((len(urls), self.embedding_dimension))
        
        # The actual encoding part needs to be synchronous
        return await asyncio.get_event_loop().run_in_executor(
            None, self._encode_images_sync, images, len(urls)
        )
    
    def _encode_images_sync(self, images: List[Optional[Image.Image]], total_count: int) -> np.ndarray:
        """Synchronous helper for encoding images."""
        valid_images = [img for img in images if img is not None]
        
        if not valid_images:
            return np.zeros((total_count, self.embedding_dimension))
        
        try:
            model, processor = self.model
            
            inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(self._device)
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings_array = image_features.cpu().numpy()
            
            # Handle None images
            full_embeddings = np.zeros((total_count, self.embedding_dimension))
            valid_idx = 0
            for i, image in enumerate(images):
                if image is not None:
                    full_embeddings[i] = embeddings_array[valid_idx]
                    valid_idx += 1
            
            return full_embeddings
            
        except Exception as e:
            logger.error(f"Error encoding images: {str(e)}")
            return np.zeros((total_count, self.embedding_dimension))
    
    def calculate_visual_similarities(
        self,
        query_embedding: np.ndarray,
        item_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarities between query image and item images.
        
        Args:
            query_embedding: Query image embedding
            item_embeddings: Matrix of item image embeddings
            
        Returns:
            Array of similarity scores
        """
        if query_embedding.size == 0 or item_embeddings.size == 0:
            return np.array([])
        
        # Reshape query for sklearn
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, item_embeddings)[0]
        
        return similarities
    
    def clean_cache(self, max_age_hours: int = 24):
        """Clean old cached images."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for cache_file in self.image_cache_dir.glob("*.jpg"):
            if current_time - cache_file.stat().st_mtime > max_age_seconds:
                try:
                    cache_file.unlink()
                    logger.info(f"Removed old cache file: {cache_file}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file: {e}")

# Singleton instance
_visual_embedding_service = None

def get_visual_embedding_service() -> VisualEmbeddingService:
    """Get or create the singleton visual embedding service instance."""
    global _visual_embedding_service
    if _visual_embedding_service is None:
        _visual_embedding_service = VisualEmbeddingService()
    return _visual_embedding_service