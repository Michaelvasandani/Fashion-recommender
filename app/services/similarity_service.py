import logging
from typing import List, Optional, Dict, Any
import numpy as np
import asyncio
import time

from app.models import EbayItem
from app.services.embedding_service import get_embedding_service
from app.services.visual_embedding_service import get_visual_embedding_service
from app.config import settings

logger = logging.getLogger(__name__)

class SimilarityService:
    """Service for combining text and visual similarities."""
    
    def __init__(self):
        self.text_service = get_embedding_service()
        self.visual_service = get_visual_embedding_service()
        
    def calculate_combined_similarity(
        self,
        query: str,
        items: List[EbayItem],
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        use_async: bool = False
    ) -> List[EbayItem]:
        """
        Calculate combined text and visual similarity scores.
        
        Args:
            query: Search query text
            items: List of EbayItem objects
            text_weight: Weight for text similarity (0-1)
            visual_weight: Weight for visual similarity (0-1)
            use_async: Whether to use async image downloading
            
        Returns:
            Items with combined similarity scores, sorted by relevance
        """
        if not items:
            return []
        
        # Normalize weights
        total_weight = text_weight + visual_weight
        if total_weight > 0:
            text_weight = text_weight / total_weight
            visual_weight = visual_weight / total_weight
        else:
            text_weight = visual_weight = 0.5
        
        start_time = time.time()
        
        # Extract data
        titles = [item.title for item in items]
        image_urls = [item.image_url for item in items]
        
        # Calculate text similarities
        logger.info(f"Calculating text similarities for {len(items)} items")
        text_similarities = self._calculate_text_similarities(query, titles)
        
        # Calculate visual similarities
        logger.info(f"Calculating visual similarities for {len(items)} items")
        if use_async:
            visual_similarities = asyncio.run(
                self._calculate_visual_similarities_async(image_urls)
            )
        else:
            visual_similarities = self._calculate_visual_similarities(image_urls)
        
        # Combine scores
        for i, item in enumerate(items):
            text_score = text_similarities[i] if i < len(text_similarities) else 0
            visual_score = visual_similarities[i] if i < len(visual_similarities) else 0
            
            # Store individual scores
            item.similarity_score = float(text_score)  # Keep text score as primary
            
            # Calculate combined score
            combined_score = (text_weight * text_score) + (visual_weight * visual_score)
            
            # Add additional metadata
            if not hasattr(item, 'similarity_details'):
                item.similarity_details = {
                    'text_score': float(text_score),
                    'visual_score': float(visual_score),
                    'combined_score': float(combined_score),
                    'text_weight': text_weight,
                    'visual_weight': visual_weight
                }
        
        # Sort by combined score
        sorted_items = sorted(
            items,
            key=lambda x: x.similarity_details.get('combined_score', 0) if hasattr(x, 'similarity_details') else 0,
            reverse=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Combined similarity calculation completed in {elapsed:.2f}s")
        
        return sorted_items
    
    def _calculate_text_similarities(self, query: str, titles: List[str]) -> np.ndarray:
        """Calculate text similarities."""
        try:
            query_embedding = self.text_service.encode_text(query)
            title_embeddings = self.text_service.encode_batch(titles)
            similarities = self.text_service.calculate_similarities(
                query_embedding,
                title_embeddings
            )
            return similarities
        except Exception as e:
            logger.error(f"Error calculating text similarities: {e}")
            return np.zeros(len(titles))
    
    def _calculate_visual_similarities(self, image_urls: List[Optional[str]]) -> np.ndarray:
        """Calculate visual similarities between first image and all others."""
        # Filter out None URLs
        valid_urls = [url for url in image_urls if url]
        
        if len(valid_urls) < 2:
            return np.zeros(len(image_urls))
        
        try:
            # Use first valid image as query
            embeddings = self.visual_service.encode_image_batch(valid_urls)
            
            if embeddings.size == 0:
                return np.zeros(len(image_urls))
            
            # Calculate similarity of each image to the first one
            query_embedding = embeddings[0]
            similarities = self.visual_service.calculate_visual_similarities(
                query_embedding,
                embeddings
            )
            
            # Map back to original indices
            full_similarities = np.zeros(len(image_urls))
            valid_idx = 0
            for i, url in enumerate(image_urls):
                if url:
                    full_similarities[i] = similarities[valid_idx]
                    valid_idx += 1
            
            return full_similarities
            
        except Exception as e:
            logger.error(f"Error calculating visual similarities: {e}")
            return np.zeros(len(image_urls))
    
    async def _calculate_visual_similarities_async(self, image_urls: List[Optional[str]]) -> np.ndarray:
        """Async version of visual similarity calculation."""
        valid_urls = [url for url in image_urls if url]
        
        if len(valid_urls) < 2:
            return np.zeros(len(image_urls))
        
        try:
            embeddings = await self.visual_service.encode_image_batch_async(valid_urls)
            
            if embeddings.size == 0:
                return np.zeros(len(image_urls))
            
            query_embedding = embeddings[0]
            similarities = self.visual_service.calculate_visual_similarities(
                query_embedding,
                embeddings
            )
            
            # Map back to original indices
            full_similarities = np.zeros(len(image_urls))
            valid_idx = 0
            for i, url in enumerate(image_urls):
                if url:
                    full_similarities[i] = similarities[valid_idx]
                    valid_idx += 1
            
            return full_similarities
            
        except Exception as e:
            logger.error(f"Error in async visual similarity calculation: {e}")
            return np.zeros(len(image_urls))
    
    def find_similar_by_image(
        self,
        query_image_url: str,
        items: List[EbayItem],
        include_text: bool = True,
        text_weight: float = 0.3,
        visual_weight: float = 0.7
    ) -> List[EbayItem]:
        """
        Find items similar to a query image.
        
        Args:
            query_image_url: URL of the query image
            items: List of items to search through
            include_text: Whether to include text similarity
            text_weight: Weight for text (if included)
            visual_weight: Weight for visual similarity
            
        Returns:
            Items sorted by similarity to query image
        """
        if not items or not query_image_url:
            return items
        
        # Get query image embedding
        query_embedding = self.visual_service.encode_image_from_url(query_image_url)
        if query_embedding is None:
            logger.error("Failed to encode query image")
            return items
        
        # Get item embeddings
        image_urls = [item.image_url for item in items]
        item_embeddings = self.visual_service.encode_image_batch(image_urls)
        
        # Calculate visual similarities
        visual_similarities = self.visual_service.calculate_visual_similarities(
            query_embedding,
            item_embeddings
        )
        
        # Add scores to items
        for i, item in enumerate(items):
            visual_score = visual_similarities[i] if i < len(visual_similarities) else 0
            
            if include_text and hasattr(item, 'similarity_score'):
                # Combine with existing text score
                text_score = item.similarity_score
                combined_score = (text_weight * text_score) + (visual_weight * visual_score)
            else:
                combined_score = visual_score
            
            item.similarity_details = {
                'visual_score': float(visual_score),
                'combined_score': float(combined_score),
                'query_image': query_image_url
            }
        
        # Sort by score
        sorted_items = sorted(
            items,
            key=lambda x: x.similarity_details.get('combined_score', 0) if hasattr(x, 'similarity_details') else 0,
            reverse=True
        )
        
        return sorted_items

# Singleton instance
_similarity_service = None

def get_similarity_service() -> SimilarityService:
    """Get or create the singleton similarity service instance."""
    global _similarity_service
    if _similarity_service is None:
        _similarity_service = SimilarityService()
    return _similarity_service