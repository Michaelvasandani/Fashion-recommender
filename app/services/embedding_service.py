import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings using Sentence-BERT."""
    
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        logger.info(f"EmbeddingService initialized with device: {self._device}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the Sentence-BERT model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device
            )
            logger.info(f"Model loaded successfully on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: The text to encode
            
        Returns:
            Numpy array of shape (embedding_dimension,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for encoding")
            return np.zeros(self.embedding_dimension)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # Unit norm for cosine similarity
            )
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return np.zeros(self.embedding_dimension)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array of shape (n_texts, embedding_dimension)
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            logger.warning("No valid texts to encode")
            return np.zeros((len(texts), self.embedding_dimension))
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            
            # Handle any texts that were filtered out
            if len(valid_texts) < len(texts):
                full_embeddings = np.zeros((len(texts), self.embedding_dimension))
                valid_idx = 0
                for i, text in enumerate(texts):
                    if text and text.strip():
                        full_embeddings[i] = embeddings[valid_idx]
                        valid_idx += 1
                return full_embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch encoding: {str(e)}")
            return np.zeros((len(texts), self.embedding_dimension))
    
    def calculate_similarities(
        self,
        query_embedding: np.ndarray,
        item_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarities between query and items.
        
        Args:
            query_embedding: Query embedding vector
            item_embeddings: Matrix of item embeddings
            
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
    
    def rank_by_similarity(
        self,
        query: str,
        items: List[dict],
        text_field: str = 'title'
    ) -> List[dict]:
        """
        Rank items by semantic similarity to query.
        
        Args:
            query: Search query
            items: List of items to rank
            text_field: Field name containing text to compare
            
        Returns:
            Items sorted by similarity score (highest first)
        """
        if not items:
            return []
        
        # Extract texts
        texts = [item.get(text_field, '') for item in items]
        
        # Generate embeddings
        query_embedding = self.encode_text(query)
        item_embeddings = self.encode_batch(texts)
        
        # Calculate similarities
        similarities = self.calculate_similarities(query_embedding, item_embeddings)
        
        # Add scores to items and sort
        for item, score in zip(items, similarities):
            item['similarity_score'] = float(score)
        
        # Sort by similarity (highest first)
        sorted_items = sorted(
            items,
            key=lambda x: x.get('similarity_score', 0),
            reverse=True
        )
        
        return sorted_items

# Singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create the singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service