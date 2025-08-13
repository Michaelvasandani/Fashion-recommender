import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time

from app.config import settings
from app.models import EbayItem
from app.services.oauth_manager import OAuthManager

logger = logging.getLogger(__name__)

class EbayAPIError(Exception):
    pass

class EbayClient:
    def __init__(self):
        self.oauth_manager = OAuthManager()
        self.base_url = settings.EBAY_BROWSE_API_ENDPOINT
        self.market_id = settings.EBAY_MARKET_ID
        self.session = requests.Session()
    
    def search_items(
        self, 
        query: str, 
        limit: int = 50,
        category_id: Optional[str] = None,
        offset: int = 0
    ) -> List[EbayItem]:
        """Search for items using eBay Browse API."""
        if not settings.is_configured:
            raise EbayAPIError("eBay API credentials not configured")
        
        # Get OAuth token
        try:
            access_token = self.oauth_manager.get_access_token()
        except Exception as e:
            raise EbayAPIError(f"Failed to obtain OAuth token: {str(e)}")
        
        # Set headers with OAuth token
        headers = {
            'Authorization': f'Bearer {access_token}',
            'X-EBAY-C-MARKETPLACE-ID': self.market_id,
            'Accept': 'application/json'
        }
        
        # Build query parameters
        params = {
            'q': query,
            'limit': min(limit, 200),  # Browse API max is 200
            'offset': offset
        }
        
        if category_id:
            params['category_ids'] = category_id
        
        # Make the API request
        try:
            response = self.session.get(
                f"{self.base_url}/item_summary/search",
                headers=headers,
                params=params,
                timeout=settings.TIMEOUT_SECONDS
            )
            
            if response.status_code == 401:
                # Token might be expired, clear and retry once
                logger.warning("Got 401, refreshing token and retrying")
                self.oauth_manager.clear_token()
                access_token = self.oauth_manager.get_access_token()
                headers['Authorization'] = f'Bearer {access_token}'
                
                response = self.session.get(
                    f"{self.base_url}/item_summary/search",
                    headers=headers,
                    params=params,
                    timeout=settings.TIMEOUT_SECONDS
                )
            
            response.raise_for_status()
            
            data = response.json()
            return self._parse_response(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"eBay Browse API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise EbayAPIError(f"Failed to fetch items from eBay: {str(e)}")
    
    def _parse_response(self, data: Dict[str, Any]) -> List[EbayItem]:
        """Parse Browse API response into EbayItem objects."""
        items = []
        
        try:
            item_summaries = data.get('itemSummaries', [])
            
            for item_data in item_summaries:
                parsed_item = self._parse_item(item_data)
                if parsed_item:
                    items.append(parsed_item)
                    
        except Exception as e:
            logger.error(f"Error parsing eBay Browse API response: {str(e)}")
            
        return items
    
    def _parse_item(self, item_data: Dict[str, Any]) -> Optional[EbayItem]:
        """Parse individual item from Browse API response."""
        try:
            item_id = item_data.get('itemId')
            title = item_data.get('title')
            
            if not item_id or not title:
                return None
            
            # Extract price
            price_data = item_data.get('price', {})
            price = float(price_data.get('value', 0))
            currency = price_data.get('currency', 'USD')
            
            # Extract shipping cost if available
            shipping_cost = None
            shipping_options = item_data.get('shippingOptions', [])
            if shipping_options:
                shipping_cost_data = shipping_options[0].get('shippingCost', {})
                if shipping_cost_data:
                    shipping_cost = float(shipping_cost_data.get('value', 0))
            
            # Extract image URL
            image = item_data.get('image', {})
            image_url = image.get('imageUrl') if image else None
            
            # Extract condition
            condition = item_data.get('condition')
            
            # Extract location
            item_location = item_data.get('itemLocation', {})
            location = item_location.get('postalCode')
            if not location and 'country' in item_location:
                location = item_location.get('country')
            
            return EbayItem(
                item_id=item_id,
                title=title,
                price=price,
                currency=currency,
                listing_url=item_data.get('itemWebUrl'),
                image_url=image_url,
                condition=condition,
                location=location,
                shipping_cost=shipping_cost,
                listing_type=item_data.get('buyingOptions', ['Unknown'])[0] if item_data.get('buyingOptions') else None,
                end_time=None  # Browse API doesn't return end time in search results
            )
            
        except Exception as e:
            logger.error(f"Error parsing item: {str(e)}")
            return None
    
    def search_items_with_combined_ranking(
        self,
        query: str,
        limit: int = 50,
        category_id: Optional[str] = None,
        offset: int = 0,
        text_weight: float = 0.5,
        visual_weight: float = 0.5,
        use_visual: bool = True
    ) -> List[EbayItem]:
        """Search items and rank them by combined text and visual similarity."""
        from app.services.similarity_service import get_similarity_service
        
        start_time = time.time()
        
        # Get regular search results
        items = self.search_items(query, limit, category_id, offset)
        
        if not items or not settings.ENABLE_SEMANTIC_SEARCH:
            return items
        
        try:
            # Get similarity service
            similarity_service = get_similarity_service()
            
            # Calculate combined similarities
            if use_visual and settings.ENABLE_VISUAL_SEARCH:
                logger.info(f"Using combined text+visual ranking for {len(items)} items")
                items = similarity_service.calculate_combined_similarity(
                    query=query,
                    items=items,
                    text_weight=text_weight,
                    visual_weight=visual_weight,
                    use_async=False  # Could make this configurable
                )
            else:
                # Fall back to text-only ranking
                logger.info(f"Using text-only ranking for {len(items)} items")
                items = self.search_items_with_semantic_ranking(
                    query, limit, category_id, offset
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Combined ranking completed in {elapsed:.2f}s")
            
            return items
            
        except Exception as e:
            logger.error(f"Error in combined ranking: {str(e)}")
            logger.warning("Falling back to text-only semantic ranking")
            return self.search_items_with_semantic_ranking(
                query, limit, category_id, offset
            )
    
    def search_items_with_semantic_ranking(
        self,
        query: str,
        limit: int = 50,
        category_id: Optional[str] = None,
        offset: int = 0
    ) -> List[EbayItem]:
        """Search items and rank them by semantic similarity."""
        from app.services.embedding_service import get_embedding_service
        
        start_time = time.time()
        
        # Get regular search results
        items = self.search_items(query, limit, category_id, offset)
        
        if not items or not settings.ENABLE_SEMANTIC_SEARCH:
            return items
        
        try:
            # Get embedding service
            embedding_service = get_embedding_service()
            
            # Extract titles
            titles = [item.title for item in items]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(titles)} items")
            query_embedding = embedding_service.encode_text(query)
            title_embeddings = embedding_service.encode_batch(titles)
            
            # Calculate similarities
            similarities = embedding_service.calculate_similarities(
                query_embedding,
                title_embeddings
            )
            
            # Add scores to items
            for item, score in zip(items, similarities):
                item.similarity_score = float(score)
            
            # Sort by similarity (highest first)
            items.sort(key=lambda x: x.similarity_score or 0, reverse=True)
            
            elapsed = time.time() - start_time
            logger.info(f"Semantic ranking completed in {elapsed:.2f}s")
            
            return items
            
        except Exception as e:
            logger.error(f"Error in semantic ranking: {str(e)}")
            logger.warning("Falling back to regular search results")
            return items
    
    def get_total_results(self, query: str, category_id: Optional[str] = None) -> int:
        """Get total number of results for a query."""
        # For Browse API, we'd need to make a search request and check the total field
        # This is less efficient than Finding API, but necessary
        try:
            access_token = self.oauth_manager.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'X-EBAY-C-MARKETPLACE-ID': self.market_id,
                'Accept': 'application/json'
            }
            
            params = {
                'q': query,
                'limit': 1,
                'offset': 0
            }
            
            if category_id:
                params['category_ids'] = category_id
            
            response = self.session.get(
                f"{self.base_url}/item_summary/search",
                headers=headers,
                params=params,
                timeout=settings.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('total', 0)
            
        except Exception as e:
            logger.error(f"Error getting total results: {str(e)}")
            return 0