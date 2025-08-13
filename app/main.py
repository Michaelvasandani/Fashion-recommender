from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import logging
import hashlib

from app.config import settings
from app.models import SearchResponse, ErrorResponse, RecommendRequest, RecommendResponse, ImageSearchRequest
from app.services.ebay_client import EbayClient, EbayAPIError
from app.services.similarity_service import get_similarity_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Secondhand Fashion Recommender API",
    description="Find secondhand fashion items from eBay",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ebay_client = EbayClient()

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Secondhand Fashion Recommender API",
        "description": "AI-powered fashion recommendations using text and visual similarity",
        "endpoints": {
            "/search": "Search with semantic and visual ranking",
            "/recommend": "Multi-modal recommendations (text/image/both)",
            "/search/by-image": "Search by image similarity",
            "/find-similar/{item_id}": "Find items similar to a specific listing",
            "/docs": "Interactive API documentation",
            "/health": "Health check"
        },
        "features": [
            "Text embeddings with Sentence-BERT",
            "Visual embeddings with CLIP",
            "Combined similarity scoring",
            "Configurable ranking weights",
            "Image-based search"
        ]
    }

@app.get("/search", response_model=SearchResponse)
async def search_items(
    q: str = Query(..., description="Search query (e.g., 'Zara jacket')", min_length=1),
    limit: int = Query(25, description="Number of results to return", ge=1, le=200),
    page: int = Query(1, description="Page number", ge=1),
    category_id: Optional[str] = Query(None, description="eBay category ID"),
    semantic_ranking: bool = Query(True, description="Enable semantic ranking of results"),
    visual_ranking: bool = Query(True, description="Enable visual similarity ranking"),
    text_weight: float = Query(0.5, description="Weight for text similarity (0-1)", ge=0, le=1),
    visual_weight: float = Query(0.5, description="Weight for visual similarity (0-1)", ge=0, le=1)
):
    if not settings.is_configured:
        raise HTTPException(
            status_code=500,
            detail="eBay API credentials not configured. Please set EBAY_APP_ID and EBAY_CERT_ID."
        )
    
    try:
        logger.info(f"Searching for: {q}, limit: {limit}, page: {page}, semantic: {semantic_ranking}, visual: {visual_ranking}")
        
        # Calculate offset from page number
        offset = (page - 1) * limit
        
        # Determine which ranking method to use
        if semantic_ranking and visual_ranking:
            # Use combined text + visual ranking
            items = ebay_client.search_items_with_combined_ranking(
                query=q,
                limit=limit,
                offset=offset,
                category_id=category_id,
                text_weight=text_weight,
                visual_weight=visual_weight,
                use_visual=True
            )
        elif semantic_ranking:
            # Use text-only semantic ranking
            items = ebay_client.search_items_with_semantic_ranking(
                query=q,
                limit=limit,
                offset=offset,
                category_id=category_id
            )
        else:
            # Use basic eBay search without ranking
            items = ebay_client.search_items(
                query=q,
                limit=limit,
                offset=offset,
                category_id=category_id
            )
        
        total_results = ebay_client.get_total_results(q, category_id)
        
        return SearchResponse(
            query=q,
            total_results=total_results,
            page_number=page,
            items_per_page=len(items),
            items=items
        )
        
    except EbayAPIError as e:
        logger.error(f"eBay API error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ebay_configured": settings.is_configured
    }

@app.get("/webhooks/account-deletion")
async def validate_webhook(
    challenge_code: str = Query(..., description="eBay challenge code for validation")
):
    """Handle eBay webhook validation challenge."""
    try:
        # Get the endpoint URL from config
        endpoint_url = settings.WEBHOOK_ENDPOINT_URL
        
        # Create the string to hash: challenge_code + verification_token + endpoint_url
        string_to_hash = challenge_code + settings.WEBHOOK_VERIFICATION_TOKEN + endpoint_url
        
        # Generate SHA256 hash
        hash_object = hashlib.sha256(string_to_hash.encode())
        challenge_response = hash_object.hexdigest()
        
        logger.info(f"Webhook validation: challenge_code={challenge_code}, response={challenge_response}")
        
        # Return the required JSON response
        return {
            "challengeResponse": challenge_response
        }
        
    except Exception as e:
        logger.error(f"Error validating webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate webhook")

@app.post("/webhooks/account-deletion")
async def handle_account_deletion(request: Request):
    """Handle eBay marketplace account deletion notifications."""
    try:
        # Get the raw body and headers
        body = await request.body()
        headers = request.headers
        
        logger.info(f"Received account deletion webhook: {headers}")
        
        # Verify the request is from eBay (basic verification)
        user_agent = headers.get("user-agent", "")
        if "ebay" not in user_agent.lower():
            logger.warning(f"Suspicious webhook request from: {user_agent}")
        
        # Parse JSON payload
        try:
            data = await request.json()
            logger.info(f"Account deletion data: {data}")
        except Exception as e:
            logger.error(f"Failed to parse webhook JSON: {e}")
            data = {}
        
        # In a real app, you would:
        # 1. Validate the webhook signature
        # 2. Delete user data from your database
        # 3. Update your records
        
        # For this fashion recommender, we don't store user data,
        # so we just log the notification
        if "userId" in data:
            logger.info(f"User {data['userId']} account deleted - no action needed")
        
        # Return success response to eBay
        return {
            "status": "success",
            "message": "Account deletion notification received",
            "timestamp": data.get("timestamp", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error processing account deletion webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")

@app.get("/webhooks/test")
def webhook_test():
    """Test endpoint to verify webhook URL is accessible."""
    return {
        "status": "ok",
        "message": "Webhook endpoint is accessible",
        "verification_token": settings.WEBHOOK_VERIFICATION_TOKEN
    }

@app.post("/recommend", response_model=RecommendResponse)
async def recommend_items(request: RecommendRequest):
    """
    Multi-modal recommendation endpoint.
    Can search by text, image URL, or both.
    """
    if not request.query and not request.image_url:
        raise HTTPException(
            status_code=400,
            detail="Either 'query' or 'image_url' must be provided"
        )
    
    try:
        similarity_service = get_similarity_service()
        
        # Determine search strategy
        if request.query and not request.image_url:
            # Text-only search
            logger.info(f"Text-only recommendation for: {request.query}")
            items = ebay_client.search_items_with_semantic_ranking(
                query=request.query,
                limit=request.limit,
                category_id=request.category_id
            )
        
        elif request.image_url and not request.query:
            # Image-only search
            logger.info(f"Image-only recommendation for: {request.image_url}")
            # First, get some base results to compare against
            base_query = "fashion clothing"  # Generic query to get items
            items = ebay_client.search_items(
                query=base_query,
                limit=request.limit * 2,  # Get more to filter
                category_id=request.category_id
            )
            # Rank by visual similarity to the query image
            items = similarity_service.find_similar_by_image(
                query_image_url=request.image_url,
                items=items,
                include_text=False,
                visual_weight=1.0
            )[:request.limit]
        
        else:
            # Combined text + image search
            logger.info(f"Combined recommendation - text: {request.query}, image: {request.image_url}")
            items = ebay_client.search_items_with_combined_ranking(
                query=request.query,
                limit=request.limit,
                category_id=request.category_id,
                text_weight=request.text_weight,
                visual_weight=request.visual_weight
            )
        
        # Add search metadata
        search_metadata = {
            "query": request.query,
            "image_url": request.image_url,
            "text_weight": request.text_weight,
            "visual_weight": request.visual_weight,
            "category_id": request.category_id,
            "result_count": len(items)
        }
        
        return RecommendResponse(
            recommendations=items,
            search_metadata=search_metadata,
            total_results=len(items)
        )
        
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/by-image", response_model=RecommendResponse)
async def search_by_image(request: ImageSearchRequest):
    """
    Search for items visually similar to a provided image.
    """
    try:
        similarity_service = get_similarity_service()
        
        # Get a broad set of fashion items
        base_query = "fashion clothing accessories"
        items = ebay_client.search_items(
            query=base_query,
            limit=request.limit * 3,  # Get extra to filter down
            category_id=None  # Search across categories
        )
        
        # Rank by visual similarity
        if request.include_text:
            # Use both visual and some text signal
            visual_weight = 1.0 - request.text_weight
            items = similarity_service.find_similar_by_image(
                query_image_url=request.image_url,
                items=items,
                include_text=True,
                text_weight=request.text_weight,
                visual_weight=visual_weight
            )
        else:
            # Pure visual search
            items = similarity_service.find_similar_by_image(
                query_image_url=request.image_url,
                items=items,
                include_text=False,
                visual_weight=1.0
            )
        
        # Return top matches
        items = items[:request.limit]
        
        search_metadata = {
            "image_url": request.image_url,
            "include_text": request.include_text,
            "text_weight": request.text_weight if request.include_text else 0,
            "visual_weight": 1.0 - request.text_weight if request.include_text else 1.0,
            "result_count": len(items)
        }
        
        return RecommendResponse(
            recommendations=items,
            search_metadata=search_metadata,
            total_results=len(items)
        )
        
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/find-similar/{item_id}", response_model=RecommendResponse)
async def find_similar_items(
    item_id: str,
    limit: int = Query(20, ge=1, le=100),
    text_weight: float = Query(0.3, ge=0, le=1),
    visual_weight: float = Query(0.7, ge=0, le=1)
):
    """
    Find items similar to a specific eBay item.
    Uses the item's title and image for similarity matching.
    """
    try:
        # First, we need to get the item details
        # For now, we'll search for the item ID and hope it comes up
        # In a real implementation, we'd have a get_item_by_id method
        
        # Search for items and find the target
        # This is a workaround - ideally we'd fetch the specific item
        search_query = item_id  # Try searching by ID
        all_items = ebay_client.search_items(query=search_query, limit=50)
        
        # Find the target item
        target_item = None
        for item in all_items:
            if item.item_id == item_id:
                target_item = item
                break
        
        if not target_item:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        
        # Search for similar items using the target's title
        similar_items = ebay_client.search_items_with_combined_ranking(
            query=target_item.title,
            limit=limit + 1,  # Get extra to exclude the original
            text_weight=text_weight,
            visual_weight=visual_weight
        )
        
        # Remove the original item from results
        similar_items = [item for item in similar_items if item.item_id != item_id][:limit]
        
        search_metadata = {
            "source_item_id": item_id,
            "source_item_title": target_item.title,
            "source_item_image": target_item.image_url,
            "text_weight": text_weight,
            "visual_weight": visual_weight,
            "result_count": len(similar_items)
        }
        
        return RecommendResponse(
            recommendations=similar_items,
            search_metadata=search_metadata,
            total_results=len(similar_items)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)