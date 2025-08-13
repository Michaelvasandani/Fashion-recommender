from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import logging
import hashlib

from app.config import settings
from app.models import SearchResponse, ErrorResponse
from app.services.ebay_client import EbayClient, EbayAPIError

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
        "endpoints": {
            "/search": "Search for secondhand items",
            "/docs": "API documentation"
        }
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)