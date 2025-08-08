from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging

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
    category_id: Optional[str] = Query(None, description="eBay category ID")
):
    if not settings.is_configured:
        raise HTTPException(
            status_code=500,
            detail="eBay API credentials not configured. Please set EBAY_APP_ID and EBAY_CERT_ID."
        )
    
    try:
        logger.info(f"Searching for: {q}, limit: {limit}, page: {page}")
        
        # Calculate offset from page number
        offset = (page - 1) * limit
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)