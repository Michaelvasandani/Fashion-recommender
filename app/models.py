from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class EbayItem(BaseModel):
    item_id: str
    title: str
    price: float
    currency: str = "USD"
    listing_url: str
    image_url: Optional[str] = None
    condition: Optional[str] = None
    location: Optional[str] = None
    shipping_cost: Optional[float] = None
    listing_type: Optional[str] = None
    end_time: Optional[datetime] = None
    similarity_score: Optional[float] = Field(None, description="Semantic similarity score (0-1)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class SearchResponse(BaseModel):
    query: str
    total_results: int
    page_number: int = 1
    items_per_page: int
    items: List[EbayItem]
    
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None