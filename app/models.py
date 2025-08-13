from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
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

class RecommendRequest(BaseModel):
    query: Optional[str] = Field(None, description="Text query for search")
    image_url: Optional[str] = Field(None, description="Image URL for visual search")
    limit: int = Field(20, description="Number of recommendations", ge=1, le=100)
    text_weight: float = Field(0.5, description="Weight for text similarity", ge=0, le=1)
    visual_weight: float = Field(0.5, description="Weight for visual similarity", ge=0, le=1)
    category_id: Optional[str] = Field(None, description="eBay category ID to filter results")
    
    @validator('query', 'image_url')
    def validate_input(cls, v, values):
        if 'query' in values and 'image_url' in values:
            if not values.get('query') and not values.get('image_url'):
                raise ValueError('Either query or image_url must be provided')
        return v

class ImageSearchRequest(BaseModel):
    image_url: str = Field(..., description="Image URL to search for similar items")
    limit: int = Field(20, description="Number of results", ge=1, le=100)
    include_text: bool = Field(True, description="Include text similarity in ranking")
    text_weight: float = Field(0.3, description="Weight for text similarity when included", ge=0, le=1)

class RecommendResponse(BaseModel):
    recommendations: List[EbayItem]
    search_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the search (query, weights, timing, etc.)"
    )
    total_results: int = Field(0, description="Total number of results found")

class SimilarityDetails(BaseModel):
    text_score: float = Field(0.0, description="Text similarity score (0-1)")
    visual_score: float = Field(0.0, description="Visual similarity score (0-1)")
    combined_score: float = Field(0.0, description="Combined weighted score (0-1)")
    text_weight: float = Field(0.5, description="Weight used for text similarity")
    visual_weight: float = Field(0.5, description="Weight used for visual similarity")