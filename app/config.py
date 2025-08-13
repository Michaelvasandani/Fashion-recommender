import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OAuth2 Credentials
    EBAY_APP_ID: str = os.getenv("EBAY_APP_ID", "")
    EBAY_CERT_ID: str = os.getenv("EBAY_CERT_ID", "")
    
    # API Endpoints (Production)
    EBAY_OAUTH_TOKEN_ENDPOINT: str = "https://api.ebay.com/identity/v1/oauth2/token"
    EBAY_BROWSE_API_ENDPOINT: str = "https://api.ebay.com/buy/browse/v1"
    
    # API Settings
    EBAY_MARKET_ID: str = "EBAY_US"
    EBAY_API_SCOPE: str = "https://api.ebay.com/oauth/api_scope"
    
    # Request Settings
    DEFAULT_PAGE_SIZE: int = 50
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: int = 30
    TOKEN_EXPIRY_BUFFER: int = 60  # Refresh token 60 seconds before expiry
    
    # Webhook Settings
    WEBHOOK_VERIFICATION_TOKEN: str = os.getenv("WEBHOOK_VERIFICATION_TOKEN", "fashionrec_webhook_2025_secure_token_abcd1234efgh5678")
    WEBHOOK_ENDPOINT_URL: str = os.getenv("WEBHOOK_ENDPOINT_URL", "https://your-app.onrender.com/webhooks/account-deletion")
    
    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    ENABLE_SEMANTIC_SEARCH: bool = os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() == "true"
    
    # Visual Embedding Settings
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    ENABLE_VISUAL_SEARCH: bool = os.getenv("ENABLE_VISUAL_SEARCH", "true").lower() == "true"
    DEFAULT_TEXT_WEIGHT: float = 0.5
    DEFAULT_VISUAL_WEIGHT: float = 0.5
    IMAGE_CACHE_DIR: str = "./image_cache"
    IMAGE_CACHE_MAX_AGE_HOURS: int = 24
    
    @property
    def is_configured(self) -> bool:
        return bool(self.EBAY_APP_ID and self.EBAY_CERT_ID)

settings = Settings()