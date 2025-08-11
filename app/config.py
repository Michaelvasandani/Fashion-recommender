import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OAuth2 Credentials
    EBAY_APP_ID: str = os.getenv("EBAY_APP_ID", "")
    EBAY_CERT_ID: str = os.getenv("EBAY_CERT_ID", "")
    
    # API Endpoints (Sandbox)
    EBAY_OAUTH_TOKEN_ENDPOINT: str = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    EBAY_BROWSE_API_ENDPOINT: str = "https://api.sandbox.ebay.com/buy/browse/v1"
    
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
    
    @property
    def is_configured(self) -> bool:
        return bool(self.EBAY_APP_ID and self.EBAY_CERT_ID)

settings = Settings()