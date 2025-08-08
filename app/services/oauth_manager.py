import base64
import time
import logging
from typing import Optional
from urllib.parse import urlencode
import requests

from app.config import settings

logger = logging.getLogger(__name__)

class OAuthManager:
    def __init__(self):
        self.client_id = settings.EBAY_APP_ID
        self.client_secret = settings.EBAY_CERT_ID
        self.token_url = settings.EBAY_OAUTH_TOKEN_ENDPOINT
        self.scope = settings.EBAY_API_SCOPE
        
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        
    def get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        if self._is_token_valid():
            return self._access_token
        
        return self._fetch_new_token()
    
    def _is_token_valid(self) -> bool:
        """Check if the current token is still valid."""
        if not self._access_token:
            return False
        
        # Check if token will expire soon (with buffer)
        return time.time() < self._token_expiry
    
    def _fetch_new_token(self) -> str:
        """Fetch a new access token from eBay OAuth endpoint."""
        logger.info("Fetching new OAuth token from eBay")
        
        # Encode credentials
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': f'Basic {encoded_credentials}'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'scope': self.scope
        }
        
        try:
            response = requests.post(
                self.token_url,
                headers=headers,
                data=urlencode(data),
                timeout=settings.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data['access_token']
            
            # Set expiry time with buffer
            expires_in = token_data.get('expires_in', 7200)
            self._token_expiry = time.time() + expires_in - settings.TOKEN_EXPIRY_BUFFER
            
            logger.info(f"Successfully obtained OAuth token, expires in {expires_in} seconds")
            return self._access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to obtain OAuth token: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"Failed to obtain OAuth token: {str(e)}")
    
    def clear_token(self):
        """Clear the cached token."""
        self._access_token = None
        self._token_expiry = 0