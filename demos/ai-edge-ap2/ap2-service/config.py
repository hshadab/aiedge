"""AP2 Service Configuration"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration for the AP2 service."""

    # Service settings
    host: str = "0.0.0.0"
    port: int = 3002
    debug: bool = True

    # AP2 Configuration
    ap2_sandbox_mode: bool = True
    ap2_merchant_id: str = "acme-corp-agent"
    ap2_api_key: Optional[str] = None

    # Google Cloud (for production AP2)
    google_cloud_project: Optional[str] = None
    google_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "AP2_"


settings = Settings()
