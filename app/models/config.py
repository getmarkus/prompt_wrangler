"""Configuration models for the prompt wrangler."""

# No type imports needed

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


# Load environment variables from .env file
load_dotenv(find_dotenv())


class Settings(BaseSettings):
    """Application settings including API keys and default parameters."""

    # OpenAI API settings
    openai_api_key: str = Field(
        default="", 
        description="OpenAI API key for accessing the API"
    )
    
    # Default model parameters
    default_model: str = Field(
        default="gpt-4o", 
        description="Default model to use for OpenAI requests"
    )
    default_temperature: float = Field(
        default=0.0, 
        description="Default temperature setting (0-1)",
        ge=0.0,
        le=1.0,
    )
    default_max_tokens: int = Field(
        default=1000, 
        description="Default maximum number of tokens for the response",
        gt=0
    )

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
