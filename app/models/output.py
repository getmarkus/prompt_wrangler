"""Output models for the prompt wrangler."""

from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class EntityOutput(BaseModel):
    """Base model for NER extraction output.
    
    This model uses open-ended field names to allow for dynamic entity extraction.
    Examples of fields: device, mask_type, add_ons, qualifier, diagnosis, etc.
    """
    
    model_config = ConfigDict(extra="allow")


class TokenUsage(BaseModel):
    """Token usage metrics from the OpenAI API response."""
    
    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    completion_tokens: int = Field(description="Number of tokens in the completion")
    total_tokens: int = Field(description="Total tokens used")


class ResponseMetrics(BaseModel):
    """Metrics for API response including timing and token usage."""
    
    start_time: datetime = Field(description="Request start time")
    end_time: datetime = Field(description="Request end time")
    response_time_ms: int = Field(description="Response time in milliseconds")
    token_usage: TokenUsage = Field(description="Token usage metrics")
    model: str = Field(description="Model used for the request")


class ProcessingOutput(BaseModel):
    """Combined output model with both extraction results and metrics."""
    
    result: EntityOutput = Field(description="Extracted entities from the input text")
    metrics: ResponseMetrics = Field(description="Performance metrics for the request")
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw response from the API for debugging purposes"
    )
