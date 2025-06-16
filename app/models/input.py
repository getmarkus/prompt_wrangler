"""Input models for the prompt wrangler."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PromptInput(BaseModel):
    """Model for input prompt data."""
    
    system_prompt: str = Field(
        default="", 
        description="System prompt for LLM context setting"
    )
    user_prompt: str = Field(
        default="", 
        description="User prompt providing specific instructions"
    )
    sample_text: str = Field(
        description="Sample text to process for NER extraction"
    )
    
    @field_validator("system_prompt", "user_prompt", "sample_text")
    def validate_not_empty(cls, v):
        """Validate that input strings are not empty."""
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class ModelParameters(BaseModel):
    """Model parameters for LLM API calls."""
    
    model: str = Field(
        default="gpt-4o", 
        description="Model name to use for API calls"
    )
    temperature: float = Field(
        default=0.0, 
        description="Temperature setting for randomness in output",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=1000, 
        description="Maximum tokens in response",
        gt=0
    )


class ProcessingInput(BaseModel):
    """Combined model for processing input including prompt and parameters."""
    
    prompt: PromptInput
    parameters: Optional[ModelParameters] = Field(
        default_factory=ModelParameters,
        description="Optional model parameters, uses defaults if not provided"
    )
