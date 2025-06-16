"""Test fixtures for prompt wrangler tests."""

import json
from datetime import datetime
from typing import Dict, Any

import pytest

from app.models.input import ModelParameters, ProcessingInput, PromptInput
from app.models.output import TokenUsage, ResponseMetrics, EntityOutput, ProcessingOutput


@pytest.fixture
def sample_system_prompt() -> str:
    """Sample system prompt for testing."""
    return (
        "You are a medical NER extraction assistant. Extract key entities from medical texts "
        "as a structured JSON with the following fields when present: device, mask_type, "
        "add_ons, qualifier, diagnosis, ordering_provider."
    )


@pytest.fixture
def sample_user_prompt() -> str:
    """Sample user prompt for testing."""
    return (
        "Extract the structured information from the following medical text. "
        "Return the data in JSON format with appropriate fields."
    )


@pytest.fixture
def sample_text() -> str:
    """Sample medical text for testing."""
    return "Patient requires a full face CPAP mask with humidifier due to AHI > 20. Ordered by Dr. Cameron."


@pytest.fixture
def sample_prompt_input(sample_system_prompt, sample_user_prompt, sample_text) -> PromptInput:
    """Sample prompt input model for testing."""
    return PromptInput(
        system_prompt=sample_system_prompt,
        user_prompt=sample_user_prompt,
        sample_text=sample_text,
    )


@pytest.fixture
def sample_model_parameters() -> ModelParameters:
    """Sample model parameters for testing."""
    return ModelParameters(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000,
    )


@pytest.fixture
def sample_processing_input(sample_prompt_input, sample_model_parameters) -> ProcessingInput:
    """Sample processing input for testing."""
    return ProcessingInput(
        prompt=sample_prompt_input,
        parameters=sample_model_parameters,
    )


@pytest.fixture
def sample_token_usage() -> TokenUsage:
    """Sample token usage for testing."""
    return TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )


@pytest.fixture
def sample_response_metrics(sample_token_usage) -> ResponseMetrics:
    """Sample response metrics for testing."""
    start_time = datetime(2025, 6, 16, 10, 0, 0)
    end_time = datetime(2025, 6, 16, 10, 0, 1)
    
    return ResponseMetrics(
        start_time=start_time,
        end_time=end_time,
        response_time_ms=1000,
        token_usage=sample_token_usage,
        model="gpt-4o",
    )


@pytest.fixture
def sample_entity_output() -> EntityOutput:
    """Sample entity output for testing."""
    return EntityOutput(
        device="CPAP",
        mask_type="full face",
        add_ons=["humidifier"],
        qualifier="AHI > 20",
        ordering_provider="Dr. Cameron",
    )


@pytest.fixture
def sample_processing_output(sample_entity_output, sample_response_metrics) -> ProcessingOutput:
    """Sample processing output for testing."""
    return ProcessingOutput(
        result=sample_entity_output,
        metrics=sample_response_metrics,
        raw_response={"choices": [{"message": {"content": "{}"}}]},
    )


@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Mock OpenAI API response for testing."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "device": "CPAP",
                        "mask_type": "full face",
                        "add_ons": ["humidifier"],
                        "qualifier": "AHI > 20",
                        "ordering_provider": "Dr. Cameron"
                    })
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
