"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.models.input import ModelParameters, ProcessingInput, PromptInput
from app.models.output import EntityOutput, ProcessingOutput, ResponseMetrics, TokenUsage


class TestInputModels:
    """Test cases for input models."""
    
    def test_prompt_input_valid(self, sample_system_prompt, sample_user_prompt, sample_text):
        """Test valid prompt input creation."""
        prompt = PromptInput(
            system_prompt=sample_system_prompt,
            user_prompt=sample_user_prompt,
            sample_text=sample_text,
        )
        
        assert prompt.system_prompt == sample_system_prompt
        assert prompt.user_prompt == sample_user_prompt
        assert prompt.sample_text == sample_text
    
    def test_prompt_input_empty_validation(self):
        """Test validation error for empty prompt inputs."""
        with pytest.raises(ValidationError):
            PromptInput(system_prompt="", user_prompt="test", sample_text="test")
            
        with pytest.raises(ValidationError):
            PromptInput(system_prompt="test", user_prompt="", sample_text="test")
            
        with pytest.raises(ValidationError):
            PromptInput(system_prompt="test", user_prompt="test", sample_text="")
    
    def test_model_parameters_defaults(self):
        """Test model parameters default values."""
        params = ModelParameters()
        
        assert params.model == "gpt-4o"
        assert params.temperature == 0.0
        assert params.max_tokens == 1000
    
    def test_model_parameters_custom(self):
        """Test custom model parameters."""
        params = ModelParameters(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
        )
        
        assert params.model == "gpt-3.5-turbo"
        assert params.temperature == 0.7
        assert params.max_tokens == 500
    
    def test_model_parameters_validation(self):
        """Test validation rules for model parameters."""
        # Temperature must be between 0 and 1
        with pytest.raises(ValidationError):
            ModelParameters(temperature=-0.1)
            
        with pytest.raises(ValidationError):
            ModelParameters(temperature=1.1)
            
        # Max tokens must be greater than 0
        with pytest.raises(ValidationError):
            ModelParameters(max_tokens=0)
    
    def test_processing_input(self, sample_prompt_input, sample_model_parameters):
        """Test processing input model."""
        processing = ProcessingInput(
            prompt=sample_prompt_input,
            parameters=sample_model_parameters,
        )
        
        assert processing.prompt == sample_prompt_input
        assert processing.parameters == sample_model_parameters
    
    def test_processing_input_default_parameters(self, sample_prompt_input):
        """Test processing input with default parameters."""
        processing = ProcessingInput(prompt=sample_prompt_input)
        
        assert processing.prompt == sample_prompt_input
        assert isinstance(processing.parameters, ModelParameters)
        assert processing.parameters.model == "gpt-4o"


class TestOutputModels:
    """Test cases for output models."""
    
    def test_entity_output_dynamic_fields(self):
        """Test entity output with dynamic fields."""
        entity = EntityOutput(
            device="CPAP",
            mask_type="full face",
            add_ons=["humidifier"],
            qualifier="AHI > 20",
            ordering_provider="Dr. Cameron",
        )
        
        assert entity.device == "CPAP"
        assert entity.mask_type == "full face"
        assert entity.add_ons == ["humidifier"]
        assert entity.qualifier == "AHI > 20"
        assert entity.ordering_provider == "Dr. Cameron"
    
    def test_token_usage(self):
        """Test token usage model."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    def test_response_metrics(self, sample_token_usage):
        """Test response metrics model."""
        metrics = ResponseMetrics(
            start_time="2023-01-01T10:00:00",
            end_time="2023-01-01T10:00:01",
            response_time_ms=1000,
            token_usage=sample_token_usage,
            model="gpt-4o",
        )
        
        assert metrics.response_time_ms == 1000
        assert metrics.token_usage == sample_token_usage
        assert metrics.model == "gpt-4o"
    
    def test_processing_output(self, sample_entity_output, sample_response_metrics):
        """Test processing output model."""
        output = ProcessingOutput(
            result=sample_entity_output,
            metrics=sample_response_metrics,
        )
        
        assert output.result == sample_entity_output
        assert output.metrics == sample_response_metrics
        assert output.raw_response is None
