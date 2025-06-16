"""Tests for OpenAI service."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.models.output import ProcessingOutput
from app.services.openai_service import OpenAIService


class TestOpenAIService:
    """Test cases for OpenAI service."""
    
    @patch("app.services.openai_service.OpenAI")
    def test_init_with_api_key(self, mock_openai):
        """Test service initialization with API key."""
        service = OpenAIService(api_key="test_key")
        
        assert service.api_key == "test_key"
        mock_openai.assert_called_once_with(api_key="test_key")
    
    @patch("app.services.openai_service.OpenAI")
    @patch("app.services.openai_service.settings")
    def test_init_with_settings(self, mock_settings, mock_openai):
        """Test service initialization with settings API key."""
        mock_settings.openai_api_key = "settings_key"
        
        service = OpenAIService()
        
        assert service.api_key == "settings_key"
        mock_openai.assert_called_once_with(api_key="settings_key")
    
    @patch("app.services.openai_service.OpenAI")
    @patch("app.services.openai_service.settings")
    def test_create_client_without_api_key(self, mock_settings, mock_openai):
        """Test client creation fails without API key."""
        # Ensure settings.openai_api_key returns None
        mock_settings.openai_api_key = None
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIService(api_key="")
    
    @patch("app.services.openai_service.instructor.patch")
    def test_client_patched_with_instructor(self, mock_patch, mock_openai_client):
        """Test that client is patched with instructor."""
        mock_patch.return_value = "patched_client"
        
        service = OpenAIService(api_key="test_key")
        
        mock_patch.assert_called_once()
        assert service.client_with_schema == "patched_client"
    
    @pytest.fixture
    def mock_openai_client(self):
        """Fixture for mocked OpenAI client."""
        with patch("app.services.openai_service.OpenAI") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client
    
    @pytest.fixture
    def mock_patched_client(self):
        """Fixture for mocked instructor-patched client."""
        with patch("app.services.openai_service.instructor.patch") as mock_patch:
            patched_client = MagicMock()
            mock_patch.return_value = patched_client
            yield patched_client
    
    def test_process_prompt_success(self, mock_openai_client, mock_patched_client, sample_processing_input, sample_entity_output, monkeypatch):
        """Test successful prompt processing."""
        # Create a mock for datetime.now to return consistent values
        mock_start_time = datetime(2025, 6, 16, 10, 0, 0)
        mock_end_time = datetime(2025, 6, 16, 10, 0, 1)
        
        datetime_mock = MagicMock()
        datetime_mock.now.side_effect = [mock_start_time, mock_end_time]
        monkeypatch.setattr("app.services.openai_service.datetime", datetime_mock)
        
        # Create an actual EntityOutput instance for the result
        entity_output = sample_entity_output
        
        # Create mock response from OpenAI
        mock_response = MagicMock()
        mock_response.model_dump.return_value = entity_output.model_dump()
        
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        
        # Set up our mock client to return the mock response
        mock_chat = MagicMock()
        mock_completions = MagicMock()
        mock_completions.create.return_value = mock_response
        mock_chat.completions = mock_completions
        mock_openai_client.chat = mock_chat
        
        # Create service with the mock clients
        service = OpenAIService(api_key="test_key", mock_client=mock_openai_client, mock_client_with_schema=mock_patched_client)
        result = service.process_prompt(sample_processing_input)
        
        # Check that the result is as expected
        assert isinstance(result, ProcessingOutput)
        assert result.metrics.model == sample_processing_input.parameters.model
        assert result.metrics.response_time_ms == 1000  # 1 second difference
        assert result.metrics.token_usage.prompt_tokens == 100
        assert result.metrics.token_usage.completion_tokens == 50
        assert result.metrics.token_usage.total_tokens == 150
        
        # Check that the API was called with expected parameters
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=sample_processing_input.parameters.model,
            messages=[
                {"role": "system", "content": sample_processing_input.prompt.system_prompt},
                {"role": "user", "content": f"{sample_processing_input.prompt.user_prompt}\n\nInput Text: {sample_processing_input.prompt.sample_text}"},
            ],
            temperature=sample_processing_input.parameters.temperature,
            max_tokens=sample_processing_input.parameters.max_tokens,
        )
    
    def test_process_prompt_error_handling(self, mock_openai_client, mock_patched_client, sample_processing_input):
        """Test error handling in process_prompt."""
        # Set up our mock client to raise an exception
        mock_chat = MagicMock()
        mock_completions = MagicMock()
        mock_completions.create.side_effect = Exception("API Error")
        mock_chat.completions = mock_completions
        mock_openai_client.chat = mock_chat
        
        # Create service with the mock clients
        service = OpenAIService(api_key="test_key", mock_client=mock_openai_client, mock_client_with_schema=mock_patched_client)
        
        # Process prompt should raise the exception
        with pytest.raises(Exception, match="API Error"):
            service.process_prompt(sample_processing_input)
