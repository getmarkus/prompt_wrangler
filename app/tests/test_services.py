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
