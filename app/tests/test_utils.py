"""Tests for utility functions."""

from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock

import pytest
from rich.console import Console

from app.utils.cli import display_results, format_metrics, read_file_contents
from app.utils.metrics import calculate_response_time, create_response_metrics


class TestMetricsUtils:
    """Test cases for metrics utilities."""
    
    def test_calculate_response_time(self):
        """Test response time calculation."""
        start_time = datetime(2025, 6, 16, 10, 0, 0)
        end_time = datetime(2025, 6, 16, 10, 0, 1)
        
        # Should be 1000ms (1 second difference)
        result = calculate_response_time(start_time, end_time)
        
        assert result == 1000
        assert isinstance(result, int)
    
    def test_create_response_metrics(self):
        """Test response metrics creation."""
        start_time = datetime(2025, 6, 16, 10, 0, 0)
        end_time = datetime(2025, 6, 16, 10, 0, 1)
        
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
        
        model = "gpt-4o"
        
        metrics = create_response_metrics(start_time, end_time, usage, model)
        
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.response_time_ms == 1000
        assert metrics.token_usage.prompt_tokens == 100
        assert metrics.token_usage.completion_tokens == 50
        assert metrics.token_usage.total_tokens == 150
        assert metrics.model == model


class TestCliUtils:
    """Test cases for CLI utilities."""
    
    def test_format_metrics(self, sample_response_metrics):
        """Test metrics formatting."""
        formatted = format_metrics(sample_response_metrics)
        
        assert "[bold]Response Time:[/bold]" in formatted
        assert "1000 ms" in formatted
        assert "[bold]Model:[/bold]" in formatted
        assert "gpt-4o" in formatted
        assert "[bold]Token Usage:[/bold]" in formatted
        assert "Prompt tokens: 100" in formatted
        assert "Completion tokens: 50" in formatted
        assert "Total tokens: 150" in formatted
    
    @patch("app.utils.cli.console")
    def test_display_results(self, mock_console, sample_processing_output):
        """Test results display formatting."""
        display_results(sample_processing_output)
        
        # Check that console.print was called with expected arguments
        mock_console.print.assert_called()
        
        # Check that the first call includes the entity header
        assert any("Extracted Entities" in str(args[0]) for args, _ in mock_console.print.call_args_list)
        
        # Check that metrics are also displayed
        assert any("Request Metrics" in str(args[0]) for args, _ in mock_console.print.call_args_list)
    
    def test_read_file_contents(self, tmp_path):
        """Test reading file contents."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_content = "This is a test file content."
        
        with open(test_file, "w") as f:
            f.write(test_content)
        
        # Read the file
        content = read_file_contents(str(test_file))
        
        assert content == test_content
    
    def test_read_file_contents_error(self):
        """Test error handling when reading non-existent file."""
        with pytest.raises(Exception):
            read_file_contents("non_existent_file.txt")
