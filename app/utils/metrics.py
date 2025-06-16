"""Metrics utilities for tracking API response time and token usage."""

import time
from datetime import datetime
from typing import Dict, Any

from app.models.output import ResponseMetrics, TokenUsage


def calculate_response_time(start_time: datetime, end_time: datetime) -> int:
    """Calculate response time in milliseconds.
    
    Args:
        start_time: Start time of the request
        end_time: End time of the request
        
    Returns:
        Response time in milliseconds
    """
    delta = end_time - start_time
    return int(delta.total_seconds() * 1000)


def create_response_metrics(
    start_time: datetime,
    end_time: datetime,
    usage: Dict[str, int],
    model: str,
) -> ResponseMetrics:
    """Create response metrics from raw data.
    
    Args:
        start_time: Start time of the request
        end_time: End time of the request
        usage: Token usage dictionary from OpenAI response
        model: Model name used for the request
        
    Returns:
        ResponseMetrics object
    """
    token_usage = TokenUsage(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )
    
    return ResponseMetrics(
        start_time=start_time,
        end_time=end_time,
        response_time_ms=calculate_response_time(start_time, end_time),
        token_usage=token_usage,
        model=model,
    )
