"""CLI helper utilities for the prompt wrangler."""

import json

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from app.models.output import ProcessingOutput, ResponseMetrics


console = Console()


def print_welcome_message() -> None:
    """Display welcome message for the prompt wrangler CLI."""
    console.print(
        Panel.fit(
            "[bold]🤖 Welcome to Prompt Wrangler[/bold]\n"
            "[italic]A tool for NER keyword extraction from medical texts[/italic]",
            border_style="blue",
        )
    )


def print_error(message: str) -> None:
    """Display error message in the CLI.
    
    Args:
        message: Error message to display
    """
    console.print(f"[bold red]Error:[/bold red] {message}")


def format_metrics(metrics: ResponseMetrics) -> str:
    """Format metrics data for display.
    
    Args:
        metrics: Response metrics to format
        
    Returns:
        Formatted metrics string
    """
    token_usage = metrics.token_usage
    
    return (
        f"[bold]Response Time:[/bold] {metrics.response_time_ms} ms\n"
        f"[bold]Model:[/bold] {metrics.model}\n"
        f"[bold]Token Usage:[/bold]\n"
        f"  • Prompt tokens: {token_usage.prompt_tokens}\n"
        f"  • Completion tokens: {token_usage.completion_tokens}\n"
        f"  • Total tokens: {token_usage.total_tokens}"
    )


def display_results(output: ProcessingOutput) -> None:
    """Display processing results in a formatted way.
    
    Args:
        output: Processing output to display
    """
    # Display extracted entities
    console.print("\n[bold green]📋 Extracted Entities:[/bold green]")
    json_str = json.dumps(output.result.model_dump(), indent=2)
    console.print(Syntax(json_str, "json", theme="monokai"))
    
    # Display metrics
    console.print("\n[bold blue]📊 Request Metrics:[/bold blue]")
    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_row(format_metrics(output.metrics))
    console.print(metrics_table)


def read_file_contents(file_path: str) -> str:
    """Read contents from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File contents as string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise typer.BadParameter(f"Could not read file: {e}")
