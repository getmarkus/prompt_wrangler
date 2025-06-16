"""Main entry point for the prompt wrangler CLI application."""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.prompt import Prompt

from app.models.config import settings
from app.models.input import ModelParameters, ProcessingInput, PromptInput
from app.services.openai_service import OpenAIService
from app.utils.cli import (
    console,
    display_results,
    print_error,
    print_welcome_message,
    read_file_contents,
)
from app.utils.logging import setup_logger


# Create Typer application
app = typer.Typer(
    help="🤖 Prompt Wrangler: A tool for NER keyword extraction from medical texts",
    add_completion=False,
)


@app.callback()
def callback(
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    """Set up application-wide settings."""
    setup_logger(log_level)


@app.command(name="extract")
def extract_entities(
    # Prompt inputs
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", help="System prompt for context setting"
    ),
    system_file: Optional[Path] = typer.Option(
        None, "--system-file", "-sf", help="File containing the system prompt"
    ),
    user_prompt: Optional[str] = typer.Option(
        None, "--user", "-u", help="User prompt with specific instructions"
    ),
    user_file: Optional[Path] = typer.Option(
        None, "--user-file", "-uf", help="File containing the user prompt"
    ),
    text: Optional[str] = typer.Option(
        None, "--text", "-t", help="Sample text to extract entities from"
    ),
    text_file: Optional[Path] = typer.Option(
        None, "--text-file", "-tf", help="File containing the sample text"
    ),
    
    # Model parameters
    model: str = typer.Option(
        settings.default_model, "--model", "-m", help="OpenAI model to use"
    ),
    temperature: float = typer.Option(
        settings.default_temperature, "--temperature", "-temp", help="Temperature setting (0-1)"
    ),
    max_tokens: int = typer.Option(
        settings.default_max_tokens, "--max-tokens", "-mt", help="Maximum tokens in response"
    ),
    
    # API key
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="OpenAI API key (overrides env variable)"
    ),
    
    # Interactive mode flag
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Run in interactive mode"
    ),
) -> None:
    """Extract named entities from medical text using LLM.
    
    If running in interactive mode, prompts will be requested from user input.
    Otherwise, prompts must be provided via options or files.
    """
    print_welcome_message()
    
    try:
        # Get API key
        openai_api_key = api_key or settings.openai_api_key
        
        if not openai_api_key:
            print_error("OpenAI API key is required. Set it with --api-key or OPENAI_API_KEY environment variable.")
            raise typer.Exit(code=1)
        
        # Gather inputs: interactive or from arguments
        if interactive:
            system_prompt_text = Prompt.ask("[bold]Enter system prompt[/bold]")
            user_prompt_text = Prompt.ask("[bold]Enter user prompt[/bold]")
            sample_text = Prompt.ask("[bold]Enter sample text[/bold]")
        else:
            # Get system prompt
            if system_file:
                system_prompt_text = read_file_contents(str(system_file))
            elif system_prompt:
                system_prompt_text = system_prompt
            else:
                print_error("System prompt is required. Provide it with --system or --system-file.")
                raise typer.Exit(code=1)
                
            # Get user prompt
            if user_file:
                user_prompt_text = read_file_contents(str(user_file))
            elif user_prompt:
                user_prompt_text = user_prompt
            else:
                print_error("User prompt is required. Provide it with --user or --user-file.")
                raise typer.Exit(code=1)
                
            # Get sample text
            if text_file:
                sample_text = read_file_contents(str(text_file))
            elif text:
                sample_text = text
            else:
                print_error("Sample text is required. Provide it with --text or --text-file.")
                raise typer.Exit(code=1)
        
        # Create input models
        prompt_input = PromptInput(
            system_prompt=system_prompt_text,
            user_prompt=user_prompt_text,
            sample_text=sample_text,
        )
        
        model_parameters = ModelParameters(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        processing_input = ProcessingInput(
            prompt=prompt_input,
            parameters=model_parameters,
        )
        
        # Process with OpenAI
        console.print("\n[bold yellow]🔄 Processing request...[/bold yellow]")
        service = OpenAIService(api_key=openai_api_key)
        result = service.process_prompt(processing_input)
        
        # Display results
        display_results(result)
        
    except typer.Exit:
        # Exit has already been handled
        pass
    except Exception as e:
        logger.exception("An error occurred")
        print_error(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
