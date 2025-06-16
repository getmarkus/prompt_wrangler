"""OpenAI API integration service."""

from datetime import datetime
from typing import Optional

import instructor
from loguru import logger
from openai import OpenAI

from app.models.config import settings
from app.models.input import ProcessingInput
from app.models.output import EntityOutput, ProcessingOutput
from app.utils.metrics import create_response_metrics


class OpenAIService:
    """Service for interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI service.
        
        Args:
            api_key: Optional API key, uses the one from settings if not provided
        """
        self.api_key = api_key or settings.openai_api_key
        self.client = self._create_client()
        self.client_with_schema = instructor.patch(self.client)
    
    def _create_client(self) -> OpenAI:
        """Create and configure OpenAI client.
        
        Returns:
            Configured OpenAI client
        """
        if not self.api_key:
            logger.error("OpenAI API key is not set")
            raise ValueError("OpenAI API key is required")
        
        return OpenAI(api_key=self.api_key)
    
    def process_prompt(self, input_data: ProcessingInput) -> ProcessingOutput:
        """Process a prompt using OpenAI API and return structured output.
        
        Args:
            input_data: Processing input with prompt and parameters
            
        Returns:
            Processing output with structured data and metrics
        """
        # Prepare the request messages
        messages = [
            {"role": "system", "content": input_data.prompt.system_prompt},
            {"role": "user", "content": f"{input_data.prompt.user_prompt}\n\nInput Text: {input_data.prompt.sample_text}"},
        ]
        
        # Set request parameters
        model = input_data.parameters.model
        temperature = input_data.parameters.temperature
        max_tokens = input_data.parameters.max_tokens
        
        logger.info(f"Processing prompt with model: {model}, temperature: {temperature}")
        
        # Record start time
        start_time = datetime.now()
        
        try:
            # Log the request details
            logger.debug(f"Sending prompt to OpenAI: {messages}")
            logger.debug(f"Using model {model} with temperature {temperature} and max_tokens {max_tokens}")
            
            # Call OpenAI API
            raw_response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract raw content from response
            raw_content = raw_response.choices[0].message.content
            logger.debug(f"Raw response content: {raw_content}")
            
            # Process the response content
            import json
            import re
            
            # Clean the raw content from any markdown code block markers
            pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
            match = re.search(pattern, raw_content)
            if match:
                json_str = match.group(1)
            else:
                # If not found in code block, try to extract any JSON from response
                pattern = r'{[\s\S]*?}'
                match = re.search(pattern, raw_content)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = raw_content
                    
            try:
                # Parse JSON and create EntityOutput
                logger.debug(f"Extracted JSON string: {json_str}")
                entity_data = json.loads(json_str)
                response = EntityOutput(**entity_data)
                logger.debug(f"Created EntityOutput: {response.model_dump()}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}")
                # Create an empty EntityOutput
                response = EntityOutput()
            
            # Record end time
            end_time = datetime.now()
            
            # Get usage data from response if available
            usage = {}
            if hasattr(raw_response, "usage"):
                usage = raw_response.usage.model_dump() if hasattr(raw_response.usage, "model_dump") else {}
                
            # Create metrics
            metrics = create_response_metrics(start_time, end_time, usage, model)
            
            return ProcessingOutput(
                result=response,
                metrics=metrics,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None
            )
            
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            raise
