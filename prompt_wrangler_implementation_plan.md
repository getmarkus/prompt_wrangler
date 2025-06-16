# Synapse Health Prompt Wrangler Challenge - Implementation Plan

## Project Overview

This document outlines the implementation plan for the Health Prompt Wrangler Challenge. The goal is to create a Named Entity Recognition (NER) keyword extraction prompt playground as a Python CLI tool.

### Core Requirements
- Create a Python CLI for NER keyword extraction from medical texts
- Use Pydantic for data modeling
- Use Instructor for structured output
- Implement two execution options: OpenAI API and KeyLLM/KeyBERT
- Include unit tests

## High-Level CLI Structure

### 1. Main CLI Interface
- Entry point with argument parsing for different modes and parameters
- Support for interactive and non-interactive modes
- Help documentation and usage examples

### 2. Input Handling
- System prompt input/configuration
- User prompt input/configuration
- Sample text input (file or direct input)
- Parameter configuration (temperature, max tokens, etc.)

### 3. Execution Pipeline
- Prompt preparation and formatting
- LLM execution (OpenAI or KeyLLM)
- Response parsing and validation
- Output formatting and display
- Performance metrics calculation and display

## Detailed CLI Features

### 1. Input Management
- **System Prompt Input**: Allow users to input or load a system prompt that instructs the LLM how to extract structured data
- **User Prompt Input**: Allow users to input or load a user prompt that contains specific extraction instructions
- **Sample Text Input**: Accept text input from:
  - Direct command line input
  - File input
  - Default example texts from the challenge document

### 2. Parameter Configuration
- **Model Selection**: Choose between OpenAI models or KeyLLM/KeyBERT
- **Temperature Control**: Adjust temperature parameter (0.0-1.0)
- **Max Tokens**: Set maximum tokens for completion
- **Top-p (Optional)**: Nucleus sampling parameter
- **Frequency Penalty (Optional)**: Control repetition
- **Presence Penalty (Optional)**: Control topic coverage
- **Configuration File Support**: Load/save parameter configurations

### 3. Execution Options
- **Execution Mode**: Toggle between:
  - OpenAI API mode (zero-shot)
  - KeyLLM mode (using KeyBERT)
- **Batch Processing**: Process multiple inputs at once
- **Iteration Mode**: Run multiple variations with parameter adjustments

### 4. Output Handling
- **Structured Output Display**: Format and present JSON output
- **Metrics Display**:
  - Token usage (prompt + completion)
  - Response time
  - Cost estimation (for OpenAI API)
- **Output Export**: Save results to JSON or CSV
- **Output Validation**: Verify output matches expected JSON schema

### 5. User Experience
- **Interactive Mode**: Step through the process with prompts
- **Non-interactive Mode**: Run with command line arguments
- **Help Documentation**: Comprehensive help with examples
- **Error Handling**: Clear error messages for API issues, validation problems, etc.

## OpenAI API Integration Approach

### 1. API Setup and Authentication
- **Environment Variables**: Store OpenAI API key in `.env` file or environment variables
- **Configuration Handling**: Create a secure config module for API credentials
- **API Client Initialization**: Initialize the OpenAI client with appropriate settings

### 2. Prompt Engineering
- **System Prompt Design**: 
  - Create a template for consistent NER extraction focused on medical equipment
  - Include instructions for JSON structure and field formatting
  - Provide examples of expected output format
- **User Prompt Formatting**:
  - Combine user instructions with sample text
  - Apply any special formatting required for effective extraction

### 3. API Call Implementation
- **Request Construction**:
  - Build the API request with system and user prompts
  - Include all selected parameters (temperature, max_tokens, etc.)
  - Set the response format to JSON to ensure structured output
- **Response Handling**:
  - Parse and validate the JSON response
  - Handle errors and edge cases
  - Extract completion tokens and usage statistics

### 4. Performance Monitoring
- **Token Counting**:
  - Count prompt tokens
  - Track response tokens
  - Calculate total token usage
- **Timing Measurements**:
  - Record start time before API call
  - Calculate total response time
  - Include network latency in measurements

### 5. Output Processing
- **JSON Validation**:
  - Validate against expected schema
  - Handle malformed JSON responses
- **Format Conversion**:
  - Convert to other formats if needed
  - Ensure consistent output structure

## KeyLLM/KeyBERT Integration Approach

### 1. Library Setup
- **Dependencies**: Add KeyBERT and KeyLLM to requirements
- **Model Initialization**: 
  - Load appropriate language models for medical terminology
  - Configure embedding models compatible with healthcare domain
  - Set up sentence transformers backend

### 2. Text Processing Pipeline
- **Preprocessing**:
  - Text cleaning and normalization
  - Sentence splitting for long inputs
  - Medical terminology recognition
- **NER Configuration**:
  - Define entity types (devices, accessories, diagnosis, etc.)
  - Set up keyword extraction parameters
  - Configure similarity thresholds

### 3. Extraction Implementation
- **Keyword Extraction**:
  - Extract candidate keywords/keyphrases
  - Filter and rank by relevance
  - Group by entity type
- **Entity Classification**:
  - Map extracted keywords to predefined entity types
  - Apply contextual rules for entity disambiguation
  - Handle negations and qualifiers

### 4. JSON Conversion
- **Schema Mapping**:
  - Map extracted entities to JSON schema
  - Apply type conversions as needed
  - Format values consistently
- **Output Validation**:
  - Ensure all required fields are present
  - Validate value formats
  - Handle missing or uncertain data

### 5. Performance Metrics
- **Execution Timing**:
  - Measure extraction time
  - Track processing steps
- **Quality Metrics**:
  - Calculate confidence scores
  - Provide extraction statistics

## Structured Output Using Instructor Library

### 1. Pydantic Schema Definition
- **Base Models**:
  - Define core Pydantic models representing medical equipment data
  - Create fields for common entities (device, features, diagnosis, provider)
  - Set appropriate field types and validation rules
- **Custom Types**:
  - Define custom validators for medical terminology
  - Implement specialized field types for measurements (SpO2, AHI)
  - Create enumeration types for common values

### 2. Instructor Integration
- **Model Patching**:
  - Apply instructor's patching to Pydantic models
  - Configure model extraction parameters
  - Set up response validation
- **Response Processing**:
  - Handle structured extraction from LLM outputs
  - Validate against schema definitions
  - Apply post-processing rules

### 3. Output Formatting
- **JSON Serialization**:
  - Configure JSON output options
  - Set format for arrays and nested objects
  - Handle special characters and medical terminology
- **Pretty Printing**:
  - Format output for CLI display
  - Highlight key fields
  - Colorize output when supported

### 4. Error Handling
- **Validation Errors**:
  - Detailed error messages for schema violations
  - Suggestions for fixing format issues
  - Fallback mechanisms for partial extraction
- **Recovery Strategies**:
  - Handle missing fields
  - Implement default values
  - Provide confidence scores for uncertain extractions

## Unit Test Coverage Plan

### 1. Test Organization
- **Test Directory Structure**:
  - Organize tests by component (input handling, API integration, output processing)
  - Include fixtures directory for sample prompts and responses
  - Set up configuration for test environments
- **Test Framework**:
  - Use pytest as the primary testing framework
  - Implement pytest fixtures for common test scenarios
  - Utilize parameterized tests for multiple input variations

### 2. Unit Test Types
- **Input Validation Tests**:
  - Test system and user prompt validation
  - Verify parameter validation (temperature, max_tokens)
  - Test file input handling
- **API Integration Tests**:
  - Mock OpenAI API responses
  - Test request formation
  - Verify error handling
- **KeyLLM Integration Tests**:
  - Test keyword extraction functionality
  - Verify entity classification
  - Test output conversion
- **Output Processing Tests**:
  - Verify JSON schema validation
  - Test pretty printing functionality
  - Verify metrics calculation

### 3. Mock and Fixture Strategy
- **Mock Objects**:
  - Create mock LLM responses
  - Mock API calls to avoid actual API usage during tests
  - Simulate different response scenarios
- **Test Fixtures**:
  - Sample medical equipment texts
  - Expected extraction results
  - Various parameter configurations

### 4. Test Coverage Goals
- **Code Coverage**:
  - Aim for >80% test coverage
  - Focus on critical extraction logic
  - Ensure error paths are tested
- **Scenario Coverage**:
  - Test all example inputs from challenge document
  - Include edge cases and error scenarios
  - Test both execution paths (OpenAI and KeyLLM)

## MVP Scope and Priorities

### 1. Core MVP Requirements
- **Essential Features**:
  - System and user prompt input
  - Sample text input handling
  - Parameter configuration (temperature, max_tokens)
  - LLM execution (OpenAI API)
  - JSON structured output
  - Token usage and response time metrics
  - Basic CLI interface with help documentation

### 2. Priority Tiers

#### Tier 1 (Must-Have)
- **Functional NER Extraction**:
  - Working OpenAI API integration
  - Basic prompt engineering for medical equipment extraction
  - JSON response validation
  - Error handling for API issues
  - Performance metrics display
  - Command-line interface with basic options

#### Tier 2 (Should-Have)
- **Enhanced Usability**:
  - Configuration file support
  - Interactive mode
  - Output formatting options
  - KeyLLM/KeyBERT integration as an alternative to OpenAI
  - File input/output support

#### Tier 3 (Nice-to-Have)
- **Advanced Features**:
  - Batch processing
  - Parameter optimization suggestions
  - Extended metrics and analytics
  - Custom schema definitions
  - Visualization of extraction results

### 3. Exclusions from MVP
- **Out of Scope**:
  - GUI interface
  - Extensive prompt libraries
  - Model fine-tuning
  - Database integration
  - Advanced analytics
  - Production deployment considerations

### 4. Implementation Timeline Targets
- **Phase 1**: Core input handling and OpenAI integration
- **Phase 2**: Structured output with instructor
- **Phase 3**: Testing and KeyLLM alternative implementation
- **Phase 4**: Documentation and additional features
