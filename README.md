# 🤖 Prompt Wrangler

A lightweight CLI tool for NER (Named Entity Recognition) keyword extraction from medical texts using LLMs.

## 📋 Overview

Prompt Wrangler is designed for AI engineers in healthcare who need to extract structured data from clinical notes. This tool provides a playground to experiment with prompt tuning, allowing you to:

- Input system and user prompts
- Paste sample medical texts
- Configure model parameters (temperature, max_tokens)
- Send requests to OpenAI's API
- Receive structured JSON output
- View token usage and response time metrics

## 🚀 Features

- **Structured Data Extraction**: Extract medical entities like devices, diagnosis, and provider information
- **Flexible Input Options**: Input prompts and text via CLI arguments, files, or interactive mode
- **Parameter Tuning**: Adjust temperature and token limits for optimal results
- **Performance Metrics**: Track token usage and response times
- **JSON Output**: Get standardized, structured output for easy parsing

### Remaining work with more time

- verify that token metrics are correct
- add support for KeyLLM/KeyBERT

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/prompt-wrangler.git
cd prompt-wrangler

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## ⚙️ Configuration

1. Copy the example environment file and add your OpenAI API key:

```bash
cp example.env .env
```

1. Edit the `.env` file with your OpenAI API key and other settings:

```properties
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_MODEL=gpt-4o
DEFAULT_TEMPERATURE=0.0
DEFAULT_MAX_TOKENS=1000
```

## 🖥️ Usage

### Basic Usage

```bash
# Using command-line arguments
prompt-wrangler extract --system "You are a medical NER extraction assistant." \
    --user "Extract entities from this text." \
    --text "Patient requires a full face CPAP mask with humidifier due to AHI > 20. Ordered by Dr. Cameron."

# Using text files for input
prompt-wrangler extract --system-file system_prompt.txt \
    --user-file user_prompt.txt \
    --text-file sample_text.txt

# Using interactive mode
prompt-wrangler extract --interactive
```

### Test case example

```bash
python -m app.main extract --system-file system_prompt.txt --user-file user_prompt.txt --text-file test_case_1.txt
```


### Command Options

```text
Options:
  -s, --system TEXT               System prompt for context setting
  -sf, --system-file PATH         File containing the system prompt
  -u, --user TEXT                 User prompt with specific instructions
  -uf, --user-file PATH           File containing the user prompt
  -t, --text TEXT                 Sample text to extract entities from
  -tf, --text-file PATH           File containing the sample text
  -m, --model TEXT                OpenAI model to use [default: gpt-4o]
  -temp, --temperature FLOAT      Temperature setting (0-1) [default: 0.0]
  -mt, --max-tokens INTEGER       Maximum tokens in response [default: 1000]
  -k, --api-key TEXT              OpenAI API key (overrides env variable)
  -i, --interactive               Run in interactive mode
  -l, --log-level TEXT            Logging level (DEBUG, INFO, WARNING, ERROR)
```

## 📊 Example Output

```json
{
  "device": "CPAP",
  "mask_type": "full face",
  "add_ons": ["humidifier"],
  "qualifier": "AHI > 20",
  "ordering_provider": "Dr. Cameron"
}
```

With metrics:

```text
Response Time: 1250 ms
Model: gpt-4o
Token Usage:
  • Prompt tokens: 85
  • Completion tokens: 40
  • Total tokens: 125
```

## 🧪 Testing

```bash
pytest
```

For test coverage report:

```bash
pytest --cov=app
```

## 🤔 Project Design Thinking

### Prompt Design

The system employs a two-part prompting strategy:

1. **System Prompt**: Sets the context for entity extraction, defining the domain and expected structure.
2. **User Prompt**: Provides specific instructions for the current extraction task.

This separation allows for reusable system prompts while customizing the specific extraction needs.

### Observability

The tool provides metrics on:

- Response time (ms)
- Token usage (prompt, completion, total)
- Model used

This helps in optimizing prompts for efficiency and cost.

### Structured Output

Structured JSON output enables:

- Easy integration with downstream systems
- Consistent format for data processing
- Clear mapping of extracted entities

## 📝 Future Enhancements

- Support for KeyLLM/KeyBERT implementation
- Output comparison between different models
- Bulk processing of multiple inputs
- Web interface option
- Fine-tuning capabilities
