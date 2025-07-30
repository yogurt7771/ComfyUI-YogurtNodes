# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-YogurtNodes is a collection of custom nodes for ComfyUI, providing image processing, LLM integration, string manipulation, and workflow utilities. The project is structured as a Python package with modular components organized by functionality.

## Development Commands

### Setup and Installation
```bash
# Install dependencies 
pip install -r requirements.txt

# Development environment setup (optional)
pip install isort flake8 black ipython
```

### Code Formatting
```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Lint with flake8
flake8 .
```

## Project Architecture

### Core Structure
```
yogurt_nodes/
├── __init__.py              # Main module exports
├── image/                   # Image processing nodes
├── io/                      # Input/output operations  
├── llm/                     # Language model integrations
├── logic/                   # Logic and control flow nodes
├── models/                  # Model selection utilities
├── number/                  # Number generation nodes
├── string/                  # String manipulation nodes
└── utils.py                 # Shared utilities
```

### Module Organization
- **Image**: Image processing (text overlay, batch operations, Poisson blending)
- **IO**: File operations, preview systems, directory management
- **LLM**: Gemini, OpenAI, OpenRouter API integrations with unified configuration
- **Logic**: Control flow nodes (switch, pack/unpack, none)
- **Models**: Model selectors for checkpoints, LoRA, ControlNet
- **String**: Text processing, regex operations, path manipulation
- **Number**: Range and numeric utilities

### ComfyUI Integration
All nodes follow ComfyUI conventions:
- Nodes use `@classmethod` methods for input/output definitions
- Categories prefixed with "YogurtNodes/"
- Return tuples matching `RETURN_TYPES`
- Display names defined in `FUNCTION` attribute

## API Key Configuration

The project implements a three-tier priority system for API keys across all LLM services:

### Priority Order
1. **Code parameter** (highest priority)
2. **api_key.json file** in `yogurt_nodes/llm/`
3. **Environment variables** (lowest priority)

### Configuration File Format
```json
{
    "gemini": "YOUR_GEMINI_API_KEY",
    "openai": "YOUR_OPENAI_API_KEY", 
    "openai_base_url": "https://api.openai.com/v1",
    "openrouter": "YOUR_OPENROUTER_API_KEY"
}
```

### Environment Variables
- `GEMINI_API_KEY` - Gemini API key
- `GOOGLE_GENAI_USE_VERTEXAI=true` - Enable Vertex AI
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_BASE_URL` - OpenAI base URL (optional)
- `OPENROUTER_API_KEY` - OpenRouter API key

## Node Development Patterns

### Base Node Structure
```python
class YourNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}, "optional": {...}}
    
    RETURN_TYPES = ("TYPE1", "TYPE2")
    FUNCTION = "your_function"
    CATEGORY = "YogurtNodes/Category"
    
    def your_function(self, ...):
        # Implementation
        return (result1, result2)
```

### LLM Client Pattern
```python
class LLMClient:
    def __init__(self, api_key: str = ""):
        # Try api_key parameter first
        if not api_key:
            # Try api_key.json file
            # Try environment variable
            # Raise error if none found
```

## File Operations

The project includes extensive I/O utilities:
- **Bridge nodes**: Preview and save operations with metadata support
- **Directory management**: Create directories with parent creation
- **Glob operations**: Pattern-based file discovery with sorting options
- **Format support**: PNG, JPEG, text files with custom extensions

## Testing and Quality

While no automated test suite is present, the devcontainer setup includes:
- Black code formatter
- isort import organizer  
- flake8 linter
- Development dependencies in setup.sh

## Dependencies

Core dependencies managed through:
- `requirements.txt` - Runtime dependencies
- `pyproject.toml` - Project metadata and publishing configuration
- `.devcontainer/setup.sh` - Development environment setup