# Fuzzy

A tool for injecting synthetic bugs into code samples to create training data for bug detection and correction models, including diffusion-based coding models.

## Overview

Fuzzy processes code samples from HuggingFace datasets, introducing various types of bugs (from minor syntax errors to severe logical issues) using large language models (LLMs). The tool generates paired data consisting of original code, buggy code, and explanations of injected bugs, ideal for training diffusion-based code models for bug detection and correction.

## Features

- Support for any HuggingFace dataset with configurable column selection
- Concurrent processing of code samples for efficient throughput
- Configurable severity levels for injected bugs
- Exponential backoff for API rate-limit handling
- Streaming dataset loading from Hugging Face
- Structured output in JSONL format

## Requirements

- Python 3.6+
- Dependencies: `numpy`, `datasets`, `openai`
- An OpenRouter API key

## Usage

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_api_key_here"

# Run with default settings (uses codeparrot/codeparrot-clean-train)
python fuzzy.py

# Run with custom options and a different dataset
python fuzzy.py --dataset "bigcode/the-stack-v2" --split "train" --column "content" --language "javascript" --group_size 50 --max_requests 1000 --model "anthropic/claude-3-opus:beta" --output_file custom_output.jsonl
```

## Arguments

- `--dataset`: HuggingFace dataset name (default: "codeparrot/codeparrot-clean-train")
- `--split`: Dataset split to use (default: "train")
- `--column`: Column name containing the code to process (default: "content")
- `--language`: Programming language of the code (default: "python", with auto-detection)
- `--group_size`: Number of concurrent requests (default: 100)
- `--max_requests`: Maximum samples to process (default: process entire dataset)
- `--model`: LLM to use for bug injection (default: deepseek/deepseek-chat:free)
- `--output_file`: Output file path (default: bug_injected_results.jsonl)

## Output Format

The tool generates a JSONL file with each line containing:
- `original_code`: The unmodified source code
- `modified_code`: Code with injected bugs
- `explanation`: Description of the injected bugs

## Applications

The generated paired data is particularly valuable for:

- Training diffusion-based coding models for bug correction
- Creating synthetic regression test data
- Evaluating code correction capabilities of LLMs
- Teaching debugging techniques through paired examples