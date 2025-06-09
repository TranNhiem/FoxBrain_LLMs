---
license: llama3.1
---

# FoxBrain Model Usage

This repository provides example code for running the FoxBrain model for mathematical coding problem-solving and general multi-tasking. This version of FoxBrain is based on LLama 3.1 70B and now features **three distinct reasoning modes** built directly into the tokenizer configuration.

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [New: Three Reasoning Modes](#new-three-reasoning-modes)
- [Helper Functions](#helper-functions)
- [System Prompt](#system-prompt)
- [Chat Template](#chat-template)
- [Deployment Methods](#deployment-methods)
  - [1. Direct Python API (Local Implementation)](#1-direct-python-api-local-implementation)
  - [2. OpenAI-compatible API Server](#2-openai-compatible-api-server)
  - [3. Advanced: Function Calling with VLLM](#3-advanced-function-calling-with-vllm)
- [Migration Guide](#migration-guide)
- [Tested FoxBrain Deployment Example](#tested-foxbrain-deployment-example)
- [Parsing Structured Outputs from FoxBrain](#parsing-structured-outputs-from-foxbrain)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [GPU & BF16 Precision](#gpu--bf16-precision)

## Installation

Ensure you have Python 3.8 or higher installed. Then, install the required dependencies using pip. You can either install the packages individually or use a requirements file.

### Using pip
```bash
pip install vllm==0.6.6.post1
pip install transformers==4.48.0 
```

## New: Three Reasoning Modes

**🆕 Major Update**: FoxBrain now includes three built-in reasoning modes that can be easily switched using the `enable_thinking` parameter in the tokenizer's `apply_chat_template()` method. No more manual system prompt management!

### Available Modes:

1. **`non-reasoning`** (Simple Mode): Basic FoxBrain assistant for general conversations
2. **`thinking`** (Thinking Mode): Enhanced reasoning without structured output
3. **`budget_thinking`** (Budget Mode): Advanced step-by-step reasoning with structured XML output and budget management

### Quick Usage Example:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-foxbrain-model-path")

# Simple mode
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True, 
    enable_thinking="non-reasoning"
)

# Thinking mode  
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True, 
    enable_thinking="thinking"
)

# Budget thinking mode (structured output)
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True, 
    enable_thinking="budget_thinking"
)
```

## Helper Functions

The helper functions below are **specifically designed for `budget_thinking` mode**, which produces structured XML output with `<step>`, `<count>`, `<answer>`, and `<reflection>` tags.

> **Note**: These functions are only needed when using `enable_thinking="budget_thinking"`. For `non-reasoning` and `thinking` modes, you can work directly with the generated text.

```python
import re

def check_patterns(response):
    """
    Check if the response contains all required XML patterns for budget_thinking mode.
    
    Args:
        response (str): The model's generated response
    
    Returns:
        str: Parsed response or 'Missing' if patterns are incomplete
    """
    patterns = {
        'answer': r'<answer>(.*?)</answer>',
        'reflection': r'<reflection>(.*?)</reflection>',
        'steps': r'<step>(.*?)</step>',
        'count': r'<count>(.*?)</count>'
    }
    
    matches = {
        'answer': re.search(patterns['answer'], response, re.DOTALL),
        'reflection': re.search(patterns['reflection'], response, re.DOTALL),
        'steps': re.findall(patterns['steps'], response, re.DOTALL),
        'count': re.findall(patterns['count'], response, re.DOTALL)
    }
    
    return "Missing" if not all([matches['answer'], matches['reflection'], matches['steps'], matches['count']]) else response

def parse_response(response):
    """
    Parse the model's response and extract key components (for budget_thinking mode).
    
    Args:
        response (str): The model's generated response
    
    Returns:
        tuple: Parsed answer, reflection, steps, and clarification
    """
    response_check = check_patterns(response)
    
    if response_check == "Missing":
        clarification_match = re.search(r'<clarification>(.*?)</clarification>', response, re.DOTALL)
        clarification = clarification_match.group(1).strip() if clarification_match else response
        return "", "", [], clarification
    else:
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        reflection_match = re.search(r'<reflection>(.*?)</reflection>', response, re.DOTALL)
        
        answer = answer_match.group(1).strip() if answer_match else ""
        reflection = reflection_match.group(1).strip() if reflection_match else ""
        steps = re.findall(r'<step>(.*?)</step>', response, re.DOTALL)
        
        return answer, reflection, steps, ""
```

## Chat Template

The FoxBrain model now includes an enhanced chat template with built-in support for the three reasoning modes. The template automatically selects the appropriate system prompt based on the `enable_thinking` parameter.

> **Default Mode**: If no `enable_thinking` parameter is specified, the model defaults to `"budget_thinking"` mode.

When using the Direct Python API approach, the enhanced template is applied automatically with the tokenizer:

```python
# Initialize tokenizer (template is built-in)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Use with different modes
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking="budget_thinking"  # or "non-reasoning" or "thinking"
)
```

## Basic Configuration

```python 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "Path to FoxBrain model checkpoint"
number_gpus = 2  # for BF16, 1 GPU for FP8 (A100 or H100 80GB Vram)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=4096, 
    # repetition_penalty=1.5,
    stop=['<|eot_id|>', '<|end_of_text|>'], 
    #stop_token_ids=[128001,128009]
)

# Initialize VLLM model
llm = LLM(
    model=model_id,
    tensor_parallel_size=number_gpus,  # Number of GPUs to use
    dtype="bfloat16",  # Use bfloat16 for better performance with Llama models
)

# Function to run inference with mode selection
def run_inference(prompt, mode="budget_thinking"):
    """
    Run inference with the specified reasoning mode.
    
    Args:
        prompt (str): User's question or prompt
        mode (str): One of "non-reasoning", "thinking", or "budget_thinking"
    
    Returns:
        str: Generated response
    """
    # Format the input in chat format for Llama models
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template with mode selection
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=mode
    )
    
    # Run inference
    outputs = llm.generate(formatted_prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return generated_text

# Example usage for different modes
question = "If a train travels at 120 km/h and another train travels at 80 km/h in the opposite direction, how long will it take for them to be 500 km apart if they start from the same station?"

# Simple mode - direct answer
simple_response = run_inference(question, mode="non-reasoning")
print(f"Simple Mode Response: {simple_response}")

# Thinking mode - enhanced reasoning
thinking_response = run_inference(question, mode="thinking")
print(f"Thinking Mode Response: {thinking_response}")

# Budget thinking mode - structured output
budget_response = run_inference(question, mode="budget_thinking")

# Parse structured output (only for budget_thinking mode)
answer, reflection, steps, clarification = parse_response(budget_response)
print(f"Answer: {answer}")
print(f"Steps: {steps}")
print(f"Reflection: {reflection}")
```

---

# VLLM Inference Implementation

VLLM offers three primary deployment methods for running your FoxBrain model. Each approach now supports the new reasoning modes.

## 1. Direct Python API (Local Implementation)

The simplest approach uses VLLM's Python API directly. This method now supports easy mode switching:

```python
# Interactive mode switching example
def interactive_foxbrain():
    """Interactive FoxBrain with mode switching capabilities."""
    
    current_mode = "budget_thinking"
    messages = []
    
    print("FoxBrain Interactive Mode")
    print("Commands: 'mode0' (simple), 'mode1' (thinking), 'mode2' (budget_thinking), 'reset', 'quit'")
    print(f"Current mode: {current_mode}")
    print("-" * 60)
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            messages = []
            print("History cleared!")
            continue
        elif user_input.lower() == 'mode0':
            current_mode = 'non-reasoning'
            messages = []
            print("Switched to Simple mode!")
            continue
        elif user_input.lower() == 'mode1':
            current_mode = 'thinking'
            messages = []
            print("Switched to Thinking mode!")
            continue
        elif user_input.lower() == 'mode2':
            current_mode = 'budget_thinking'
            messages = []
            print("Switched to Budget Thinking mode!")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Use tokenizer's chat template with the selected mode
        prompt = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False, 
            enable_thinking=current_mode
        )
        
        # Generate
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        print(f"Assistant: {generated_text}")
        
        # Parse structured output for budget_thinking mode
        if current_mode == "budget_thinking":
            answer, reflection, steps, clarification = parse_response(generated_text)
            if answer:  # Only show parsed output if structured elements are found
                print("\n--- Parsed Output ---")
                print(f"Final Answer: {answer}")
                if steps:
                    print(f"Steps taken: {len(steps)}")
                print(f"Reflection: {reflection}")
        
        messages.append({"role": "assistant", "content": generated_text})

# Run the interactive session
interactive_foxbrain()
```

## 2. OpenAI-compatible API Server

For applications requiring a REST API, VLLM provides an OpenAI-compatible server that supports the new reasoning modes through custom parameters.

### Server Setup

Start the VLLM OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "Path to FoxBrain model checkpoint" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16
```

### Client Usage with Mode Selection

```python
from openai import OpenAI

# Configure the client to use your VLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

def chat_with_mode(question, mode="budget_thinking"):
    """Send a request with the specified reasoning mode."""
    
    # Create messages (no system prompt needed - handled by tokenizer)
    messages = [{"role": "user", "content": question}]
    
    # Add mode as extra_body parameter (if supported by your VLLM version)
    # Note: Mode selection through API may require custom implementation
    response = client.chat.completions.create(
        model="Path to FoxBrain model checkpoint",
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        # Custom parameter for mode (implementation dependent)
        extra_body={"enable_thinking": mode}
    )
    
    return response.choices[0].message.content

# Example usage
question = "What is the capital of France?"

# Try different modes
simple_answer = chat_with_mode(question, "non-reasoning")
thinking_answer = chat_with_mode(question, "thinking") 
budget_answer = chat_with_mode(question, "budget_thinking")
```

> **Note**: Mode selection through the OpenAI API may require additional server configuration or custom parameter handling depending on your VLLM version.

## 3. Advanced: Function Calling with VLLM

Function calling now works seamlessly with all three reasoning modes. The `budget_thinking` mode is particularly effective for complex function calling scenarios.

### Enhanced Server Setup

```bash
vllm serve "Path to FoxBrain model checkpoint" \
    --dtype auto \
    --api-key "your-api-key" \
    --port 8883 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.97
```

### Function Calling with Mode Awareness

```python
def test_function_calling_with_mode(question, mode="budget_thinking"):
    """Test function calling with different reasoning modes."""
    
    # Tools definition (same as before)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Make request (mode can be specified if API supports it)
    response = client.chat.completions.create(
        model="Path to FoxBrain model checkpoint",
        messages=[{"role": "user", "content": question}],
        tools=tools,
        tool_choice="auto",
        temperature=0.5,
        max_tokens=2048,
        # Custom parameter for mode (if supported)
        extra_body={"enable_thinking": mode}
    )
    
    # Handle tool calls as before...
    # (Tool calling logic remains the same)
```

## Migration Guide

### For Existing Users

If you have existing code using manual system prompts, here's how to migrate:

#### Old Approach (Still Works):
```python
# Old way - manual system prompt management
DEFAULT_SYSTEM_PROMPT = """You are FoxBrain AI by Foxconn..."""

messages = [
    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": user_question}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

#### New Approach (Recommended):
```python
# New way - mode-based system prompt selection
messages = [{"role": "user", "content": user_question}]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True,
    enable_thinking="budget_thinking"  # Selects appropriate system prompt automatically
)
```

### Breaking Changes: None!

- **Existing code continues to work** without modification
- **Helper functions remain the same** for `budget_thinking` mode
- **API compatibility is maintained**

### Recommended Updates:

1. **Remove manual system prompts** and use `enable_thinking` parameter
2. **Use mode-specific parsing** only for `budget_thinking` mode
3. **Update deployment scripts** to leverage new mode capabilities

---

# Parsing Structured Outputs from FoxBrain

The parsing functions are **specifically designed for `budget_thinking` mode** outputs. For other modes, you can work directly with the generated text.

```python
def parse_response_enhanced(response, mode="budget_thinking"):
    """
    Enhanced parsing function that handles different modes appropriately.
    
    Args:
        response (str): The model's generated response
        mode (str): The reasoning mode used ("non-reasoning", "thinking", "budget_thinking")
    
    Returns:
        dict: Parsed components based on mode
    """
    if mode == "budget_thinking":
        # Use structured parsing for budget_thinking mode
        patterns = {
            "budget": r'<count>([^<]+)</count>',
            "steps": r'<step>([^<]+)</step>',
            "answers": r'<answer>([^<]+)</answer>',
            "reflections": r'<reflection>([^<]+)</reflection>',
            "clarifications": r'<clarification>([^<]+)</clarification>',
            "quality": r'<reward>([^<]+)</reward>'
        }
        
        parsed = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                parsed[key] = [match.strip() for match in matches]
        
        return parsed
    
    else:
        # For non-reasoning and thinking modes, return the response as-is
        return {"response": response.strip()}

def display_parsed_response(parsed, mode="budget_thinking"):
    """Display parsed components based on mode."""
    
    if mode == "budget_thinking":
        print(f"\n=== Budget Thinking Mode Output ===")
        
        if "budget" in parsed:
            print(f"🔢 STEP BUDGET: {parsed['budget'][0]}")
        
        if "steps" in parsed:
            print("\n📋 STEPS:")
            for i, step in enumerate(parsed["steps"], 1):
                print(f"  {i}. {step}")
        
        if "answers" in parsed:
            print("\n✅ FINAL ANSWER:")
            print(f"  {parsed['answers'][0]}")
        
        if "reflections" in parsed:
            print("\n🤔 REFLECTION:")
            print(f"  {parsed['reflections'][0]}")
        
        if "quality" in parsed:
            print("\n⭐ QUALITY SCORE:")
            print(f"  {parsed['quality'][0]}")
    
    else:
        print(f"\n=== {mode.title()} Mode Output ===")
        print(parsed.get("response", "No response content"))

# Usage example
response_content = generated_text
current_mode = "budget_thinking"  # or "thinking" or "non-reasoning"

parsed_components = parse_response_enhanced(response_content, mode=current_mode)
display_parsed_response(parsed_components, mode=current_mode)
```

---

## Performance Optimization

Performance recommendations remain the same, with additional considerations for mode selection:

### Mode Performance Characteristics:

- **`non-reasoning`**: Fastest, most direct responses
- **`thinking`**: Moderate processing time, enhanced reasoning
- **`budget_thinking`**: Slower but most thorough, structured output

### Deployment Tips:

1. **Use `non-reasoning` mode** for simple Q&A and general conversation
2. **Use `thinking` mode** for complex reasoning without structured output needs
3. **Use `budget_thinking` mode** for mathematical problems, coding tasks, and scenarios requiring step-by-step analysis

### GPU Memory Requirements:

- **70B parameter model**: 80GB GPU memory for FP8, or 2x40GB GPUs for BF16
- **Tensor Parallelism**: Use `--tensor-parallel-size` for multi-GPU setups
- **Memory optimization**: Consider quantization for constrained environments

```bash
# Optimized deployment command
vllm serve "Path to FoxBrain model checkpoint" \
    --dtype auto \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.97 \
    --enable-chunked-prefill
```

---

## Troubleshooting

### Common Issues and Solutions:

1. **Mode not working**: Ensure you're using the latest tokenizer configuration with built-in mode support
2. **Structured output missing**: Only `budget_thinking` mode produces XML-structured output
3. **Performance issues**: Choose appropriate mode for your use case
4. **Memory errors**: Adjust `--gpu-memory-utilization` and `--max-model-len`

### Mode Selection Guidelines:

- **Simple queries**: Use `non-reasoning` mode
- **Complex reasoning**: Use `thinking` mode
- **Step-by-step analysis**: Use `budget_thinking` mode
- **Function calling**: `budget_thinking` mode recommended for complex scenarios

For additional support and examples, refer to the included example scripts and documentation.
