---
license: llama3.1
---

# FoxBrain Model Usage

This repository provides example code for running the FoxBrain model for mathematical coding problem-solving and general multi-tasking. This version of FoxBrain is based on LLama 3.1 70B.

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Helper Functions](#helper-functions)
- [Usage](#usage)
  - [Direct Python API](#1-direct-python-api-local-implementation)
  - [OpenAI-compatible API Server](#2-openai-compatible-api-server)
  - [Function Calling with VLLM](#3-advanced-function-calling-with-vllm)
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

## Helper Functions

Below are the helper functions used for parsing the model's generated responses:
```python
import re

def check_patterns(response):
    """
    Check if the response contains all required XML patterns.
    
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
    Parse the model's response and extract key components.
    
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

## System Prompt

The default system prompt used for FoxBrain (can be customized based on your needs):

```DEFAULT_SYSTEM_PROMPT = """You are a FoxBrain AI Assistant created and Developed by Foxconn (鴻海研究院). When given a human question related to multiple choices, as an expert & helpful reasoning assistant, your task is to provide a detailed answer following the instructions template below:

**Instructions:**

1. **Determine Budget**: Based on the question's complexity, decide on an appropriate step budget as an integer between 1 and 9 (e.g., 1 to 3 for an easy question, 3 to 6 for a medium question, 6 to 9 for a very difficult question). Reset the counter between `<count>` and `</count>` to this budget.

2. **Step-by-Step Solution**: Provide a logical, step-by-step solution, including code snippets where appropriate.
   - Enclose each step within `<step>` and `</step>` tags.
   - After each step, decrement the budget within `<count>` and `</count>` tags.
   - Stop when the budget reaches 0; you don't have to use all steps.

3. **Self-Reflection**: If unsure how to proceed based on self-reflection and reward, decide if you need to revise previous steps.
   - Enclose self-reflections within `<reflection>` and `</reflection>` tags.
   - Enclose the quality score within `<reward>` and `</reward>` tags.

4. **Final Answer**: After completing the reasoning steps, synthesize the steps into the final comprehensive detailed answer within `<answer>` and `</answer>` tags.

5. **Evaluation**: Provide a critical, honest, and subjective self-evaluation of your reasoning process within `<reflection>` and `</reflection>` tags.

6. **Quality Score**: Assign a quality score as a float between 0.0 and 1.0 within `<reward>` and `</reward>` tags.

**Example Format:**
<count> [starting budget] </count>
<step> [Step 1 with code snippets if necessary] </step>
<count> [remaining budget] </count>
<step> [Step 2 with code snippets if necessary] </step>
<reflection> [Evaluation of steps so far] </reflection>
<reward> [Quality score] </reward>
<count> [remaining budget] </count>
...
<answer>
[Final comprehensive answer]
</answer>
<reflection> [Final evaluation] </reflection>
<reward> [Quality score] </reward>"""

# System Prompt can be optional or left empty
# DEFAULT_SYSTEM_PROMPT = ""
```

## Basic Configuration

```python 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id="Path to Model Weight"
number_gpus = 2 ## for BF16 1 GPU for FP8 (A100 or H100 80GB Vram)
# Sampling parameters
tokenizer = AutoTokenizer.from_pretrained(model_id_tokenizer)
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

# Function to run inference
def run_inference(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    # Format the input in chat format for Llama models
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template to format the conversation
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Run inference
    outputs = llm.generate(formatted_prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return generated_text

# Example usage
question = "If a train travels at 120 km/h and another train travels at 80 km/h in the opposite direction, how long will it take for them to be 500 km apart if they start from the same station?"
response = run_inference(question)

# Parse the response to extract structured information
answer, reflection, steps, clarification = parse_response(response)

print(f"Answer: {answer}")
print(f"Steps: {steps}")
print(f"Reflection: {reflection}")
```

---

# VLLM Inference Implementation

VLLM offers three primary deployment methods for running your FoxBrain model. Each approach has different advantages depending on your use case:

## 1. Direct Python API (Local Implementation)

The simplest approach is to use VLLM's Python API directly within your applications. This method provides the most control and is ideal for:
- Custom Python applications
- Backend services
- Batch processing
- Research and experimentation

The code example above demonstrates this approach. Key components include:

- **Model Initialization**: Load the model with your specific hardware configuration
- **Input Formatting**: Properly format inputs using the chat template
- **Inference**: Generate completions with your specified parameters
- **Response Parsing**: Extract structured information from the model output

### Code Example Walkthrough

```python
# Initialize the model with hardware settings
llm = LLM(
    model="/path/to/FoxBrain_model",
    tensor_parallel_size=2,  # Distribute across 2 GPUs
    dtype="bfloat16",        # Use BF16 precision
)

# Format input with chat template
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Run inference
outputs = llm.generate(formatted_prompt, sampling_params)
generated_text = outputs[0].outputs[0].text
```

---

## 2. OpenAI-compatible API Server

For applications already built with the OpenAI API or when you need a standardized REST interface, VLLM provides an OpenAI-compatible server. This approach is ideal for:
- Web applications
- Cross-language applications
- Integration with existing OpenAI-based tools
- Services requiring a REST API

### Server Setup

Start the VLLM OpenAI-compatible server with:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/FoxBrain_model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --chat-template /path/to/llama31_chattemplate.jinja
```

### Client Usage

Once the server is running, use the OpenAI client to interact with it:

```python
from openai import OpenAI

# Configure the client to use your VLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Match your VLLM port
    api_key="dummy-key"  # The API key doesn't matter for local VLLM servers
)

# Send a request
response = client.chat.completions.create(
    model="/path/to/FoxBrain_model",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=2048,
)

# Process the response
answer = response.choices[0].message.content
```

---

## 3. Advanced: Function Calling with VLLM

VLLM supports function/tool calling capabilities similar to OpenAI's implementation. This allows your FoxBrain model to:
- Make calls to external tools and APIs
- Solve problems requiring external data
- Perform calculations or data lookups
- Process the results of these calls

This approach builds on the OpenAI-compatible API server with additional configuration.

### Server Setup with Function Calling

Enable function calling by adding specific flags when starting the server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/FoxBrain_model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \  # Enable tool choice
    --tool-call-parser llama3_json \  # Use llama3 JSON format
    --chat-template /path/to/llama31_chattemplate.jinja
```

### Function Calling Client Implementation

```python
# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
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

# Request with tools
response = client.chat.completions.create(
    model="/path/to/FoxBrain_model",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What's the weather in Taipei?"}
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0.7,
    max_tokens=2048,
)

# Handle tool calls
assistant_message = response.choices[0].message

if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
    # Process tool calls
    for tool_call in assistant_message.tool_calls:
        function_call = tool_call.function
        
        # Call your actual function and get a result
        result = "Weather in Taipei: 25°C, Partly Cloudy"  # Example result
        
        # Send the result back to the model
        follow_up = client.chat.completions.create(
            model="/path/to/FoxBrain_model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What's the weather in Taipei?"},
                assistant_message,
                {"role": "tool", "tool_call_id": tool_call.id, "name": function_call.name, "content": result}
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        
        # Get the final response
        final_response = follow_up.choices[0].message.content
```

---

# Performance Optimization

For optimal performance with VLLM, consider these tips based on your deployment method:

## Performance Tips

1. **GPU Memory**: For 70B parameter models like FoxBrain, you'll need at least 80GB of GPU memory for FP8 precision or 2x40GB GPUs for BF16 precision.

2. **Tensor Parallelism**: Use `--tensor-parallel-size` to distribute the model across multiple GPUs. This reduces per-GPU memory requirements but may slightly impact throughput.

3. **Quantization**: For memory-constrained environments, consider quantization:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model /path/to/FoxBrain_model \
       --quantization awq \
       --dtype bfloat16
   ```

4. **Batch Size**: Adjust the `--max-model-len` parameter to control context length and `--gpu-memory-utilization` (default 0.9) to control memory usage.

5. **Continuous Batching**: VLLM uses continuous batching by default, which is more efficient than traditional batching. You can adjust the maximum batch size with `--max-num-batched-tokens`.

---

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**: Reduce tensor parallel size, use quantization, or try a smaller model.

2. **Slow Inference**: Check GPU utilization, ensure enough CPU cores for preprocessing, and consider using a higher quantization precision.

3. **Invalid JSON from Function Calls**: This can happen with complex prompts. Add parsing and error handling in your application to handle and correct malformed JSON.

4. **Context Length Issues**: If you're seeing truncated outputs, increase the `--max-model-len` parameter when starting the server.

For more information, refer to the [VLLM documentation](https://vllm.readthedocs.io/).
