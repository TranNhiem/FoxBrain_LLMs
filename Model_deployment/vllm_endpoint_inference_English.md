---
license: llama3.1
---

# FoxBrain Model Usage

This repository provides example code for running the FoxBrain model for mathematical coding problem-solving and general multi-tasking. This version of FoxBrain is based on LLama 3.1 70B.

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Helper Functions](#helper-functions)
- [System Prompt](#system-prompt)
- [Chat Template](#chat-template)
- [Deployment Methods](#deployment-methods)
  - [1. Direct Python API (Local Implementation)](#1-direct-python-api-local-implementation)
  - [2. OpenAI-compatible API Server](#2-openai-compatible-api-server)
  - [3. Advanced: Function Calling with VLLM](#3-advanced-function-calling-with-vllm)
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

## Chat Template

The FoxBrain model requires proper message formatting using a Llama 3.1 compatible chat template. The template file `llama31_chattemplate.jinja` is essential for correct model operation:

1. Download the chat template file from the repository
2. Save it to an accessible location on your system
3. Specify its path when starting the VLLM server:

```bash
--chat-template "Path to llama31_chattemplate.jinja"
```

The chat template is crucial for:
- Properly formatting the conversation history for the model
- Ensuring consistent handling of system prompts, user messages, and assistant responses
- Supporting function/tool calls with the expected message format

When using the Direct Python API approach, the template is applied automatically with the tokenizer:

```python
# Make sure to initialize the tokenizer with the same template
tokenizer = AutoTokenizer.from_pretrained(model_id)
# If needed, specify the chat template path
# tokenizer.chat_template = open("Path to llama31_chattemplate.jinja").read()

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## Basic Configuration

```python 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id="Path to FoxBrain model checkpoint"
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
    --model "Path to FoxBrain model checkpoint" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --chat-template "Path to llama31_chattemplate.jinja"
```

> **Note**: The `--host 0.0.0.0` parameter makes the server accessible from any IP address. For security in production environments, you might want to bind to a specific IP address. The `--port` parameter (8000 in this example) determines which port the server listens on.

### Client Usage

Once the server is running, use the OpenAI client to interact with it:

```python
from openai import OpenAI

# Configure the client to use your VLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Match your server's IP/hostname and port
    api_key="dummy-key"  # The API key doesn't matter for local VLLM servers
)

# Send a request
response = client.chat.completions.create(
    model="Path to FoxBrain model checkpoint",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=2048,
 stop=['<|eot_id|>', '<|end_of_text|>', '<|end_header_id|>'],
)

# Process the response
answer = response.choices[0].message.content
```

> **Important**: The `base_url` in your client configuration must match the server's IP address and port. If your server is running on a remote machine with IP 192.168.1.100 and port 8000, your base_url should be `http://192.168.1.100:8000/v1`. For local testing, use `localhost` or `127.0.0.1`.

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
    --model "Path to FoxBrain model checkpoint" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template "Path to llama31_chattemplate.jinja"
```

> **Note**: When deploying across different machines, replace `--host 0.0.0.0` with your server's IP address if you want to restrict access. Then update all client `base_url` values to point to this IP address (e.g., `http://server-ip-address:8000/v1`).

### Tested FoxBrain Deployment Example

The following command has been tested and verified to work efficiently with the FoxBrain model for function calling:

```bash
vllm serve "Path to FoxBrain model checkpoint" \
    --dtype auto \
    --api-key "your-api-key" \
    --port 8883 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template "Path to llama31_chattemplate.jinja" \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.97
```

This command includes:
- Automatic dtype detection for optimal performance
- API exposure on port 8883 with a custom API key
- Function calling enabled with the Llama3 JSON parser
- A large context window set to 32K tokens
- Distribution of the model across 2 GPUs
- Maximization of GPU memory usage at 97%

The example Python script `foxbrain_function_calling_example.py` demonstrates how to connect to this VLLM server and use function calling capabilities with FoxBrain. The script includes implementations for weather lookup and calculator functions, as well as robust JSON parsing for handling function call responses.

> **Important**: The `llama31_chattemplate.jinja` file is critical for proper function calling with the FoxBrain model. Make sure to download this template file from the repository and specify its correct path in your VLLM command. Without the proper chat template, the model may not properly understand or generate function calls.

Here's a key portion of the client code from this example:

```python
from openai import OpenAI
import json
import re

# Configure client to use your VLLM server
client = OpenAI(
    base_url="http://127.0.0.1:8883/v1",  # Match your server's IP/hostname and port
    api_key="your-api-key"  # Match your API key
)

> **Important**: When deploying across machines:
> - For local testing on the same machine: use `base_url="http://127.0.0.1:PORT/v1"` or `base_url="http://localhost:PORT/v1"`
> - For connecting from another machine: use `base_url="http://SERVER_IP:PORT/v1"` where SERVER_IP is the IP address of the server running VLLM
> - Always ensure the port number matches the one specified in your VLLM server command (8883 in the example above)

# Define tool functions
def get_weather(location: str, unit: str = "celsius"):
    """Simulated weather function for testing"""
    weather_data = {
        "San Francisco": {"celsius": "18°C", "fahrenheit": "64°F", "condition": "Foggy"},
        "New York": {"celsius": "22°C", "fahrenheit": "72°F", "condition": "Sunny"},
        "Tokyo": {"celsius": "25°C", "fahrenheit": "77°F", "condition": "Partly Cloudy"},
        "London": {"celsius": "16°C", "fahrenheit": "61°F", "condition": "Rainy"},
    }
    
    # Default response for unknown locations
    if location not in weather_data:
        return f"Weather data for {location} in {unit}: 20°C/68°F, Sunny"
    
    data = weather_data[location]
    temp = data[unit]
    condition = data["condition"]
    
    return f"Weather data for {location}: {temp}, {condition}"

# Map function names to functions
tool_functions = {
    "get_weather": get_weather,
    "calculator": calculator  # Calculator function implementation omitted for brevity
}

# Define tools in OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g., 'San Francisco'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                },
                "required": ["location"]
            }
        }
    }
    # Calculator tool definition omitted for brevity
]

# Example of calling the API and handling tool calls
def test_function_calling(question):
    # Define the FoxBrain system prompt (with structured output instructions)
    system_prompt = """You are a FoxBrain AI Assistant created and Developed by Foxconn (鴻海研究院)...."""
    
    # Make completion request with tools
    response = client.chat.completions.create(
        model="Path to FoxBrain model checkpoint",  # Your model path
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.5,
        max_tokens=2048,
        stop=['<|eot_id|>', '<|end_of_text|>', '<|end_header_id|>'],
        seed=42,
    )
    
    # Get the assistant's message
    assistant_message = response.choices[0].message
    
    # Handle tool calls if present
    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            function_call = tool_call.function
            
            # Parse arguments and call the function
            arguments = json.loads(function_call.arguments)
            function = tool_functions[function_call.name]
            result = function(**arguments)
            
            # Send the function result back to the model
            follow_up_response = client.chat.completions.create(
                model="Path to FoxBrain model checkpoint",  # Your model path
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    assistant_message,
                    {"role": "tool", "tool_call_id": tool_call.id, "name": function_call.name, "content": result}
                ],
                temperature=0.5,
                max_tokens=2048,
                stop=['<|eot_id|>', '<|end_of_text|>', '<|end_header_id|>'],
                seed=42,
            )
            
            # Get the final response after incorporating tool results
            final_response = follow_up_response.choices[0].message.content
```

## Function Calling Client Implementation

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
    model="Path to FoxBrain model checkpoint",
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
            model="Path to FoxBrain model checkpoint",
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

# Parsing Structured Outputs from FoxBrain

The FoxBrain model has been fine-tuned to produce structured outputs with specific tags like `<step>`, `<reflection>`, and `<answer>`. When working with the model, you'll often want to extract these components for further processing. Here's a robust parsing function that can be used with the FoxBrain outputs:

```python
def parse_response(response):
    """Parse structured elements from FoxBrain model responses."""
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
            parsed[key] = matches
    
    return parsed

def display_parsed_response(parsed, title="Parsed Response"):
    """Display parsed components in a structured format."""
    print(f"\n=== {title} ===")
    
    if not parsed:
        print("No structured elements found in the response.")
        return
    
    if "budget" in parsed:
        print(f"🔢 STEP BUDGET: {parsed['budget'][0].strip()}")
    
    if "steps" in parsed:
        print("\n📋 STEPS:")
        for i, step in enumerate(parsed["steps"], 1):
            print(f"  {i}. {step.strip()}")
    
    if "answers" in parsed:
        print("\n✅ ANSWER:")
        print(f"  {parsed['answers'][0].strip()}")
    
    if "reflections" in parsed:
        print("\n🤔 REFLECTION:")
        print(f"  {parsed['reflections'][0].strip()}")
    
    if "quality" in parsed:
        print("\n⭐ QUALITY SCORE:")
        print(f"  {parsed['quality'][0].strip()}")
```

### Example Usage:

```python
# After receiving a response from the model
response_content = assistant_message.content
parsed_components = parse_response(response_content)
display_parsed_response(parsed_components)

# For function calling responses
if has_tool_calls:
    # Extract just the answer component for cleaner tool processing
    answers = parsed_components.get("answers", [])
    if answers:
        print(f"Using extracted answer for tool processing: {answers[0]}")
```

This parsing approach is particularly useful when combining FoxBrain's structured output capabilities with function calling, as it allows you to:

1. Prioritize specific components like the final `<answer>` for decision-making
2. Track how FoxBrain manages its step budget during complex reasoning
3. Extract reflections that explain the model's approach to problem-solving
4. Isolate components for different downstream processing needs

---

## Performance Optimization

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

4. **Batch Size**: Adjust the `--max-model-len`
