---
license: llama3.1
---

# FoxBrain Model Deployment with Huggingface

This guide provides instructions for deploying and using the FoxBrain model using Huggingface's Transformers library. The FoxBrain model is based on LLama 3.1 70B and can be deployed using various methods depending on your use case.

> **Note**: FoxBrain models are built on Llama 3.1 and include the chat template directly in the tokenizer configuration. When loading the model with Huggingface, the correct chat template is automatically available, simplifying deployment and ensuring correct formatting of inputs.

## Table of Contents

- [Installation](#installation)
- [Helper Functions](#helper-functions)
- [Deployment Methods](#deployment-methods)
  - [Local Inference with Transformers](#1-local-inference-with-transformers)
  - [Inference with Accelerate](#2-inference-with-accelerate)
  - [Text Generation Inference (TGI)](#3-text-generation-inference-tgi)
- [API Integration](#api-integration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Installation

Ensure you have Python 3.8 or higher installed. Install the required dependencies:

```bash
# Basic Transformers setup
pip install transformers==4.48.0
pip install torch>=2.0.0
pip install accelerate
pip install bitsandbytes  # For quantization

# Text Generation Inference (optional)
pip install text-generation
```

## Helper Functions

These helper functions are useful for parsing the model's structured output:

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

FoxBrain works best with this system prompt (can be customized):

```python
DEFAULT_SYSTEM_PROMPT = """You are a FoxBrain AI Assistant created and Developed by Foxconn (鴻海研究院). When given a human question related to multiple choices, as an expert & helpful reasoning assistant, your task is to provide a detailed answer following the instructions template below:

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
"""
```

---

## Deployment Methods

Huggingface offers several deployment methods for FoxBrain, each with different advantages:

### 1. Local Inference with Transformers

The simplest approach for direct Python integration. Best for:
- Single-user applications
- Research and development
- Testing and fine-tuning workflows

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList

# Define custom stopping criteria for the tokens that signal the end of generation
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Load model and tokenizer
model_path = "/path/to/FoxBrain_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Llama 3.1-based models like FoxBrain typically have the chat template included
# We can access it directly from the tokenizer
print(f"Tokenizer has chat template: {tokenizer.chat_template is not None}")

# Set up a fallback chat template only if needed (should not be necessary for FoxBrain)
if tokenizer.chat_template is None:
    print("Warning: Chat template not found in tokenizer, using fallback template")
    tokenizer.chat_template = """
    {% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
    {% else %}
    {% set loop_messages = messages %}
    {% set system_message = "" %}
    {% endif %}
    {% if system_message != "" %}
    <s>{{ system_message }}</s>
    {% endif %}
    {% for message in loop_messages %}
    <s>{{ message['role'] }}: {{ message['content'] }}</s>
    {% endfor %}
    <s>assistant: 
    """

# Get token IDs for stop sequences
stop_words = ['<|eot_id|>', '<|end_of_text|>', '<|end_header_id|>']
stop_token_ids = [tokenizer.encode(word, add_special_tokens=False)[-1] for word in stop_words]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

# Create a text-generation pipeline
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    stopping_criteria=stopping_criteria
)

# Function to generate responses
def generate_response(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Format using chat template
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
    response = gen_pipeline(chat_prompt)[0]['generated_text']
    
    # Extract the assistant's response part
    response = response.split("assistant:")[-1].strip()
    
    return response

# Example usage
question = "If a train travels at 120 km/h and another train travels at 80 km/h in the opposite direction, how long will it take for them to be 500 km apart if they start from the same station?"
response = generate_response(question)

# Parse the response
answer, reflection, steps, clarification = parse_response(response)
print(f"Answer: {answer}")
print(f"Steps: {steps}")
```

---

### 2. Inference with Accelerate

For larger models that don't fit in a single GPU's memory, use Accelerate for distributed inference:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, infer_auto_device_map

# Initialize accelerator
accelerator = Accelerator()

# Load model with automatic device mapping
model_path = "/path/to/FoxBrain_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Get the optimal device map
device_map = infer_auto_device_map(
    AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map=None,  # Just for device map computation
    )
)

# Load with the computed device map
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True
)

# Check if chat template is available (should be for FoxBrain)
print(f"Tokenizer has chat template: {tokenizer.chat_template is not None}")

# Get token IDs for stop sequences
stop_words = ['<|eot_id|>', '<|end_of_text|>', '<|end_header_id|>']
stop_token_ids = [tokenizer.encode(word, add_special_tokens=False)[-1] for word in stop_words]

# Generate response function (similar to the previous example)
def generate_with_accelerate(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Format using chat template
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids
    
    # Move input to the appropriate device
    input_ids = input_ids.to(accelerator.device)
    
    # Generate tokens
    output_ids = model.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # Add stop tokens directly to generate method
        eos_token_id=stop_token_ids
    )
    
    # Decode and return only the new tokens
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response
```

---

### 3. Text Generation Inference (TGI)

For production deployments with high throughput, use Huggingface's Text Generation Inference server:

#### Server Setup

```bash
# Install the TGI server
pip install text-generation-server

# Start the TGI server
text-generation-launcher \
    --model-id /path/to/FoxBrain_model \
    --port 8080 \
    --hostname 0.0.0.0 \
    --max-total-tokens 4096 \
    --max-input-length 2048 \
    --max-batch-prefill-tokens 4096 \
    --trust-remote-code \
    --stopping-criteria '<|eot_id|>','<|end_of_text|>','<|end_header_id|>'
```

#### Client Usage

```python
from text_generation import Client

# Connect to the TGI server
client = Client("http://localhost:8080")

def generate_with_tgi(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    # Create messages format
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Generate response
    response = client.chat(
        messages=messages,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        stop_sequences=["<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>"]
    )
    
    return response.choices[0].message.content

# Example usage
question = "What's the best way to implement a binary search tree in Python?"
response = generate_with_tgi(question)

# Parse the structured response
answer, reflection, steps, clarification = parse_response(response)
```

---

## API Integration

You can also deploy FoxBrain to the Huggingface Inference API and access it remotely:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/your-username/FoxBrain"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Example usage with chat completion
output = query_api({
    "inputs": {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Explain how to implement merge sort in Python"}
        ]
    },
    "parameters": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True
    }
})

# Process the response
response = output["generated_text"]
answer, reflection, steps, clarification = parse_response(response)
```

---

## Performance Optimization

For optimal performance when deploying FoxBrain with Huggingface:

### Memory Optimization

1. **4-bit Quantization**: Reduce memory usage dramatically with minimal quality loss
   ```python
   from transformers import BitsAndBytesConfig
   
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True
   )
   
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       quantization_config=quantization_config,
       device_map="auto"
   )
   ```

2. **Flash Attention**: Enable faster attention computation
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
       device_map="auto"
   )
   ```

3. **Model Sharding**: Distribute model across multiple GPUs
   ```python
   device_map = {
       "model.embed_tokens": 0,
       "model.layers.0": 0,
       "model.layers.1": 0,
       "model.layers.2": 0,
       # ... assign layers to different GPUs
       "model.layers.30": 1,
       "model.layers.31": 1,
       "model.norm": 1,
       "lm_head": 1
   }
   
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map=device_map,
       torch_dtype=torch.bfloat16
   )
   ```

### Inference Speed

1. **Batch Processing**: Process multiple prompts simultaneously
   ```python
   prompts = ["Question 1", "Question 2", "Question 3"]
   inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
   
   with torch.no_grad():
       outputs = model.generate(**inputs, max_new_tokens=512)
   
   responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
   ```

2. **KV Caching**: For chat applications, reuse KV cache
   ```python
   # First prompt
   inputs = tokenizer("First message", return_tensors="pt").to("cuda")
   with torch.no_grad():
       outputs = model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True)
       past_key_values = outputs.past_key_values
   
   # Follow-up with KV cache
   new_inputs = tokenizer("Follow-up message", return_tensors="pt").to("cuda")
   with torch.no_grad():
       outputs = model.generate(
           **new_inputs, 
           max_new_tokens=50,
           past_key_values=past_key_values
       )
   ```

---

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Use quantization (4-bit or 8-bit)
   - Distribute model across multiple GPUs
   - Reduce batch size or sequence length
   - Use gradient checkpointing during fine-tuning

2. **Slow Generation**
   - Enable Flash Attention if your GPU supports it
   - Use KV caching for multiple generations
   - Adjust `num_beams` or switch to greedy decoding
   - For TGI, increase the number of shards to utilize more GPUs

3. **Token Length Issues**
   - Properly set `max_new_tokens` and `max_total_tokens`
   - For long conversations, use a sliding window approach
   - Compress or summarize context when exceeding token limits

4. **Unexpected Output Format**
   - Ensure the chat template is correctly set
   - Verify system prompt is properly formatted
   - Check tokenizer configuration matches model requirements

5. **TGI Server Issues**
   - Verify port availability and permissions
   - Check CUDA/GPU compatibility
   - Ensure sufficient disk space for model loading

For more information, refer to the [Huggingface documentation](https://huggingface.co/docs/transformers/index) and [TGI documentation](https://huggingface.co/docs/text-generation-inference/index).
