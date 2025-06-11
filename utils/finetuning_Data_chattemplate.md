# Chat Templates Guide

A comprehensive guide for preparing training data in various chat formats for language model fine-tuning.

## Overview

This repository provides standardized templates and examples for four popular chat formats used in language model training. All formats use JSON Lines (.jsonl) format where each line contains a complete training example.

## Supported Formats

- **[Alpaca Chat Format](#1-alpaca-chat-format)** - Simple instruction/response structure
- **[Conversation Template Format](#2-conversation-template-format)** - Multi-turn conversations with roles
- **[Tool Use Format](#3-tool-use-format)** - Function calling and tool usage scenarios
- **[RAG Format](#4-rag-retrieval-augmented-generation-format)** - Context-based question answering

## 1. Alpaca Chat Format

### Structure
```json
{"instruction": "...", "input": "...", "response": "..."}
```

### Field Descriptions
- **instruction**: The task or question being asked
- **input**: Additional context or data (can be empty string if not needed)
- **response**: The expected output or answer

### Examples

**English:**
```json
{"instruction": "What is the capital of France?", "input": "", "response": "The capital of France is Paris."}
```

**Traditional Chinese:**
```json
{"instruction": "法國的首都是什麼？", "input": "", "response": "法國的首都是巴黎。"}
```

## 2. Conversation Template Format

### Structure
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Field Descriptions
- **messages**: Array of message objects representing the conversation
- **role**: Can be "system", "user", or "assistant"
- **content**: The actual message content

### Examples

**English:**
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
```

**Traditional Chinese:**
```json
{"messages": [{"role": "system", "content": "你是一個有用的助手。"}, {"role": "user", "content": "法國的首都是什麼？"}, {"role": "assistant", "content": "法國的首都是巴黎。"}]}
```

## 3. Tool Use Format

### Structure
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant with access to the following tools: \n\n[TOOL_DEFINITIONS_IN_JSON_SCHEMA_FORMAT]"}, {"role": "user", "content": "..."}, {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "tool_name", "arguments": "{...}"}}]}, {"role": "tool", "tool_call_id": "call_123", "content": "{...}"}, {"role": "assistant", "content": "..."}]}
```

### Field Descriptions
- **messages**: Array of message objects representing the tool usage conversation
- **role**: Can be "system", "user", "assistant", or "tool"
- **content**: The actual message content
- **tool_calls**: Array of tool call objects (for assistant role)
- **tool_call_id**: Reference ID linking tool response to tool call
- **function**: Contains tool name and arguments

### Examples

**English:**
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant with access to the following tools: \n\n[TOOL_DEFINITIONS_IN_JSON_SCHEMA_FORMAT]"}, {"role": "user", "content": "What's the weather in London?"}, {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"}}]}, {"role": "tool", "tool_call_id": "call_123", "content": "{\"temperature\": 15, \"unit\": \"celsius\", \"description\": \"Cloudy\"}"}, {"role": "assistant", "content": "The weather in London is 15 degrees Celsius and cloudy."}]}
```

**Traditional Chinese:**
```json
{"messages": [{"role": "system", "content": "你是一個有用的助手，可以使用以下工具：\n\n[工具定義的JSON架構格式]"}, {"role": "user", "content": "倫敦的天氣如何？"}, {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"}}]}, {"role": "tool", "tool_call_id": "call_123", "content": "{\"temperature\": 15, \"unit\": \"celsius\", \"description\": \"Cloudy\"}"}, {"role": "assistant", "content": "倫敦的天氣是攝氏15度，多雲。"}]}
```

## 4. RAG (Retrieval-Augmented Generation) Format

### Structure
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant. Answer the user's question only based on the provided document context. If the answer is not in the document, state that you don't know.", "context": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Field Descriptions
- **messages**: Array of message objects representing the context-based conversation
- **role**: Can be "system", "user", or "assistant"
- **content**: The actual message content
- **context**: Additional field in system message containing the source document or context information

### Examples

**English:**
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant. Answer the user's question only based on the provided document context. If the answer is not in the document, state that you don't know.", "context": "The Eiffel Tower was built in 1889 for the 1889 Universal Exposition in Paris, France. It was designed by Gustave Eiffel. It is 330 meters tall."}, {"role": "user", "content": "When was the Eiffel Tower constructed?"}, {"role": "assistant", "content": "The Eiffel Tower was constructed in 1889."}]}
```

**Traditional Chinese:**
```json
{"messages": [{"role": "system", "content": "你是一個有用的助手。請僅根據提供的文件內容回答用戶的問題。如果答案不在文件中，請說明你不知道。", "context": "艾菲爾鐵塔建於1889年，為1889年巴黎世界博覽會而建。它由古斯塔夫·艾菲爾設計，高度為330米。"}, {"role": "user", "content": "艾菲爾鐵塔是什麼時候建造的？"}, {"role": "assistant", "content": "艾菲爾鐵塔建於1889年。"}]}
```

## File Structure Examples

### Alpaca Format (data.jsonl)
```
{"instruction": "What is the capital of France?", "input": "", "response": "The capital of France is Paris."}
{"instruction": "法國的首都是什麼？", "input": "", "response": "法國的首都是巴黎。"}
```

### Conversation Format (data.jsonl)
```
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "system", "content": "你是一個有用的助手。"}, {"role": "user", "content": "法國的首都是什麼？"}, {"role": "assistant", "content": "法國的首都是巴黎。"}]}
```

### Tool Use Format (data.jsonl)
```
{"messages": [{"role": "system", "content": "You are a helpful assistant with access to the following tools: \n\n[TOOL_DEFINITIONS_IN_JSON_SCHEMA_FORMAT]"}, {"role": "user", "content": "What's the weather in London?"}, {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"}}]}, {"role": "tool", "tool_call_id": "call_123", "content": "{\"temperature\": 15, \"unit\": \"celsius\", \"description\": \"Cloudy\"}"}, {"role": "assistant", "content": "The weather in London is 15 degrees Celsius and cloudy."}]}
{"messages": [{"role": "system", "content": "你是一個有用的助手，可以使用以下工具：\n\n[工具定義的JSON架構格式]"}, {"role": "user", "content": "倫敦的天氣如何？"}, {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"}}]}, {"role": "tool", "tool_call_id": "call_123", "content": "{\"temperature\": 15, \"unit\": \"celsius\", \"description\": \"Cloudy\"}"}, {"role": "assistant", "content": "倫敦的天氣是攝氏15度，多雲。"}]}
```

### RAG Format (data.jsonl)
```
{"messages": [{"role": "system", "content": "You are a helpful assistant. Answer the user's question only based on the provided document context. If the answer is not in the document, state that you don't know.", "context": "The Eiffel Tower was built in 1889 for the 1889 Universal Exposition in Paris, France. It was designed by Gustave Eiffel. It is 330 meters tall."}, {"role": "user", "content": "When was the Eiffel Tower constructed?"}, {"role": "assistant", "content": "The Eiffel Tower was constructed in 1889."}]}
{"messages": [{"role": "system", "content": "你是一個有用的助手。請僅根據提供的文件內容回答用戶的問題。如果答案不在文件中，請說明你不知道。", "context": "艾菲爾鐵塔建於1889年，為1889年巴黎世界博覽會而建。它由古斯塔夫·艾菲爾設計，高度為330米。"}, {"role": "user", "content": "艾菲爾鐵塔是什麼時候建造的？"}, {"role": "assistant", "content": "艾菲爾鐵塔建於1889年。"}]}
```

## Best Practices

1. **Consistency**: Maintain consistent formatting across all entries
2. **Clarity**: Make instructions clear and specific
3. **Completeness**: Ensure responses fully address the instruction
4. **Variety**: Include diverse types of tasks and complexity levels
5. **Quality**: Verify that responses are accurate and helpful

## Usage

1. Choose the appropriate format for your use case
2. Create a `.jsonl` file following the format structure
3. Add one JSON object per line (no commas between lines)
4. Validate your JSON structure before training
5. Use the examples as templates for your specific domain

## Contributing

Feel free to contribute additional examples or improvements to the templates. Please ensure all examples follow the established format standards.

## License

This guide is provided under [MIT License](LICENSE).