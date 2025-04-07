---
license: llama3.1
---

# FoxBrain 模型使用指南

本文檔提供了運行 FoxBrain 模型的示例代碼，該模型專為數學編程問題解決和多任務處理而設計。此版本的 FoxBrain 基於 LLama 3.1 70B 模型開發。

## 目錄

- [安裝指南](#安裝指南)
- [概述](#概述)
- [輔助函數](#輔助函數)
- [系統提示詞](#系統提示詞)
- [對話模板](#對話模板)
- [部署方法](#部署方法)
  - [1. 直接使用 Python API（本地實現）](#1-直接使用-python-api本地實現)
  - [2. OpenAI 兼容 API 伺服器](#2-openai-兼容-api-伺服器)
  - [3. 進階：使用 VLLM 進行函數調用](#3-進階使用-vllm-進行函數調用)
- [測試過的 FoxBrain 部署範例](#測試過的-foxbrain-部署範例)
- [解析 FoxBrain 結構化輸出](#解析-foxbrain-結構化輸出)
- [性能優化](#性能優化)
- [常見問題排解](#常見問題排解)
- [GPU 和 BF16 精度](#gpu-和-bf16-精度)

## 安裝指南

確保您已安裝 Python 3.8 或更高版本。然後，使用 pip 安裝所需的依賴項。您可以單獨安裝軟件包或使用 requirements 文件。

### 使用 pip
```bash
pip install vllm==0.6.6.post1
pip install transformers==4.48.0 
```

## 輔助函數

以下是用於解析模型生成回應的輔助函數：
```python
import re

def check_patterns(response):
    """
    檢查回應是否包含所有必需的 XML 標籤。
    
    參數：
        response (str)：模型生成的回應
    
    返回：
        str：解析後的回應，如果模式不完整則返回 'Missing'
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
    解析模型的回應並提取關鍵組件。
    
    參數：
        response (str)：模型生成的回應
    
    返回：
        tuple：解析後的答案、反思、步驟和說明
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

## 系統提示詞

FoxBrain 使用的默認系統提示詞（可根據需要自定義）：

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

# 系統提示詞可以選擇性使用或留空
# DEFAULT_SYSTEM_PROMPT = ""
```

## 對話模板

FoxBrain 模型需要使用 Llama 3.1 兼容的對話模板來正確格式化消息。模板文件 `llama31_chattemplate.jinja` 對於正確操作至關重要：

1. 從代碼庫下載對話模板文件
2. 將其保存到系統中可訪問的位置
3. 啟動 VLLM 伺服器時指定其路徑：

```bash
--chat-template "llama31_chattemplate.jinja 的路徑"
```

對話模板對以下方面至關重要：
- 正確格式化模型的對話歷史
- 確保一致處理系統提示詞、用戶消息和助手回應
- 支持函數/工具調用的預期消息格式

當使用直接 Python API 方法時，模板會通過分詞器自動應用：

```python
# 確保使用相同的模板初始化分詞器
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 如果需要，指定對話模板路徑
# tokenizer.chat_template = open("llama31_chattemplate.jinja 的路徑").read()

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## 基本配置

```python 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id="FoxBrain 模型檢查點的路徑"
number_gpus = 2 ## 對於 BF16 需要 1 GPU，對於 FP8 需要 A100 或 H100 80GB Vram
# 採樣參數
tokenizer = AutoTokenizer.from_pretrained(model_id_tokenizer)
sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=4096, 
            # repetition_penalty=1.5,
            stop=['<|eot_id|>', '<|end_of_text|>'], 
            #stop_token_ids=[128001,128009]
        )

# 初始化 VLLM 模型
llm = LLM(
    model=model_id,
    tensor_parallel_size=number_gpus,  # 使用的 GPU 數量
    dtype="bfloat16",  # 使用 bfloat16 提高 Llama 模型性能
)

# 運行推理的函數
def run_inference(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    # 以聊天格式格式化 Llama 模型的輸入
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # 應用對話模板格式化對話
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 運行推理
    outputs = llm.generate(formatted_prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return generated_text

# 使用範例
question = "如果一列火車以 120 公里/小時的速度行駛，另一列火車以 80 公里/小時的速度朝相反方向行駛，如果它們從同一車站出發，需要多長時間才能相距 500 公里？"
response = run_inference(question)

# 解析回應以提取結構化信息
answer, reflection, steps, clarification = parse_response(response)

print(f"答案: {answer}")
print(f"步驟: {steps}")
print(f"反思: {reflection}")
```

---

# VLLM 推理實現

VLLM 提供三種主要部署方法來運行您的 FoxBrain 模型。每種方法都有不同的優勢，取決於您的使用場景：

## 1. 直接使用 Python API（本地實現）

最簡單的方法是直接在應用程序中使用 VLLM 的 Python API。此方法提供最多控制權，適合：
- 自定義 Python 應用程序
- 後端服務
- 批處理
- 研究和實驗

上面的代碼示例演示了這種方法。主要組件包括：

- **模型初始化**：使用特定硬件配置加載模型
- **輸入格式化**：使用對話模板正確格式化輸入
- **推理**：使用指定參數生成完成
- **回應解析**：從模型輸出中提取結構化信息

### 代碼示例詳解

```python
# 使用硬件設置初始化模型
llm = LLM(
    model="FoxBrain_模型的路徑",
    tensor_parallel_size=2,  # 分佈在 2 個 GPU 上
    dtype="bfloat16",        # 使用 BF16 精度
)

# 使用對話模板格式化輸入
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 運行推理
outputs = llm.generate(formatted_prompt, sampling_params)
generated_text = outputs[0].outputs[0].text
```

---

## 2. OpenAI 兼容 API 伺服器

對於已經使用 OpenAI API 構建的應用程序或需要標準化 REST 接口時，VLLM 提供了 OpenAI 兼容伺服器。此方法適合：
- Web 應用程序
- 跨語言應用程序
- 與現有 OpenAI 基礎工具集成
- 需要 REST API 的服務

### 伺服器設置

使用以下命令啟動 VLLM OpenAI 兼容伺服器：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "FoxBrain 模型檢查點的路徑" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --chat-template "llama31_chattemplate.jinja 的路徑"
```

> **注意**：`--host 0.0.0.0` 參數使伺服器可從任何 IP 地址訪問。在生產環境中出於安全考慮，您可能希望綁定到特定 IP 地址。`--port` 參數（本例中為 8000）確定伺服器監聽的端口。

### 客戶端使用

伺服器運行後，使用 OpenAI 客戶端與其交互：

```python
from openai import OpenAI

# 配置客戶端使用您的 VLLM 伺服器
client = OpenAI(
    base_url="http://localhost:8000/v1",  # 匹配您伺服器的 IP/主機名和端口
    api_key="dummy-key"  # 對於本地 VLLM 伺服器，API 密鑰不重要
)

# 發送請求
response = client.chat.completions.create(
    model="FoxBrain 模型檢查點的路徑",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "法國的首都是什麼？"}
    ],
    temperature=0.7,
    max_tokens=2048,
)

# 處理回應
answer = response.choices[0].message.content
```

> **重要**：客戶端配置中的 `base_url` 必須與伺服器的 IP 地址和端口匹配。如果您的伺服器在 IP 為 192.168.1.100、端口為 8000 的遠程機器上運行，您的 base_url 應為 `http://192.168.1.100:8000/v1`。對於本地測試，使用 `localhost` 或 `127.0.0.1`。

---

## 3. 進階：使用 VLLM 進行函數調用

VLLM 支持類似於 OpenAI 實現的函數/工具調用功能。這允許您的 FoxBrain 模型：
- 調用外部工具和 API
- 解決需要外部數據的問題
- 執行計算或數據查詢
- 處理這些調用的結果

這種方法在 OpenAI 兼容 API 伺服器基礎上進行了額外配置。

### 使用函數調用的伺服器設置

通過添加特定標誌啟用函數調用來啟動伺服器：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "FoxBrain 模型檢查點的路徑" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template "llama31_chattemplate.jinja 的路徑"
```

> **注意**：在跨不同機器部署時，如果要限制訪問，請將 `--host 0.0.0.0` 替換為伺服器的 IP 地址。然後更新所有客戶端的 `base_url` 值以指向此 IP 地址（例如，`http://server-ip-address:8000/v1`）。

### 測試過的 FoxBrain 部署範例

以下命令已經過測試並驗證，可以高效地運行具有函數調用功能的 FoxBrain 模型：

```bash
vllm serve "FoxBrain 模型檢查點的路徑" \
    --dtype auto \
    --api-key "您的 API 密鑰" \
    --port 8883 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template "llama31_chattemplate.jinja 的路徑" \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.97
```

此命令包括：
- 自動 dtype 檢測以獲得最佳性能
- 在端口 8883 上公開 API，帶有自定義 API 密鑰
- 啟用函數調用並使用 Llama3 JSON 解析器
- 設置較大的上下文窗口（32K 標記）
- 在 2 個 GPU 上分佈模型
- 最大化 GPU 內存使用率（97%）

示例 Python 腳本 `foxbrain_function_calling_example.py` 演示了如何連接到此 VLLM 伺服器並使用 FoxBrain 的函數調用功能。該腳本包括天氣查詢和計算器函數的實現，以及用於處理函數調用回應的強大 JSON 解析功能。

> **重要**：`llama31_chattemplate.jinja` 文件對於 FoxBrain 模型的正確函數調用至關重要。確保從代碼庫下載此模板文件並在 VLLM 命令中指定其正確路徑。如果沒有正確的對話模板，模型可能無法正確理解或生成函數調用。

以下是此示例中客戶端代碼的關鍵部分：

```python
from openai import OpenAI
import json
import re

# 配置客戶端使用您的 VLLM 伺服器
client = OpenAI(
    base_url="http://127.0.0.1:8883/v1",  # 匹配您伺服器的 IP/主機名和端口
    api_key="您的 API 密鑰"  # 匹配您的 API 密鑰
)

> **重要**：跨機器部署時：
> - 在同一台機器上進行本地測試：使用 `base_url="http://127.0.0.1:PORT/v1"` 或 `base_url="http://localhost:PORT/v1"`
> - 從另一台機器連接：使用 `base_url="http://SERVER_IP:PORT/v1"`，其中 SERVER_IP 是運行 VLLM 的伺服器 IP 地址
> - 始終確保端口號與 VLLM 伺服器命令中指定的端口號匹配（上例中為 8883）

# 定義工具函數
def get_weather(location: str, unit: str = "celsius"):
    """模擬的天氣函數用於測試"""
    weather_data = {
        "San Francisco": {"celsius": "18°C", "fahrenheit": "64°F", "condition": "多霧"},
        "New York": {"celsius": "22°C", "fahrenheit": "72°F", "condition": "晴朗"},
        "Tokyo": {"celsius": "25°C", "fahrenheit": "77°F", "condition": "多雲"},
        "London": {"celsius": "16°C", "fahrenheit": "61°F", "condition": "下雨"},
    }
    
    # 未知位置的默認回應
    if location not in weather_data:
        return f"{location} 的天氣數據（{unit}）：20°C/68°F，晴朗"
    
    data = weather_data[location]
    temp = data[unit]
    condition = data["condition"]
    
    return f"{location} 的天氣數據：{temp}，{condition}"

# 將函數名映射到函數
tool_functions = {
    "get_weather": get_weather,
    "calculator": calculator  # 計算器函數實現省略以簡化
}

# 以 OpenAI 格式定義工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "獲取指定位置的當前天氣",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名稱，例如 'San Francisco'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "溫度單位"}
                },
                "required": ["location"]
            }
        }
    }
    # 計算器工具定義省略以簡化
]

# 調用 API 並處理工具調用的示例
def test_function_calling(question):
    # 定義 FoxBrain 系統提示詞（帶結構化輸出指示）
    system_prompt = """您是由鴻海研究院創建和開發的 FoxBrain AI 助手...."""
    
    # 使用工具發送完成請求
    response = client.chat.completions.create(
        model="FoxBrain 模型檢查點的路徑",  # 您的模型路徑
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
    
    # 獲取助手的消息
    assistant_message = response.choices[0].message
    
    # 處理存在的工具調用
    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            function_call = tool_call.function
            
            # 解析參數並調用函數
            arguments = json.loads(function_call.arguments)
            function = tool_functions[function_call.name]
            result = function(**arguments)
            
            # 將函數結果發送回模型
            follow_up_response = client.chat.completions.create(
                model="FoxBrain 模型檢查點的路徑",  # 您的模型路徑
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
            
            # 獲取納入工具結果後的最終回應
            final_response = follow_up_response.choices[0].message.content
```

## 函數調用客戶端實現

```python
# 定義可用工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "獲取指定位置的當前天氣",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名稱"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

# 帶工具的請求
response = client.chat.completions.create(
    model="FoxBrain 模型檢查點的路徑",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "台北的天氣如何？"}
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0.7,
    max_tokens=2048,
)

# 處理工具調用
assistant_message = response.choices[0].message

if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
    # 處理工具調用
    for tool_call in assistant_message.tool_calls:
        function_call = tool_call.function
        
        # 調用您的實際函數並獲取結果
        result = "台北天氣：25°C，多雲"  # 示例結果
        
        # 將結果發送回模型
        follow_up = client.chat.completions.create(
            model="FoxBrain 模型檢查點的路徑",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "台北的天氣如何？"},
                assistant_message,
                {"role": "tool", "tool_call_id": tool_call.id, "name": function_call.name, "content": result}
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        
        # 獲取最終回應
        final_response = follow_up.choices[0].message.content
```

---

# 解析 FoxBrain 結構化輸出

FoxBrain 模型已經經過微調，能夠生成帶有特定標籤的結構化輸出，如 `<step>`、`<reflection>` 和 `<answer>`。在使用模型時，您經常需要提取這些組件進行進一步處理。以下是可用於 FoxBrain 輸出的強大解析函數：

```python
def parse_response(response):
    """解析 FoxBrain 模型回應中的結構化元素。"""
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

def display_parsed_response(parsed, title="解析後的回應"):
    """以結構化格式顯示解析後的組件。"""
    print(f"\n=== {title} ===")
    
    if not parsed:
        print("回應中未找到結構化元素。")
        return
    
    if "budget" in parsed:
        print(f"🔢 步驟預算: {parsed['budget'][0].strip()}")
    
    if "steps" in parsed:
        print("\n📋 步驟:")
        for i, step in enumerate(parsed["steps"], 1):
            print(f"  {i}. {step.strip()}")
    
    if "answers" in parsed:
        print("\n✅ 答案:")
        print(f"  {parsed['answers'][0].strip()}")
    
    if "reflections" in parsed:
        print("\n🤔 反思:")
        print(f"  {parsed['reflections'][0].strip()}")
    
    if "quality" in parsed:
        print("\n⭐ 質量評分:")
        print(f"  {parsed['quality'][0].strip()}")
```

### 使用範例:

```python
# 從模型接收回應後
response_content = assistant_message.content
parsed_components = parse_response(response_content)
display_parsed_response(parsed_components)

# 對於函數調用回應
if has_tool_calls:
    # 提取答案組件以進行更清晰的工具處理
    answers = parsed_components.get("answers", [])
    if answers:
        print(f"用於工具處理的提取答案: {answers[0]}")
```

這種解析方法在結合 FoxBrain 的結構化輸出功能與函數調用時特別有用，因為它允許您：

1. 優先處理最終 `<answer>` 等特定組件以進行決策
2. 跟踪 FoxBrain 在複雜推理過程中如何管理其步驟預算
3. 提取解釋模型解決問題方法的反思
4. 隔離組件以滿足不同的下游處理需求

---

## 性能優化

對於 VLLM 的最佳性能，根據您的部署方法考慮以下提示：

## 性能提示

1. **GPU 內存**：對於像 FoxBrain 這樣的 70B 參數模型，您需要至少 80GB 的 GPU 內存用於 FP8 精度，或者 2x40GB GPU 用於 BF16 精度。

2. **張量並行**：使用 `--tensor-parallel-size` 在多個 GPU 上分佈模型。這減少了每 GPU 的內存需求，但可能略微影響吞吐量。

3. **量化**：對於內存受限的環境，考慮量化：
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model "FoxBrain 模型的路徑" \
       --quantization awq \
       --dtype bfloat16
   ```

4. **批量大小**：調整 `--max-model-len`（默認為 16384，可設置為最大上下文長度 128k）參數控制上下文長度，並使用 `--gpu-memory-utilization`（默認為 0.97）控制內存使用。

5. **連續批處理**：VLLM 默認使用連續批處理，這比傳統批處理更有效。您可以使用 `--max-num-batched-tokens` 調整最大批大小。

---

## 常見問題排解

常見問題及解決方案：

1. **內存不足**：減少張量並行大小，使用量化，或嘗試較小的模型。

2. **推理緩慢**：檢查 GPU 利用率，確保有足夠的 CPU 核心進行預處理，並考慮使用更高的量化精度。

3. **函數調用中的無效 JSON**：這可能發生在複雜提示中。在應用程序中添加解析和錯誤處理，以處理並糾正格式錯誤的 JSON。

4. **上下文長度問題**：如果您看到輸出被截斷，在啟動伺服器時增加 `--max-model-len` 參數。

更多信息，請參考 [VLLM 文檔](https://vllm.readthedocs.io/)。