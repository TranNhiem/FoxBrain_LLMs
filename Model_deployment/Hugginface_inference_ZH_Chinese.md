---
license: llama3.1
---

# 使用 Huggingface 部署 FoxBrain 模型

本指南提供使用 Huggingface 的 Transformers 庫部署和使用 FoxBrain 模型的說明。FoxBrain 模型基於 LLama 3.1 70B，可以根據您的使用情境採用不同的部署方法。

> **注意**：FoxBrain 模型基於 Llama 3.1 並在分詞器配置中直接包含聊天模板。使用 Huggingface 載入模型時，正確的聊天模板會自動可用，簡化部署流程並確保輸入格式正確。

## 目錄

- [安裝](#安裝)
- [輔助函數](#輔助函數)
- [部署方法](#部署方法)
  - [使用 Transformers 進行本地推理](#1-使用-transformers-進行本地推理)
  - [使用 Accelerate 進行推理](#2-使用-accelerate-進行推理)
  - [文本生成推理 (TGI)](#3-文本生成推理-tgi)
- [API 整合](#api-整合)
- [性能優化](#性能優化)
- [故障排除](#故障排除)

## 安裝

確保您已安裝 Python 3.8 或更高版本。安裝所需的依賴項：

```bash
# 基本 Transformers 設置
pip install transformers==4.48.0
pip install torch>=2.0.0
pip install accelerate
pip install bitsandbytes  # 用於量化

# 文本生成推理 (可選)
pip install text-generation
```

## 輔助函數

這些輔助函數有助於解析模型的結構化輸出：

```python
import re

def check_patterns(response):
    """
    檢查回應是否包含所有必需的 XML 模式。
    
    參數:
        response (str): 模型生成的回應
    
    返回:
        str: 解析後的回應，如果模式不完整則返回 'Missing'
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
    
    參數:
        response (str): 模型生成的回應
    
    返回:
        tuple: 解析後的答案、反思、步驟和澄清
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

## 系統提示

FoxBrain 使用以下系統提示效果最佳（可自定義）：

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

## 部署方法

Huggingface 提供了多種部署 FoxBrain 的方法，每種方法都有不同的優勢：

### 1. 使用 Transformers 進行本地推理

最簡單的 Python 直接整合方法。最適合：
- 單用戶應用程序
- 研究與開發
- 測試和微調工作流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList

# 定義自定義停止條件，用於標記生成結束的標記
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# 載入模型和分詞器
model_path = "/path/to/FoxBrain_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 基於 Llama 3.1 的 FoxBrain 模型通常包含聊天模板
# 我們可以直接從分詞器訪問它
print(f"分詞器包含聊天模板：{tokenizer.chat_template is not None}")

# 僅在需要時設置備用聊天模板（對於 FoxBrain 應該不必要）
if tokenizer.chat_template is None:
    print("警告：在分詞器中未找到聊天模板，使用 FoxBrain 聊天格式化")
    # 定義自定義聊天格式化函數而不使用模板
    def apply_chat_template(messages):
        formatted_chat = ""

        eos_token = "<|eot_id|>"  
        if messages and messages[0]["role"] == "system":
            # 更新第一個系統消息的內容
            previous_content = messages[0]["content"]
            messages[0]["content"] = "<|begin_of_text|>system<|start_header_id|>\n" + previous_content + eos_token
        elif messages and messages[0]["role"] != "system":
            # 如果沒有系統消息，則在開頭插入一個新的系統消息
            messages.insert(0, {
                "role": "system",
                "content": "You are FoxBrain assistant"
            })

        eos_token_ = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        # 循環處理消息以應用模板
        for message in messages:
            role = message['role']
            if role == "user":
                content = "<|start_header_id|>user<|end_header_id|>\n\n" + message['content'] + eos_token_
                formatted_chat += f'{content}'
            elif role == "assistant":
                content = message['content'] + eos_token + "\n"
                formatted_chat += f'\n{content}'
            else:
                content = message['content']
                formatted_chat += f'{content}'

        return formatted_chat
    
    # 使用我們的自定義函數覆蓋分詞器的 apply_chat_template 方法
    tokenizer.apply_chat_template = lambda messages, tokenize=False, add_generation_prompt=False, **kwargs: apply_chat_template(messages)

# 獲取停止序列的標記 ID
stop_words = ['<|eot_id|>', '<|end_of_text|>', '<|end_header_id|>']
stop_token_ids = [tokenizer.encode(word, add_special_tokens=False)[-1] for word in stop_words]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

# 創建文本生成管道
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

# 生成回應的函數
def generate_response(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # 使用聊天模板格式化
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 生成回應
    response = gen_pipeline(chat_prompt)[0]['generated_text']
    
    # 提取助手的回應部分
    response = response.split("assistant:")[-1].strip()
    
    return response

# 使用示例
question = "如果一列火車以 120 公里/小時的速度行駛，另一列火車以 80 公里/小時的速度朝相反方向行駛，如果它們從同一車站出發，需要多長時間才能相距 500 公里？"
response = generate_response(question)

# 解析回應
answer, reflection, steps, clarification = parse_response(response)
print(f"答案: {answer}")
print(f"步驟: {steps}")
```

---

### 2. 使用 Accelerate 進行推理

對於無法裝入單個 GPU 內存的較大模型，使用 Accelerate 進行分佈式推理：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, infer_auto_device_map

# 初始化加速器
accelerator = Accelerator()

# 使用自動設備映射加載模型
model_path = "/path/to/FoxBrain_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 獲取最佳設備映射
device_map = infer_auto_device_map(
    AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map=None,  # 僅用於設備映射計算
    )
)

# 使用計算的設備映射加載
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True
)

# 檢查聊天模板是否可用（對於 FoxBrain 應該可用）
print(f"分詞器包含聊天模板：{tokenizer.chat_template is not None}")

# 獲取停止序列的標記 ID
stop_words = ['<|eot_id|>', '<|end_of_text|>', '<|end_header_id|>']
stop_token_ids = [tokenizer.encode(word, add_special_tokens=False)[-1] for word in stop_words]

# 生成回應函數（類似於前面的示例）
def generate_with_accelerate(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # 使用聊天模板格式化
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 標記化輸入
    input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids
    
    # 將輸入移動到適當的設備
    input_ids = input_ids.to(accelerator.device)
    
    # 生成標記
    output_ids = model.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        # 直接將停止標記添加到生成方法
        eos_token_id=stop_token_ids
    )
    
    # 解碼並僅返回新標記
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response
```

---

### 3. 文本生成推理 (TGI)

對於需要高吞吐量的生產部署，使用 Huggingface 的文本生成推理服務器：

#### 服務器設置

```bash
# 安裝 TGI 服務器
pip install text-generation-server

# 啟動 TGI 服務器
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

#### 客戶端使用

```python
from text_generation import Client

# 連接到 TGI 服務器
client = Client("http://localhost:8080")

def generate_with_tgi(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    # 創建消息格式
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # 生成回應
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

# 使用示例
question = "用 Python 實現二叉搜索樹的最佳方法是什麼？"
response = generate_with_tgi(question)

# 解析結構化回應
answer, reflection, steps, clarification = parse_response(response)
```

---

## API 整合

您還可以將 FoxBrain 部署到 Huggingface 推理 API 並遠程訪問：

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/your-username/FoxBrain"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 使用聊天完成的示例
output = query_api({
    "inputs": {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "解釋如何用 Python 實現合併排序"}
        ]
    },
    "parameters": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "stop": ["<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>"]
    }
})

# 處理回應
response = output["generated_text"]
answer, reflection, steps, clarification = parse_response(response)
```

---

## 性能優化

為了在使用 Huggingface 部署 FoxBrain 時獲得最佳性能：

### 記憶體優化

1. **4 位元量化**：大幅減少記憶體使用，並最小化質量損失
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

2. **Flash Attention**：啟用更快的注意力計算
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
       device_map="auto"
   )
   ```

3. **模型分割**：將模型分佈到多個 GPU 上
   ```python
   device_map = {
       "model.embed_tokens": 0,
       "model.layers.0": 0,
       "model.layers.1": 0,
       "model.layers.2": 0,
       # ... 將層分配到不同的 GPU
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

### 推理速度

1. **批處理**：同時處理多個提示
   ```python
   prompts = ["問題 1", "問題 2", "問題 3"]
   inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
   
   with torch.no_grad():
       outputs = model.generate(**inputs, max_new_tokens=512)
   
   responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
   ```

2. **KV 快取**：對於聊天應用程序，重用 KV 快取
   ```python
   # 第一個提示
   inputs = tokenizer("第一條消息", return_tensors="pt").to("cuda")
   with torch.no_grad():
       outputs = model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True)
       past_key_values = outputs.past_key_values
   
   # 使用 KV 快取的後續提示
   new_inputs = tokenizer("後續消息", return_tensors="pt").to("cuda")
   with torch.no_grad():
       outputs = model.generate(
           **new_inputs, 
           max_new_tokens=50,
           past_key_values=past_key_values
       )
   ```

---

## 故障排除

常見問題及解決方案：

1. **記憶體不足錯誤**
   - 使用量化（4 位元或 8 位元）
   - 將模型分佈到多個 GPU 上
   - 減少批次大小或序列長度
   - 在微調過程中使用梯度檢查點

2. **生成速度慢**
   - 如果您的 GPU 支持，啟用 Flash Attention
   - 對多次生成使用 KV 快取
   - 調整 `num_beams` 或切換到貪婪解碼
   - 對於 TGI，增加分片數量以利用更多 GPU

3. **標記長度問題**
   - 正確設置 `max_new_tokens` 和 `max_total_tokens`
   - 對於長時間對話，使用滑動窗口方法
   - 超出標記限制時壓縮或總結上下文

4. **意外輸出格式**
   - 確保聊天模板正確設置
   - 驗證系統提示格式正確
   - 檢查分詞器配置是否符合模型要求

5. **TGI 服務器問題**
   - 驗證端口可用性和權限
   - 檢查 CUDA/GPU 兼容性
   - 確保有足夠的磁盤空間用於模型加載

更多信息，請參考 [Huggingface 文檔](https://huggingface.co/docs/transformers/index) 和 [TGI 文檔](https://huggingface.co/docs/text-generation-inference/index)。
