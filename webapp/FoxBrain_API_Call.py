'''
TranNhiem 2024/06 -- FoxBrain Call API 
This function call model through Request URL generate 
'''

import subprocess
import os
import requests
import json


### Helper function to format the chat messages and Print out the response
# DEFAULT_SYSTEM_PROMPT = """\nYou are FoxBrain AI assistant developed by Foxconn AI Research Center, designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity. You will using Traditional Chinese, English or Others to response as long as the same Language with user query input.  If user's question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
#             If you don't know the answer to a question, please don't share false information."""

DEFAULT_SYSTEM_PROMPT = """\nYou are FoxBrain AI assistant developed by Foxconn AI Research Center, You are designed to help users find detailed and comprehensive information, always aiming to provide answers in such a manner that users don't need to search elsewhere for clarity."""

def apply_chat_template(example):
    messages = example["messages"]
    formatted_chat = ""

    eos_token = "<|eot_id|>"  
    if messages and messages[0]["role"] == "system":
        # Update the content of the first system message
        previous_content =  messages[0]["content"] #"<|begin_of_text|>system<|start_header_id|>\n"+
        messages[0]["content"] = "<|begin_of_text|>system<|start_header_id|>\n" + previous_content +eos_token
    ## Add an empty system message if there is no initial system message
    elif messages and messages[0]["role"] != "system":
        # Insert a new system message at the beginning if there isn't one
        messages.insert(0, {"role": "system", "content": f"<|begin_of_text|>system<|start_header_id|>{DEFAULT_SYSTEM_PROMPT}<|eot_id|>"})
        
    # Define your end-of-sentence token here
    eos_token_="<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    # Loop through the messages to apply the template
    for i, message in enumerate(messages):
        role = message['role']
        
        if role =="user":
            content = "<|start_header_id|>user<|end_header_id|>\n\n" + message['content'] + eos_token_
            formatted_chat += f'{content}'
        
        elif  role =="assistant":
            content = message['content'] + eos_token +"\n"
            formatted_chat += f'\n{content}' 
        else :
            content = message['content']
            formatted_chat += f'{content}' 

            
    return formatted_chat


def print_response(response):
    if response.status_code == 200:
        all_content = []
        try:
            # Iterate over the response stream line by line
            for line in response.iter_lines():
                if line:
                    try:
                        # Decode line and convert to JSON
                        response_json = json.loads(line.decode('utf-8'))
                        # Extract "response" content if it exists
                        content = response_json.get("response", "")
                        if content:
                            all_content.append(content)
                            # Print content as it arrives
                            print(content, end='', flush=True)
                    except json.JSONDecodeError as e:
                        # Handle JSON decoding errors
                        print("Error parsing JSON:", e)
            # Ensure the final print is on a new line
            print()
        except Exception as e:
            # Handle other exceptions that may occur
            print("An error occurred:", e)
        finally:
            # Join all content into a single string and return it
            return ' '.join(all_content)
    
    else:
        # Handle non-200 status codes
        print("Request failed with status code:", response.status_code)
        return ""

messages = {
"messages": [
    {"role": "system", "content": f"{DEFAULT_SYSTEM_PROMPT}"}
]
}
## Optional 
# Append the user message
messages["messages"].append({"role": "user", "content": "你好嗎 ：）？"})
# Append the assistant response
messages["messages"].append({"role": "assistant", "content": "你好！我很好，謝謝你問 :) 你今天過得怎麼樣？"})  
## Now user Question or Input  
messages["messages"].append({"role": "user", "content": f"能告訴我一些關於你自己的事嗎？"})

input_prompt = apply_chat_template(messages)


url = "http://40.84.133.133:8889/api/generate"



text_output = ""
temperature_=0.7
typical_p=0.5

while text_output == "":
    payload = {
        "model": "FoxBrain_8B_202406",
        "prompt": input_prompt,
        "stream": True, 
        "options": {
            # "seed": 123,
            "temperature": 0.7, ## 
            #"top_k": 50,
            # "top_p": 0.7,
            # # "tfs_z": 0.5,
            "typical_p": 0.5,
            # # "repeat_last_n": 33,
            # # "repeat_penalty": 1.2,
            # "presence_penalty": 0.4,
            # "frequency_penalty": 0.7,
        }
    }
   
    response = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, stream=True)

    respone_output=print_response(response)
    text_output = respone_output
    temperature_ = random.uniform(0.6, 1.0) 
    typical_p_= random.uniform(0.4, 1.0)
print("*************************************************************")
