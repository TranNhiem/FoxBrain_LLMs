import os
import json
import openai
import time
import pandas as pd

class Chatgpt:
    def __init__(self, api_key):
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview" 
        openai.api_base = "https://sslgroupservice.openai.azure.com/"
        openai.api_key = api_key

    def inference(self, messages):
        output = openai.ChatCompletion.create(
            engine="gpt-4",
            messages=messages,
            temperature=0.3
            )
        return output['choices'][0]['message']

class DataGenerator:
    def __init__(self, api_key):
        self.model = Chatgpt(api_key)
    
    def generate(self, instruction='', history=False):
        messages=[
            {"role": "system", "content": "You are a translation helper that translates English paragraphs into Traditional Chinese."}
            ]
        data = []
        try:
            messages.append({"role": "user", "content": f"{instruction}"})
            output = dict(self.model.inference(messages))
            if history:
                messages.append(output)
            else:
                del messages[1]
            
            data.append(output["content"])

            print(f"------------ out start ------------ \n")
            print(data[0])
            print(f"------------ out end ------------ \n")

        except Exception as e: 
            print(e)
            data = ['']
        return data[0]



def translate(data, translator):
    
    translation = []
    for i, row in data.iterrows():
        document_eng = row['article']
        summary_eng = row['highlights']
        
        summary_tw = translator.generate(instruction = f'Translate the following English paragraph into Traditional Chinese:\n\n{summary_eng}')
        time.sleep(3)
        document_tw = translator.generate(instruction = f'Translate the following English paragraph into Traditional Chinese:\n\n{document_eng}')
        time.sleep(3)
        
        dct = {
            "document": document_tw,
            "summary": summary_tw
        }
        
        translation.append(dct)
    
        with open(f'cnn_tw.json', 'w') as f:
                json.dump(translation, f, ensure_ascii=False, indent=4)

data = pd.read_csv('cnn_long_eng.csv')
translator = DataGenerator('97f22a7a32ff4ff4902003896f247ca2')
translate(data, translator)