#pip install openai==0.27.0
#pip install google

import os
import json
import openai
from googlesearch import search
import time
import gradio as gr
from datetime import datetime



api_key = "97f22a7a32ff4ff4902003896f247ca2"


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
            temperature=0.2
            )
        return output['choices'][0]['message']

currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year



class Bot:
    def __init__(self, api_key, system_prompt):
        self.model = Chatgpt(api_key)
        self.system_prompt = system_prompt
        self.messageList = [{"role": "system", "content": self.system_prompt}]

    def rag(self, rag_input=''):
        self.messageList.append({"role": "assistant", "content": rag_input})

    def generate(self, message='', history=False):
        resp = ''
        try:
            self.messageList.append({"role": "user", "content": message})
            output = dict(self.model.inference(self.messageList))
            if history:
                self.messageList.append(output)
            else:
                del self.messageList[-1]
            resp = output["content"]

        except Exception as e: 
            print(e)
        
        return resp



def google_search_augmented_input(message):
    vanilla_bot = Bot('97f22a7a32ff4ff4902003896f247ca2',system_prompt='')
    response = vanilla_bot.generate(message)
    response_aug = response

    t_start = time.time()
    query_bot = Bot(api_key,system_prompt=prompt_query)
    
    # generate search query
    query = query_bot.generate(f"使用者訊息:\n{message}")
    if '<是>' in query:
        query = query.strip('<<是>>')
        print('******'*5,f"\n進行網頁搜尋:{query}\n",'******'*5)
        results = search(query, stop=20, lang="zh-tw", pause=0.1, country="Taiwan")

        # extract info from each url
        success = 0
        collect = ""
        for url in results:
            if success < 5:
                try:
                    # check if extractable
                    article = Article(url, language='zh')                    

                except Exception as e:
                    print(e)
            
        # summarize search results
        if success > 0:
            summ_bot = Bot(api_key,system_prompt=prompt_summ)
            input_to_summ_bot = f"{message}\n\n{'------'*10}搜尋結果{'------'*10}\n\n{collect}"
            
        else:
            print('no web content is collected')
            pass

    else:
        print('******'*5, "No Google search", '******'*5)
    t_end = time.time()
    print('------'*10,f'\naug_response:\n{response_aug}\n','------'*10)
    print('------'*10,f'\nresponse:\n{response}\n','------'*10)
    print(f'RAG time: {t_end - t_start:.2f} sec')
    return response_aug, response

#google_search_augmented_input("今年的總統大選是誰勝出?")
google_search_augmented_input("今年3月美國聖荷西有什麼有趣的科技展覽?")

#bot = Bot(api_key,system_prompt="you are a helpful assistant.")
#message = prompt_extract
#print(bot.generate(message=f"Improve the instruction (in traditional Chinese):\n{message}"))

"""
if __name__ == "__main__":
    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column():
                chat1 = gr.Textbox(label=f"with Google search information retrieval")
            with gr.Column():
                chat2 = gr.Textbox(label="direct response")

        with gr.Column():
            chat_input = gr.Textbox(placeholder="input your message here", label="Your message")
            enter_button = gr.Button(value="enter")
            enter_button.click(google_search_augmented_input, inputs=[chat_input], outputs=[chat1, chat2])
        
        demo.launch(share=True)
"""