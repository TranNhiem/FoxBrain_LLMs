
'''
TranNhiem@ 2023/12/05
This is demo of FoxBrain Beta Version using Ctranslate2 for optimal Inference and Webapp via Gradio

## Installation Requirement 
pip install ctranslate2==3.22.0
pip install gradio==3.45.0
pip install sentencepiece==0.1.99
pip install openai==0.27.0
pip install googlesearch-python
pip install google
pip install newspaper3k
'''

import os
import sys
import json 
import re

import ctranslate2
import sentencepiece as spm
import gradio as gr

import openai
from googlesearch import search
from datetime import datetime
from newspaper import Article
import nltk
nltk.download('punkt')


model_dir="/data/rick/pretrained_weights/ctranslate/FoxBrain_Beta_SFT"
print("Loading the model...")
generator = ctranslate2.Generator(model_dir, device="cuda") # device_index=[0, 1, 2, 3]
sp = spm.SentencePieceProcessor(os.path.join(model_dir, "tokenizer.model"))

## Default System Prompt NLP General Tasks
DEFAULT_SYSTEM_PROMPT = """\n You are FoxBrain developed by Tran Nhiem (Rick) and (Project Lead by Professor Li, Yung-Hui) at Foxconn.  As a helpful assistant, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
        answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content and response the language as the same as User Input Langauge. Please ensure\
        that your responses are socially unbiased and positive in nature and response the same language as Human input.\
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
        correct. If you don't know the answer to a question, please don't share false information. Important Note, You are designed, developed, and created by Foxconn HHRIAI, Not by OpenAI. """

###*************************************************
### Section 0 Google search RAG
###*************************************************

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
    augmented_message = message
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
                    article.download()

                    bot = Bot(api_key,system_prompt=prompt_extract)
                    
                    # else:
                    #     
                    success += 1
                    collect += f"Google搜尋結果:\n {extract_info.strip('<<是>>')}\n\n"

                except Exception as e:
                    print(e)
            
        # summarize search results
        if success > 0:
            summ_bot = Bot(api_key,system_prompt=prompt_summ)
            input_to_summ_bot = f"{message}\n\n{'------'*10}搜尋結果{'------'*10}\n\n{collect}"
            print('*********'*5,f"\ninput_to_summ_bot:\n{input_to_summ_bot}\n",'*********'*5)
            summary = summ_bot.generate(message=input_to_summ_bot)
            if "<否>" in summary:
                pass
            else:
                augmented_message = f"{message}\n\nBelow text is the real time search information, please reply user message with this information:\n\'{summary.strip('<<是>>')}\'"
        else:
            print('no web content is collected')
            pass

    else:
        print('******'*5, "No Google search", '******'*5)
        
    print('------'*10,f'\naug_response:\n{augmented_message}\n','------'*10)
    return augmented_message



###*************************************************
### Section 1 Helper Functions
###*************************************************

## 1. Function to log user interactions
def log_interaction(user_id, user_input, bot_response):
    log_entry = {
        "user_id": user_id,
        "input": user_input,
        "response": bot_response
    }
    # Assuming you're logging to a JSON file
    log_file = 'user_interactions_log.json'

    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        # File exists and is non-empty
        with open(log_file, 'r+') as file:
            data = json.load(file)
            data.append(log_entry)
            file.seek(0)
            json.dump(data, file, indent=4)
    else:
        # File doesn't exist or is empty
        with open(log_file, 'w') as file:
            json.dump([log_entry], file, indent=4)

## 2. Helper to read the HTML 
def read_content(file_path) :
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

## 3. Function to Run the Inference Model 
def predict(session_id, message, chatbot, temperature, top_k,top_p, max_output_tokens, repetition_penalty):
    aug_message = google_search_augmented_input(message)
    system_prompt= f"<s>[INST] <<SYS>> \n{DEFAULT_SYSTEM_PROMPT}\n You can maintain the relevance and Engage conversation by reviewing the chat history between you (Foxbrain) and the User.  Please discard or omit the history content information seems off-topic, redundant, or doesn't contribute meaningfully to the context.\n <</SYS>>\n\n"
    print("session_id", session_id)
    if session_id =="": 
        session_id = datetime.now().strftime("%Y%m%d%H")

    ##  Check the History Conversation
    if len(chatbot) >= 15:
        chatbot = chatbot[-10:]

    ## Calculate the Total Words to Reduce input tokens
    total_words = sum(len(sentence.split()) for pair in chatbot for sentence in pair)
    # If total_words exceeds 3000, truncate the earlier parts of the conversation
    if total_words > 1000:
        remaining_words = 0
        truncated_conversation = []

        for pair in chatbot:
            if remaining_words + len(' '.join(pair)) <= 500:
                truncated_conversation.append(pair)
                remaining_words += len(' '.join(pair).split())
            else:
                break
        # Update the conversation with the truncated version
        chatbot = truncated_conversation

    if len(chatbot)<1: 
        input_prompt_=""
        input_prompt =system_prompt+  input_prompt_ + str(aug_message) + " [/INST] " 
    else: 
        input_prompt_=""
        for interaction in chatbot:
            print("This is chatbot length", len(chatbot))

            # Replace patterns and strip tags
            output_string_1 = re.sub(r'👤:', 'User:', interaction[0])#Human:
            # print(output_string_1)
            output_string_1=re.sub(r'[/INST]', '', output_string_1)
            output_string_1=re.sub(r'</s><s>', '', output_string_1)
            # print(output_string_1)
            output_string_2 = re.sub(r'FoxBrain 😃:', 'FoxBrain:', interaction[1])
            output_string_2=re.sub(r'[/INST]', '', output_string_2)
            output_string_2=re.sub(r'</s><s>', '', output_string_2)
           # Append the current interaction to the accumulated interactions
            input_prompt_ += str(output_string_1) + "\n\n" + str(output_string_2) + "\n\n"
        print("chat History", input_prompt_)
        input_prompt =system_prompt +"\n\n Here is the history conversation between you (FoxBrain) and User:\n\n" + input_prompt_ + "</s><s> [INST] "+ str(aug_message) +  " [/INST] " 
        print('***'*10,input_prompt, '***'*10)
    ## Tokenize the Model via Sentencepiece
    prompt_tokens = sp.encode(input_prompt, out_type=str)

    #system_prompt_tokens=sp.encode(input_prompt_, out_type=str)
    response = generator.generate_tokens(
        prompt_tokens,
        # static_prompt=system_prompt_tokens,
        max_length=max_output_tokens,
        sampling_temperature=temperature,
        sampling_topk=top_k,
        sampling_topp=top_p,
        repetition_penalty=repetition_penalty,

    )

    def generate_words(sp, step_results):   
        tokens_buffer = []

        for step_result in step_results:
            is_new_word = step_result.token.startswith("▁")

            if is_new_word and tokens_buffer:
                word = sp.decode(tokens_buffer)
                if word:
                    yield word
                tokens_buffer = []

            tokens_buffer.append(step_result.token_id)

        if tokens_buffer:
            word = sp.decode(tokens_buffer)
            if word:
                yield word


    text_output = ""
    for word in generate_words(sp, response):
        # if text_output:
        word = " " + word
        #print(word, end="", flush=True)
        text_output += word
    
    # Adding Fun Icon
    message = "👤: " + message

    text_output= re.sub(r'FoxBrain:', '', text_output)
    text_output= re.sub(r'FoxBrain :', '', text_output)
    text_output= re.sub(r'User:', '', text_output)
    text_output= re.sub(r'User :', '', text_output)

    # Define the pattern to search for
    pattern = "developed by OpenAI"
    pattern_2 = "我是 OpenAI"
    # Check if the pattern exists in the text
    if re.search(pattern, text_output):
        # Replace 'OpenAI' with 'Foxconn AI'
        text_output = re.sub(r"OpenAI", "Foxconn AI", text_output)
    elif re.search(pattern_2, text_output):
        # Replace 'OpenAI' with 'Foxconn AI'
        text_output = re.sub(r"OpenAI", "Foxconn AI", text_output)
    else:
        # If the pattern is not found, keep the text as it is
        text_output = text_output
    
    bot_message = "FoxBrain 😃: " + text_output#.strip()
    # Log interaction with the user_id
    log_interaction(session_id , message, bot_message)
    print("----------------------------Bot Message Append --------------------")
    print(bot_message)
    print("---------------------------- End Bot Message --------------------")
    chatbot.append((message, bot_message ))

    return "",chatbot


###*************************************************
### Section 1 Gradio App Interface
###*************************************************

css = """.toast-wrap { display: none !important } """
title = "FoxBrain Assistant"

with gr.Blocks() as demo:
    #gr.Markdown("""<h1><center> SIF-LLM Assistant (Alpha Released)  </center></h1>""")
    gr.HTML(read_content("./html_header.html"))

    chatbot = gr.Chatbot(label="FoxBrain Assistant.").style(height=300)
    
    ## Help to Create the Unique user ID
    """gr.HTML(
        <script>
        function generateUUID() {
            // Generate a simple UUID
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        // Store UUID in local storage or generate a new one
        if (!localStorage.getItem("sessionUUID")) {
            localStorage.setItem("sessionUUID", generateUUID());
        }

        // Function to set the UUID to the hidden session ID input
        function setSessionID() {
            var sessionInput = document.getElementById('session-id-input');
            if (sessionInput) {
                sessionInput.value = localStorage.getItem("sessionUUID");
            } else {
                // If the element is not found, try again after a short delay
                setTimeout(setSessionID, 100);
            }
        }

        // Set the UUID on window load
        window.onload = setSessionID;
        </script>
    )
    """

    ## Define the hidden text input for the session ID
    session_id = gr.Textbox( label="Session ID", visible=False, elem_id="session-id-input")

    with gr.Row():
        message = gr.Textbox(show_label=False, placeholder="Enter your prompt and press enter", visible=True)
    state = gr.State()
    
    
    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    #It is a value between 1.0 and infinity, where 1.0 means no penalty
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.2,
                        step=0.1,
                        interactive=True,
                        label="repetition_penalty",
                        info="Penalize repetition — 1.0 to disable.",
                    )
            with gr.Column():
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                        info="Higher values produce more diverse outputs",
                    )
            with gr.Column():
                with gr.Row():
                    top_p = gr.Slider(
                                        label="Top-p (More Balance)",
                                        value=0.98,
                                        minimum=0.0,
                                        maximum=1,
                                        step=0.01,
                                        interactive=True,
                                        info=(
                                            "Sample from the smallest possible set of tokens whose cumulative probability "
                                            "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                        ),
                        )
            
            # with gr.Column():
            #     with gr.Row():
            #         ## two values penalty_alpha & top_k use to set Contrastive Decoding for LLM 
            #         penalty_alpha = gr.Slider(
            #             minimum=0.001,
            #             maximum=1.0,
            #             value=0.4,# Values 0.0 mean equal to gready_search
            #             step=0.05,
            #             interactive=True,
            #             label="penalty_alpha (More Creative)",
            #             info="balances the model confidence and the degeneration penalty",
            #         )
            with gr.Column():
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=0.0,
                        maximum=100,
                        value=20,## Top number of candidates 
                        step=2,
                        interactive=True,
                        label="Top_k",
                        info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                    )
            with gr.Column():
                with gr.Row():      
                    max_output_tokens = gr.Slider(
                        minimum=1024,
                        maximum=3048,
                        value=2048,
                        step=100,
                        interactive=True,
                        label="Max output tokens",
                        info=" sets the limit on the number of words in the Language Model's response."
                    )

    message.submit(predict, inputs=[session_id, message, chatbot, temperature, top_k, top_p, max_output_tokens,repetition_penalty ], outputs=[message, chatbot], queue=True) #repetition_penalty,temperature, top_p, penalty_alpha, top_k, max_output_tokens,base_model_input, conversation_style


    gr.HTML("""
                <div class="footer">
                    <p style="align-items: center; margin-bottom: 7px;" >
                    </p>
                    <div style="text-align: Center; font-size: 1.5em; font-weight: bold; margin-bottom: 0.5em;">
                        <div style="
                            display: inline-flex; 
                            gap: 0.6rem; 
                            font-size: 1.0rem;
                            justify-content: center;
                            margin-bottom: 10px;
                            ">
                        <p style="align-items: center; margin-bottom: 7px;" >
                            Hallooo (it's me!! (Tran Nhiem) Rick) i'm in charge of this development: I'm actively working to improve its accuracy, performance, & reliability. Your valuable feedback and suggestions are very valuable & will help TO enhance the Future Development.
                            <a href="https://docs.google.com/spreadsheets/d/1ToK4CBd9-AU0bVKV6c0A5lNaV4FUinsosJxETn1zo3g/edit?usp=sharing" style="text-decoration: underline;" target="_blank"> 🙌  Share your feedback here </a> ; 
                            <a href="https://docs.google.com/presentation/d/1JCiF13WNijo58f9xFXFVQz_MN7BgjhDtGPmOuvrwbhw/edit?usp=sharing" style="text-decoration: underline;" target="_blank"> 🙌 or upload screenshot images here</a> 
                        </p>
                       
                        </div>
                """
                )


demo.queue(max_size=128, concurrency_count=20)
demo.launch(debug=True, server_port=12345, share=True)