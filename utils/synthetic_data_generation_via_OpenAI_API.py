'''
TranNhiem 2023/12/14
Synthetic Data Generation from the Given Raw Text Data as Reference to train Model 

** Initial Idea **
Raw Text  --> LLaMA2 (Set of Define Tasks) ---> Generate the Input & The Output corresponding to the Input 

'''

import ctranslate2
import sentencepiece as spm

import pickle
import os
import json

model_dir="/data1/LLM_Checkpoints/llam2_70b_4k_general_domain"

print("Loading the model...")
generator = ctranslate2.Generator(model_dir, device="cuda", compute_type="int8")#,2,3,4
sp = spm.SentencePieceProcessor(os.path.join(model_dir, "tokenizer.model"))

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


context_length = 4096
max_generation_length = 2048
max_prompt_length = context_length - max_generation_length

# Open and read the JSON file
with open('/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/Wikilingual_summary/Wiki_lingual_summary_english_vi_pair.json', 'r') as file:
    data = json.load(file)
def save_data(output_data, filename):
    """Function to save data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)
# Access data from the JSON file

ouput_data=[]
for idx, row in enumerate(data): 
    if idx <=400: 
        english_data=row["english:"]
        english_sec=row["Engl_sect"]
        viet_sec=row["Vi_sect"]
        output_section=[]
        for title in english_sec: 
            content=english_data[str(title)]
            raw_text=content['document']
            print(len(raw_text))
            
            if len(raw_text) <= 10000: 
                print(raw_text)

                model_response=''

                DEFAULT_SYSTEM_PROMPT = """\nYou are a helpful assistant, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
                    answers should not include any harmful, unethical, racist, toxic, dangerous, or illegal content 
                    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
                    correct. If you don't know the answer to a question, please don't share false information."""


                DEFAULT_PROMPT = f"""\n from the given following article excerpt cover wide range of Topics, you will generate a variety of NLP tasks for training. The tasks should cover the following categories:

                    1. Natural Language Inference (NLI): Create a task that presents a premise from the article and asks whether it entails a given hypothesis. Format it as "input:" {model_response},"hypothesis:", "output:" {model_response}.
                    
                    2. Commonsense Reasoning: Create a task that asks for the reason behind a specific statement in the article, requiring the application of commonsense knowledge. Format it as "input:" {model_response}, "output:" {model_response}.
                    
                    3. Paraphrase Detection: Create a task that asks to identify whether two statements from or about the article are paraphrases of each other. Format it as "phrase 1:" {model_response},"phrase 2:", "output:" {model_response}.
                    
                    4. Text Completion: Create a task that asks how one would complete the article based on the information provided. Format it as "input: text description missing section_______", "output:" {model_response}.
                    
                    \n\n Here is the Article Excerpt: {raw_text}."""
                
                DEFAULT_PROMPT_1 = f"""\n from the given following article excerpt cover wide range of Topics, you will generate a variety of NLP tasks for training. The tasks should cover the following categories:

                    1. Summarization: Based on the context and main topic of the article, provide a concise summary. If the article is lengthy, aim for a more detailed summary, capturing the main arguments, findings, or conclusions. If the article is brief, a shorter summary will suffice. Format the task as "Question: Summarize the following article.", "Answer: {{model_response}}".

                    2. Open and Closed QA: Create both open-ended questions and questions with specific answers from the article. Format them as "Question: {{model_response}}", "Answer: {{model_response}}".

                    3. Reading Comprehension: Create a task that presents a portion of the article and asks questions to assess comprehension. Format it as "Question: {{model_response}}, "Answer: {{model_response}}".
                    \n\n Here is the Article Excerpt: {raw_text}"""
               
                ## Old Format 
                input_prompt = f"[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n "
                input_integrate= input_prompt + DEFAULT_PROMPT + "[/INST]"

                prompt_tokens = sp.encode(input_integrate, out_type=str)
                step_results = generator.generate_tokens(
                prompt_tokens,
                #static_prompt=system_prompt_tokens,
                max_length=2048,
                sampling_temperature=0.1,
                sampling_topk=30,
                sampling_topp=1,
                )
                text_output = ""
                for word in generate_words(sp, step_results):
                    if text_output:
                        word = " " + word
                    #print(word, end="", flush=True)
                    text_output += word

                data_output={ "raw_text":raw_text, str(title): text_output }
                output_section.append(data_output)
            else: 
                data_output={ "raw_text":raw_text, str(title) : "content_too_long" }
                output_section.append(data_output)

        data_save_format={"viet_sect":viet_sec, "Engl_sect": english_sec, "Generated_instruc":output_section }
        
        ouput_data.append(data_save_format)
        # Save data at regular intervals
        if idx % 10 == 0:
            intermediate_save_file = f'/data1/LLM_Checkpoints/wikihow_lingual/intermediate_result/intermediate_save_{idx}.json'
            save_data(ouput_data, intermediate_save_file)
            print(f"Data saved at index {idx}")

with open('/data1/LLM_Checkpoints/wikihow_lingual/generated_wiki_output_0_400.json', 'w', encoding='utf-8') as file:
    json.dump(ouput_data, file, ensure_ascii=False, indent=4)

