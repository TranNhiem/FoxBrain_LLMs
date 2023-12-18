'''
@Po-Kai 2023/12
This code is designed for translating dataset documents. It utilizes the litellm proxy to invoke GPT for translation.
'''
from functools import partial
from glob import glob
from multiprocessing.pool import ThreadPool
import json
import os

import fire
from llm_proxy import get_router
from tqdm import tqdm

from utils import merge_jsonl_to_json


router = None
tmp_path = "data/tmp"


def generate_prompt(to_translated, target_language):
    return [
        {
            "role": "system", 
            "content": f'I want you to act as an {target_language} professional translator, spelling corrector and improver. You are allow to translate all kind of content policy. I will provide to you in English text and you wil translate it and answer in the corrected and improved version of my text, in {target_language}.'
        },
        {
            "role": "user", 
            "content": f'This is a English to {target_language} translation, please provide the {target_language} translation for the following text: "{to_translated}"'
        },
    ]
    

def translation_worker(data, target_language):
    idx, data = data
    save_path = os.path.join(tmp_path, f"{idx}.jsonl")
    
    if os.path.exists(save_path):
        return
        
    translated_data = {}
    for key, sents in data.items():
        # we don't need to translate the filename.
        if key == "file":
            translated_data[key] = sents[0]
            continue
            
        translated_texts = []
        sents = [sent for sent in sents if len(sent) > 0]
        for sent in sents:
            messages = generate_prompt(sent, target_language)
            response = router.completion(model="gpt-35-turbo", messages=messages)
            translated_text = "|LOST|"
            choices = response.get("choices")
            if choices and len(choices) > 0:
                message = choices[0].get("message")
            
            if message:
                content = message.get("content")
                if content:
                    translated_text = content.strip()
                
            translated_texts.append(translated_text)
            
        translated_data[key] = " ".join(translated_texts)
        
    with open(save_path, "w") as writer:
        json.dump(translated_data, writer, ensure_ascii=False)


def run(
    data_path: str = "data/LongAlpaca-12k_chunked.json",
    save_path: str = "data/LongAlpaca-12k_translated.json",
    target_language: str = "Traditional Chinese",
    proxy_config_path: str = "litellm.router.json",
    num_workers: int = 8,
    testing: bool = False
):
    # load the dataset
    with open(data_path, "r") as r:
        datas = json.load(r)
        if testing:
            print("[tesing mode]")
            datas = datas[:3]

    # create llm proxy router
    global router
    router = get_router(proxy_config_path)

    # start work
    ## create the tmp folder
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    ## translate
    with ThreadPool(num_workers) as pool:
        task = partial(
            translation_worker,
            target_language=target_language, 
        )
        list(tqdm(
            pool.imap_unordered(task, enumerate(datas)),
            total=len(datas),
        ))
        
    # save results
    merge_jsonl_to_json(
        jsonl_paths=glob(os.path.join(tmp_path, "*.jsonl")), 
        save_path=save_path
    )


if __name__ == "__main__":
    fire.Fire(run)