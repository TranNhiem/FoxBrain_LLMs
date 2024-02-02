'''
@Po-Kai 2023/12
This code is designed for translating dataset documents. It utilizes the litellm proxy to invoke GPT for translation.
'''
from functools import partial
from glob import glob
from multiprocessing.pool import ThreadPool
import json
import os
from typing import List

import fire
from tqdm import tqdm

from utils.data_utils import merge_jsonl_to_json
from utils.openai.translator import OpenAITranslate

translator = None
tmp_path = "data/tmp"
    

def translation_worker(data, to_translated_keys):
    idx, data = data
    save_path = os.path.join(tmp_path, f"{idx}.jsonl")
    
    if os.path.exists(save_path):
        return
        
    for key in to_translated_keys:
        sents = data[key]
        sents = [sent for sent in sents if len(sent) > 0]
        
        translated_texts = []
        for sent_idx, sent in enumerate(sents):
            translated_text = translator.translate(sent)
            if translated_text in ["|LOST|", "|ERR|"]:
                print(f"Got some error on index-sent_idx: {idx}-{sent_idx}!")
                continue
            
            translated_texts.append(translated_text)
            
        translated_text = " ".join(translated_texts)
        
        data[key] = translated_text

    with open(save_path, "w") as writer:
        json.dump(data, writer, ensure_ascii=False)


def run(
    data_path: str = "data/LongAlpaca-12k_chunked.json",
    save_path: str = "data/LongAlpaca-12k_translated.json",
    target_language: str = "Traditional_Chinese",
    proxy_config_path: str = "../utils/openai/litellm.router.json",
    to_translated_keys: List[str] = ["output"],
    num_workers: int = 8,
    testing: bool = False,
):
    # load the dataset
    with open(data_path, "r") as r:
        datas = json.load(r)
        if testing:
            print("[tesing mode]")
            datas = datas[:3]

    # create openai translator
    global translator
    translator = OpenAITranslate(
        direction=f"English->{target_language}",
        proxy_config_path=proxy_config_path
    )

    # start work
    ## create the tmp folder
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    ## translate
    with ThreadPool(num_workers) as pool:
        task = partial(
            translation_worker,
            to_translated_keys=to_translated_keys
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