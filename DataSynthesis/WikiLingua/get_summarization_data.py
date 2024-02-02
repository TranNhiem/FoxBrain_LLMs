"""
Author: Po-Kai Chen
Date: 1/7/2024
"""
import json
import random

from datasets import load_dataset
from tqdm import tqdm

from prompts import get_prompt


random.seed(2024)

with open("wikilingua_data_zhTW.json", "r") as r:
    datas = json.load(r)
    
datas = datas["train"]

# Reserve the part of datas, which have language pair data
pair_datas = []
nonpair_datas = []
for data in datas:
    if data["source_language"] != data["target_language"]:
        pair_datas.append(data)
    else:
        nonpair_datas.append(data)


# Collect the bidirection pair data
print(len(pair_datas))
align_datas = []
for i in range(0, len(pair_datas), 2):
    align_datas.append({
        "paragraph": {
            pair_datas[i]["source_language"]: pair_datas[i]["source"],
            pair_datas[i]["target_language"]: pair_datas[i+1]["source"],
        },
        "summary": {
            pair_datas[i]["source_language"]: pair_datas[i+1]["target"],
            pair_datas[i]["target_language"]: pair_datas[i]["target"],       
        }
    })

tmp = []
for nonpair_data in nonpair_datas:
    tmp.append({
        "lang": nonpair_data["source_language"],
        "title": nonpair_data["references"][0],
        "paragraph": nonpair_data["source"],
        "summary": nonpair_data["target"],
    })
nonpair_datas = tmp


# Define the prompt for multilingual summarization task
multilingual_summarization_prompts = get_prompt("summarization")


def remove_lang_require(selected_prompt, prompt_lang):
    if prompt_lang == "en":
        selected_prompt = selected_prompt.replace(" in {lang}", "").replace(" {lang}", "")
    else:
        selected_prompt = selected_prompt.replace("用{lang}", "").replace("{lang}", "")
    return selected_prompt 


# Generating data
datas = []
for pair in tqdm(align_datas):
    src_lang = random.choice(["en", "zh"])
    prompt_lang = random.choice(["en", "zh", "zh"]) # let zh 2x weight
    tgt_lang = random.choice(["en", "zh"])
    
    src = pair["paragraph"][src_lang]
    tgt = pair["summary"][tgt_lang]
        
    prompts = multilingual_summarization_prompts[prompt_lang]
    selected_prompt = random.choice(prompts)
    without_lang = False
    if (src_lang == tgt_lang == prompt_lang) or src_lang == tgt_lang:
        # remove lang require
        selected_prompt = remove_lang_require(selected_prompt, prompt_lang)
        without_lang = True

    if without_lang:
        input = selected_prompt.format(paragraph=src)
    else:
        if prompt_lang == "en":
            lang = "Chinese" if tgt_lang == "zh" else "English"
        else:
            lang = "中文" if tgt_lang == "zh" else "英文"
        input = selected_prompt.format(lang=lang, paragraph=src)
    
    output = tgt
    
    datas.append({
        "input": input,
        "output": output
    })


for pair in tqdm(nonpair_datas):
    lang = pair["lang"]
    src = pair["paragraph"]
    tgt = pair["summary"]

    prompt_lang = random.choice(["en", "zh"])
    prompts = multilingual_summarization_prompts[prompt_lang]
    selected_prompt = random.choice(prompts)
    selected_prompt = remove_lang_require(selected_prompt, prompt_lang)
    input = selected_prompt.format(paragraph=src)
    output = tgt
    
    datas.append({
        "input": input,
        "output": output
    })

# Save data
with open("wikilingua_summarization_task_data.json", "w") as w:
    json.dump(datas, w, indent=4, ensure_ascii=False)