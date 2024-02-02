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
for data in datas:
    if data["source_language"] != data["target_language"]:
        pair_datas.append(data)


# Collect the bidirection pair data
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


# Define the prompt for translation task
translation_prompts = get_prompt("translation")


# Generating data
datas = []
for pair in tqdm(align_datas):
    src_lang = random.choice(["en", "zh", "zh"])
    
    article = pair["paragraph"][src_lang]
    output = pair["paragraph"]["en" if src_lang == "zh" else "zh"]
    tgt_lang = "英文" if src_lang == "zh" else random.choice(["中文", "繁體中文"])
    selected_prompt = random.choice(translation_prompts)
    input = selected_prompt.format(lang=tgt_lang, article=article)
    
    datas.append({
        "input": input,
        "output": output
    })

# Save data
with open("wikilingua_translation_task_data.json", "w") as w:
    json.dump(datas, w, indent=4, ensure_ascii=False)