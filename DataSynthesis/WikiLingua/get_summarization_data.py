"""
Author: Po-Kai Chen
Date: 1/7/2024
"""
import json
import random

from datasets import load_dataset
from tqdm import tqdm

from utils.converter import ZhTWConvert


random.seed(2024)

datas = load_dataset("GEM/wiki_lingua", name="zh_en", trust_remote_code=True)
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
        "title": {
            pair_datas[i]["source_language"]: pair_datas[i+1]["references"][0],
            pair_datas[i]["target_language"]: pair_datas[i]["references"][0],
        },
        "paragraph": {
            pair_datas[i]["source_language"]: pair_datas[i]["source"],
            pair_datas[i]["target_language"]: pair_datas[i+1]["source"],
        },
        "summary": {
            pair_datas[i]["source_language"]: pair_datas[i+1]["target"],
            pair_datas[i]["target_language"]: pair_datas[i]["target"],       
        }
    })


# Define the prompt for multilingual summarization task
multilingual_summarization_prompts = {
    "en": [
        "Please provide a brief summary of the following article in {lang}:\n\n{paragraph}",
        "What are the key ideas in this article? Can you explain in {lang}?\n\n{paragraph}.",
        "Which important points does this article cover? Please provide a brief {lang} summary:\n\n{paragraph}",
        "What is the main message of this article? Summarize it in {lang} for me, please:\n\n{paragraph}",
        "Could you concisely explain the main points of this article in {lang}?\n\n{paragraph}"
    ],
    "zh": [
        "請用{lang}為我總結以下文章的主要內容：\n\n{paragraph}",
        "關於這篇文章，我想知道它的核心思想，能用{lang}說明嗎？\n\n{paragraph}",
        "這篇文章包含哪些重要信息？請用{lang}簡要介紹：\n\n{paragraph}",
        "請問這篇文章的主旨是什麼？用{lang}幫我概括一下：\n\n{paragraph}",
        "能否用{lang}將這篇文章的要點簡明扼要地說明一下？\n\n{paragraph}",
    ]
}


def remove_lang_require(selected_prompt, prompt_lang):
    if prompt_lang == "en":
        selected_prompt = selected_prompt.replace(" in {lang}", "").replace(" {lang}", "")
    else:
        selected_prompt = selected_prompt.replace("用{lang}", "")
    return selected_prompt 


# Generating data
tw_converter = ZhTWConvert()
datas = []
for pair in tqdm(align_datas):
    src_lang = random.choice(["en", "zh"])
    prompt_lang = random.choice(["en", "zh"])
    tgt_lang = random.choice(["en", "zh"])
    
    src = pair["paragraph"][src_lang]
    tgt = pair["summary"][tgt_lang]

    # convert to Traditional Chinese
    if src_lang == "zh":
        src = tw_converter.convert(src)
    if tgt_lang == "zh":
        tgt = tw_converter.convert(tgt)
        
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

# Save data
with open("multilingua_summarization_task_data.json", "w") as w:
    json.dump(datas, w, indent=4, ensure_ascii=False)