"""
Author: Po-Kai Chen
Date: 1/7/2024
"""
import json
import random

from datasets import load_dataset
from tqdm import tqdm


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
multilingual_summarization_prompts = {
    "en": [
        "Please provide a brief {lang} summary of this article: \n\n{paragraph}",
        "Can you highlight the main points of this article in {lang} for me? \n\n{paragraph}",
        "I need an {lang} overview of the key themes in this article: \n\n{paragraph}",
        "What are the essential elements of this article? Please summarize in {lang}: \n\n{paragraph}",
        "Could you condense the main arguments of this article in {lang}? \n\n{paragraph}",
        "What does this article primarily discuss? Give me an {lang} summary: \n\n{paragraph}",
        "Please outline the major findings of this article in {lang}: \n\n{paragraph}",
        "Can you distill the core message of this article in {lang}? \n\n{paragraph}",
        "I'd like a concise {lang} explanation of this article's main idea: \n\n{paragraph}",
        "What's the gist of this article? Summarize it in {lang} for me: \n\n{paragraph}",
        "Please provide an {lang} rundown of the critical points in this article: \n\n{paragraph}",
        "Can you provide a quick {lang} synopsis of this article? \n\n{paragraph}",
        "I'm looking for an {lang} interpretation of the main points of this article: \n\n{paragraph}",
        "What are the highlights of this article? Please summarize them in {lang}: \n\n{paragraph}",
        "Can you give me an {lang} summary of the essential insights from this article? \n\n{paragraph}",
        "I need an {lang} perspective on the key arguments of this article: \n\n{paragraph}",
        "What is the primary focus of this article? Please explain in {lang}: \n\n{paragraph}",
        "Could you give me an {lang} brief on the main topics covered in this article? \n\n{paragraph}",
        "What are the significant contributions of this article? Summarize them in {lang}, please: \n\n{paragraph}",
        "I'd appreciate an {lang} summary of the overarching themes in this article: \n\n{paragraph}",
    ],
    "zh": [
        "請用{lang}對這篇文章做一個簡潔的總結：\n\n{paragraph}",
        "我需要了解這篇文章的主要觀點，能用{lang}幫忙說明嗎？\n\n{paragraph}",
        "請用{lang}幫我梳理一下這篇文章的核心內容：\n\n{paragraph}",
        "這篇文章談了哪些重點？請用{lang}為我總結：\n\n{paragraph}",
        "能否用{lang}對這篇文章的要旨做一個概述？\n\n{paragraph}",
        "我想了解這篇文章涉及的主題，請用{lang}簡述：\n\n{paragraph}",
        "這篇文章的重要資訊是什麼？請用{lang}指出：\n\n{paragraph}",
        "請用{lang}幫我提取這篇文章的精華部分：\n\n{paragraph}",
        "這篇文章主要講了些什麼？請用{lang}總結一下：\n\n{paragraph}",
        "關於這篇文章，我想要一個{lang}的精簡說明：\n\n{paragraph}",
        "請問這篇文章的主要論點是什麼？用{lang}簡述一下：\n\n{paragraph}",
        "這篇文章的重點是什麼？請用{lang}幫我概括：\n\n{paragraph}",
        "請用{lang}為我解釋這篇文章的基本觀點：\n\n{paragraph}",
        "這篇文章包含了哪些關鍵訊息？請用{lang}簡要介紹：\n\n{paragraph}",
        "這篇文章的主要貢獻是什麼？請用{lang}說明：\n\n{paragraph}",
        "請用{lang}幫我總結一下這篇文章的主要發現：\n\n{paragraph}",
        "這篇文章的精髓能用{lang}表達出來嗎？\n\n{paragraph}",
        "關於這篇文章，我需要一個{lang}的快速概覽：\n\n{paragraph}",
        "這篇文章的核心要義是什麼？請用{lang}簡單描述：\n\n{paragraph}",
        "我想獲得這篇文章的總體評述，能用{lang}提供嗎？\n\n{paragraph}"
    ]
}


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