"""
Author: Po-Kai Chen
Date: 1/10/2024
"""
from glob import glob
import json
import re


paths = glob("tmp/*.txt")

print("Num of files:", len(paths))


maybe_task_names = []
datas = {}
for path in paths:
    with open(path, "r") as r:
        data = r.read()
    maybe_task_name = [line.strip() for line in re.findall(r"(?:\n\n|^).+：\n", data)]
    idx = path.split("/")[-1][:-len(".txt")]
    datas[idx] = re.split("\n+", data)
    maybe_task_names.extend(maybe_task_name)


guess_keys = {
    "NLI": r"自然語言推理.*[:：]",
    "Open_QA": r"開放[型式]問[答題].*[:：]",
    "Close_QA": r"(\s+|^)封閉[型式]問[答題].*[:：]",
    "QA": r"開放和封閉[型式]問[答題].*[:：]",
    "Reading_omprehension": r"閱讀理解.*[:：]",
    "Commonsense_reasoning": r"常識推理.*[:：]",
    "Text_completion": r"文章填充.*[:：]",
    "Paraphrase_detection": r"[釋释][義义][檢检][測测].*[:：]",
}


# 找尋可能沒被納入規則(guess_keys)的東西。
for line_idx, maybe_task in enumerate(set(maybe_task_names)):
    find = False
    for guess_key_pattern in guess_keys.values():
        if re.findall(guess_key_pattern, maybe_task):
            find = True
            break
            
    if not find:        
        print(f"{line_idx}\t{maybe_task}")



# get multilingual articles
with open("data/wikilingua_pair_data.json", "r") as r:
    contents = json.load(r)

task_names = list(guess_keys.keys())
task_name_patterns = list(guess_keys.values())
results = {task_name: [] for task_name in task_names if task_name not in ["Close_QA", "Open_QA"]}
for _id, sents in datas.items():

    # find region per task
    all_task = []
    for sent_idx, sent in enumerate(sents):
        for task_name, pattern in zip(task_names, task_name_patterns):
            if re.findall(pattern, sent):
                all_task.append((sent_idx, task_name))
                break

    # split the region for task
    tmp = {}
    for idx in range(len(all_task)):
        if idx < len(all_task)-1:
            content = sents[all_task[idx][0]+1:all_task[idx+1][0]]
        else:
            content = sents[all_task[idx][0]+1:]
        tmp[all_task[idx][1]] = content # "\n".join(content)

    if "NLI" in tmp and len(tmp["NLI"]) % 3 != 0:
        print(".", end="")
    elif "NLI" in tmp:
        for idx in range(0, len(tmp["NLI"]), 3):
            assert re.findall(r"(\d\.\s)?輸入(：|:)", tmp["NLI"][idx]), tmp["NLI"]
            assert re.findall(r"假設(：|:)", tmp["NLI"][idx+1]), tmp["NLI"]
            assert re.findall(r"輸出(：|:)", tmp["NLI"][idx+2]), tmp["NLI"]
            results["NLI"].append(
                {
                    "input": re.sub("(\d\.\s)?輸入(：|:)", "", tmp["NLI"][idx].strip()),
                    "hypothesis": re.sub("假設(：|:)", "", tmp["NLI"][idx+1].strip()),
                    "output": re.sub("輸出(：|:)", "", tmp["NLI"][idx+2].strip()),
                    "article": contents[_id]
                }
            )

    if "Commonsense_reasoning" in tmp and len(tmp["Commonsense_reasoning"]) % 2 != 0 and len(tmp["Commonsense_reasoning"]) > 1:
        print("<", end="")
    elif "Commonsense_reasoning" in tmp:
        if re.match(r"範例\d(:|：)", tmp["Commonsense_reasoning"][0]):
            tmp["Commonsense_reasoning"] = [i for i in tmp["Commonsense_reasoning"] if not re.match("範例\d(:|：)", i)]
        for idx in range(0, len(tmp["Commonsense_reasoning"]), 2):
            try:
                assert re.findall(r"(\d\.\s)?輸入(：|:)", tmp["Commonsense_reasoning"][idx]), tmp["Commonsense_reasoning"]
                assert re.findall(r"輸出(：|:)", tmp["Commonsense_reasoning"][idx+1]), tmp["Commonsense_reasoning"]
            except:
                print("<", end="")
            else:
                results["Commonsense_reasoning"].append(
                    {
                        "input": re.sub("(\d\.\s)?輸入(：|:)", "", tmp["Commonsense_reasoning"][idx].strip()),
                        "output": re.sub("輸出(：|:)", "", tmp["Commonsense_reasoning"][idx+1].strip()),
                        "article": contents[_id]
                    }
                )
                 
    if "Reading_omprehension" in tmp and len(tmp["Reading_omprehension"]) % 2 != 0:     
        print(">", end="")
    elif "Reading_omprehension" in tmp:
        for idx in range(0, len(tmp["Reading_omprehension"]), 2):
            try:
                assert re.findall(r"(\d\.\s)?(問題|範例)(\d)?(：|:)", tmp["Reading_omprehension"][idx]), tmp["Reading_omprehension"]
                assert re.findall(r"答案(\d)?(：|:)", tmp["Reading_omprehension"][idx+1]), tmp["Reading_omprehension"]
            except:
                print(">", end="")
            else:
                results["Reading_omprehension"].append(
                    {
                        "input": re.sub(r"(\d\.\s)?(問題|範例)(\d)?(：|:)", "", tmp["Reading_omprehension"][idx].strip()),
                        "output": re.sub(r"答案(\d)?(：|:)", "", tmp["Reading_omprehension"][idx+1].strip()),
                        "article": contents[_id]
                    }
                )

    if "Text_completion" in tmp and len(tmp["Text_completion"]) % 2 != 0 and len(tmp["Text_completion"]) > 1:
        print(":", end="")
    elif "Text_completion" in tmp:
        for idx in range(0, len(tmp["Text_completion"]), 2):
            try:
                assert re.findall(r"(\d\.\s)?輸入(：|:)", tmp["Text_completion"][idx]), tmp["Text_completion"]
                assert re.findall(r"輸出(：|:)", tmp["Text_completion"][idx+1]), tmp["Text_completion"]
            except:
                print(":", end="")
            else:
                _input = re.sub("(\d\.\s)?輸入(：|:)", "", tmp["Text_completion"][idx].strip())
                if _input != "缺少部分的文本描述_______":
                    results["Text_completion"].append(
                        {
                            "input": _input,
                            "output": re.sub("輸出(：|:)", "", tmp["Text_completion"][idx+1].strip()),
                            "article": contents[_id]
                        }
                    )

    if "Paraphrase_detection" in tmp and len(tmp["Paraphrase_detection"]) % 3 != 0:
        print("]", end="")
    elif "Paraphrase_detection" in tmp:
        for idx in range(0, len(tmp["Paraphrase_detection"]), 3):
            assert re.findall(r"(\d\.\s)?片(語|段)\d(:|：)(\d\.\s)?", tmp["Paraphrase_detection"][idx]), tmp["Paraphrase_detection"]
            assert re.findall(r"片(語|段)\d(:|：)", tmp["Paraphrase_detection"][idx+1]), tmp["Paraphrase_detection"]
            assert re.findall(r"輸出(：|:)", tmp["Paraphrase_detection"][idx+2]), tmp["Paraphrase_detection"]
            results["Paraphrase_detection"].append(
                {
                    "input": re.sub("(\d\.\s)?片(語|段)\d(:|：)(\d\.\s)?", "", tmp["Paraphrase_detection"][idx].strip()),
                    "input2": re.sub("片(語|段)\d(:|：)", "", tmp["Paraphrase_detection"][idx+1].strip()),
                    "output": re.sub("輸出(：|:)", "", tmp["Paraphrase_detection"][idx+2].strip()),
                    "article": contents[_id]
                }
            )

    for qa_key in ["Open_QA", "Close_QA", "QA"]:
        if qa_key in tmp and len(tmp[qa_key]) % 2 == 0:
            for idx in range(0, len(tmp[qa_key]), 2):
                try:
                    assert re.findall(r"(\d\.\s)?問題\d?(:|：)(\d\.\s)?", tmp[qa_key][idx]), tmp[qa_key][idx]
                    assert re.findall(r"答案\d?(:|：)", tmp[qa_key][idx+1]), tmp[qa_key]
                except:
                    print("[", end="")
                else:
                    results["QA"].append(
                        {
                            "input": re.sub(r"(\d\.\s)?問題\d?(:|：)(\d\.\s)?", "", tmp[qa_key][idx].strip()),
                            "output": re.sub(r"答案\d?(:|：)", "", tmp[qa_key][idx+1].strip()),
                            "article": contents[_id]
                        }
                    )

with open("data/wikilingua_gpt4_nlp_tasks.json", "w") as w:
    json.dump(results, w, indent=4, ensure_ascii=False)