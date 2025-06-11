import json
import os

def modify_task_config(model_name, system_prompt, suffix_inst):

    config_path = 'tasks_config.json'
    save_config_path = f'./configs/tasks_config_{model_name}.json'
    with open('../results/model_7c_chat_result.json') as ref:
                model_7c_data = json.load(ref)

    with open(config_path,'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            data[i]['model_name'] = model_name
            data[i]['system_prompt'] = system_prompt
            data[i]['suffix_inst'] = suffix_inst
            
            key = data[i]['scenario']
            #key = key.replace('XSum_TC','XSum_TC_5k')
            #key = key.replace('BB_Penguins_in_a_Table_TC','PenguinsInTable_TC')
            #task_size = len(model_7c_data['results'][key]) - 1
            if 'DRCD' in key:
                with open('../data/DRCD_Test/test.json') as ref:
                    ref = json.load(ref)
                task_size = len(ref)
            elif 'XSum_TC' in key:
                task_size = 5000
            elif 'IMDB_TC' in key:
                task_size = 5000
            elif 'TTQA' in key:
                task_size = 103
            elif 'BB_Penguins' in key:
                task_size = 149                
            
            data[i]['eval_num_samples'] = task_size

    with open(save_config_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def modify_tmmlu_config(model_name, system_prompt, suffix_inst):

    config_path = 'tmmlu_task_config.json'
    save_config_path = f'./configs/tmmlu_task_config_{model_name}.json'

    with open('../results/model_7c_chat_result.json') as f:
        model_7c_data = json.load(f)

    with open(config_path,'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            data[i]['model_name'] = model_name

            key = 'TMMLU_' + data[i]['subject']
            task_size = len(model_7c_data['results'][key])
            
            data[i]['eval_num_samples'] = task_size
            data[i]['system_prompt'] = system_prompt
            data[i]['suffix_inst'] = suffix_inst

    with open(save_config_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

#"""
def tmmlu_config():

    config_path = 'tmmlu_task_config_old.json'
    save_config_path = f'tmmlu_task_config.json'

    with open('../results/model_7c_chat_result.json') as f:
        model_7c_data = json.load(f)

    with open(config_path,'r') as f:
        data = json.load(f)

    configs = []
    tmmlu_data_path = '../data/TMMLU/subjects'
    config_format = data[0]
    n = 0
    for i in os.listdir(tmmlu_data_path):
        config_format['subject'] = i
        key = 'TMMLU_' + i
        task_size = len(model_7c_data['results'][key])
        n += task_size
        
        config_format['eval_num_samples'] = task_size
        a = config_format.copy()
        configs.append(a)

    print(f'tmmlu total quesion: {n}')
    with open(save_config_path, 'w') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)
tmmlu_config()
#"""

def modify_task_translation_config(model_name, system_prompt, suffix_inst):

    config_path = 'tasks_config_translation.json'
    save_config_path = f'./configs/tasks_config_translation_{model_name}.json'

    with open(config_path,'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            data[i]['model_name'] = model_name
            data[i]['system_prompt'] = system_prompt
            data[i]['suffix_inst'] = suffix_inst

    with open(save_config_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def modify_task_cnn_config(model_name, system_prompt, suffix_inst):

    config_path = 'tasks_config_cnn.json'
    save_config_path = f'./configs/tasks_config_cnn_{model_name}.json'

    with open(config_path,'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            data[i]['model_name'] = model_name
            data[i]['system_prompt'] = system_prompt
            data[i]['suffix_inst'] = suffix_inst

    with open(save_config_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


fox_prompt = "[INST] <<SYS>>\n\nyou are a Foxbrain AI assistant developmen by Tran Rick, designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity.\
            When given tasks, approach them step-by-step, always justifying your actions for the user. If you encounter multiple-choice questions, first output the correct answer, then delve into why other options are incorrect.\
            breaking down even complex tasks into simpler, understandable terms.\
            Additionally, consider yourself well-versed in every language, capable of translating and explaining language tasks effortlessly. When presented with task definitions or samples, dissect them into key components, clarifying each segment with relevant examples. \
            Your overarching goal is to be a reliable source of knowledge, translating any instruction or task into actionable and easily digestible information.\
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
            correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n"
fox_inst = '[/INST]'

taiwanllm_prompt = ""

for i in [f'v{k}' for k in range(12)]:#,'gpt3.5','twllama_v0','twllama_v1','twllama_v2']:
    modify_task_cnn_config(i, fox_prompt, fox_inst)
    modify_task_config(i, fox_prompt, fox_inst)
    modify_tmmlu_config(i, fox_prompt, fox_inst)
    modify_task_translation_config(i, fox_prompt, fox_inst)