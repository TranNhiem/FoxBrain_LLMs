import os
import re
from typing import Dict
from torch.utils.data import Dataset
from rich import print
import json
import numpy as np
import pandas as pd

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))


class Scenario(Dataset):
    name: str = None


class DRCD(Dataset):
    name: str = "DRCD"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/DRCD_Test/test.json", **kwargs):
        raw_data = json.load(open(data_path, "r"))
        self._js_ds = [dict(id=str(i), **obj) for i, obj in raw_data.items()]
        
    def __len__(self):
        return len(self._js_ds)

    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._js_ds[idx]

        context = sample['paragraph']
        question = sample['question']
        references = sample['references']
        idx = sample['id']

        return {'context': context, 'question': question, 'references': references, 'id': str(idx)}
"""
class DRCD(Scenario):
    name: str = "DRCD"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/DRCD_Test/test.json", **kwargs):
        
        all_samples = []
        with open(data_path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                all_samples.append(json.loads(line))
        all_samples = all_samples[0]
        print(len(all_samples))
        self._clean_samples = all_samples

    def __len__(self):
        return len(self._clean_samples)

    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._clean_samples[str(idx)]
        context = sample['paragraph']
        question = sample['question']
        references = [sample["references"]]
        idx = str(idx)

        return {'context': context, 'question': question, 'references': references, 'id': str(idx)}
"""

class FGC(Scenario):
    name: str = "FGC"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/FGC_Test/preprocessed_FGC_official_final.json", **kwargs):
        raw_data = json.load(open(data_path, "r"))
        self._js_ds = [dict(id=i, **obj) for i, obj in raw_data.items()]

    def __len__(self):
        return len(self._js_ds)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._js_ds[idx]

        context = sample['paragraph']
        question = sample['question']
        references = sample['references']

        # Construction question
        # question = f"請根據以下內容回答問題，且答案需盡可能簡短。注意：答案必須為內容的子字串。\n\n{context}\n\n問題： {sample_question}\n\n"

        return {'context': context, 'question': question, 'references': references, 'id': str(sample['id'])}


class TTQA(Scenario):
    name: str = "TTQA"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/TTQA/TTQA_mc_2.0.0.json", **kwargs):
        raw_data = json.load(open(data_path, "r", encoding="utf-8"))
        self._js_ds = [dict(id=i, **obj) for i, obj in raw_data.items()]
        
    def __len__(self):
        return len(self._js_ds)

    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._js_ds[idx]

        choices = sample['choices']
        sample_question = sample['question']
        answer = sample['answer']

        # Construct multiple choice question
        _map_num_to_alph = {i:a for i, a in zip(range(4), 'ABCD')}
        choice_message = ";".join([f"({_map_num_to_alph[i]}) {tg}" for i, tg in enumerate(choices)])

        choice = _map_num_to_alph[answer]
        choice_text = choices[answer]
        references = [choice, f"({choice}) {choice_text}", choice_text]
        question = f"問題: {sample_question} \n\n請從以下選項中選擇並回答: {choice_message}\n"

        return {'question': question, 'references': references, 'id': str(sample['id'])}


class TMMLU(Scenario):
    name: str = "TMMLU"
    def __init__(self, data_dir: str = f"{_CUR_DIR}/../data/TMMLU/subjects", subject: str = None, **kwargs):
        assert subject is not None, f"subject = {subject} invalid"
        self.name = f"{self.name}/{subject}"
        data_path = f"{data_dir}/{subject}/data.csv"
        self._df = pd.read_csv(data_path)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._df.iloc[idx]

        ref = sample['content.A']
        question = sample['content.Q']
        references = [ref]
        return {'question': question, 'references': references, 'id': int(idx)}


class XSumTC(Scenario):
    name: str = "XSum_TC"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/XSum_TC_5k/test_sub5000.csv", **kwargs):
        self._df = pd.read_csv(data_path)

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._df.iloc[idx]

        context = sample['document']
        references = [sample['summary']]

        return {'context': context, 'references': references, 'id': str(idx)}

class EN2CH(Scenario):
    name: str = "EN2CH"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/COCT/coct.csv", **kwargs):
        self._df = pd.read_csv(data_path)

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._df.iloc[idx]

        context = sample['en']
        references = [sample['ch']]

        return {'context': context, 'references': references, 'id': str(idx)}

class CH2EN(Scenario):
    name: str = "CH2EN"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/COCT/coct.csv", **kwargs):
        self._df = pd.read_csv(data_path)

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._df.iloc[idx]

        context = sample['ch']
        references = [sample['en']]

        return {'context': context, 'references': references, 'id': str(idx)}

class CNN(Scenario):
    name: str = "CNN"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/CNN/cnn_tw.csv", **kwargs):
        self._df = pd.read_csv(data_path)

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._df.iloc[idx]

        context = sample['document']
        references = [sample['summary']]

        return {'context': context, 'references': references, 'id': str(idx)}

class IMDBTC(Scenario):
    name:str = "IMDB_TC"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/IMDB_TC/test.csv", **kwargs):
        self._df = pd.read_csv(data_path)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._df.iloc[idx]
        # sample = self._hf_ds[idx]

        context = sample['text']
        label = sample['label']
        _lzh = {0: "負面", 1: "正面"}[label]
        references = [_lzh]

        question = f"請閱讀以下評論，並回答此評論是正面還是負面，如果是正面，請回答\'正面\';，如果是負面，請回答\'負面\'：\n\n評論：{context}\n\n"

        return {'question': question, 'references': references, 'id': str(idx)}


class BigBenchPenguinsInATableTC(Scenario):
    name: str = "BB_Penguins_in_a_Table_TC"
    def __init__(self, data_path: str = f"{_CUR_DIR}/../data/PenguinsInTable_TC/data.json", **kwargs):
        raw_data = json.load(open(data_path, "r"))
        self._js_ds = [dict(id=k, **v) for k, v in raw_data.items()]

    def __len__(self):
        return len(self._js_ds)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        sample = self._js_ds[idx]

        sample_q = sample['question']
        if sample_q.endswith('回答：'):
            sample_q = sample_q.rstrip('回答：')
    
        # Convert to multiple choice
        _map_num_to_alph = {i:a for i, a in zip(range(5), 'ABCDE')}
        mc_targets = sample['choices']
        choice_message = ";".join([f"({_map_num_to_alph[i]}) {tg}" for i, tg in enumerate(mc_targets)])
        
        answer = sample['answer']
        ref = sample['answer_str']
        choice = _map_num_to_alph[answer]

        references = [ref, f"({choice}): {ref}"]
        question = f"{sample_q} \n請從以下選項中選擇並回答: {choice_message}\n"

        return {'question': question, 'references': references, 'id': str(sample['id'])}


if __name__ == '__main__':
    ds = BigBenchPenguinsInATableTC()
    fgc = FGC()
    drcd = DRCD()
    ttqa = TTQA()
    xsum = XSumTC()
    imdb = IMDBTC()
    cnn = CNN()
    en2ch = EN2CH()
    ch2en = CH2EN()

    print(ds[10])
    print(fgc[0])
    print(drcd[3])
    print(ttqa[10])
    print(xsum[10])
    print(cnn[10])
    print(en2ch[10])
    print(ch2en[10])
    print(imdb[10])

    kwargs = {"subject": "食品檢驗分析", "happy": "halloween!"}
    tmmlu = TMMLU(**kwargs)
    print(tmmlu[10])
