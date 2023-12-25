import re
import os
import json
from typing import List, Dict
from functools import partial
from glob import glob
from pprint import pprint

from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
import numpy as np
import pandas as pd


def prefix_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0
    
    return 1 if pred.strip().startswith(gold.strip()) else 0


def rouge_tc_score(gold: str, pred: str, rouge_type: str, scorer: "RougeCalculator") -> float:
    if rouge_type == "rouge1":
        return scorer.rouge_1(summary=gold, references=pred)
    elif rouge_type == "rouge2":
        return scorer.rouge_2(summary=gold, references=pred)
    elif rouge_type == "rougeL":
        return scorer.rouge_l(summary=gold, references=pred)
    else:
        raise KeyError(f"No rouge_type = {rouge_type}")


def get_rouge_tc_function(rouge_type: str = "rouge2") -> callable:
    scorer = RougeCalculator(stemming=True, lang="zh")
    return partial(rouge_tc_score, scorer=scorer, rouge_type=rouge_type)

def get_rouge_en_function(rouge_type: str = "rouge2") -> callable:
    scorer = RougeCalculator(stemming=True, lang="en")
    return partial(rouge_tc_score, scorer=scorer, rouge_type=rouge_type)

def get_bleu_tc_score(gold: str, pred: str) -> float:
    bleu_f = BLEUCalculator(lang="zh")
    score = bleu_f.bleu(gold, pred)
    #if not score < 1:
    #    print(f'score:{score}\nref:\n{gold}pred:\n{pred}')
    #    print('-----------------------------------------')
    return score if score < 1 else 0

def get_bleu_en_score(gold: str, pred: str) -> float:
    bleu_f = BLEUCalculator()
    score = bleu_f.bleu(gold, pred)
    #if not score < 1:
    #    print(f'score:{score}\nref:\n{gold}pred:\n{pred}')
    #    print('-----------------------------------------')
    return score if score < 1 else 0


class Task:
    def __init__(self, dir, n):
        self._prepare_data(dir, n)

    def _prepare_data(self, dir, n):
        # create self._gold_dict
        raise NotImplementedError
    
    def _get_response_dict(self, list_of_response):
        response_dict = {}
        for data in list_of_response:
            response_dict[str(data['id'])] = data['response']
        return response_dict

    def evaluate(self, list_of_response: List[Dict]) -> Dict:
        # return metrics
        raise NotImplementedError


class ChoiceTask(Task):
    CHOICES = None

    def _extract_choice(self, response, choices=None):
        raise NotImplementedError
    
    def evaluate(self, list_of_response: List[Dict]) -> Dict:
        correct_list = []
        gold_dict= self._gold_dict

        response_dict = self._get_response_dict(list_of_response)

        for idx in self._gold_dict.keys():
            choice = self._extract_choice(
                response_dict[idx], 
                choices=self._choices_dict[idx] if hasattr(self, '_choices_dict') else None
            )
            correct_list.append(1 if choice == gold_dict[idx] else 0)
        return {
            'accuracy': np.mean(correct_list)
        }


class MultipleChoiceTask(ChoiceTask):
    CHOICES = "ABCDE"

    def _extract_choice(self, response, choices=None):
        if len(response.strip()) == 0:
            return -1

        patterns = [
            r"^\s*([A-Ea-e])",
            r"^\s*\(([A-Ea-e])\)",
            r"^[選|选]\(?([A-Ea-e])\)?",
            r"^[選|选][項|项|擇|择]\(?([A-Ea-e])\)?",
            r"[選|选][擇|择]\s*\(?([A-Ea-e])\)?",
            r"答案[選|选][項|项|擇|择][為|为]\s*\(?([A-Ea-e])\)?",
            r"答案是\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[為|为]\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[應|应][為|为]\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案是[：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[應|应][該|该]是[：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"正[確|确]的[選|选|一][项|項][應|应|應該|应该]*是\s*\(?([A-Ea-e])\)?",
            r"正[確|确]的[项|項]是\s*\(?([A-Ea-e])\)?",
            r"正[確|确]的?[陳|陈]述[應|应][該|该]*是\s*\(?([A-Ea-e])\)?",
            r"答案[為|为][：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[應|应][為|为][：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[：|:]\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案是[：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[應|应][該|该]是[：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[為|为][：|:]\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[應|应][為|为][：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[：|:]\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案可能是[：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[選|选][擇|择][：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"答案[選|选][擇|择][：|:]?\s*[選|选]?[項|项|擇|择]?\s*\(?([A-Ea-e])\)?",
            r"[應|应][該|该]*[選|选][擇|择][：|:]?\s*\(?([A-Ea-e])\)?",
            r"[應該|应该]是[：|:]?\s*\(?([A-Ea-e])\)?",
            r"正確的敘述是[：|:]?\s*\(?([A-Ea-e])\)?",
            r"正確.{0,8}是[選項|选项|選擇|选择]?[：|:]?\s*\(?([A-Ea-e])\)?",
            r"最恰當的選項為[：|:]?\s*\(?([A-Ea-e])\)?",
            r"[選|选][項|项|擇|择]\s*[：|:]?\s*\(?([A-Ea-e])\)?\s*是?正[確|确]",
            r'\(?([A-Ea-e])\)?\s*[應該|应该]?[是|為|为]正[確|确]',
            r'回答\s*[：|:]\s*\(?([A-Ea-e])\)?'
        ]
        found_set = set()
        for p in patterns:
            for char in re.findall(p, response):
                found_set.add(char.upper())
        if len(found_set) == 1:
            char = found_set.pop()
            return self.CHOICES.index(char.upper())  
        
        # when the response has only one english char and the char is A/B/C/D
        # high possible the char is the answer
        en_chars = re.findall(r'[a-zA-Z]', response)
        if len(en_chars) == 1:
            char = en_chars[0].upper()
            if char in self.CHOICES:
                return self.CHOICES.index(char.upper())  
            
        if choices:
            for j, choice_str in enumerate(choices):
                if prefix_exact_match(choice_str, response) > 0:
                    return j

        return -1


class QuestionAnsweringTask(Task):
    _metric_fns = {"prefix_exact_match": prefix_exact_match}
    
    def evaluate(self, list_of_response: List[Dict]) -> Dict:
        gold_dict= self._gold_dict
        response_dict = self._get_response_dict(list_of_response)

        metrics = {}
        for m_name, metric_fn in self._metric_fns.items():
            vals = []
            for idx in self._gold_dict.keys():
                references = [r.strip() for r in gold_dict[idx]]
                pred = response_dict[idx].strip()
                vals.append(np.max([metric_fn(ref, pred) for ref in references]))
            metrics[m_name] = np.mean(vals)
        return metrics


class SummaryTask(Task):
    _metric_fns = {"Rouge-1": get_rouge_tc_function("rouge1"),
                   "Rouge-2": get_rouge_tc_function("rouge2"),
                   "Rouge-L": get_rouge_tc_function("rougeL")}

    def evaluate(self, list_of_response: List[Dict]) -> Dict:
        gold_dict= self._gold_dict
        response_dict = self._get_response_dict(list_of_response)

        metrics = {}
        for m_name, metric_fn in self._metric_fns.items():
            vals = []
            for idx in self._gold_dict.keys():
                vals.append(metric_fn(gold_dict[idx], response_dict[idx]))
            metrics[m_name] = np.mean(vals)
        lengths = []
        for idx in self._gold_dict.keys():
            lengths.append(len(response_dict[idx]))
        metrics['length'] = np.mean(lengths)
        return metrics

class TranslationTask_EN2CH(Task):
    _metric_fns = {"Bleu": get_bleu_tc_score,
                   "Rouge-1": get_rouge_tc_function("rouge1"),
                   "Rouge-2": get_rouge_tc_function("rouge2"),
                   "Rouge-L": get_rouge_tc_function("rougeL")}

    def evaluate(self, list_of_response: List[Dict]) -> Dict:
        gold_dict= self._gold_dict
        response_dict = self._get_response_dict(list_of_response)

        metrics = {}
        for m_name, metric_fn in self._metric_fns.items():
            vals = []
            for idx in self._gold_dict.keys():
                vals.append(metric_fn(gold_dict[idx], response_dict[idx]))
            metrics[m_name] = np.mean(vals)

            gt = []
            pred = []
            for idx in self._gold_dict.keys():
                pred.append(response_dict[idx])
                gt.append(gold_dict[idx])
            
            df = pd.DataFrame.from_dict({'pred':pred,'gt':gt})
            metrics['Length'] = df['pred'].str.count(pat='[\u4e00-\u9fff]').to_numpy().mean()
            metrics['Length GT'] = df['gt'].str.count(pat='[\u4e00-\u9fff]').to_numpy().mean()
        return metrics

class TranslationTask_CH2EN(Task):
    _metric_fns = {"Bleu": get_bleu_en_score,
                   "Rouge-1": get_rouge_en_function("rouge1"),
                   "Rouge-2": get_rouge_en_function("rouge2"),
                   "Rouge-L": get_rouge_en_function("rougeL")}

    def evaluate(self, list_of_response: List[Dict]) -> Dict:
        gold_dict= self._gold_dict
        response_dict = self._get_response_dict(list_of_response)

        metrics = {}
        for m_name, metric_fn in self._metric_fns.items():
            vals = []
            for idx in self._gold_dict.keys():
                vals.append(metric_fn(gold_dict[idx], response_dict[idx]))
            metrics[m_name] = np.mean(vals)
            
            gt = []
            lengths = []
            for idx in self._gold_dict.keys():
                lengths.append(len(response_dict[idx].split(' ')))
                gt.append(len(gold_dict[idx].split(' ')))
            metrics['Length'] = np.mean(lengths)
            metrics['Length GT'] = np.mean(gt)
        return metrics

class TTQATask(MultipleChoiceTask):
    def _prepare_data(self, dir, n):
        data = json.load(open(f'{dir}/TTQA_mc_2.0.0.json'))
        self._gold_dict = {}
        self._choices_dict = {}
        for idx in data:
            self._gold_dict[str(idx)] = data[idx]['answer']
            self._choices_dict[str(idx)] = data[idx]['choices']


class TMMLUTask(MultipleChoiceTask):
    def _prepare_data(self, dir, n):
        df = pd.read_csv(f'{dir}/data.csv')
        self._gold_dict = {}
        for i, row in df.iterrows():
            self._gold_dict[str(i)] = self.CHOICES.index(row['content.A'])


class PenguinsInTableTCTask(MultipleChoiceTask):
    def _prepare_data(self, dir, n):
        data = json.load(open(f'{dir}/data.json'))
        self._gold_dict = {}
        self._choices_dict = {}
        for idx in data:
            self._gold_dict[str(idx)] = data[idx]['answer']
            self._choices_dict[str(idx)] = data[idx]['choices']


class IMDBTCTask(ChoiceTask):
    CHOICES = '負正'

    def _prepare_data(self, dir, n):
        df = pd.read_csv(f'{dir}/test.csv')
        self._gold_dict = {}
        for i, row in df.iterrows():
            if int(i) > n:
                pass
            else:
                self._gold_dict[str(i)] = int(row['label'])

    def _extract_choice(self, response, choices=None):
        if len(response.strip()) == 0:
            return -1

        patterns = [
            r"^\s*([正|負])面",            
        ]
        found_set = set()
        for p in patterns:
            for char in re.findall(p, response):
                found_set.add(char)
        if len(found_set) == 1:
            char = found_set.pop()
            return self.CHOICES.index(char)  

        return -1


class XSumTCTask(SummaryTask):
    def _prepare_data(self, dir, n):
        df = pd.read_csv(f'{dir}/test_sub5000.csv')
        self._gold_dict = {}
        for i, row in df.iterrows():
            if int(i) > n:
                pass
            else:
                self._gold_dict[str(i)] = str(row['summary'])

class CNNTask(SummaryTask):
    def _prepare_data(self, dir, n):
        df = pd.read_csv(f'{dir}/cnn_tw.csv')
        self._gold_dict = {}
        for i, row in df.iterrows():
            if int(i) > n:
                pass
            else:
                self._gold_dict[str(i)] = str(row['summary'])

class EN2CH(TranslationTask_EN2CH):
    def _prepare_data(self, dir, n):
        df = pd.read_csv(f'{dir}/coct.csv')
        self._gold_dict = {}
        for i, row in df.iterrows():
            if int(i) > n:
                pass
            else:
                self._gold_dict[str(i)] = str(row['ch'])

class CH2EN(TranslationTask_CH2EN):
    def _prepare_data(self, dir, n):
        df = pd.read_csv(f'{dir}/coct.csv')
        self._gold_dict = {}
        for i, row in df.iterrows():
            if int(i) > n:
                pass
            else:
                self._gold_dict[str(i)] = str(row['en'])

class DRCDTask(QuestionAnsweringTask):
    def _prepare_data(self, dir, n):
        data = json.load(open(f'{dir}/test.json'))
        self._gold_dict = {}
        for idx in data:
            if int(idx) > n:
                pass
            else:
                self._gold_dict[str(idx)] = data[idx]['references']


class FGCTask(QuestionAnsweringTask):
    def _prepare_data(self, dir, n):
        data = json.load(open(f'{dir}/preprocessed_FGC_official_final.json'))
        self._gold_dict = {}
        for idx in data:
            self._gold_dict[str(idx)] = data[idx]['references']


EVALUATION_ITEMS = [
    ['XSum_TC_5k', XSumTCTask('./data/XSum_TC_5k/', 5000)], 
    ['DRCD', DRCDTask('./data/DRCD_Test/', 3493)], 
    ['TTQA', TTQATask('./data/TTQA/', 103)],
    ['IMDB_TC', IMDBTCTask('./data/IMDB_TC/', 5000)],
    ['PenguinsInTable_TC', PenguinsInTableTCTask('./data/PenguinsInTable_TC', 0)],
    ['CH2EN', CH2EN('./data/COCT', 100)],
    ['EN2CH', EN2CH('./data/COCT', 100)],
    ['CNN', CNNTask('./data/CNN', 162)],
    *[[f'TMMLU_{subject}', TMMLUTask(f'./data/TMMLU/subjects/{subject}/', 0)]
      for subject in os.listdir(f'./data/TMMLU/subjects/')]
    #['FGC', FGCTask('./data/FGC_Test')],
]


def evaluate_all(result_path):
    results = json.load(open(result_path))
    metrics = {}
    for name, task in EVALUATION_ITEMS:
        if name == 'BB_Penguins_in_a_Table_TC':
            name = 'PenguinsInTable_TC'
        #print(results['results'].keys())
        if name in results['results']:
            list_of_response = results['results'][name]
            metrics[name] = task.evaluate(list_of_response)
        else:
            print(f'Task \"{name}\" result missing in \"{result_path}\"')
            metrics[name] = {'result missing':0}
    return metrics


#if __name__ == '__main__':
def evaluate_all_task(model_list):
    for model in model_list:
        path = f'./results/{model}_result.json'
        print(f'================ {model} ================')
        show = {}
        metrics = evaluate_all(path)
        metrics['TMMLU_Avg'] = {'accuracy': np.mean([metrics[k]['accuracy'] for k in metrics if 'TMMLU' in k])}
        for cate in LD:
            metrics[f'TMMLU_{cate}'] = {'accuracy': np.mean([metrics[k]['accuracy'] for k in metrics if k.replace('TMMLU_','') in LD[cate]])}

        for i in ['DRCD','XSum_TC_5k', 'PenguinsInTable_TC','TMMLU_Avg','TTQA','IMDB_TC']:# + ['TMMLU_'+cate for cate in LD]:
            show[i] = metrics[i]

            # overall performance
            #"""
            rouge = 0
            if i == 'XSum_TC_5k' or i == 'CNN':
                for rogue_type in show[i]:
                    if 'rouge' in rogue_type:
                        rouge += show[i][rogue_type]
                print(rouge)
            else:
                for j in show[i]:
                    print(show[i][j])
            #"""
        
        # rouge-1, rouge-2, rouge-L
        """
        for i in metrics['XSum_TC_5k']:
            print(metrics['XSum_TC_5k'][i])
        print('==========')
        """

        """# TMMLU subject scores
        for i in ['TMMLU_'+cate for cate in LD]:
            for j in metrics[i]:
                print(metrics[i][j])
        print('==========')
        """
def evaluate_summary(model_list):
    data_dict = {'XSum_TC_5k':{}, 'CNN':{}}
    for model in model_list:
        path = f'./results/{model}_result.json'
        metrics = evaluate_all(path)

        data_dict['XSum_TC_5k'][model] = [metrics['XSum_TC_5k'][metric_name] for metric_name in metrics['XSum_TC_5k']]
        data_dict['CNN'][model] = [metrics['CNN'][metric_name] for metric_name in metrics['CNN']]
    
    row_names = [metric_name for metric_name in metrics['XSum_TC_5k']]
    xsum = pd.DataFrame(data_dict['XSum_TC_5k'], index=row_names)
    cnn = pd.DataFrame(data_dict['CNN'], index=row_names)
    print(xsum)
    print(cnn)
    if not os.path.exists('./results/csv/'):
        os.mkdir('./results/csv/')
    xsum.to_csv(f'./results/csv/summary_xsum.csv')
    cnn.to_csv(f'./results/csv/summary_cnn.csv')

def evaluate_translation(model_list):
    data_dict = {'EN2CH':{}, 'CH2EN':{}}
    for model in model_list:
        path = f'./results/{model}_result.json'
        metrics = evaluate_all(path)

        for task_name in data_dict:
            data_dict[task_name][model] = [metrics[task_name][metric_name] for metric_name in metrics[task_name]]
    
    row_names = [metric_name for metric_name in metrics[task_name]]
    a = pd.DataFrame(data_dict['EN2CH'], index=row_names)
    b = pd.DataFrame(data_dict['CH2EN'], index=row_names)
    if not os.path.exists('./results/csv/'):
        os.mkdir('./results/csv/')
    a.to_csv(f'./results/csv/translation_EN2CH.csv')
    b.to_csv(f'./results/csv/translation_CH2EN.csv')

def get_overall_performace(model_list, tasks):
    data_dict = {}
    #tasks_name = [tasks_name_dict[i] for i in tasks]
    for model in model_list:
        path = f'./results/{model}_result.json'
        metrics = evaluate_all(path)
        metrics['TMMLU_Avg'] = {'accuracy': np.mean([metrics[k]['accuracy'] for k in metrics if 'TMMLU' in k])}
        
        values = []
        for task in tasks:
            # check if the result is empty
            if metrics[task] == {'result missing':0}:
                    values.append(0)
            else:
                # select summarization task
                if task == 'XSum_TC_5k' or task == 'CNN':
                    rouge = 0
                    for rogue_type in metrics[task]:
                        #print(rogue_type)
                        if 'Rouge-2' in rogue_type:
                            rouge += metrics[task][rogue_type]
                    values.append(rouge)
                # select translation task
                elif '2' in task:
                    values.append(metrics[task]['Bleu'])
                else:
                    for v_name in metrics[task]:
                        values.append(metrics[task][v_name])
        values.append(np.array(values).mean())
        string_values = []
        for v in values:
            v = v*100
            v = f'{v:.2f}%'
            string_values.append(v)
        model_name_dict = {f'v{i}': f'FoxBrain ver.{i}' for i in range(20)}
        for i in range(20):
            model_name_dict[f'twllama_v{i}'] = f'TaiwanLlama ver.{i}'
        model_name_dict['model_7c_chat'] = 'MediaTek'
        model_name_dict['gpt3.5'] = 'ChatGPT'

        data_dict[model_name_dict[model]] = string_values
    df = pd.DataFrame(data_dict, index=(tasks+['avg'])).T
    print(df)
    if not os.path.exists('./results/csv/'):
        os.mkdir('./results/csv/')
    df.to_csv(f'./results/csv/overall_performance.csv')


#tasks_name_dict = {'DRCD':'Reading Comprehension', 'CNN':'Long Document Summarization', 'EN2CH':'English2Chinese Translation', 'CH2EN':'Chinese2English Translation', 'XSum_TC_5k':'One Sentence Summarization', 'PenguinsInTable_TC':'Table Data Understanding', 'TMMLU_Avg':'Academic Knowledge', 'TTQA':'Taiwan Specific Knowledge', 'IMDB_TC':'Opinion Mining'}
#tasks_name_dict = {i:i for i in tasks_name_dict}

# model : 'v0','v1','v2','v3','v4','v5','twllama_v0','twllama_v1','twllama_v2','model_7c_chat','gpt3.5'
# tasks : 'DRCD','XSum_TC_5k', 'CNN', 'EN2CH', 'CH2EN', 'PenguinsInTable_TC','TMMLU_Avg','TTQA','IMDB_TC'
tasks = ['DRCD','XSum_TC_5k', 'CNN', 'EN2CH', 'CH2EN', 'PenguinsInTable_TC','TMMLU_Avg','TTQA','IMDB_TC']
model_list = ['v5','v8','v9','v10','v11']#,'model_7c_chat','gpt3.5']
get_overall_performace(model_list,tasks)


#get_overall_performace(['v4', 'twllama_v1','gpt3.5'], tasks,tasks_name_dict)
#evaluate_summary(['v4', 'twllama_v1', 'gpt3.5']) # ,'twllama_v1'
#evaluate_translation(['v4', 'twllama_v1', 'gpt3.5'])