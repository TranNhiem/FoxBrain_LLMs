import json, os, numpy as np
from BCEmbedding import RerankerModel, EmbeddingModel
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from evaluate import Task
from config_task import *

class BCE_Task(Task):
    def __init__(self, config=None):
        self.config = config
    
    def create_folder(self, is_gt=False):
        # 建立資料夾
        try:
            os.mkdir(self.config['save_path'])
        except:
            pass
        if is_gt:
            try:
                os.mkdir(f"{self.config['save_path']}/groundtruth")
            except:
                pass
            try:
                os.mkdir(f"{self.config['save_path']}/groundtruth/{self.config['target_task']}")
            except:
                pass
        else:
            try:
                os.mkdir(f"{self.config['save_path']}/{self.config['file_name']}")
            except:
                pass
            try:
                os.mkdir(f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}")
            except:
                pass
        
    def Rerank_calculate(self, dict1, dict2, file_name='predict_scores.json'):
        # 建立資料夾
        self.create_folder()

        sentence_pairs = list()

        for id, query in dict1.items():
            sentence_pairs.append((query, dict2[str(id)]))
        
        # load model
        model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")

        # 計算Scores & Save
        scores = model.compute_score(sentence_pairs, batch_size=256)
        with open(f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}/{file_name}", 'w', encoding='utf-8') as f:
            json.dump(scores, f)

    def Embedding_calculate(self, dict1, dict2, file_name='predict_scores.json', mode='pre', batch_size=256):
        '''
        mode:pre:預測模式，計算模型的score/gt:groundtruth模式，計算groundtruth的score/eval:單純計算，不儲存.json
        dict format: {'id':'sentence'...}
        '''
        # 建立資料夾
        if mode != 'eval':
            if mode == 'gt':
                self.create_folder(True)
            else:
                self.create_folder(False)

        sentences_1 = list()
        sentences_2 = list()

        for id, sentences in dict1.items():
            sentences_1.append(sentences)
            sentences_2.append(dict2[str(id)])

        model = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1")

        vec1 = torch.FloatTensor(model.encode(sentences_1, batch_size=batch_size))
        print(vec1.shape)
        vec2 = torch.FloatTensor(model.encode(sentences_2, batch_size=batch_size))
        print(vec2.shape)

        # 計算Scores & Save
        scores = F.cosine_similarity(vec1, vec2, dim=1).tolist()
        if mode == 'pre':
            path = f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}/{file_name}"
        elif mode == 'gt':
            path = f"{self.config['save_path']}/groundtruth/{self.config['target_task']}/{file_name}"
        else:
            return scores
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(scores, f)

    def plot_hist(self, data_name="pre", json_path=None, target_path=None):
        '''
        mode:pre:預測模式，計算模型的score/gt:groundtruth模式，計算groundtruth的score/eval:單純計算，不儲存.json
        '''
        if data_name == "pre":
            path = f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}/predict_scores.json"
            target_path = f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}/hist.png"
        elif data_name == "org":
            path = f"{self.config['save_path']}/groundtruth/{self.config['target_task']}/groundtruth_scores.json"
            target_path = f"{self.config['save_path']}/groundtruth/{self.config['target_task']}/org_hist.png"
        elif data_name == "eval":
            path = json_path
            target_path = target_path
            
        with open(path, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        scores = np.array(scores)
        # 算出平均值
        mean = np.mean(scores)
        # 算出標準差
        std = np.std(scores)
        print(path)
        print(f"mean: {mean}, std: {std}")
        # 畫出分佈圖
        plt.hist(scores, bins=100)
        if data_name == "pre":
            plt.title(f"{self.config['file_name']} {self.config['target_task']}\nmean: {mean}, std: {std}")
        elif data_name == "org":
            plt.title(f"GroundTruth {self.config['target_task']}\nmean: {mean}, std: {std}")
        elif data_name == "eval":
            plt.title(f"mean: {mean}, std: {std}")
        plt.savefig(target_path)
        # 清除之前的分佈圖
        plt.clf()

    # def plot_sub_hist(self):
    #     with open(f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}/groundtruth_scores.json", 'r', encoding='utf-8') as f:
    #         groundtruth_scores = json.load(f)
    #     with open(f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}/predict_scores.json", 'r', encoding='utf-8') as f:
    #         predict_scores = json.load(f)
    #     # 取出跟原本的分數差異
    #     scores = np.array(groundtruth_scores) - np.array(predict_scores)
    #     # 只拿預測的分數
    #     scores = np.array(scores)
    #     # 算出平均值
    #     mean = np.mean(scores)
    #     # 算出標準差
    #     std = np.std(scores)
    #     print(f"mean: {mean}, std: {std}")
    #     # 畫出分佈圖
    #     plt.hist(scores, bins=100)
    #     plt.title(f"{self.config['file_name']} {self.config['target_task']}\nmean: {mean}, std: {std}")
    #     plt.savefig(f"{self.config['save_path']}/{self.config['file_name']}/{self.config['target_task']}/sub_hist.png")
    #     # 清除之前的分佈圖
    #     plt.clf()

def catch_worse_data(data_path, score_path, task=None, save_path='./worst_data.txt', threshold=0.4):
    '''
    找出threshold以下的資料
    '''
    # load data
    with open(data_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    with open(score_path, 'r', encoding='utf-8') as f:
        scores = json.load(f)

    datas = datas[task]

    with open(save_path, 'w', encoding='utf-8') as f: 
        for i in range(len(datas)):
            if scores[i] < threshold:
                f.write(f"ID：{datas[i]['id']}\nReferences：{datas[i]['references']}\nResponse：{datas[i]['response']}\nScore：{scores[i]}\n")

def evaluate_translate_Embed(configs):
    for config in configs:
        bce = BCE_Task(config)

        # load data
        with open(f"{config['path']}/{config['file_name']}/{config['target_task']}/result.json", 'r', encoding='utf-8') as f:
            datas = json.load(f)
        datas = datas[config['target_task']]

        querys = bce._get_querys_dict(datas, config['beginning'], config['ending'])

        if config['Mode'] != "groundtruth":
            response_dict = bce._get_response_dict(datas)
            # calculate cosine similarity
            bce.Embedding_calculate(querys, response_dict)
            # plot
            bce.plot_hist()
        else:
            references_dict = bce._get_references_dict(datas)
            # calculate cosine similarity
            bce.Embedding_calculate(querys, references_dict, "groundtruth_scores.json", is_gt=True)
            # plot
            bce.plot_hist(data_name="org")

def evaluate_summary_Embed(configs):
    for config in configs:
        bce = BCE_Task(config)

        # load data
        with open(f"{config['path']}/{config['file_name']}/{config['target_task']}/result.json", 'r', encoding='utf-8') as f:
            datas = json.load(f)
        datas = datas[config['target_task']]

        response_dict = bce._get_response_dict(datas)
        references_dict = bce._get_references_dict(datas)
        # calculate cosine similarity
        bce.Embedding_calculate(references_dict, response_dict)
        # plot
        bce.plot_hist()

def find_score_by_ID(task, id):
    for model in ['foxbrain_yi6b_11epc', 'breeze_7b', 'yi_6b']: #, 'gpt3.5'
        path = f'./results/BCEEmbed/{model}/{task}/predict_scores.json'
        with open(path, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        print(f"{model}, ID:{str(id)}, Score:{scores[id]}")
    # gt_path = f'./results/BCEEmbed/groundtruth/{task}/groundtruth_scores.json'
    # with open(gt_path, 'r', encoding='utf-8') as f:
    #     scores = json.load(f)
    # print((f"GT, ID:{str(id)}, Score:{scores[id]}"))

def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    # print(torch.cuda.device_count())

    # # for CH -> EN
    # evaluate_translate_Embed(CH2EN_configs)

    # # for EN -> CH
    # evaluate_translate_Embed(EN2CH_configs)

    # # for CNN
    # evaluate_summary_Embed(CNN_configs)

    # # for XSum_TC
    # evaluate_summary_Embed(XSum_TC_configs)

    task = "CNN"
    model = "breeze_7b"#"yi_6b" #   "foxbrain_yi6b_11epc"# 
    data_path = f"./inference/outputs/{model}/{task}/result.json"
    score_path = f"./results/BCEEmbed/{model}/{task}/predict_scores.json"
    save_path = f"./results/BCEEmbed/{model}/{task}/worse_data.txt"

    # catch_worse_data(data_path, score_path, task=task, save_path=save_path, threshold=0.7)
    find_score_by_ID(task, 124)



if __name__ == "__main__":
    main()