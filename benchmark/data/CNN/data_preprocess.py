import os
import pandas as pd
import numpy as np
import json


def preprocess(file_path, save_path):
    data = json.load(open(file_path, 'r', encoding="utf8"))

    docs = []
    summs = []

    for i in data:
        doc = i['document']
        summ = i['summary']

        # cleaning
        if (len(doc) > 20) and (len(summ) > 20):
            docs.append(doc)
            summs.append(summ)
        else:
            pass
    print(f'------------------ Available data: {len(summs)} ! ------------------')

    data_dict = {'document': docs, 'summary': summs}
    df = pd.DataFrame(data=data_dict)
    df.to_csv(save_path)

file_path = 'cnn_tw.json'
save_path = 'cnn_tw.csv'
preprocess(file_path, save_path)