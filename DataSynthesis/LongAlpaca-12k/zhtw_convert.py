'''
@Po-Kai 2023/12
Consider the translated sentence still have some simplified Chinese, we need to further convert the result into traditional Chinese.
'''
import json
from typing import List

import fire
from tqdm import tqdm

from utils.converter import ZhTWConvert


def run(
    data_path: str = "data/LongAlpaca-12k_translated.json",
    save_path: str = "data/LongAlpaca-12k_zh-tw.json",
    to_translated_keys: List[str] = ["output"],
):
    # load data
    with open(data_path, "r") as r:
        datas = json.load(r)

    # convert to zh_TW
    convert = ZhTWConvert()
    converts_datas = []

    
    for key in to_translated_keys:
        to_translated = [data[key] for data in datas]
        translated = convert.batch_convert(to_translated)
        
        for data, translated_sent in zip(datas, translated):
            data[key] = translated_sent

    # save results
    with open(save_path, "w") as writer:
        json.dump(datas, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(run)
