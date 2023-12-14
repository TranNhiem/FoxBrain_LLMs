'''
@Po-Kai 2023/12
Consider the translated sentence still have some simplified Chinese, we need to further convert the result into traditional Chinese.
'''
import json

import fire
from tqdm import tqdm

from tools import ZHConvert


def run(
    data_path: str = "data/LongAlpaca-12k_translated.json",
    save_path: str = "data/LongAlpaca-12k_zhTW.json",
):
    # load data
    with open(data_path, "r") as r:
        datas = json.load(r)

    # convert to zh_TW
    convert = ZHConvert()
    converts_datas = []
    for data in tqdm(datas):
        batch = list(data.items())
        convertds = convert.batch_convert([sample[1] for sample in batch])
        
        new_data = {}
        for (key, _), convertd in zip(batch, convertds):
            new_data[key] = convertd
            
        converts_datas.append(new_data)

    # save results
    with open(save_path, "w") as writer:
        json.dump(converts_datas, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(run)
