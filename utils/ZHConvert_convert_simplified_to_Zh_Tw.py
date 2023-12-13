"""
@Po-Kai 2023/12

ZHConvert is a tool for traditional and simplified Chinese conversion and localization. 
Compared to OpenCC, it provides better localization for terms and words.
"""
from itertools import chain
from multiprocessing.pool import ThreadPool
from typing import List

from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry


class ZHConvert(object):

    def __init__(self, style="Taiwan", retry_times=3):
        self.api_endpoint = "https://api.zhconvert.org/convert"
        self.style = style

        retry_strategy = Retry(
            total=retry_times,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        self.session = session

    def _batch_convert_worker(self, batch):
        batch = "|||".join(batch)
        convert_text = self.convert(batch)
        convert_texts = convert_text.split("|||")
        return convert_texts

    def _batch_iter(self, texts: List[str], batch_size):
        for idx in range(0, len(texts), batch_size):
            yield texts[idx:idx+batch_size]

    def convert(self, text):
        response = self.session.post(
            self.api_endpoint, 
            data={"text": text, "converter": self.style}
        )
        convert_text = response.json()["data"]["text"]
        return convert_text
  
    def batch_convert(self, texts: List[str], batch_size=1000, num_workers=4):
        with ThreadPool(num_workers) as pool:
            parallel_results = pool.map(
                self._batch_convert_worker,
                self._batch_iter(texts, batch_size)
            )
        convert_all = list(chain(*parallel_results))
        return convert_all


if __name__ == "__main__":
    zh_sents = [
        "计算年化收益率",
        "做好准备工作",
        "寻找健康食品的优惠",
        "关注更便宜的健康食材",
        "在家制作食物",
    ]
    convert = ZHConvert()
    resutls = convert.batch_convert(zh_sents)
    print(resutls)