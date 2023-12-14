'''
@Po-Kai 2023/12

This code implements for prechunking the LongAlpaca-12k data.
'''
import json
from functools import partial
from multiprocessing.pool import Pool

import fire
from tqdm import tqdm

from utils import chunk_text_to_max_tokens


def parallel_worker(data, max_tokens=512):
    new_data = {}
    for key, text in data.items():
        chunks = chunk_text_to_max_tokens(text, max_tokens)
        new_data[key] = chunks
    return new_data


def run(
    data_path: str = "data/LongAlpaca-12k.json",
    save_path: str = "data/LongAlpaca-12k_chunked.json",
    max_tokens: int = 512,
    num_worker: int = 8
):
    # load data
    with open(data_path, "r") as r:
        datas = json.load(r)

    # chunking data
    with Pool(num_worker) as pool:
        worker_fn = partial(parallel_worker, max_tokens=max_tokens)
        results = list(tqdm(
            pool.imap_unordered(worker_fn, datas),
            total=len(datas),
        ))

    # save results
    with open(save_path, "w") as writer:
        json.dump(results, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(run)