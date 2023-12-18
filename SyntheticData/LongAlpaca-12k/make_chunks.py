'''
@Po-Kai 2023/12

This code implements for prechunking the LongAlpaca-12k data.
'''
import json
from functools import partial
from multiprocessing.pool import Pool
from typing import List

import fire
from tqdm import tqdm

from utils import chunk_text_to_max_tokens


def parallel_worker(data, to_chunk_keys, max_tokens=512):
    for key in to_chunk_keys:
        chunks = chunk_text_to_max_tokens(data[key], max_tokens)
        data[key] = chunks
    return data


def run(
    data_path: str = "data/LongAlpaca-12k.json",
    save_path: str = "data/LongAlpaca-12k_chunked.json",
    to_chunk_keys: List[str] = ["output"],
    max_tokens: int = 512,
    num_worker: int = 8
):
    # load data
    with open(data_path, "r") as r:
        datas = json.load(r)

    # chunking data
    with Pool(num_worker) as pool:
        worker_fn = partial(
            parallel_worker, 
            to_chunk_keys=to_chunk_keys, 
            max_tokens=max_tokens
        )
        results = list(tqdm(
            pool.imap_unordered(worker_fn, datas),
            total=len(datas),
        ))

    # save results
    with open(save_path, "w") as writer:
        json.dump(results, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(run)