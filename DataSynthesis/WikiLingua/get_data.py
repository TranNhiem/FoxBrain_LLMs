"""
Author: Po-Kai Chen
Date: 1/7/2024
"""
import json

from datasets import load_dataset
from tqdm import tqdm

from utils.converter import ZhTWConvert


data = load_dataset("GEM/wiki_lingua", name="zh_en", trust_remote_code=True)
tw_converter = ZhTWConvert()

all_data = {}
for split in data.keys():
    print(f"Processing on {split} subset...")
    gem_ids, to_converted, keys  = [], [], []
    for item in data[split]:
        gem_id = item["gem_id"]
        if item["source_language"] == "zh":
            to_converted.append(item["source"])
            gem_ids.append(gem_id)
            keys.append("source")

        if item["target_language"] == "zh":
            to_converted.append(item["target"])
            gem_ids.append(gem_id)
            keys.append("target")

    print(f"Coverting...")
    converted = tw_converter.batch_convert(to_converted)
    new_data = list(data[split])
    for item in new_data:
        curr_gem_id = item["gem_id"]
        if curr_gem_id in gem_ids:
            indices = [idx for idx, gem_id in enumerate(gem_ids) if gem_id == curr_gem_id]
            
            for idx in indices:
                if keys[idx] == "references":
                    item[keys[idx]] = [converted[idx]]
                else:
                    item[keys[idx]] = converted[idx]
                
    all_data[split] = new_data
    print(f"All done on {split} subset!")

# Save data
with open("wikilingua_data_zhTW.json", "w") as w:
    json.dump(all_data, w, indent=4, ensure_ascii=False)