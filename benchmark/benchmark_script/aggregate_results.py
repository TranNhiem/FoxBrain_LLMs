import os
import argparse
import json
import glob
from tqdm import tqdm
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default=None, type=str)
    parser.add_argument("--model_name", default=None, type=str)
    args = parser.parse_args()

    result_dir = args.result_dir
    model_name = args.model_name

    result_jpath = glob.glob(f"{result_dir}/**/result.json")
    if os.path.exists(f"{result_dir}/TMMLU"):
        result_jpath.extend(glob.glob(f"{result_dir}/TMMLU/**/result.json"))
    print(result_jpath)


    # write result into existing log, if there's one
    dest = f"{_CUR_DIR}/../results/{model_name}_result.json"
    if os.path.exists(dest):
        outputs = json.load(open(dest, "r"))
    else:
        outputs = {'name': model_name, 'link': "", "describe": "", "results": {}}
    
    #outputs = {'name': model_name, 'link': "", "describe": "", "results": {}}
    # start writing, will cover existing ds_name(CNN...) results
    for jpath in tqdm(result_jpath):
        # load inference result
        jdata = json.load(open(jpath, "r"))
        print(jdata.keys())
        ds_name = list(jdata.keys())[0]
        
        # Remove query and references
        new_jdata = [dict(id=s['id'], response=s['response']) for s in jdata[ds_name]]
        if ds_name == 'XSum_TC':
            ds_name = "XSum_TC_5k"
        elif ds_name == 'BB_Penguins_in_a_Table_TC':
            ds_name = "PenguinsInTable_TC"
        ds_name = ds_name.replace('TMMLU/', 'TMMLU_')
        outputs['results'][ds_name] = new_jdata
    
    # Save to result folder
    dest = f"{_CUR_DIR}/../results/{model_name}_result.json"
    json.dump(outputs, open(dest, "w"), indent=4, ensure_ascii=False)
    print(f".... save aggregated results to {dest}")





