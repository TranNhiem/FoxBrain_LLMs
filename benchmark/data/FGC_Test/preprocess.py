import json
import pandas as pd

path = '0000.parquet'

data = pd.read_parquet(path)
data = data.to_json(orient="records")
with open('data.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False)

data = json.load(open('data.json'))

preprocessed = {}
for inst in data:
    print(inst)
    paragraph_str = inst['DTEXT']
    for qa in inst['QUESTIONS']:
        question_str = qa['QTEXT']
        references = [qa['ANSWER']]
        preprocessed[str(len(preprocessed))] = {
            'paragraph': paragraph_str,
            'question': question_str,
            'references': references
        }

with open('preprocessed_FGC_official_final.json', 'w') as fw:
    json.dump(preprocessed, fw, ensure_ascii=False, indent=4)
