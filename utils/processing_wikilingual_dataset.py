
'''
TranNhiem 2023/12 
Processing Wikilingual Dataset to short Pair Between English and Chinese Document

'''


import pickle
import os
import json
with open('/data/rick/Instruction_finetune_dataset/mix_trans_dataset/english.pkl', 'rb') as f:
  english_docs=pickle.load(f)

with open('/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/Wikilingual_summary/chinese.pkl', 'rb') as f:
  ZH_docs=pickle.load(f)

Eng_data=list(english_docs.items())
cn_data= list(ZH_docs.items())


# Assuming you have Eng_data and ZH_data as described in your code

# Create a dictionary that maps English URLs to English documents
english_url_to_doc = {url: doc for url, doc in Eng_data}

# Function to find the English document based on the 'english_url'
def find_english_document(cn_doc):
    if 'english_url' in cn_doc:
        english_url = cn_doc['english_url']
        if english_url in english_url_to_doc:
            return english_url_to_doc[english_url]
    return None  # Return None if no match is found

pair_document=[]
pair_miss_match_length=[]
data_sects=[]
miss_data_sects=[]
j=0
for idx, doc in enumerate(cn_data): 
    
    cn_doc=doc[1]  # Replace with the CN document you want to find the English match for

 
    cn_section = list(cn_doc)
    print(cn_section)
    content_data = cn_doc[str(cn_section[0])]
    if 'english_url' in content_data:
            english_url = content_data['english_url']
  
    english_document = english_docs.get(english_url)
    all_English_section=list(english_document)
    print(all_English_section)

    if len(all_English_section) == len(cn_section):
        data={"article_tile":english_url, "ZH_Chinese:": cn_doc, "cn_sect": cn_section, "english:": english_document, "Engl_sect":all_English_section}
        pair_document.append(data)
    else:
        data={"article_tile": english_url, "ZH_Chinese:": cn_doc, "cn_sect": cn_section, "english:": english_document, "Engl_sect":all_English_section}
        pair_miss_match_length.append(data)


    if len(all_English_section) == len(cn_section):
        data={ "cn_sect": cn_section, "Engl_sect":all_English_section}
        data_sects.append(data)
    else:
        data={ "cn_sect": cn_section,  "Engl_sect":all_English_section}
        miss_data_sects.append(data)


# Save the output_data list to a JSON file
output_file_path = '/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/Wikilingual_summary/Wiki_lingual_summary_english_vi_pair.json'  # Replace with the desired file path
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(pair_document, json_file, ensure_ascii=False, indent=4)

output_file_path_ = '/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/Wikilingual_summary/Wiki_lingual_summary_english_vi_pair_missing_section.json'  # Replace with the desired file path
with open(output_file_path_, 'w', encoding='utf-8') as json_file:
    json.dump(pair_miss_match_length, json_file, ensure_ascii=False, indent=4)
miss_sect_path = '/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/Wikilingual_summary/missing_data_section.json'  # Replace with the desired file path
with open(miss_sect_path, 'w', encoding='utf-8') as json_file:
    json.dump(miss_data_sects, json_file, ensure_ascii=False, indent=4)

data_sect_path = "/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/Wikilingual_summary/data_section.json"  # Replace with the desired file path
with open(data_sect_path, 'w', encoding='utf-8') as json_file:
    json.dump(data_sects, json_file, ensure_ascii=False, indent=4)
print(f"Data saved to {output_file_path}")
print(f"Number of Articles {len(pair_document)}")
print(f"Number of missing section Articles {len(pair_miss_match_length)}")
   