'''
@TranNhiem 2023/12

This design implementation of OpenCC: convert each sentence of Simplified Chinese to ZH_TW Chinese
 
'''
from concurrent.futures import ThreadPoolExecutor
import json

import opencc
import pandas as pd


# convert the converter into a global variable to achieve a remarkable 600x speedup.
converter = opencc.OpenCC('s2t.json')


# Function to convert text from simplified to traditional Chinese
def convert_simplified_to_traditional(text):
    return converter.convert(text)


file_path = "/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/firefly_instruct/firefly_1M1format.json"


## Dealing with Convert 
def load_input_data(INPUT_TASKS_PATH):
    with open(INPUT_TASKS_PATH, 'r') as json_file:
        content = json_file.read()

    # Split the content by line and remove empty lines
    json_objects = [line for line in content.splitlines() if line.strip()]

    df_list = []  # List to store DataFrames for each JSON object

    # Iterate through the JSON objects, load and convert them into DataFrames
    for index, json_object in enumerate(json_objects):
        try:
            data = json.loads(json_object)
            df = pd.DataFrame([data], index=[index])  # Convert JSON object to DataFrame with index
            df_list.append(df)  # Append DataFrame to list
        except (json.JSONDecodeError, ValueError) as err:
            print(f"Error parsing JSON Object {index + 1}: {err}")

    # Concatenate the DataFrames in the list into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)
    print(f"Complete Loaded {len(final_df)} JSON objects.")
    return final_df

# Load input data as DataFrame
def load_input_data_(INPUT_TASKS_PATH):
    """
    Load input data from a JSON file and return as a DataFrame.

    Args:
        INPUT_TASKS_PATH (str): Path to the input JSON file.

    Returns:
        pd.DataFrame: Input data as a DataFrame.
    """
    with open(INPUT_TASKS_PATH, "rb") as f:
        json_data = json.loads(f.read())
    return pd.DataFrame(json_data)

input_data = load_input_data_(file_path)

# Convert the JSON data in parallel
def convert_text(df, start, end, subset=True, output_file="traditional_chinese.json"):
    if subset:
        subset_df = df.iloc[start:end]#.copy()
    else:
        subset_df = df#.copy()
    
    #original_subset_df = subset_df.copy()  # Make a copy of the original subset DataFrame


    instructions = subset_df['instruction'].tolist()
    inputs = subset_df['input'].tolist()
    outputs = subset_df['output'].tolist()

    with ThreadPoolExecutor() as executor:
        instructions = list(executor.map(convert_simplified_to_traditional, instructions))
        inputs = list(executor.map(convert_simplified_to_traditional, inputs))
        outputs = list(executor.map(convert_simplified_to_traditional, outputs))

    subset_df['instruction'] = instructions
    subset_df['input'] = inputs
    subset_df['output'] = outputs

    # Save the converted DataFrame to a JSON file
    subset_df.to_json(output_file, orient='records', force_ascii=False, indent=4)
    # Save the original and converted DataFrames to separate files
    # original_file = "original_data.json"
    # original_subset_df.to_json(original_file, orient='records', force_ascii=False)

    return subset_df

# Example usage:
start = 0
end = 10
output_file = "/data/rick/Instruction_finetune_dataset/cn_Instruct_dataset/firefly_instruct/ZH_TW_firefly_1M1format.json"
converted_data = convert_text(input_data, start, end,subset=False, output_file=output_file)
