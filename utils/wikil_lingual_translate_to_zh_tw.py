import re
import os
import sys
import multiprocessing

# import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import json 
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from data_synth_utils.openai.translator import OpenAITranslate
from data_synth_utils.converter import ZhTWConvert


translator = OpenAITranslate(
    direction="English->Traditional_Chinese",
    proxy_config_path="data_synth_utils/openai/litellm.router.json"
)
converter = ZhTWConvert()
num_workers = 8

def truncate_message(message, max_words=1500):
    words = message.split()
    total_words = len(words)
    # print(total_words)

    if total_words > max_words:
        sections = []  # List to store the sections of the message
        current_section = []  # List to store words for the current section

        word_count = 0  # Counter for the number of words in the current section

        # Iterate over each word in the original message
        for word in words:
            # Add the word to the current section
            current_section.append(word)
            word_count += 1  # Increment the word count

            # If the current word count exceeds the max_words limit
            if word_count >= max_words:
                # If the word ends with a period
                if word.endswith('.'):
                    # Save the current section to the sections list
                    sections.append(' '.join(current_section))
                    # Start a new section
                    current_section = []
                    word_count = 0  # Reset the word count
                else:
                    # Find the last period in the current section
                    section_str = ' '.join(current_section)
                    last_period_index = section_str.rfind('.')
                    # Split the section at the last period
                    sections.append(
                        section_str[:last_period_index + 1].strip())
                    # Start a new section with the remaining words
                    current_section = section_str[last_period_index + 2:].split()
                    word_count = len(current_section)

        # Append the last section if it's not empty
        if current_section:
            sections.append(' '.join(current_section))

        return sections
    else:
        # If the total number of words is less than or equal to max_words
        return [message]

# # Test with a long message

def clean_text(text):

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before and after punctuation marks
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    # Trim leading and trailing spaces
    text = text.strip()
    return text

def clean_message(message):
    # Replace multiple consecutive newlines with a single space
    message = re.sub(r'\n+', ' ', message)
    # Replace spaces followed by a period with just a period
    message = message.replace(" .", ".").strip()
    return message

def translate_section(section, direction): 
    ## Implement OpenAI Section here
    try:
        translated = translator.translate(section)
        translated = converter.convert(translated)
    except Exception as e:  # This can catch all exceptions including OOM.
        print(f"Error occurred with error: {str(e)}")
        # You can also append a placeholder or log the error for further inspection
        translated = "[Translation Failed]"

    return translated


def translation(message, direction="English->Traditional_Chinese"):
    
    max_words = 1024
    message = clean_message(message)
    message = clean_text(message)
    sections = truncate_message(message, max_words=max_words)
    translated_sections = []

    # Partition sections into chunks
    section_chunks = sections
    for section in sections:
        translated = translate_section(section, direction)
        translated_sections.append(translated)
    return ' '.join(translated_sections)

def parallel_translation(each_section):
    tasks_process = each_section["task_process"]
    tasks_process_translated = {}

    for task, content in tasks_process.items():
        if isinstance(content, dict):
            input_text = content.get('input', '')
            output_text = content.get('output', '')
            translated_input = translation(
                input_text, 
                direction='English->Traditional_Chinese'
            )
            translated_output = translation(
                output_text, 
                direction='English->Traditional_Chinese'
            )

            tasks_process_translated[task] = {
                "input": translated_input, "output": translated_output
            }

        elif isinstance(content, list):
            translated_pairs = []
            for pair in content:
                translated_pair = {}
                for key in pair:
                    text = pair[key]
                    translated_text = translation(
                        text, 
                        direction='English->Traditional_Chinese'
                    )
                    # Check word count condition
                    translated_pair[key] = translated_text
                translated_pairs.append(translated_pair)

            tasks_process_translated[task] = translated_pairs
    # Add the translated tasks to the new structure
    return {
        "title": each_section['title'], 
        "task_process_translate": tasks_process_translated
    }


# Function to save data
def save_intermediate_data(data, iteration, base_path):
    data_file = f"{base_path}/tmp_intermediate_{iteration}.json"
    with open(data_file, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Load the data
file_path = "./Wikilingual_task_v2_final.json"
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Iterating through each conversation and translating
new_data = []
for idx, conversation in tqdm(enumerate(conversations_data), total=len(conversations_data)):
    if  idx <= 4000:
        process_data = conversation["Processed_generated_instruc"]
        En_sec=conversation['Engl_sect']

        with ThreadPool(num_workers) as pool:
            process_data_translate = pool.map(parallel_translation, process_data)
        data = {
            "Engl_sect":En_sec,
            "Processed_generated_instruc_translate":process_data_translate
        }
        new_data.append(data)
        
        # Every 1000 iterations, save the current state
        if idx % 50 == 0:
            print(f"Saving intermediate results at iteration {idx}")
            save_intermediate_data(new_data, idx, "./tmp")


# Save the updated data
save_file_path = "./Wikilingual_task_v2_final_translated_4k_6k5.json"
with open(save_file_path, 'w', encoding="utf-8") as outfile:
    json.dump(new_data, outfile,ensure_ascii=False, indent=4)