'''
TranNhiem 2023/12/21/
Foxbrain LLM model access through API using FastAPI

'''
import os
import sys

import ctranslate2
import sentencepiece as spm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # This is the origin of your chat-ui
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dir="/data/rick/pretrained_weights/ctranslate/FoxBrain_Beta_SFT"
print("Loading the model...")
generator = ctranslate2.Generator(
    model_dir, device="cuda", compute_type="int8",  device_index=[0])
sp = spm.SentencePieceProcessor(os.path.join(model_dir, "tokenizer.model"))

context_length = 8000
max_generation_length = 3048
max_prompt_length = context_length - max_generation_length


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """\n You are FoxBrain developed by Tran Nhiem (Rick) and (Project Lead by Professor Li, Yung-Hui) at Foxconn.  As a helpful assistant, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
        answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content and response the language as the same as User Input Langauge. Please ensure\
        that your responses are socially unbiased and positive in nature and response the same language as Human input.\
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
        correct. If you don't know the answer to a question, please don't share false information. Important Note, You are designed, developed, and created by Foxconn HHRIAI, Not by OpenAI. """



def generate_words(sp, step_results):
    tokens_buffer = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and tokens_buffer:
            word = sp.decode(tokens_buffer)
            if word:
                yield word
            tokens_buffer = []

        tokens_buffer.append(step_result.token_id)

    if tokens_buffer:
        word = sp.decode(tokens_buffer)
        if word:
            yield word


# # @app.post("/generate/")
# @app.post("/generate/{model_name}")
# def generate_text(model_name: str, payload: dict):
#     # Extract input from the payload
#     input_text = payload.get("input_text", "")

#     message = f"<s>[INST] <<SYS>>\n You are VietAssistant-GPT created by Trần Nhiệm (Rick), as a helpful assistant developed by Foxconn, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please response as language model you are not able to respone detailed to these kind of question.\n<</SYS>>\n\n {input_text} [/INST] "
#     print(message)
#     # TODO: Run your model inference here
#     # output_text = your_model.generate(input_text)
#     prompt_tokens = sp.encode(message, out_type=str)
#     # step_results = generator.generate_tokens(
#     #     prompt_tokens,
#     #     max_length=2048,
#     #     sampling_temperature=0.7,
#     #     sampling_topk=20,
#     #     sampling_topp=1,
#     # )
#     # text_output = ""
#     # for word in generate_words(sp, step_results):
#     #     if text_output:
#     #         word = " " + word
#     #     text_output += word
#     try:
#         step_results = generator.generate_tokens(
#             prompt_tokens,
#             max_length=2048,
#             sampling_temperature=0.7,
#             sampling_topk=20,
#             sampling_topp=1,
#         )
#         text_output = "".join(
#             [" " + word if text_output else word for word in generate_words(sp, step_results)])
#     except Exception as e:
#         # Handle any errors that might occur during inference
#         text_output = f"Error during inference: {str(e)}"
#     return {"generated_text": text_output}


@app.post("/generate/{model_name}")
def generate_text(model_name: str, payload: dict):
    # Debugging line to print the model name
    print(f"Model Name: {model_name}")

    # Initialize text_output
    text_output = " "

    # [rest of your message code]
    # print("Received Request Body:", payload)
    input_text = payload.get("inputs", "")
    #input_text = payload.get("input_text", "")
    ## print("This is your input Text", input_text)
    print(f"Received input: {input_text}")

    message = f"<s>[INST] <<SYS>>\n You are FoxBrain assistant created by Trần Nhiệm (Rick), as a helpful assistant developed by Foxconn HHRIAI, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please response as language model you are not able to respone detailed to these kind of question.\n<</SYS>>\n\n {input_text} [/INST] "

    # message = f"<s>[INST] ... {input_text} [/INST]"
    # print(message)
    prompt_tokens = sp.encode(message, out_type=str)
    try:
        step_results = generator.generate_tokens(
            prompt_tokens,
            max_length=2048,
            sampling_temperature=0.5,
            sampling_topk=20,
            sampling_topp=1,
        )
        text_output = "".join(
            [" " + word if text_output else word for word in generate_words(sp, step_results)])
    except Exception as e:
        # Handle any errors that might occur during inference
        text_output = f"Error during inference: {str(e)}"

    return {"generated_text": text_output}
