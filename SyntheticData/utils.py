'''
@Po-Kai 2023/12

This is for text preprocessing.
1. chunk text to max length, it helps us adhere to the maximum length limit of GPT without losing any text.
'''
import nltk
import tiktoken


def count_tokens(text, encoding_name="cl100k_base"):
    ### Returns the number of tokens in a text string.
    # encoding_name:
    #     cl100k_base: gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    #     p50k_base: Codex models, text-davinci-002, text-davinci-003
    #     r50k_base(or gpt2):	GPT-3 models like davinci
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def chunk_text_to_max_tokens(text, max_tokens=512):
    # Count tokens in the text
    num_of_tokens = count_tokens(text)

    # Truncate text if it exceeds the maximum tokens
    if num_of_tokens > max_tokens:
        # Split the text into sentences
        sents = nltk.sent_tokenize(text)  
        nft_sents = [count_tokens(sent) for sent in sents]
        chunks = []
        curr_size = 0
        curr_sents = []
        for nft_sent, sent in zip(nft_sents, sents):
            if (curr_size+nft_sent) <= max_tokens:
                curr_size += nft_sent
                curr_sents.append(sent)
            else:
                chunk = " ".join(curr_sents)
                chunks.append(chunk)
                # assume all sentence of length is not over the length limit.
                curr_size = nft_sent
                curr_sents = [sent]

        if curr_sents:
            chunk = " ".join(curr_sents)
            chunks.append(chunk)
    else:
        chunks = [text]

    return chunks