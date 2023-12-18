#!/bin/bash

# prpare data: make chunks
python make_chunks.py --max_tokens 1536

# tackle data: translate by litellm(gpt3.5)
python translate.py --testing=False

# consider the translated sentence still have some simplified Chinese,
# we need to further convert the result into traditional Chinese.
python zhtw_convert.py