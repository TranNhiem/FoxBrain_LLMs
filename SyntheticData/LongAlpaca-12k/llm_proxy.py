'''
@Po-Kai 2023/12
This code is designed to proxy multiple Azure API keys and perform load balancing.
'''
import json

from litellm import Router


def get_router(config_path):
    with open(config_path, "r") as r:
        config = json.load(r)
    router = Router(model_list=config)
    return router