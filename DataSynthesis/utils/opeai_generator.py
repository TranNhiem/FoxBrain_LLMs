'''
@Po-Kai 2023/12
A more convenient and user-friendly generator tool implemented using the OpenAI API.
'''
from .llm_proxy import get_router


class OpenAIGenerator(object):

    def __init__(
        self,
        prompt_factory,
        proxy_config_path="./litellm.router.json"
    ):
        self.prompt_factory = prompt_factory
        self.router = get_router(proxy_config_path)

    def __call__(self, prompt_args, model_name="gpt-35-turbo"):
        messages = self.prompt_factory(**prompt_args)

        result_text = "|LOST|"
        try:
            response = self.router.completion(model=model_name, messages=messages)
        except Exception as e:
            print(f"Got some error\nErr message: {e}")
            result_text = "|ERR|"
        else: 
            choices = response.get("choices")
            if choices and len(choices) > 0:
                message = choices[0].get("message")
                if message:
                    content = message.get("content")
                    if content:
                        result_text = content.strip() 

        return result_text