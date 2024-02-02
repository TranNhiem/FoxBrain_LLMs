'''
@Po-Kai 2023/12
A more convenient and user-friendly generator tool implemented using the OpenAI API.
'''
from .llm_proxy import get_router

        
class OpenAIGenerator(object):

    def __init__(self, proxy_config_path, *arg, **kwargs):
        self.prompt_factory = self._get_prompt_factory(*arg, **kwargs)
        self.router = get_router(proxy_config_path)
        self.model_name = "gpt-35-turbo"

    def __call__(self, **prompt_kwargs):
        messages = self.prompt_factory(**prompt_kwargs)

        result_text = "|LOST|"
        try:
            response = self.router.completion(model=self.model_name, messages=messages)
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

    def set_model_name(self, model_name):
        self.model_name = model_name
    def _get_prompt_factory(self, *arg, **kwargs):
        raise NotImplementedError