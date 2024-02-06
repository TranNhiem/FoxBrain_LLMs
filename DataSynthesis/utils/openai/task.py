'''
@Po-Kai 2023/12
A more convenient and user-friendly tool for using the OpenAI API.
'''
import json

from .cores.base import OpenAIGenerator


class OpenAIJSONPrompt(object):
    
    def __init__(self, prompt_json_path):
        self.prompt = self._generate_prompt(prompt_json_path)
    
    def _generate_prompt(self, json_path):
        with open(json_path, "r") as r:
            prompt_dict = json.load(r)
            assert "system" in prompt_dict, 'Missing key: "system"'
            assert "user" in prompt_dict, 'Missing key: "user"'
        return prompt_dict

    def __call__(self, **prompt_kwargs):
        return [
            {"role": "system", "content": self.prompt["system"]},
            {"role": "user", "content": self.prompt["user"].format(**prompt_kwargs)},
        ]


class OpenAIJSONGenerator(OpenAIGenerator):

    def __init__(self, *arg, prompt_json_path, **kwargs):
        self.prompt_json_path = prompt_json_path
        super().__init__(*arg, **kwargs)
        
    def _get_prompt_factory(self):
        return OpenAIJSONPrompt(self.prompt_json_path)

    def generate(self, **prompt_kwargs):
        return self(**prompt_kwargs)