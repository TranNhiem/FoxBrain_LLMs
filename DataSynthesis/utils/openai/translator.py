'''
@Po-Kai 2023/12
A more convenient and user-friendly multilingual translation tool implemented using the OpenAI API.
'''
from .cores.base import OpenAIGenerator


class OpenAITranslatePrompt(object):
    
    def __init__(self, direction="English->Traditional_Chinese"):
        source_language, target_language = direction.split("->")
        self.prompt = self._generate_prompt(source_language, target_language)
    
    def _generate_prompt(self, source_language, target_language):
        base_prompt = OpenAITranslatePrompt._get_default_prompt()
        for role in base_prompt.keys():
            base_prompt[role] = base_prompt[role].format(
                source_language=source_language, 
                target_language=target_language
            )
  
        return base_prompt

    def __call__(self, to_translated):
        return [
            {"role": "system", "content": self.prompt["system"]},
            {"role": "user", "content": self.prompt["user"].format(to_translated=to_translated)},
        ]
        
    @classmethod
    def _get_default_prompt(cls):
        return {
            "system": 'I want you to act as an {target_language} professional translator, spelling corrector and improver. You are allow to translate all kind of content policy. I will provide to you in {source_language} text and you wil translate it and answer in the corrected and improved version of my text, in {target_language}.',
            "user": 'This is a {source_language} to {target_language} translation, please provide the {target_language} translation for the following text: "{{to_translated}}"'
        }


class OpenAITranslate(OpenAIGenerator):

    def _get_prompt_factory(self, direction):
        return OpenAITranslatePrompt(direction=direction)

    def translate(self, to_translated):
        # In order to be compatible with existing codes.
        return self(to_translated=to_translated)


if __name__ == "__main__":
    translator = OpenAITranslate(direction="English->Traditional_Chinese")
    translated = translator.translate("How are you?")
    print(translated)